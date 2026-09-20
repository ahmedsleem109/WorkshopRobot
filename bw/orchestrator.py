"""T9 -- the orchestrator: a hand-written state machine over the robot's skills.

    command  ->  parse (tool, destination)  ->  NAV_RACK -> LOCATE -> GRASP -> [STOW] -> NAV_DEST
             ->  PLACE | HANDOFF  ->  DONE

Skills (each returns, the machine decides):
    navigate_to(station)   Layer 3 walking policy via bw/locomotion/navigate.walk_to
    locate(description)    Layer 1: vlm.point on 3 wrist views -> depth -> 3D (bw/perception/locate)
    grasp(tool)            Layer 2: the SmolVLA policy ("vla") or the scripted demonstrator
    place(table)           Layer 2, same two backends
    handoff()              lower the tool into the human's tray and let go
    stow_arm()             Cartesian pull-in before crossing the walkway step
    ask_human(question)    a callable; in simulation a scripted "human" (bw/eval harness)

Every transition is taken on a GATE the robot can actually observe (no ground truth): base
reached the station (walk_to's own estimate), locate returned a point, gripper_state() ==
"holding" (finger travel + bilateral pad contact), the gripper is empty after a release. Each
state has a bounded number of attempts and an on_failure transition; the machine never loops.
Scoring (did the RIGHT tool end in the RIGHT place) is bw/task/spec.py's job, outside this file.

Recovery paths (T10):
    1 tool not in the rack  -> locate None on the scan -> floor sweep with the head camera ->
                               found: ask the human to put it back / not found: report
    2 dropped mid-carry     -> gripper_state leaves "holding" during a walk -> ask the human to
                               return it to the rack -> walk back -> locate -> grasp again
    4 ambiguous "wrench"    -> both wrenches are candidates -> ask_human which one
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from bw.locomotion.navigate import walk_to
from bw.manip.ik import ArmIK
from bw.manip import scripted_grasp as sg
from bw.manip import scripted_place as sp
from bw.sim.workshop import PLACE_STATION, RACK_STATION, TABLES
from bw.task.language import TOOL_SYNONYMS

DETOUR = 0.9                      # m: how far off the straight line a replan steps (T10 #3)
VIA_STEP = (0.3, 0.0, np.pi)      # past the walkway end; measured: 17/20 reach the human (x=-0.3, crossing at speed: 16/24)
DESC = {"wrench_10mm": "10mm wrench", "wrench_13mm": "13mm wrench", "pliers": "pliers",
        "tape_roll": "roll of tape", "screwdriver": "screwdriver"}
HUMAN_WORDS = ("bring me", "hand me", "give me", "i need", "fetch", "can you get me",
               "get me", "bring the")


@dataclass
class Parsed:
    tool: str | None                 # None: ambiguous or unknown
    candidates: list[str]
    dest: str                        # "human" | "table_a" | "table_b"


def parse(command: str) -> Parsed:
    c = command.lower()
    hits = []
    for tool, syns in TOOL_SYNONYMS.items():
        for s in syns:
            if re.search(r"\b" + re.escape(s.lower()) + r"\b", c):
                hits.append((len(s), tool))
    tools = sorted({t for _, t in hits}, key=lambda t: -max(n for n, u in hits if u == t))
    if not tools and re.search(r"\b(wrench|spanner)\b", c):
        tools = ["wrench_10mm", "wrench_13mm"]
        cand, tool = tools, None
    elif re.search(r"\b(wrench|spanner)\b", c) and not re.search(r"\d|small|big|ten|thirteen", c):
        cand, tool = ["wrench_10mm", "wrench_13mm"], None
    else:
        cand, tool = tools[:1], (tools[0] if tools else None)
    dest = "human"
    for table, info in TABLES.items():
        for a in info["aliases"]:
            if a in c and not (a == "the other table" and table == "table_a"):
                dest = table
    if dest == "human" and re.search(r"\b(put|place|move|set|carry|transfer|goes)\b", c):
        dest = "table_b"                 # "put the pliers on the table" -> the other table
    return Parsed(tool, cand, dest)


@dataclass
class Result:
    success: bool
    reason: str
    tool: str | None
    dest: str
    log: list = field(default_factory=list)
    asked: list = field(default_factory=list)
    retargets: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self, sim, backend: str = "scripted", grounding: str = "vlm", human=None,
                 max_grasp: int = 2, verbose: bool = False):
        assert backend in ("scripted", "vla") and grounding in ("vlm", "oracle")
        self.sim, self.backend, self.grounding = sim, backend, grounding
        self.ik = ArmIK(sim.m)
        self.human = human or (lambda q, ctx: None)
        self.max_grasp = max_grasp
        self.verbose = verbose
        self.rng = np.random.default_rng(0)
        self.on_walk_tick = None        # scenario hook: called during walks (drop injection)
        self.interrupt = None           # scenario hook: () -> a new command, or None (T10 #5)
        self._pick = None               # the last grasp's plan (site / R / approach)

    # ----------------------------------------------------------------- skills
    def _log(self, res: Result, state: str, **kw):
        e = {"t": round(self.sim.time, 1), "state": state, **kw}
        res.log.append(e)
        if self.verbose:
            print("   ", e, flush=True)

    def navigate_to_replan(self, station, res: Result, name: str, tries: int = 2) -> bool:
        """navigate_to, plus ONE detour per side if the walk stalls without falling (T10 #3).

        The robot has no obstacle perception: a box on the route shows up as a walk that runs out
        of time with the base still far from the station. The recovery is geometric -- step
        sideways off the straight line and approach from there, first one side, then the other.
        """
        start = self.sim.d.qpos[:2].copy()          # where the blocked walk began
        if self.navigate_to(station, res, name):
            return True
        for k in range(tries):
            if not self.sim.loco.is_stable():
                return False                      # it fell: not an obstacle
            d = np.array(station[:2]) - start
            n = float(np.linalg.norm(d))
            if n < 1e-3:
                return False
            d = d / n
            side = 1.0 if k % 2 == 0 else -1.0
            # offset from the ORIGINAL line, not from the pose the robot is wedged in: measured,
            # a detour planned from the stuck pose routed straight back into the box
            way = start + 0.5 * n * d + np.array([-d[1], d[0]]) * side * DETOUR
            way = (float(np.clip(way[0], 1.5, 4.2)), float(np.clip(way[1], -1.4, 1.4)),
                   float(np.arctan2(d[1], d[0])))
            self._log(res, "REPLAN", attempt=k + 1, side="left" if side > 0 else "right",
                      via=[round(v, 2) for v in way[:2]])
            self.back_off(0.6)          # a blocked walk ends WEDGED against the box; walk_to's
                                        # own 0.4 m backup is not enough to free the gait
            if self.navigate_to(way, res, f"{name}_detour{k}") and                     self.navigate_to(station, res, name):
                return True
        return False

    def back_off(self, dist: float = 1.0, timeout_s: float = 8.0):
        """Straight back at the policy's back-up speed until `dist` or `timeout_s`."""
        loco = self.sim.loco
        x0 = self.sim.d.qpos[:2].copy()
        t0 = self.sim.time
        while (float(np.linalg.norm(self.sim.d.qpos[:2] - x0)) < dist
               and self.sim.time - t0 < timeout_s and loco.is_stable()
               and self.sim.d.qpos[0] < 4.05):        # do not back into the bench (x 4.45)
            loco.set_velocity(-0.25, 0.0, 0.0)
            self.sim.physics_step(loco.n_substeps)
        loco.set_velocity(0.0, 0.0, 0.0)
        self.sim.settle(0.4)

    def navigate_to(self, station, res: Result, name: str) -> bool:
        t0 = time.time()
        hook = self.on_walk_tick
        # the route to the human needs a big LEFT turn at the rack, and the mirrored left turn
        # fell 3/3 with a tool held (session 5): turn clockwise the long way round instead
        w = walk_to(self.sim, station, turn_sign=-1 if name in ("via_step", "human") else None,
                    on_phase=(lambda ph: hook(self, ph)) if hook else None)
        res.timings[name] = res.timings.get(name, 0.0) + time.time() - t0
        self._log(res, "NAV", to=name, ok=w["success"], fell=w["fell"], err_mm=w["err_xy_mm"])
        return bool(w["success"])

    def _point_fn(self):
        if self.grounding == "oracle":
            return None
        from bw.perception.vlm import point
        return point

    def locate(self, tool: str, res: Result):
        from bw.perception.locate import capture_views, locate_in_views
        from bw.perception.vlm import point as vlm_point
        sim = self.sim
        t0 = time.time()
        views = capture_views(sim)
        fn = self._oracle_point(tool, views) if self.grounding == "oracle" else vlm_point
        loc = locate_in_views(views, DESC[tool], fn, sim.d.qpos[0:3].copy(), sim.d.qpos[3:7].copy())
        res.timings["locate"] = res.timings.get("locate", 0.0) + time.time() - t0
        self._log(res, "LOCATE", tool=tool, found=loc is not None,
                  world=None if loc is None else np.round(loc.world, 3).tolist())
        return loc

    def _oracle_point(self, tool: str, views):
        """Ground-truth pointing (ablation only): project the tool's grasp point (the visible
        part, above the rack plates) into THAT view's image; None if outside the frame."""
        by_img = {id(v.rgb): v for v in views}
        p = sg.grasp_point(self.sim, tool)

        def fn(rgb, _desc):
            v = by_img[id(rgb)]
            h, w = rgb.shape[:2]
            pc = v.cam_mat.T @ (p - v.cam_pos)          # camera frame: -z forward (MuJoCo)
            if pc[2] > -0.05:
                return None
            u = v.K[0, 0] * pc[0] / -pc[2] + v.K[0, 2]
            vv = -v.K[1, 1] * pc[1] / -pc[2] + v.K[1, 2]
            if not (0 <= u < w and 0 <= vv < h):
                return None
            if v.depth[int(vv), int(u)] < -pc[2] - 0.03:  # something nearer in front of it
                return None
            return (float(u), float(vv))
        return fn

    def identify(self, loc, candidates) -> str | None:
        """Which tool body did grounding point at? The scripted backend then grasps THAT one --
        so a grounding error shows up as a wrong-object failure, as it would on hardware."""
        best, bd = None, 0.06              # rack slots are 0.12 m apart: match in the plane
        for n in self.sim.tool_body:
            d = float(np.linalg.norm(sg.grasp_point(self.sim, n)[:2] - loc.world[:2]))
            if d < bd:
                best, bd = n, d
        return best

    def grasp(self, tool: str, instruction: str, res: Result) -> bool:
        t0 = time.time()
        if self.backend == "scripted":
            g = sg.run_grasp(self.sim, self.ik, tool, self.rng)
            if g.get("site"):
                self._pick = g                # where it came from, for return_to_rack
        else:
            from bw.policy.vla import run_skill
            self.sim.lock_stance()
            run_skill(self.sim, instruction, max_s=16.0)
            self.sim.settle(1.0)
            g = {}
        holding = self.sim.gripper_state() == "holding"
        res.timings["grasp"] = res.timings.get("grasp", 0.0) + time.time() - t0
        self._log(res, "GRASP", tool=tool, holding=holding, scripted=g.get("reason"))
        return holding

    def place(self, table: str, tool: str, instruction: str, res: Result) -> bool:
        t0 = time.time()
        if self.backend == "scripted":
            sp.run_place(self.sim, self.ik, tool, table, self.rng)
        else:
            from bw.policy.vla import run_skill
            self.sim.lock_stance()
            run_skill(self.sim, instruction, max_s=14.0)
            self.sim.settle(2.0)
        empty = self.sim.gripper_state() != "holding"
        res.timings["place"] = res.timings.get("place", 0.0) + time.time() - t0
        self._log(res, "PLACE", table=table, released=empty)
        return empty

    def handoff(self, tool: str, res: Result) -> bool:
        r = sp.run_handoff(self.sim, self.ik, tool, self.rng)
        self._log(res, "HANDOFF", **r)
        return self.sim.gripper_state() != "holding"

    def ask_human(self, question: str, res: Result, **ctx):
        a = self.human(question, {"orch": self, **ctx})
        res.asked.append((question, a))
        self._log(res, "ASK_HUMAN", q=question, a=a)
        return a

    def floor_sweep(self, tool: str, res: Result):
        """Head-camera look for a tool that is not in the rack (recovery 1). Point only -- the
        arm cannot reach the floor, so a find is reported to the human, not grasped."""
        from bw.perception.locate import View, locate_in_views
        sim = self.sim
        views = []
        for _ in range(1):
            cpos, cmat = sim.camera_pose("head")
            views.append(View(rgb=sim.render("head", size=(512, 512)),
                              depth=sim.render("head", depth=True, size=(512, 512)),
                              K=sim.camera_intrinsics("head", size=(512, 512)),
                              cam_pos=cpos, cam_mat=cmat))
        fn = None if self.grounding == "oracle" else self._point_fn()
        if fn is None:
            loc = None
            p = sim.gt_tool_pos(tool)
            if p[2] < 0.3:
                class L:
                    world = p
                loc = L()
        else:
            loc = locate_in_views(views, DESC[tool], fn, sim.d.qpos[:3].copy(), sim.d.qpos[3:7].copy())
        self._log(res, "FLOOR_SWEEP", found=loc is not None)
        return loc

    def _new_command(self, res: Result, tool: str | None):
        """T10 #5: a new command typed mid-task. Only a change of TARGET counts."""
        if self.interrupt is None:
            return None
        c = self.interrupt()
        if not c:
            return None
        q = parse(c)
        if q.tool is None or q.tool == tool:
            return None
        self._log(res, "RETARGET", command=c, tool=q.tool, dest=q.dest)
        return q

    # ----------------------------------------------------------------- machine
    def run(self, command: str) -> Result:
        sim = self.sim
        p = parse(command)
        res = Result(False, "", p.tool, p.dest)
        self._log(res, "PARSE", command=command, tool=p.tool, cand=p.candidates, dest=p.dest)
        tool = p.tool
        if tool is None:
            if not p.candidates:
                self.ask_human(f"I don't know which tool '{command}' means.", res)
                res.reason = "unknown_tool"
                return res
            a = self.ask_human("Which one: " + " or ".join(DESC[c] for c in p.candidates) + "?",
                               res, candidates=p.candidates)
            tool = parse(a or "").tool if a else None
            if tool is None:
                res.reason = "ambiguous_unresolved"
                return res
        res.tool = tool
        instr_pick = f"pick up the {DESC[tool]}"
        dest_name = {"human": None, "table_a": "the workbench", "table_b": "the side table"}[p.dest]
        instr_place = f"put the {DESC[tool]} on {dest_name}" if dest_name else None

        state, grasps, returns = "NAV_RACK", 0, 0
        while True:
            if state == "NAV_RACK":
                at_rack = np.linalg.norm(sim.d.qpos[:2] - np.array(RACK_STATION[:2])) < 0.08
                if not at_rack and not self.navigate_to(RACK_STATION, res, "rack"):
                    res.reason = "nav_rack_failed"
                    return res
                state = "LOCATE"
            elif state == "LOCATE":
                sim.lock_stance()
                loc = self.locate(tool, res)
                if loc is None:
                    state = "RECOVER_MISSING"
                    continue
                target = self.identify(loc, [tool])
                if target is None:
                    self._log(res, "IDENTIFY", target=None)
                    state = "RECOVER_MISSING"
                    continue
                state = "GRASP"
            elif state == "RECOVER_MISSING":
                f = self.floor_sweep(tool, res)
                if f is not None:
                    a = self.ask_human(f"The {DESC[tool]} is on the floor and I can't reach it. "
                                       "Could you put it back in the rack?", res, tool=tool)
                    if a == "done" and returns < 1:
                        returns += 1
                        state = "LOCATE"
                        continue
                    res.reason = "missing_on_floor"
                else:
                    self.ask_human(f"I can't find the {DESC[tool]}.", res)
                    res.reason = "missing"
                return res
            elif state == "GRASP":
                grasps += 1
                ok = self.grasp(target if self.backend == "scripted" else tool, instr_pick, res)
                if ok:
                    state = "TO_DEST"
                elif grasps < self.max_grasp:
                    sim.move_arm(np.concatenate([sg.SCAN_Q[:6], [sg.GRIPPER_OPEN]]), 1.5)
                    state = "LOCATE"
                else:
                    res.reason = "grasp_failed"
                    return res
            elif state == "TO_DEST":
                q = self._new_command(res, tool)
                if q is not None:
                    # holding the wrong tool now: set it down in table A's zone (the only
                    # release the skills can do besides the handoff), then start over
                    res.retargets.append((tool, q.tool))
                    if sim.gripper_state() == "holding":
                        # put the unwanted tool BACK IN ITS SLOT, from where the robot stands.
                        # Setting it down in table A's zone instead needs the table A station
                        # (the zone is out of reach from the rack station) and the 0.6 m hop
                        # back is beyond the policy's sidestep -- measured, session 5.
                        if self.backend == "scripted" and self._pick is not None:
                            r = sg.return_to_rack(sim, self.ik, tool, self._pick, self.rng)
                            self._log(res, "RETURN_TO_RACK", **r)
                            if not r["success"]:
                                res.reason = "retarget_release_failed"
                                return res
                        else:
                            res.reason = "retarget_release_failed"
                            return res
                    p, tool = q, q.tool
                    res.tool, res.dest = tool, q.dest
                    instr_pick = f"pick up the {DESC[tool]}"
                    dest_name = {"human": None, "table_a": "the workbench",
                                 "table_b": "the side table"}[p.dest]
                    instr_place = f"put the {DESC[tool]} on {dest_name}" if dest_name else None
                    grasps = 0
                    state = "NAV_RACK"
                    continue
                if p.dest == "human":
                    sg.stow_arm(sim, self.ik)
                    ok = (self.navigate_to(VIA_STEP, res, "via_step")
                          and self._still_holding(res)
                          and self.navigate_to(sp.handoff_station(), res, "human"))
                else:
                    ok = self.navigate_to_replan(PLACE_STATION[p.dest], res, p.dest)
                # a FALL is not a drop: check the base first, or a fall that also opened the
                # jaws is reported as a dropped tool (3 trials in the session-5 drop suite)
                if not ok and not self.sim.loco.is_stable():
                    res.reason = "nav_dest_failed"
                    return res
                if not self._still_holding(res):
                    state = "RECOVER_DROP"
                    continue
                if not ok:
                    res.reason = "nav_dest_failed"
                    return res
                sim.settle(0.3)
                state = "DELIVER"
            elif state == "RECOVER_DROP":
                a = self.ask_human(f"I dropped the {DESC[tool]}. Could you put it back in the "
                                   "rack?", res, tool=tool)
                if a != "done" or returns >= 1:
                    res.reason = "dropped"
                    return res
                returns += 1
                grasps = 0
                state = "NAV_RACK"
            elif state == "DELIVER":
                ok = (self.handoff(tool, res) if p.dest == "human"
                      else self.place(p.dest, tool, instr_place, res))
                res.success = bool(ok)
                res.reason = "delivered" if ok else "release_failed"
                return res

    def _still_holding(self, res: Result) -> bool:
        h = self.sim.gripper_state() == "holding"
        if not h:
            self._log(res, "GATE", holding=False)
        return h
