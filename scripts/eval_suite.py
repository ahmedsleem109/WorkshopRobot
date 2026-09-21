"""T11 -- the end-to-end evaluation suite: the orchestrator (bw/orchestrator.py) on full tasks.

    render_venv\\Scripts\\python.exe scripts\\eval_suite.py [--suite nominal|transfer|missing|drop|ambiguous|all]
        [--n N] [--backend scripted|vla] [--grounding vlm|oracle] [--out F.json]

Every trial starts with the robot at the rack station, all five tools in the rack unless the
scenario says otherwise, and a typed command. Success is judged by bw/task/spec.py-style ground
truth AFTER the episode, never by the orchestrator's own verdict:
    human delivery  the NAMED tool rests in the human's handoff tray
    transfer        the NAMED tool rests in the NAMED table's zone
    obstacle (S3)   a box appears on the route to table B: success = the tool still reaches the
                    zone (the robot detours around it)
    retarget (S5)   a new command arrives mid-task: success = the SECOND tool is delivered and
                    the first is not in the tray (the robot sets it down in table A's zone)
    missing (S1)    the tool is absent: success = the robot reports it and delivers NOTHING
    drop (S2)       the gripper is opened mid-walk; the scripted human puts the tool back in the
                    rack when asked; success = it is still delivered
    ambiguous (S4)  "bring me the wrench" with both wrenches present; the human answers; success
                    = the answered wrench is delivered and the robot ASKED before acting
Each failure is tagged with its cause (the orchestrator's reason, or wrong_object when the
robot delivered a different tool): the failure-cause table is the headline result.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.manip.scripted_grasp import SCAN_Q
from bw.manip.scripted_place import in_handoff_tray
from bw.orchestrator import DESC, Orchestrator
from bw.sim.workshop import (GRASP_TOOLS, PLACE_ZONE, RACK_STATION, TABLES, TOOL_NAMES,
                             in_place_zone, place_zone_z)
from bw.sim.workshop_sim import GRIPPER_OPEN, WorkshopSim

WORDS = {"wrench_10mm": "10mm wrench", "wrench_13mm": "13mm wrench", "pliers": "pliers",
         "tape_roll": "tape", "screwdriver": "screwdriver"}


class Human:
    """The scripted person the robot can ask. Remembers where each tool stood at reset so it
    can put a dropped / missing one back in the rack when asked."""

    def __init__(self, sim, answer_tool=None):
        self.sim, self.answer_tool = sim, answer_tool
        self.home = {n: sim.d.qpos[sim.tool_qadr[n]:sim.tool_qadr[n] + 7].copy() for n in TOOL_NAMES}
        self.asked = 0

    def __call__(self, question, ctx):
        self.asked += 1
        q = question.lower()
        if q.startswith("which one") and self.answer_tool:
            return f"the {WORDS[self.answer_tool]}"
        if "put it back in the rack" in q:
            tool = ctx.get("tool")
            if tool and self.home.get(tool) is not None and self.home[tool][2] > 0.5:
                qa, da = self.sim.tool_qadr[tool], self.sim.tool_dadr[tool]
                self.sim.d.qpos[qa:qa + 7] = self.home[tool]
                self.sim.d.qvel[da:da + 6] = 0
                self.sim.settle(1.0)
                return "done"
            return "no"
        return None


def trial(sim, orch, suite, seed):
    tool = GRASP_TOOLS[seed % len(GRASP_TOOLS)]
    rng = np.random.default_rng(50_000 + 1000 * seed + TOOL_NAMES.index(tool))
    present = set(TOOL_NAMES)
    if suite == "missing":
        present.discard(tool)
    sim.reset(rng, target=None if suite == "missing" else tool, arm_q=SCAN_Q,
              base_pose=RACK_STATION, tools_present=present)
    human = Human(sim)
    orch.human = human
    orch.on_walk_tick = None
    orch.interrupt = None
    # Reseed the ORCHESTRATOR too, not just the scene. Its rng drives every scripted skill's
    # randomised move durations, and it used to carry on from wherever the previous trial left
    # it -- so a trial's outcome depended on which trials ran before it in the same process and
    # a seed did not identify a trial. Measured 2026-09-20, same code and seeds: `--suite drop`
    # alone scored 9/10 while `--suite transfer,drop` scored drop 6/10, and retarget seed 5
    # failed in a batch but passed on its own. Now every trial is reproducible in isolation.
    orch.rng = np.random.default_rng(70_000 + 1000 * seed + TOOL_NAMES.index(tool))
    first = None
    dest = "human"
    if suite in ("nominal", "missing", "drop"):
        cmd = f"bring me the {WORDS[tool]}"
    elif suite == "transfer":
        dest = ("table_a", "table_b")[seed % 2]
        alias = TABLES[dest]["aliases"][seed % 3]
        cmd = f"put the {WORDS[tool]} on {alias}"
    elif suite == "retarget":
        # T10 #5: the person changes their mind after the robot has the first tool
        other = {"wrench_10mm": "wrench_13mm", "wrench_13mm": "wrench_10mm"}.get(
            tool, GRASP_TOOLS[(seed + 2) % len(GRASP_TOOLS)])
        cmd = f"bring me the {WORDS[tool]}"
        fired = {"done": False}

        def interrupt():
            if fired["done"]:
                return None
            fired["done"] = True
            return f"actually, the {WORDS[other]}"
        orch.interrupt = interrupt
        first, tool = tool, other
    elif suite == "ambiguous":
        tool = ("wrench_10mm", "wrench_13mm")[seed % 2]
        human.answer_tool = tool
        cmd = "bring me the wrench"
    if suite == "obstacle":
        # T10 #3: a box is put on the route to table B once the tool is in the jaws
        dest = "table_b"
        cmd = f"put the {WORDS[tool]} on the side table"
        state = {"done": False}

        def put_box(o, phase):
            if not state["done"] and o.sim.gripper_state() == "holding":
                state["done"] = True
                p = np.array([3.15, -0.28, 0.2])   # on the first leg of the route to table B
                o.sim.m.body_pos[o.sim.m.body("obstacle").id] = p
                o.sim.d.mocap_pos[o.sim.obstacle_mocap] = p
        orch.on_walk_tick = put_box
    if suite == "drop":
        state = {"done": False}

        def drop(o, phase):
            if not state["done"] and phase == "goto_far" and o.sim.gripper_state() == "holding":
                state["done"] = True
                q = o.sim.arm_target.copy()
                q[6] = GRIPPER_OPEN
                o.sim.set_arm_target(q)
        orch.on_walk_tick = drop
    t0 = time.time()
    r = orch.run(cmd)
    wall = time.time() - t0
    # --- ground-truth scoring
    in_dest = [n for n in TOOL_NAMES
               if (in_handoff_tray(sim.gt_tool_pos(n)) if dest == "human"
                   else in_place_zone(sim.gt_tool_pos(n), dest)) and sim.held_tool() != n]
    delivered = tool if tool in in_dest else (in_dest[0] if in_dest else None)
    # Where the tool actually ended up, always recorded: without it a failed trial says only
    # "not delivered", and the place distribution's tail (1 transfer in 10) cannot be told from
    # a fall or a drop without re-running it by hand.
    tp = sim.gt_tool_pos(tool)
    gt = {"tool_pos": [round(float(x), 3) for x in tp], "held": sim.held_tool(),
          "base": [round(float(x), 3) for x in sim.d.qpos[0:2]]}
    if dest != "human":
        cx, cy = PLACE_ZONE[dest]
        gt["zone_d"] = [round(float(tp[0] - cx), 3), round(float(tp[1] - cy), 3)]
        gt["dz"] = round(float(tp[2] - place_zone_z()), 3)
    if suite == "retarget":
        # the SECOND tool must be delivered, and the first must not be in the tray
        ok = delivered == tool and first not in in_dest and bool(r.retargets)
        cause = ("ok" if ok else ("no_retarget" if not r.retargets else
                                  ("wrong_object" if delivered else r.reason)))
        return {"suite": suite, "seed": seed, "tool": tool, "first": first, "command": cmd,
                "success": bool(ok), "cause": cause, "orch_reason": r.reason,
                "delivered": delivered, "asked": r.asked, "retargets": r.retargets,
                "sim_s": round(sim.time, 1), "wall_s": round(wall, 1), "gt": gt,
                "timings": {k: round(v, 1) for k, v in r.timings.items()}}
    if suite == "missing":
        ok = delivered is None and r.reason.startswith("missing")
        cause = "ok" if ok else ("wrong_object" if delivered else r.reason)
    else:
        ok = delivered == tool
        if suite == "ambiguous":
            ok = ok and human.asked >= 1
        cause = "ok" if ok else ("wrong_object" if delivered not in (None, tool) else r.reason)
    return {"suite": suite, "seed": seed, "tool": tool, "command": cmd, "success": bool(ok),
            "cause": cause, "orch_reason": r.reason, "delivered": delivered, "asked": r.asked,
            "sim_s": round(sim.time, 1), "wall_s": round(wall, 1), "gt": gt,
            "timings": {k: round(v, 1) for k, v in r.timings.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="nominal")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--backend", default="scripted")
    ap.add_argument("--grounding", default="oracle")
    ap.add_argument("--out", default=None)
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    suites = (["nominal", "transfer", "missing", "drop", "ambiguous", "retarget", "obstacle"] if args.suite == "all"
              else args.suite.split(","))
    if args.grounding == "vlm":
        from bw.perception.vlm import ensure_server
        ensure_server()
    sim = WorkshopSim()
    sim.attach_locomotion()
    orch = Orchestrator(sim, backend=args.backend, grounding=args.grounding, verbose=args.v)
    rows = []
    for s in suites:
        for seed in range(args.start, args.start + args.n):
            row = trial(sim, orch, s, seed)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if args.out:
                # mkdir first: a run into a fresh output directory used to complete every trial and
                # then throw FileNotFoundError writing the results away (job 325, session 7 -- 20
                # trials lost).
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(rows, indent=0))
    if args.grounding == "vlm":
        # Never leave the card held: twice this session a model process outlived its job and the
        # queue runner -- which waits for a free GPU by design -- stalled behind it.
        from bw.perception import vlm
        print("vlm server stopped:", vlm.stop_server())
    print("\nsuite        success   failure causes")
    for s in suites:
        rs = [r for r in rows if r["suite"] == s]
        causes = Counter(r["cause"] for r in rs if not r["success"])
        print(f"  {s:10s} {sum(r['success'] for r in rs):3d}/{len(rs):<3d}  {dict(causes)}")
    lat = defaultdict(list)
    for r in rows:
        for k, v in r["timings"].items():
            lat[k].append(v)
    print("wall-clock per layer (median s):", {k: round(float(np.median(v)), 1) for k, v in lat.items()})


if __name__ == "__main__":
    main()
