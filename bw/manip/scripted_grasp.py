"""Scripted IK grasp controller -- the demonstrator SmolVLA is fine-tuned on.

Tools stand in the bench rack (see bw/sim/workshop.py for why a flat tray was abandoned),
so every grasp is HORIZONTAL: approach along the robot's heading at bench height with the
jaws closing sideways across the handle, close, lift straight up out of the slot, carry.
That is the middle of the Z1's workspace once it sits on the Go2 pedestal, and no part of
the gripper can reach a surface on the way in.

Every waypoint is solved with damped least-squares IK on the live model; the servos execute
linear joint-space interpolation between waypoints, and each 10 Hz command is exactly the
action recorded for the VLA. Ground-truth object poses are used here ON PURPOSE: this is
the demonstrator, not the policy.
"""

from __future__ import annotations

import numpy as np

from bw.manip.ik import ArmIK
from bw.sim.workshop import GRASP_Z, rack_floor_z
from bw.sim.workshop_sim import GRIPPER_CLOSED, GRIPPER_OPEN, WorkshopSim

# Half-width of the grasped section ACROSS the jaws (lateral), per tool.
GRASP_HALF_WIDTH = {"wrench_10mm": 0.0065, "wrench_13mm": 0.008, "screwdriver": 0.0125,
                    "pliers": 0.0075, "tape_roll": 0.0035}
# Squeeze per tool, as a FORCE (N); the close command is derived from it: the finger servo
# pushes GRIP_KP x (contact travel - command), and contact travel is GRASP_HALF_WIDTH.
# One squeeze does not fit all, measured 2026-09-19 (scripts/_grip_force_probe.py, transfer
# benchmarks, finger traces):
#   * ~20 N (the old -8 mm command at kp 1200): the 13 mm wrench creeps through the pads at a
#     median 55 mm/s with the arm held still; pliers fall out in transit.
#   * ~60 N: 13 mm wrench 3.4 mm/s, pliers grasp 21/25 -> 25/25. But the 10 mm wrench's thin
#     section is EXTRUDED -- finger travel closes 7.6 -> 7.3 mm over half a second, then the
#     tool squirts out and the jaws slam shut (3/25 in transit) -- and the tape roll's curved
#     rim wedges out of the jaws (transfer 21/25 -> 13/25).
# A real gripper is run the same way (force per object), and the command is in the action.
# Lift duration per tool (s). The tape roll is held at a point on its rim, so lifting swings
# it ~90 deg in the jaws to hang below that point; fast, the swing jams the ring against the
# rack plates (scripts/grasp_diagnose.py: failures show 120-790 N of rack contact). Slower:
# grasp 19/25 -> 21/25. What is left is NOT the lift: all 4 remaining failures start with the
# ring already pinched against a plate at 75-120 N when the jaws close; every grasp that starts
# under 12 N succeeds. Next fix is in the approach/close, not here.
LIFT_S = {"tape_roll": (2.5, 3.0)}
GRIP_FORCE = {"wrench_10mm": 20.0, "wrench_13mm": 60.0, "screwdriver": 60.0,
              "pliers": 60.0, "tape_roll": 18.0}


def grip_close_cmd(name: str) -> float:
    from bw.sim.build_models import GRIP_KP
    return max(GRIPPER_CLOSED, GRASP_HALF_WIDTH[name] - GRIP_FORCE[name] / GRIP_KP)
PRE = 0.13               # stand-off along the approach before closing in
RETREAT = 0.13           # enough to clear the rack; further only shakes the tool
HOLD_VERIFY = 4.0        # seconds the tool must stay in the jaws AFTER the motion ends.
                         # 2 s was too lenient and hid a real failure: the pliers passed it
                         # 24/25 and then dropped at ~2.3 s (see GRASP_Z in workshop.py).
TAPE_PHI = np.radians(45.0)   # tape roll: grasp this far ABOVE the ring's equator. Swept in
                              # scripts/_tape_sweep.py (8 seeds, held-2s): 0 deg 2/8 (jaws hit
                              # the rack, site_err 13.9 mm), 15 6/8, 30 7/8, 45 8/8 (site_err
                              # 1.2 mm), 60 5/8, 70 1/8 -- clearance above the plates buys the
                              # left half of the curve, the wall's apparent width across a
                              # laterally-closing jaw (7 mm / cos phi) costs the right half.
LIFT = 0.18              # straight up, clearing the rack plates (base + RACK_H = 0.055)
STAGE_UP = 0.12          # how far ABOVE the pre-grasp point the staging waypoint sits
PITCH = (0.0, 0.15, -0.15, 0.3)
SCAN_Q = np.array([0.0, 1.5, -2.1, 1.5, 0.0, 0.0, GRIPPER_OPEN])   # scripts/find_scan_pose.py
CARRY_Q = np.array([0.0, 0.65, -1.05, 0.35, 0.0, 0.0, GRIPPER_CLOSED])


def grasp_rot(approach: np.ndarray, width_dir: np.ndarray) -> np.ndarray:
    """ee rotation: +x along the approach, +z along the jaw (closing) axis."""
    x = approach / np.linalg.norm(approach)
    z = width_dir - np.dot(width_dir, x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def tape_rim_dir(side: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Unit radial direction to the rim point TAPE_PHI above the ring's equator, on `side`,
    projected into the ring's own plane (`axis` is the ring's normal) so a leaning roll is
    followed rather than assumed away."""
    want = np.cos(TAPE_PHI) * side + np.sin(TAPE_PHI) * np.array([0.0, 0.0, 1.0])
    u = want - np.dot(want, axis) * axis
    n = np.linalg.norm(u)
    return side if n < 1e-6 else u / n


def tape_radial(sim: WorkshopSim) -> np.ndarray:
    """Unit radial direction to the TOP of the ring, in the ring's own plane (follows a lean).

    The tape roll is pinched RADIALLY at its crown: the jaw axis runs along this direction, one
    pad inside the hole, one on top, pads flat on the 7 mm wall. It replaced a lateral-closing
    grasp at 45 deg above the equator (TAPE_PHI), which was measured failing on legs 8/25: the
    26 mm tall pad's upper corner reaches past the wall into the ring's crown, so the pad TIP
    hit the ring face at ~310 N during the approach (scripts probe, pinned and on legs). A
    pinned base absorbs that; a standing robot is knocked back 20-40 mm and closes short.
    Held at the crown the ring already hangs below the grip, so the lift does not swing it."""
    R = sim.d.xmat[sim.tool_body["tape_roll"]].reshape(3, 3)
    n = R[:, 2]
    up = np.array([0.0, 0.0, 1.0])
    u = up - np.dot(up, n) * n
    return u / np.linalg.norm(u)


def grasp_point(sim: WorkshopSim, name: str) -> np.ndarray:
    """World position of the grasp on a standing tool: on the tool's own long axis at
    GRASP_Z above the rack floor (above the rack plates), following whatever lean the tool
    has settled into."""
    pos = sim.gt_tool_pos(name)
    R = sim.d.xmat[sim.tool_body[name]].reshape(3, 3)
    if name == "tape_roll":
        # Ring standing on edge: grasp the rim, where the wall is 7 mm thick across the jaws.
        # NOT at the equator (TAPE_PHI = 90 deg), which is where the ring's own centre sits --
        # 45 mm above the rack floor, i.e. 5 mm BELOW the 50 mm plate tops, so the jaws reach
        # the rack before they reach the tape: site_err measured 8-19 mm against ~1 mm for
        # every other tool, and the pads closed on nothing in 5 of 8 episodes. TAPE_PHI is the
        # angle up from the equator; at 45 deg the rim point clears the plates by 26 mm and
        # the wall still presents 9.9 mm across a laterally-closing jaw.
        return pos + tape_radial(sim) * 0.0415
    up = -R[:, 0]                              # the tool's +x points down into the slot...
    if up[2] < 0:                              # ...except the screwdriver, which stands
        up = -up                               # handle-down (see stand_quat in workshop.py)
    if up[2] < 0.5:                            # knocked over / lying down: grasp its centre
        return pos
    along = (rack_floor_z() + GRASP_Z[name]) - pos[2]
    return pos + up * (along / up[2])


def plan_grasp(sim: WorkshopSim, ik: ArmIK, name: str, rng: np.random.Generator):
    gp = grasp_point(sim, name)
    sh = sim.d.xpos[sim.m.body("arm_link02").id]
    heading = np.arctan2(gp[1] - sh[1], gp[0] - sh[0])
    q_now = sim.arm_q()[:6]
    best = None
    # The tape roll's radial pinch needs a LEVEL approach: the jaw axis is orthogonalised
    # against the approach, so a pitched approach tilts the pads off the flat crown wall and
    # they close on an edge (one pad at ~200 N, the other barely touching; pinned 7/12).
    for pitch in ((0.0,) if name == "tape_roll" else PITCH):
        pitch = pitch + rng.uniform(-0.03, 0.03)
        a = np.array([np.cos(pitch) * np.cos(heading), np.cos(pitch) * np.sin(heading),
                      -np.sin(pitch)])
        lateral = np.array([-np.sin(heading), np.cos(heading), 0.0])
        ws = (lateral, -lateral)
        if name == "tape_roll":
            r = tape_radial(sim)
            ws = (r, -r)
        for w in ws:
            Rg = grasp_rot(a, w)
            site = gp          # symmetric parallel jaw: the ee site IS the grasp centre
            qg, ep, er, ok = ik.solve_multi(sim.d, site, Rg, q_init=q_now)
            if not ok:
                continue
            qp, _, _, okp = ik.solve(sim.d, site - PRE * a, Rg, q_init=qg)
            ql, _, _, okl = ik.solve(sim.d, site + [0, 0, LIFT], Rg, q_init=qg)
            if not (okp and okl):
                continue
            # Staging pose: the pre-grasp point, raised. The arm reaches THIS first, so the
            # long joint-space swing from the scan pose descends in front of the rack instead
            # of sweeping through it. MEASURED: without it the gripper knocked a NEIGHBOURING
            # tool into the target during `reach_pre` -- the target had already moved 50-68 mm
            # before the jaws were anywhere near it (scripts/_knock_probe.py, seeds 17 and 21).
            qs, _, _, oks = ik.solve(sim.d, site - PRE * a + [0, 0, STAGE_UP], Rg, q_init=qp)
            cost = np.abs(qp - q_now).sum() + 0.5 * abs(qg[5])
            if best is None or cost < best[0]:
                best = (cost, dict(Rg=Rg, site=site, a=a, q_pre=qp, q_grasp=qg, q_lift=ql,
                                   q_stage=qs if oks else None))
        if best is not None:
            break
    return None if best is None else best[1]


def cartesian(sim, ik, q_start, p0, p1, Rg, grip, duration, record, rate_hz=10.0):
    steps = max(2, int(round(duration * rate_hz)))
    q = q_start
    for k in range(1, steps + 1):
        p = p0 + (p1 - p0) * (k / steps)
        q, ep, er, ok = ik.solve(sim.d, p, Rg, q_init=q, iters=150)
        sim.move_arm(np.concatenate([q, [grip]]), 1.0 / rate_hz, record, rate_hz)
    return q


def to_pregrasp(sim: WorkshopSim, ik: ArmIK, name: str, rng: np.random.Generator,
                record=None) -> dict | None:
    """Transport only: open the jaws, swing to the staging pose, arrive at the PRE-GRASP pose.
    Returns `plan_grasp`'s plan, or None if the IK failed.

    This exists so a LEARNED policy can be scored on the part of the task it is actually being
    trained for (session 7). Measured over 60 pick episodes, 91% of joint 1's squared motion and
    100% of joint 6's is in the leading transport swing (27-28 mrad per tick against 3-5 mrad during
    the fine approach), and the trained policy's error on j1 -- 29 mrad -- is the size of the swing
    rather than of the approach. Transport is an IK problem the scripted stack solves exactly, and
    in the full pipeline its target comes from `locate()` (median 10.7 mm), so handing the VLA the
    approach and the grasp is the division of labour the architecture already assumes.

    It deliberately repeats the opening phases of `run_grasp` instead of refactoring them out:
    run_grasp is the 125/125 demonstrator and is not worth disturbing for an experiment."""
    sim.lock_stance()
    plan = plan_grasp(sim, ik, name, rng)
    if plan is None:
        return None
    op = min(GRIPPER_OPEN, GRASP_HALF_WIDTH[name] + 0.012 + rng.uniform(0.0, 0.006))

    def arm7(q6, grip):
        return np.concatenate([q6, [grip]])

    sim.move_arm(arm7(sim.arm_q()[:6], op), rng.uniform(0.3, 0.5), record)
    if plan["q_stage"] is not None:
        sim.move_arm(arm7(plan["q_stage"], op), rng.uniform(1.4, 1.9), record)
    sim.move_arm(arm7(plan["q_pre"], op), rng.uniform(1.0, 1.4), record)
    return plan


def run_grasp(sim: WorkshopSim, ik: ArmIK, name: str, rng: np.random.Generator,
              record=None, on_phase=None) -> dict:
    """Execute a full grasp of `name`. `record(action7)` is called at 10 Hz.

    `on_phase(label)` is called as each stage BEGINS -- the diagnostic in
    scripts/grasp_diagnose.py uses it to attribute a slip to a stage, and the data
    collector (T6) can store it as a per-frame label.
    """
    ph = on_phase if on_phase is not None else (lambda _label: None)
    closed = grip_close_cmd(name)
    sim.lock_stance()
    ph("plan")
    plan = plan_grasp(sim, ik, name, rng)
    if plan is None:
        return {"success": False, "reason": "ik_fail"}
    z0 = sim.gt_tool_pos(name)[2]
    plan_tool = sim.gt_tool_pos(name)
    # Open only as far as the part needs, plus ~1.5 cm of margin each side.
    op = min(GRIPPER_OPEN, GRASP_HALF_WIDTH[name] + 0.012 + rng.uniform(0.0, 0.006))
    Rg, site, a = plan["Rg"], plan["site"], plan["a"]

    def arm7(q6, grip):
        return np.concatenate([q6, [grip]])

    ph("open")
    sim.move_arm(arm7(sim.arm_q()[:6], op), rng.uniform(0.3, 0.5), record)
    if plan["q_stage"] is not None:
        ph("stage")
        sim.move_arm(arm7(plan["q_stage"], op), rng.uniform(1.4, 1.9), record)
    ph("reach_pre")
    sim.move_arm(arm7(plan["q_pre"], op), rng.uniform(1.0, 1.4), record)
    # Re-point from close range before descending: tools settle in their slots for a second
    # or two after the scene is built, so a plan made at reset can be a centimetre stale.
    site = grasp_point(sim, name)
    ph("approach")
    q = cartesian(sim, ik, plan["q_pre"], sim.ee_pos(), site, Rg, op,
                  rng.uniform(0.8, 1.2), record)
    sim.move_arm(arm7(q, op), rng.uniform(0.3, 0.5), record)         # dwell: servos settle
    diag = {"site_err": np.round(sim.ee_pos() - site, 4).tolist(),
            "tool_shift": round(float(np.linalg.norm(sim.gt_tool_pos(name) - plan_tool)), 4),
            "open": round(op, 2)}
    ph("close")
    sim.move_arm(arm7(q, closed), rng.uniform(0.6, 0.9), record)
    sim.move_arm(arm7(q, closed), 0.3, record)
    # One re-approach if the jaws closed on nothing: the tool leans inside its slot, so a
    # plan made from its settled pose can still miss by a centimetre. A demonstrator that
    # retries is also what the orchestrator's on_failure transition does later.
    diag["retried"] = False
    if not _pad_contacts(sim, name):
        diag["retried"] = True
        ph("regrasp")
        sim.move_arm(arm7(q, op), 0.4, record)
        deeper = grasp_point(sim, name) + 0.012 * a
        q = cartesian(sim, ik, q, sim.ee_pos(), deeper, Rg, op, 0.8, record)
        sim.move_arm(arm7(q, op), 0.3, record)
        sim.move_arm(arm7(q, closed), 0.7, record)
        sim.move_arm(arm7(q, closed), 0.3, record)
        site = deeper
    diag["pad_contacts"] = sorted(_pad_contacts(sim, name))
    diag["grip_q"] = round(float(sim.arm_q()[6]), 4)
    ph("lift")
    q = cartesian(sim, ik, q, site, site + np.array([0, 0, LIFT]), Rg, closed,
                  rng.uniform(*LIFT_S.get(name, (1.2, 1.6))), record)
    ph("lift_settle")
    sim.move_arm(arm7(q, closed), 0.4, record)      # settle before translating
    diag["lift_only"] = round(float(sim.gt_tool_pos(name)[2] - z0), 3)
    diag["ee_lift"] = round(float(sim.ee_pos()[2] - site[2]), 3)
    diag["pads_after_lift"] = sorted(_pad_contacts(sim, name))
    # Retreat straight back from the rack before folding the arm in: swinging a held tool
    # over the rack in joint space knocks it out of the jaws on the way past.
    up = site + np.array([0, 0, LIFT])
    ph("retreat")
    q = cartesian(sim, ik, q, up, up - RETREAT * np.array([a[0], a[1], 0.0]), Rg, closed,
                  rng.uniform(1.6, 2.1), record)
    diag["after_retreat"] = round(float(sim.gt_tool_pos(name)[2] - z0), 3)
    # The demonstration ENDS here: tool lifted clear of the rack and retracted, gripper still
    # closed. Folding the arm into a travel pose is the orchestrator's stow_arm(), not part of
    # the grasp skill -- and a joint-space fold with a tool in the jaws was measured dropping
    # it, because re-rolling the wrist levers the tool out of the pads.
    ph("hold")
    sim.move_arm(np.concatenate([q, [closed]]), 0.6, record)
    # VERIFY, do not record: hold the arm still for HOLD_VERIFY seconds and require the tool
    # to still be there. Scoring at the instant the motion stops is what made the old
    # benchmark read 69% while only 31% of those grasps survived two more seconds -- the tool
    # was sliding through the jaws the whole time (see the option block in bw/sim/workshop.py).
    ph("verify")
    sim.settle(HOLD_VERIFY)
    ph("end")
    lifted = sim.gt_tool_pos(name)[2] - z0
    held = sim.held_tool()
    ok = lifted > 0.08 and held == name
    diag["site"] = [float(v) for v in site]
    diag["R"] = [[float(v) for v in row] for row in Rg]
    diag["approach"] = [float(v) for v in a]
    return {"success": bool(ok), "lifted": round(float(lifted), 3), "held": held, **diag,
            "reason": "ok" if ok else ("wrong_object" if held not in (None, name) else "not_held")}


def return_to_rack(sim: WorkshopSim, ik: ArmIK, name: str, pick, rng, record=None) -> dict:
    """Put a held tool BACK in the slot it came from -- the grasp, reversed.

    `pick` is the dict `run_grasp` returned (its `site`, `R` and `approach`). The robot does not
    move: the slot is right in front of it, so this is the cheapest way to get rid of a tool the
    person no longer wants (T10 #5). Walking somewhere to put it down is not: the station-to-
    station hop is 0.6 m sideways, and the policy's sidestep only manages ~0.1 m of it.
    """
    sim.lock_stance()
    site = np.asarray(pick["site"], float)
    Rg = np.asarray(pick["R"], float)
    a = np.asarray(pick["approach"], float)
    grip = float(sim.arm_target[6])
    q = sim.arm_q()[:6]
    above = site + np.array([0.0, 0.0, LIFT])
    back = above - RETREAT * np.array([a[0], a[1], 0.0])
    q = cartesian(sim, ik, q, sim.ee_pos(), back, Rg, grip, rng.uniform(1.0, 1.3), record)
    q = cartesian(sim, ik, q, sim.ee_pos(), above, Rg, grip, rng.uniform(1.4, 1.8), record)
    q = cartesian(sim, ik, q, sim.ee_pos(), site, Rg, grip, rng.uniform(1.6, 2.0), record)
    sim.move_arm(np.concatenate([q, [grip]]), 0.3, record)
    op = min(GRIPPER_OPEN, GRASP_HALF_WIDTH[name] + 0.014)
    sim.move_arm(np.concatenate([q, [op]]), 0.6, record)
    q = cartesian(sim, ik, q, sim.ee_pos(), site - PRE * a, Rg, op, rng.uniform(1.2, 1.6), record)
    sim.settle(1.0)
    standing = sim.gt_tool_pos(name)[2] > rack_floor_z() + 0.02
    return {"success": bool(sim.held_tool() is None and standing),
            "held": sim.held_tool(), "pos": [round(float(v), 3) for v in sim.gt_tool_pos(name)]}


def _pad_contacts(sim: WorkshopSim, name: str) -> set[str]:
    tb = sim.tool_body[name]
    out = set()
    for i in range(sim.d.ncon):
        c = sim.d.contact[i]
        bodies = (sim.m.geom_bodyid[c.geom1], sim.m.geom_bodyid[c.geom2])
        if tb not in bodies:
            continue
        for g in (c.geom1, c.geom2):
            if sim.m.geom_bodyid[g] in (sim.mover_body, sim.stator_body):
                out.add(sim.m.geom(g).name or str(g))
    return out


STOW_B = np.array([0.26, 0.0, 0.62])     # ee target, BASE frame, walking with a tool. (0.15, 0, 0.52):
                                         # step-down 7/12 vs 17/20 -- tucking in does not help


def stow_arm(sim: WorkshopSim, ik: ArmIK, record=None, duration: float = 1.5) -> np.ndarray:
    """Pull the held tool in over the body for walking: a CARTESIAN move of the ee to STOW_B
    (base frame) keeping the grasp orientation -- a joint-space fold re-rolls the wrist and was
    measured levering tools out of the jaws. Brings the payload's CoM back toward the trunk,
    for the step down off the walkway on the way to the human."""
    b = sim.d.qpos[:3].copy()
    yaw = sim.get_base_yaw()
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    target = b + Rz @ STOW_B
    R = sim.d.site_xmat[sim.ee_site].reshape(3, 3).copy()
    grip = float(sim.arm_target[6])
    return cartesian(sim, ik, sim.arm_q()[:6], sim.ee_pos(), target, R, grip, duration, record)
