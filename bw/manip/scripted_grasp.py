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
PRE = 0.13               # stand-off along the approach before closing in
RETREAT = 0.13           # enough to clear the rack; further only shakes the tool
LIFT = 0.18              # straight up, clearing the rack plates (base + RACK_H = 0.055)
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


def grasp_point(sim: WorkshopSim, name: str) -> np.ndarray:
    """World position of the grasp on a standing tool: on the tool's own long axis at
    GRASP_Z above the rack floor (above the rack plates), following whatever lean the tool
    has settled into."""
    pos = sim.gt_tool_pos(name)
    R = sim.d.xmat[sim.tool_body[name]].reshape(3, 3)
    if name == "tape_roll":
        # Ring standing on edge: grasp the side wall facing the robot, where the wall runs
        # vertically and is 7 mm thick across the jaws.
        sh = sim.d.xpos[sim.m.body("arm_link02").id]
        side = R[:, 1] if (pos - sh)[1] < 0 else -R[:, 1]
        return pos + side * 0.0415
    up = -R[:, 0]                              # the tool's +x points down into the slot
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
    for pitch in PITCH:
        pitch = pitch + rng.uniform(-0.03, 0.03)
        a = np.array([np.cos(pitch) * np.cos(heading), np.cos(pitch) * np.sin(heading),
                      -np.sin(pitch)])
        lateral = np.array([-np.sin(heading), np.cos(heading), 0.0])
        for w in (lateral, -lateral):
            Rg = grasp_rot(a, w)
            site = gp          # symmetric parallel jaw: the ee site IS the grasp centre
            qg, ep, er, ok = ik.solve_multi(sim.d, site, Rg, q_init=q_now)
            if not ok:
                continue
            qp, _, _, okp = ik.solve(sim.d, site - PRE * a, Rg, q_init=qg)
            ql, _, _, okl = ik.solve(sim.d, site + [0, 0, LIFT], Rg, q_init=qg)
            if not (okp and okl):
                continue
            cost = np.abs(qp - q_now).sum() + 0.5 * abs(qg[5])
            if best is None or cost < best[0]:
                best = (cost, dict(Rg=Rg, site=site, a=a, q_pre=qp, q_grasp=qg, q_lift=ql))
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


def run_grasp(sim: WorkshopSim, ik: ArmIK, name: str, rng: np.random.Generator,
              record=None) -> dict:
    """Execute a full grasp of `name`. `record(action7)` is called at 10 Hz."""
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

    sim.move_arm(arm7(sim.arm_q()[:6], op), rng.uniform(0.3, 0.5), record)
    sim.move_arm(arm7(plan["q_pre"], op), rng.uniform(1.8, 2.6), record)
    # Re-point from close range before descending: tools settle in their slots for a second
    # or two after the scene is built, so a plan made at reset can be a centimetre stale.
    site = grasp_point(sim, name)
    q = cartesian(sim, ik, plan["q_pre"], sim.ee_pos(), site, Rg, op,
                  rng.uniform(0.8, 1.2), record)
    sim.move_arm(arm7(q, op), rng.uniform(0.3, 0.5), record)         # dwell: servos settle
    diag = {"site_err": np.round(sim.ee_pos() - site, 4).tolist(),
            "tool_shift": round(float(np.linalg.norm(sim.gt_tool_pos(name) - plan_tool)), 4),
            "open": round(op, 2)}
    sim.move_arm(arm7(q, GRIPPER_CLOSED), rng.uniform(0.6, 0.9), record)
    sim.move_arm(arm7(q, GRIPPER_CLOSED), 0.3, record)
    # One re-approach if the jaws closed on nothing: the tool leans inside its slot, so a
    # plan made from its settled pose can still miss by a centimetre. A demonstrator that
    # retries is also what the orchestrator's on_failure transition does later.
    diag["retried"] = False
    if not _pad_contacts(sim, name):
        diag["retried"] = True
        sim.move_arm(arm7(q, op), 0.4, record)
        deeper = grasp_point(sim, name) + 0.012 * a
        q = cartesian(sim, ik, q, sim.ee_pos(), deeper, Rg, op, 0.8, record)
        sim.move_arm(arm7(q, op), 0.3, record)
        sim.move_arm(arm7(q, GRIPPER_CLOSED), 0.7, record)
        sim.move_arm(arm7(q, GRIPPER_CLOSED), 0.3, record)
        site = deeper
    diag["pad_contacts"] = sorted(_pad_contacts(sim, name))
    diag["grip_q"] = round(float(sim.arm_q()[6]), 4)
    q = cartesian(sim, ik, q, site, site + np.array([0, 0, LIFT]), Rg, GRIPPER_CLOSED,
                  rng.uniform(1.2, 1.6), record)
    sim.move_arm(arm7(q, GRIPPER_CLOSED), 0.4, record)      # settle before translating
    diag["lift_only"] = round(float(sim.gt_tool_pos(name)[2] - z0), 3)
    diag["ee_lift"] = round(float(sim.ee_pos()[2] - site[2]), 3)
    diag["pads_after_lift"] = sorted(_pad_contacts(sim, name))
    # Retreat straight back from the rack before folding the arm in: swinging a held tool
    # over the rack in joint space knocks it out of the jaws on the way past.
    up = site + np.array([0, 0, LIFT])
    q = cartesian(sim, ik, q, up, up - RETREAT * np.array([a[0], a[1], 0.0]), Rg, GRIPPER_CLOSED,
                  rng.uniform(1.6, 2.1), record)
    diag["after_retreat"] = round(float(sim.gt_tool_pos(name)[2] - z0), 3)
    # The demonstration ENDS here: tool lifted clear of the rack and retracted, gripper still
    # closed. Folding the arm into a travel pose is the orchestrator's stow_arm(), not part of
    # the grasp skill -- and a joint-space fold with a tool in the jaws was measured dropping
    # it, because re-rolling the wrist levers the tool out of the pads.
    sim.move_arm(np.concatenate([q, [GRIPPER_CLOSED]]), 0.6, record)
    lifted = sim.gt_tool_pos(name)[2] - z0
    held = sim.held_tool()
    ok = lifted > 0.08 and held == name
    return {"success": bool(ok), "lifted": round(float(lifted), 3), "held": held, **diag,
            "reason": "ok" if ok else ("wrong_object" if held not in (None, name) else "not_held")}


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
