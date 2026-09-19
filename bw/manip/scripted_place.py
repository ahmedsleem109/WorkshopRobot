"""Scripted `place` skill -- the second half of the two-table transfer (T2.3).

`run_grasp` ends with the tool lifted clear of the rack and still in the jaws, hanging
VERTICALLY: a standing tool is grasped horizontally across its handle, so its long axis
comes out of the rack pointing down. Placing it on a bare table therefore has to decide what
to do with that orientation, and `PLACE_ROLL` is that decision -- a rotation applied to the
grasp orientation about the APPROACH axis, which lays the tool from hanging down (0 deg) to
lying flat across the gripper (90 deg). Nothing else about the grip changes, so the tool
cannot be levered out of the pads on the way.

The axis matters and was measured (scripts/reach_audit.py): rolling about the jaw (closing)
axis is unreachable past ~45 deg at any standing distance, because it demands a full wrist
re-solve, while rolling about the approach axis is FREE -- on the Z1 that is joint 6 alone,
and reachability is then identical at 0 and 90 deg. Rolling the wrong way is what made the
first version of this skill fail IK on every single episode.

The release height is computed, not guessed: the tool's pose in the gripper frame is read
from the live sim, carried through the planned gripper orientation, and the lowest corner of
its collision geometry is put PLACE_CLEAR above the table top.
"""

from __future__ import annotations

import mujoco
import numpy as np

from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import cartesian, grasp_rot
from bw.sim.workshop import PLACE_ZONE, ZONE_HALF, place_zone_z
from bw.sim.workshop_sim import GRIPPER_OPEN, WorkshopSim

# PLACE_ROLL is COMPUTED per episode, not fixed: roll about the approach axis until the held
# tool hangs directly BELOW the gripper. Set it to a number (radians) to override.
#
# Why this is the rule and not a tuned constant. The tool's offset from the gripper is
# whatever the grasp left it as, and it differs per tool: measured in the gripper frame, a
# 13 mm wrench sits 49 mm one way and the PLIERS sit 92 mm the OTHER way -- the pliers stick
# out of the jaws rather than hanging from them, because they are grasped near one end. Any
# offset that is horizontal at release has to be paid for by the base: the gripper must stand
# that much further out than the target. For the pliers that put the required gripper pose
# 220 mm past the zone, outside the arm's reach, and IK happily returned a saturated solution
# that landed the tool 190 mm short -- 0/4 placements, while every other tool was fine.
# Rolling the offset vertical costs nothing (joint 6 alone) and removes the problem at its
# source instead of moving the furniture.
PLACE_ROLL = None
# Releasing the tool laid FLAT (90 deg on from hanging) was tried and is worse: over 24
# transfers per table the median landing error rose from 20-24 mm to 57-59 mm and table A
# fell from 83% to 67% -- the lower jaw is then under the tool, the descent stops on the
# pad instead of the tool, and the drop is longer. Tools are released hanging and allowed
# to topple.
PLACE_JITTER = 0.020            # m: aim scatter inside the zone; swept, see plan_place
# The topple is NOT random, it is a coin with two known faces. MEASURED 2026-09-19
# (scripts/topple_diagnose.py, table B, 12 seeds per tool, distance along the approach axis,
# + = away from the robot): a released tool either stays standing (~0 mm) or topples BACK
# TOWARD THE ROBOT -- wrench_13mm median -95 mm (almost always falls), wrench_10mm /
# screwdriver / tape_roll 0 mm or -75..-98 mm, pliers 0 mm; every topple had the same sign,
# and the tool's lean at release predicted it (cos = +1.00 on the 13 mm wrench). Aimed at
# the centre, a 95 mm topple leaves 5 mm of a 100 mm half-zone: that was 24 of the 53
# transfer failures. Shifting the aim by about half the topple centres BOTH outcomes.
PLACE_AIM_SHIFT = 0.045         # m, along the approach axis, away from the robot
PLACE_CLEAR = 0.004             # m: tool's lowest point above the table at release
PLACE_LIFT = 0.16               # m: height of the pre-place waypoint above the release pose
PLACE_RETRACT = 0.12            # m: up, AFTER backing off (see run_place)
PLACE_BACKOFF = 0.11            # m: straight back along the approach, first thing after
                                # opening -- a finger sits inside the tape roll's hole
HOLD_VERIFY = 2.0               # the placed tool must stay put this long (matches the grasp)


def _rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def tool_lowest_dz(sim: WorkshopSim, name: str, R_tool: np.ndarray) -> float:
    """Lowest collision-geometry z of `name` relative to its body origin, in orientation
    `R_tool`. Negative when the tool hangs below its own origin. Same corner enumeration as
    WorkshopSim._standing_offsets, which is what put the tools ON the rack floor rather than
    8 mm through the bench."""
    m = sim.m
    lowest = 0.0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != sim.tool_body[name] or m.geom_contype[g] == 0:
            continue
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
        Rg = Rg.reshape(3, 3)
        centre, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner = centre + half * [sx, sy, sz]
                    lowest = min(lowest, float((R_tool @ (m.geom_pos[g] + Rg @ corner))[2]))
    return lowest


def _on_table(sim: WorkshopSim, name: str) -> bool:
    """Is the tool touching a table top (either table)?"""
    tb = sim.tool_body[name]
    for i in range(sim.d.ncon):
        c = sim.d.contact[i]
        b1, b2 = sim.m.geom_bodyid[c.geom1], sim.m.geom_bodyid[c.geom2]
        if tb not in (b1, b2):
            continue
        g = c.geom2 if b1 == tb else c.geom1
        if (sim.m.geom(g).name or "") in ("bench_top", "table_b_top"):
            return True
    return False


def _descend_to_contact(sim, ik, name, q_start, p_target, R, grip, record,
                        overshoot=0.030, steps=22, rate_hz=10.0):
    """Lower the gripper toward `p_target` (and `overshoot` past it) until the tool rests on
    the table or the arm stops making progress. Returns the final joint solution."""
    p0 = sim.ee_pos()
    p1 = np.array(p_target, float) - np.array([0.0, 0.0, overshoot])
    q = q_start
    last_z = p0[2]
    stalled = 0
    for k in range(1, steps + 1):
        p = p0 + (p1 - p0) * (k / steps)
        q, _, _, _ = ik.solve(sim.d, p, R, q_init=q, iters=150)
        sim.move_arm(np.concatenate([q, [grip]]), 1.0 / rate_hz, record, rate_hz)
        z = sim.ee_pos()[2]
        step_z = abs(p1[2] - p0[2]) / steps
        stalled = stalled + 1 if (last_z - z) < 0.3 * step_z else 0
        last_z = z
        if _on_table(sim, name) or stalled >= 2:
            break
    return q


def place_rolls(sim: WorkshopSim, name: str):
    """Candidate rolls about the approach axis, best first: the one that hangs the tool
    straight down, then neighbours, then the override/fallbacks."""
    if PLACE_ROLL is not None:
        return [PLACE_ROLL, -PLACE_ROLL]
    p_ee = sim.ee_pos()
    R_ee = sim.d.site_xmat[sim.ee_site].reshape(3, 3)
    v = R_ee.T @ (sim.gt_tool_pos(name) - p_ee)
    n = float(np.linalg.norm(v))
    if n < 1e-4:
        return [0.0, np.pi / 2, -np.pi / 2, np.pi]
    uy, uz = v[1] / n, v[2] / n
    # grasp_rot puts ee +y DOWN, so "tool below the gripper" means the rolled v has a
    # positive ee-y component and no ee-z component: tan(theta) = -uz/uy.
    theta = np.arctan2(-uz, uy)
    return [theta, theta + np.radians(20), theta - np.radians(20),
            theta + np.radians(40), theta - np.radians(40), 0.0, np.pi]


def release_pose(sim: WorkshopSim, name: str, target_xy, R_place: np.ndarray):
    """Gripper position that puts `name` at `target_xy`, just above the table, GIVEN how the
    tool is held RIGHT NOW. Read live, never cached: a long tool swings in the jaws during
    the approach, and a stale reading is a straight translation error at the release -- the
    pliers landed 187 mm from their target because the pose was computed before the arm
    moved."""
    p_ee = sim.ee_pos()
    R_ee = sim.d.site_xmat[sim.ee_site].reshape(3, 3)
    v = R_ee.T @ (sim.gt_tool_pos(name) - p_ee)
    R_rel = R_ee.T @ sim.d.xmat[sim.tool_body[name]].reshape(3, 3)
    dz = tool_lowest_dz(sim, name, R_place @ R_rel)
    p_tool = np.array([target_xy[0], target_xy[1], place_zone_z() + PLACE_CLEAR - dz])
    return p_tool - R_place @ v


def plan_place(sim: WorkshopSim, ik: ArmIK, name: str, table: str, rng):
    """Gripper pose that leaves `name` resting inside `table`'s zone."""
    zx, zy = PLACE_ZONE[table]
    sh = sim.d.xpos[sim.m.body("arm_link02").id]
    # Aim near the centre, not anywhere in the zone. A tool released standing TOPPLES as it
    # lands and travels while doing so, so the aim point and the topple share one error
    # budget; spending half of it on variety is what put ~20% of placements over the line.
    # ... and aim PLACE_AIM_SHIFT PAST the centre, away from the robot: the topple has a
    # direction (see PLACE_AIM_SHIFT).
    h0 = np.arctan2(zy - sh[1], zx - sh[0])
    tx = zx + PLACE_AIM_SHIFT * np.cos(h0) + rng.uniform(-PLACE_JITTER, PLACE_JITTER)
    ty = zy + PLACE_AIM_SHIFT * np.sin(h0) + rng.uniform(-PLACE_JITTER, PLACE_JITTER)

    heading = np.arctan2(ty - sh[1], tx - sh[0])
    a = np.array([np.cos(heading), np.sin(heading), 0.0])      # horizontal approach
    lateral = np.array([-np.sin(heading), np.cos(heading), 0.0])

    Rg = grasp_rot(a, lateral)
    best = None
    for roll in place_rolls(sim, name):
        R_place = Rg @ _rot_about(np.array([1.0, 0.0, 0.0]), roll)
        p_release = release_pose(sim, name, (tx, ty), R_place)
        q_rel, ep, _, ok = ik.solve_multi(sim.d, p_release, R_place, q_init=sim.arm_q()[:6])
        # `ok` alone is not enough: a saturated solve still reports ok while sitting 200 mm
        # away, which is exactly how the pliers failed silently.
        if not ok or ep > 0.01:
            continue
        q_pre, epp, _, okp = ik.solve(sim.d, p_release + [0, 0, PLACE_LIFT], R_place,
                                      q_init=q_rel)
        if not okp or epp > 0.01:
            continue
        # FIRST feasible candidate wins -- place_rolls() is already ordered by how nearly
        # the roll hangs the tool straight down, and that, not joint economy, is what decides
        # whether the release pose is inside the arm's reach.
        best = dict(R=R_place, p_release=p_release, q_release=q_rel, q_pre=q_pre,
                    target=(tx, ty), approach=a, roll_deg=round(float(np.degrees(roll)), 1))
        break
    return best


def _lean_xy(sim: WorkshopSim, name: str) -> np.ndarray:
    """Horizontal direction the held tool's UPPER end points (its long axis is body x)."""
    ax = sim.d.xmat[sim.tool_body[name]].reshape(3, 3)[:, 0]
    up = ax if ax[2] > 0 else -ax
    return up[:2].copy()


def _servo_xy(sim: WorkshopSim, ik: ArmIK, name: str, target_xy, R, grip, record,
              passes: int = 4, tol: float = 0.004):
    """Nudge the gripper horizontally until the HELD tool is over `target_xy`."""
    q = sim.arm_q()[:6]
    for _ in range(passes):
        err = np.array(target_xy, float) - sim.gt_tool_pos(name)[:2]
        if float(np.linalg.norm(err)) < tol:
            break
        p = sim.ee_pos() + np.array([err[0], err[1], 0.0])
        q_new, ep, _, ok = ik.solve(sim.d, p, R, q_init=q, iters=200)
        if not ok or ep > 0.01:
            break
        q = q_new
        sim.move_arm(np.concatenate([q, [grip]]), 0.45, record)
    return q


def run_place(sim: WorkshopSim, ik: ArmIK, name: str, table: str, rng,
              record=None, on_phase=None) -> dict:
    """Place the held tool inside `table`'s marked zone. Assumes the tool IS held."""
    ph = on_phase if on_phase is not None else (lambda _l: None)
    ph("place_plan")
    if sim.held_tool() != name:
        return {"success": False, "reason": "not_holding"}
    plan = plan_place(sim, ik, name, table, rng)
    if plan is None:
        return {"success": False, "reason": "place_ik_fail"}
    grip = float(sim.arm_target[6])
    R, p_rel, a = plan["R"], plan["p_release"], plan["approach"]

    ph("place_pre")
    sim.move_arm(np.concatenate([plan["q_pre"], [grip]]), rng.uniform(1.6, 2.2), record)

    # CLOSED LOOP, not prediction. Everything above is a good enough starting pose; where the
    # tool actually ends up is then corrected by looking at it. A one-shot release pose can
    # only be as good as the tool's assumed pose in the jaws, and that assumption does not
    # hold: a long tool held near one end PIVOTS as the arm moves, so both the offset and the
    # roll that were true at plan time are stale by the time the arm arrives -- re-solving the
    # position alone moved the pliers' release pose by 230-280 mm, and re-solving the roll as
    # well made it worse. Nudging the gripper by the tool's own measured xy error converges
    # regardless, in two or three passes, and it is indifferent to how the tool is held.
    #
    # This uses gt_tool_pos, like the rest of the demonstrator, ON PURPOSE: the demonstrator
    # is allowed ground truth, the policy trained on it is not.
    ph("place_align")
    q = _servo_xy(sim, ik, name, plan["target"], R, grip, record)
    # Re-aim from the tool's MEASURED lean, now that it hangs where the arm will release it.
    # It topples the way its upper end points (cos = +1.00 over 25 transfers), so the shift
    # goes the other way. The plan's approach-axis shift is only the prior: under a 60 N
    # squeeze the pliers hang leaning AWAY from the robot and toppled 90 mm away, so a fixed
    # shift pushed them further out. Nearly vertical (< ~1 deg) -> keep the prior.
    # The tape roll is excluded: a ring has no long axis to lean, and it does not topple.
    lean = _lean_xy(sim, name)
    if name != "tape_roll" and 0.015 < np.linalg.norm(lean) < 0.5:
        zx, zy = PLACE_ZONE[table]
        jit = np.array(plan["target"]) - (np.array([zx, zy]) + PLACE_AIM_SHIFT * a[:2])
        tgt = np.array([zx, zy]) - PLACE_AIM_SHIFT * lean / np.linalg.norm(lean) + jit
        plan["target"] = (float(tgt[0]), float(tgt[1]))
        q = _servo_xy(sim, ik, name, plan["target"], R, grip, record)
    p_rel = release_pose(sim, name, plan["target"], R)
    diag_repoint = round(float(np.linalg.norm(
        sim.gt_tool_pos(name)[:2] - np.array(plan["target"]))), 4)

    # Descend to CONTACT, not to a computed height. The predicted release pose is only as
    # good as the tool's assumed pose in the gripper; aiming BELOW the surface and stopping
    # on contact removes the dependency, and it is what the task list asks for.
    ph("place_descend")
    q = _descend_to_contact(sim, ik, name, q, p_rel, R, grip, record)
    sim.move_arm(np.concatenate([q, [grip]]), 0.3, record)
    diag = {"align_err_mm": round(1000 * float(np.linalg.norm(
                sim.gt_tool_pos(name)[:2] - np.array(plan["target"]))), 1),
            "target": [round(v, 3) for v in plan["target"]],
            "touched": _on_table(sim, name), "align_before_mm": round(1000 * diag_repoint, 1),
            "roll_deg": plan["roll_deg"]}

    ph("release")
    sim.move_arm(np.concatenate([q, [GRIPPER_OPEN]]), rng.uniform(0.5, 0.8), record)
    sim.move_arm(np.concatenate([q, [GRIPPER_OPEN]]), 0.3, record)
    # Retract BACKWARDS first, then up. Straight up hooks the tape roll every time: once the
    # jaws open, a finger is sitting inside the ring's hole, and lifting carries the ring with
    # it -- measured, the ring ended 94 mm above the table (~= PLACE_RETRACT) in 4 of 4
    # episodes, still hanging on a finger.
    ph("place_retract")
    back = sim.ee_pos() - PLACE_BACKOFF * np.array([a[0], a[1], 0.0])
    q = cartesian(sim, ik, q, sim.ee_pos(), back, R, GRIPPER_OPEN, rng.uniform(0.8, 1.1),
                  record)
    up = sim.ee_pos() + np.array([0.0, 0.0, PLACE_RETRACT])
    q = cartesian(sim, ik, q, sim.ee_pos(), up, R, GRIPPER_OPEN, rng.uniform(0.8, 1.1), record)

    # VERIFY, not recorded: the tool must still be in the zone after the arm has gone.
    # "Stable for 2 s" is read as AT REST AT THE END, not as having never moved: a tool
    # released just above the table topples out of its carried orientation, and that topple
    # is part of placing, not a failure. What matters is where it ends up and that it has
    # stopped -- so the test is the tool's SPEED after the hold, the same threshold
    # WorkshopSim.settle_static uses on the rack.
    ph("place_verify")
    p_before = sim.gt_tool_pos(name).copy()
    sim.settle(HOLD_VERIFY)
    p_after = sim.gt_tool_pos(name)
    from bw.sim.workshop import in_place_zone
    inside = in_place_zone(p_after, table)
    dadr = sim.tool_dadr[name]
    speed = float(np.linalg.norm(sim.d.qvel[dadr:dadr + 3]))
    settled = speed < 0.002
    empty = sim.held_tool() is None
    ph("place_end")
    ok = bool(inside and settled and empty)
    return {"success": ok, "in_zone": bool(inside), "settled": bool(settled),
            "gripper_empty": bool(empty),
            "drift": round(float(np.linalg.norm(p_after - p_before)), 4),
            "speed_mm_s": round(1000 * speed, 1),
            "pos": [round(float(x), 3) for x in p_after],
            "dist_from_centre": round(float(np.linalg.norm(
                p_after[:2] - np.array(PLACE_ZONE[table]))), 3),
            **diag,
            "reason": "ok" if ok else ("outside_zone" if not inside else
                                       ("still_held" if not empty else "not_settled"))}
