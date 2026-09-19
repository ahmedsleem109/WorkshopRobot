"""Walk the base to a station pose with the Layer 3 policy -- the replacement for
WorkshopSim.teleport_base in the two-table transfer.

Only `Locomotion.set_velocity` is used, i.e. exactly the interface the orchestrator (T9) gets:
no base pose is ever written. The route is a fixed, obstacle-free sequence that fits the
workshop's 3 m walkway:

    backup    straight back off the table the robot is facing (it starts with its nose
              ~10 cm from the bench and the arm over it; turning there sweeps the bench)
    turn      in place, to face the pre-station point
    goto      walk to the pre-station point, PRE_DIST behind the station along its heading,
              steering with wz
    align     in place, to the station heading
    approach  slow forward creep onto the station, correcting lateral error with vy
    stop      zero command until the base is still

Each phase ends on a MEASURED condition (distance / heading / speed), never on time, with a
per-phase timeout so a policy that cannot track a command fails loudly instead of hanging.
"""

from __future__ import annotations

import numpy as np

PRE_DIST = 0.55          # m behind the station the final approach starts from
BACKUP = 0.40            # m backed off the table before turning
TOL_POS, TOL_YAW = 0.03, np.radians(3.0)
# The policy has a DEADBAND, measured (scripts/nav_tracking.py): a command under ~0.2 m/s or a
# pure yaw under ~0.3 rad/s is executed as "stand". A proportional controller therefore stalls
# a few degrees / centimetres short of every goal -- the first walk timed out 4.2 deg from a
# 4 deg tolerance. So the navigator only ever asks for motion it will get: turns at >= W_MIN,
# the final creep at a steady V_CREEP, and it stops on the mark instead of slowing into it.
W_MIN, V_CREEP = 0.45, 0.30
STALL_S, KICK_V, COAST = 0.6, 0.45, 0.04
V_BACK, TOL_FINE, COAST_FINE = -0.25, 0.03, 0.03
# Yaw is only trimmed when it is really off: a few-degree in-place turn slides the base ~10 cm
# sideways, and the place skill plans with IK from the live base pose anyway.
TOL_FINE_YAW, FINE_ROUNDS = np.radians(8.0), 6
V_SIDE = 0.2             # m/s sidestep command (achieves ~0.10-0.14)
FAR_EXTRA = 0.50         # m: the big-turn waypoint is this much further back than `pre`


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def walk_to(sim, station, backup: float = BACKUP, on_phase=None, timeout_s: float = 25.0,
            turn_sign: int | None = None) -> dict:
    """Walk `sim` (WorkshopSim with attach_locomotion()) to station (x, y, yaw).

    turn_sign=-1 turns in place only CLOCKWISE (the long way round when needed), for a
    policy that has learned one turn direction but not the other -- measured on the nav2
    checkpoints, turn right tracks from ~4M steps while turn left still stands.
    """
    loco = sim.loco
    assert loco is not None, "attach_locomotion() first"
    ph = on_phase or (lambda _l: None)
    xs, ys, yaws = station
    h = np.array([np.cos(yaws), np.sin(yaws)])
    pre = np.array([xs, ys]) - PRE_DIST * h
    t_start = sim.time
    log = []

    def pose():
        p = loco.get_base_pose()
        return p.pos[:2].copy(), p.yaw

    def tick(cmd):
        loco.set_velocity(*cmd)
        sim.physics_step(loco.n_substeps)
        if not loco.is_stable():
            raise RuntimeError("fell")

    def run(name, cmd_fn, done_fn, limit):
        ph(name)
        t0 = sim.time
        while not done_fn():
            if sim.time - t0 > limit:
                log.append((name, "timeout"))
                return False
            tick(cmd_fn())
        log.append((name, round(sim.time - t0, 2)))
        return True

    try:
        # --- backup: straight back, heading held
        xy0, yaw0 = pose()
        ok = run("backup",
                 lambda: (-0.25, 0.0, 1.5 * _wrap(yaw0 - pose()[1])),
                 lambda: np.linalg.norm(pose()[0] - xy0) >= backup, 6.0) if backup > 0 else True

        # --- the route: two waypoints on the station's heading line. The BIG turn happens at
        # `far`; walking far -> pre then corrects lateral error while arriving already facing
        # the station, so no large in-place turn is left before the final creep. Turning in
        # place is not perfectly in place -- a 270 deg turn at `pre` walked the base 12-15 cm
        # sideways, and the creep that followed started off the line.
        far = np.array([xs, ys]) - (PRE_DIST + FAR_EXTRA) * h

        def turn_cmd(target_fn):
            e = _wrap(target_fn() - pose()[1])
            if turn_sign is not None and np.sign(e) != turn_sign and abs(e) > np.radians(8):
                return (0.0, 0.0, 0.6 * turn_sign)          # the long way round
            w = float(np.clip(1.5 * e, -0.9, 0.9))
            return (0.0, 0.0, float(np.sign(w) * max(abs(w), W_MIN)))

        def bearing_to(p):
            xy, _ = pose()
            d = p - xy
            return float(np.arctan2(d[1], d[0]))

        def goto(p, name, tol):
            def cmd():
                xy, yaw = pose()
                dist = float(np.linalg.norm(p - xy))
                e = _wrap(bearing_to(p) - yaw)
                # Face it first, in place -- but only from afar: within ~25 cm the bearing to
                # the point swings wildly, and chasing it spun the base 90 deg on arrival.
                if abs(e) > np.radians(35) and dist > 0.25:
                    return turn_cmd(lambda: bearing_to(p))
                if abs(e) > np.radians(90):             # overshot it: arrived
                    return (0.0, 0.0, 0.0)
                vx = float(np.clip(0.8 * dist, V_CREEP, 0.6)) * max(0.0, np.cos(e))
                return (max(vx, V_CREEP), 0.0, float(np.clip(2.0 * e, -0.9, 0.9)))
            def arrived():
                xy, yaw = pose()
                d = p - xy
                return (np.linalg.norm(d) < tol or
                        (np.linalg.norm(d) < 0.25 and abs(_wrap(np.arctan2(d[1], d[0]) - yaw)) > np.radians(90)))
            return run(name, cmd, arrived, 15.0)

        ok = ok and run("turn", lambda: turn_cmd(lambda: bearing_to(far)),
                        lambda: abs(_wrap(bearing_to(far) - pose()[1])) < np.radians(10), 12.0)
        ok = ok and goto(far, "goto_far", 0.10)
        ok = ok and goto(pre, "goto_pre", 0.05)

        # --- trim the heading (small by construction)
        ok = ok and run("align", lambda: turn_cmd(lambda: yaws),
                        lambda: abs(_wrap(yaws - pose()[1])) < np.radians(5), 10.0)

        # --- approach: creep forward, vy for lateral error, wz for heading
        n = np.array([-h[1], h[0]])

        def errs():
            xy, yaw = pose()
            d = np.array([xs, ys]) - xy
            return float(d @ h), float(d @ n), _wrap(yaws - yaw)

        # STRAIGHT creep, no steering: at 0.25 m/s this policy's arcs slide sideways (a
        # 0.5 rad/s arc measured -0.17 m/s of lateral slip). Lateral error is corrected with a
        # SIDESTEP instead, now that run 3 has one (+0.14 / -0.10 m/s at 1.8M steps): the
        # arrival at `pre` can be up to ~20 cm off the line, and the first straight-only creep
        # left 17-40 cm of it at the station.
        # STALL KICK. Mid-creep, carrying the arm, the final run-3 policy was measured slowing
        # into its stand state and NOT restarting at 0.25 m/s (standing still 0.39 m short
        # for 6 s, no contact anywhere); from a fresh stand it walks at 0.25 fine. So: if the
        # base has been still for STALL_S while told to move, command KICK_V briefly.
        stall = {"t": 0.0, "kick": 0}

        def creep_cmd():
            _, lat, _ = errs()
            vy = float(np.clip(2.0 * lat, -V_SIDE, V_SIDE))
            speed = float(np.linalg.norm(sim.d.qvel[:2]))
            stall["t"] = stall["t"] + loco.dt if speed < 0.03 else 0.0
            if stall["t"] > STALL_S:
                stall["kick"], stall["t"] = int(0.3 / loco.dt), 0.0
            if stall["kick"] > 0:
                stall["kick"] -= 1
                return (KICK_V, vy, 0.0)
            return (V_CREEP, vy, 0.0)

        # Stop COAST early: the base keeps going ~2-5 cm after the command drops to zero.
        ok = ok and run("approach", creep_cmd,
                        lambda: errs()[0] < COAST or abs(errs()[1]) > 0.30, 8.0)
        def still():
            return np.linalg.norm(sim.d.qvel[:2]) < 0.03 and abs(sim.d.qvel[5]) < 0.05

        # --- FINE ALIGNMENT: stop, measure, fix the worst error with one short burst, repeat.
        # Corrections interfere: a yaw trim followed by a stop coasted the final run-3 policy
        # 11 cm sideways and 10 deg back (table B, 5/5 episodes). So each burst is followed by
        # a full stop and a fresh measurement, and only then the next correction is chosen.
        run("stop", lambda: (0.0, 0.0, 0.0), still, 3.0)
        def burst(name, base_cmd, done):
            # Same stall kick as the creep: from a standstill, back-up and the right sidestep
            # often do not start (measured: table A overshoots were left 8-11 cm past the mark
            # because -0.25 m/s never got the gait going). Boost x1.6 for 0.3 s after 0.6 s still.
            st = {"t": 0.0, "kick": 0}
            base_cmd = np.array(base_cmd, float)

            def cmd():
                moving = np.linalg.norm(sim.d.qvel[:2]) > 0.03 or abs(sim.d.qvel[5]) > 0.1
                st["t"] = 0.0 if moving else st["t"] + loco.dt
                if st["t"] > STALL_S:
                    st["kick"], st["t"] = int(0.3 / loco.dt), 0.0
                if st["kick"] > 0:
                    st["kick"] -= 1
                    return tuple(1.6 * base_cmd)
                return tuple(base_cmd)
            return run(name, cmd, done, 4.0)

        # Yaw FIRST (at most twice), then only lateral / along -- never yaw again: at table A an
        # in-place yaw trim also pushed the base 5-8 cm FORWARD toward the bench, so a loop that
        # alternated yaw and back-up oscillated and ended 9-14 cm past the mark.
        for k in range(2):
            eyaw = errs()[2]
            if abs(eyaw) <= TOL_FINE_YAW:
                break
            sg = np.sign(eyaw)
            burst(f"fine_yaw{k}", (0.0, 0.0, sg * W_MIN), lambda: sg * errs()[2] < np.radians(1.5))
            run(f"stop_yaw{k}", lambda: (0.0, 0.0, 0.0), still, 3.0)
        for k in range(FINE_ROUNDS):
            along, lat, _ = errs()
            if abs(lat) > TOL_FINE:
                sg = np.sign(lat)
                burst(f"fine{k}_side", (0.0, sg * V_SIDE, 0.0), lambda: sg * errs()[1] < 0.01)
            elif abs(along) > TOL_FINE:
                sg = np.sign(along)
                burst(f"fine{k}_along", (V_CREEP if sg > 0 else V_BACK, 0.0, 0.0),
                      lambda: sg * errs()[0] < COAST_FINE)
            else:
                break
            run(f"stop{k}", lambda: (0.0, 0.0, 0.0), still, 3.0)
        fell = False
    except RuntimeError:
        ok, fell = False, True

    xy, yaw = pose()
    err_xy = float(np.linalg.norm(xy - np.array([xs, ys])))
    err_yaw = float(np.degrees(abs(_wrap(yaws - yaw))))
    d_end = np.array([xs, ys]) - xy
    return {"success": bool(ok and not fell), "fell": fell,
            "along_mm": round(1000 * float(d_end @ h), 1),
            "lateral_mm": round(1000 * float(d_end @ np.array([-h[1], h[0]])), 1),
            "err_xy_mm": round(1000 * err_xy, 1), "err_yaw_deg": round(err_yaw, 1),
            "time_s": round(sim.time - t_start, 1), "phases": log}
