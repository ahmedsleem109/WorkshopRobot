"""T2.2 reachability audit: which base poses serve which table, and at what gripper roll?

The orchestrator needs these as navigation goals, and the place skill needs to know that its
nominal station is not simply out of reach. For each table this sweeps the standing distance
and lateral offset of the base, and the PLACE_ROLL applied to the gripper, and reports the
fraction of zone targets that IK can reach.

    render_venv\\Scripts\\python.exe scripts\\reach_audit.py [--table table_b]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, grasp_rot
from bw.manip.scripted_place import PLACE_LIFT, _rot_about
from bw.sim.workshop import PLACE_STATION, PLACE_ZONE, ZONE_HALF, place_zone_z
from bw.sim.workshop_sim import WorkshopSim

ROLLS = (0, 30, 45, 60, 90)
BACK = (0.40, 0.48, 0.55, 0.62, 0.70)      # base distance behind the zone centre
LATERAL = (-0.12, 0.0, 0.12)


def targets(table, n=5):
    zx, zy = PLACE_ZONE[table]
    j = ZONE_HALF - 0.045
    out = [(zx, zy)]
    for dx in (-j, j):
        for dy in (-j, j):
            out.append((zx + dx, zy + dy))
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="table_b")
    ap.add_argument("--axis", default="z", choices=("x", "z"),
                    help="ee axis the place roll turns about: z = the jaw (closing) axis, "
                         "x = the approach axis, which on the Z1 is joint 6 alone")
    args = ap.parse_args()

    sim = WorkshopSim()
    ik = ArmIK(sim.m)
    zx, zy = PLACE_ZONE[args.table]
    _, _, yaw = PLACE_STATION[args.table]
    # unit vector from the zone back toward the base, in the ground plane
    back = np.array([-np.cos(yaw), -np.sin(yaw), 0.0])
    side = np.array([-np.sin(yaw), np.cos(yaw), 0.0])

    print(f"table {args.table}: zone at ({zx:.2f}, {zy:.2f}), station yaw {np.degrees(yaw):.0f} deg, "
          f"roll about ee {args.axis}")
    print(f"{'back':>6} {'lat':>6} | " + " ".join(f"{r:>5}" for r in ROLLS) + "   (reachable / 5 zone targets)")
    best = None
    for b in BACK:
        for lat in LATERAL:
            base = np.array([zx, zy, 0.0]) + back * b + side * lat
            row = []
            for roll in ROLLS:
                sim.reset(np.random.default_rng(0), target="wrench_10mm", arm_q=SCAN_Q,
                          base_pose=(base[0], base[1], yaw))
                okc = 0
                for tx, ty in targets(args.table):
                    a = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                    lateral_dir = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
                    ax = np.array([1.0, 0, 0]) if args.axis == "x" else np.array([0, 0, 1.0])
                    R = grasp_rot(a, lateral_dir) @ _rot_about(ax, np.radians(roll))
                    p = np.array([tx, ty, place_zone_z() + 0.05])
                    _, _, _, ok1 = ik.solve_multi(sim.d, p, R, q_init=SCAN_Q[:6])
                    _, _, _, ok2 = ik.solve_multi(sim.d, p + [0, 0, PLACE_LIFT], R,
                                                  q_init=SCAN_Q[:6])
                    okc += bool(ok1 and ok2)
                row.append(okc)
                if best is None or okc > best[0]:
                    best = (okc, b, lat, roll)
            print(f"{b:6.2f} {lat:6.2f} | " + " ".join(f"{v:5d}" for v in row))
    print(f"\nbest: {best[0]}/5 at back {best[1]:.2f} m, lateral {best[2]:+.2f} m, "
          f"roll {best[3]} deg")


if __name__ == "__main__":
    main()
