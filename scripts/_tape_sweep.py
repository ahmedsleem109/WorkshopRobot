"""Where on the ring should the tape roll be grasped? Sweep TAPE_PHI, the angle above the
ring's equator, against the honest held-for-2s criterion.

The equator (phi = 0) is where the old code grasped, and it sits 5 mm below the rack plate
tops -- the jaws reach the rack before the tape. Higher up the rim clears the plates but the
wall presents a wider, more oblique face to a laterally-closing jaw (7 mm / cos phi).

    render_venv\\Scripts\\python.exe scripts\\_tape_sweep.py [seeds]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.manip import scripted_grasp as sg
from bw.manip.ik import ArmIK
from bw.sim.workshop import TOOL_NAMES, RACK_BASE, RACK_H, rack_floor_z
from bw.sim.workshop_sim import WorkshopSim

R_RING = 0.0415


def main(seeds=8):
    sim = WorkshopSim()
    ik = ArmIK(sim.m)
    ti = TOOL_NAMES.index("tape_roll")
    plate_top = rack_floor_z() + RACK_H
    print(f"plate top {plate_top:.3f}, ring centre ~{rack_floor_z() + R_RING:.3f}")
    print(f"{'phi deg':>8} {'clear mm':>9} {'jaw width mm':>13} {'held 2 s':>9} {'site_err mm':>12}")
    for deg in (0, 15, 30, 45, 60, 70, 80):
        sg.TAPE_PHI = np.radians(deg)
        ok, errs = 0, []
        for seed in range(seeds):
            rng = np.random.default_rng(1000 * seed + ti)
            sim.reset(rng, target="tape_roll", arm_q=SG_SCAN)
            r = sg.run_grasp(sim, ik, "tape_roll", rng)
            ok += bool(r["success"])
            if "site_err" in r:
                errs.append(1000 * float(np.linalg.norm(r["site_err"])))
        clear = 1000 * (rack_floor_z() + R_RING * (1 + np.sin(np.radians(deg))) - plate_top)
        width = 7.0 / max(np.cos(np.radians(deg)), 1e-3)
        print(f"{deg:8d} {clear:9.1f} {width:13.1f} {ok:4d}/{seeds:<4d} "
              f"{np.median(errs) if errs else float('nan'):12.1f}", flush=True)


SG_SCAN = None
if __name__ == "__main__":
    SG_SCAN = sg.SCAN_Q
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
