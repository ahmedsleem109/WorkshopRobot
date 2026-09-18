"""T1.1 follow-up: is the drop caused by frictional CREEP in the solver, not by a weak grip?

The diagnostic (scripts/grasp_diagnose.py) shows the tool sliding through the jaws at a
near-constant ~12 mm/s under a CONSTANT 25 N per pad with pad friction 2.0 -- including
while the arm is completely stationary. Coulomb friction cannot do that (25 N x 2.0 is
~50 N of tangential capacity against a 0.45 N tool), so it is solver slip.

This sweeps the two knobs that govern it -- `impratio` (friction stiffness relative to
normal) and `noslip_iterations` (the dedicated post-pass that removes exactly this drift)
-- and reports success AND the measured creep, so the fix is judged by mechanism, not only
by score.

    render_venv\\Scripts\\python.exe scripts\\_creep_sweep.py [seeds]
"""
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp
from bw.sim.workshop import GRASP_TOOLS, TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim
from grasp_diagnose import Probe, analyse

GRID = [(10, 0), (10, 3), (10, 10), (50, 0), (100, 0), (100, 3), (100, 10)]


def run_config(sim, ik, impratio, noslip, seeds):
    sim.m.opt.impratio = impratio
    sim.m.opt.noslip_iterations = noslip
    out = []
    for seed in range(seeds):
        for tool in GRASP_TOOLS:
            ti = TOOL_NAMES.index(tool)
            rng = np.random.default_rng(1000 * seed + ti)
            sim.reset(rng, target=tool, arm_q=SCAN_Q)
            probe = Probe(sim, tool)
            try:
                r = run_grasp(sim, ik, tool, rng, on_phase=probe.set_phase)
            finally:
                probe.close()
            s = analyse(probe.rows)
            s.update(tool=tool, success=r["success"], seed=seed)
            out.append(s)
    return out


def main(seeds=4):
    sim = WorkshopSim()
    ik = ArmIK(sim.m)
    print(f"baseline model: impratio={sim.m.opt.impratio} noslip={sim.m.opt.noslip_iterations} "
          f"cone={sim.m.opt.cone}")
    print(f"{'impratio':>9} {'noslip':>7} {'success':>9} {'creep_med':>10} {'creep_max':>10}   per tool")
    for impratio, noslip in GRID:
        t0 = time.time()
        res = run_config(sim, ik, impratio, noslip, seeds)
        ok = sum(r["success"] for r in res)
        creeps = [r["slip_mm_final"] for r in res if r["slip_mm_final"] is not None]
        per = {t: sum(r["success"] for r in res if r["tool"] == t) for t in GRASP_TOOLS}
        print(f"{impratio:9.0f} {noslip:7d} {ok:4d}/{len(res):<4d} {np.median(creeps):10.1f} "
              f"{max(creeps):10.1f}   {per}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
