"""Scripted-grasp trial: per-tool success and a failure-stage breakdown.

    render_venv\Scripts\python.exe scripts\try_grasp.py [seeds]

Stages: ik (no reachable plan) -> no_grip (pads never closed on the tool) ->
no_lift (gripped but never left the rack) -> dropped (lifted, lost during the retreat).
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
from bw.sim.workshop import TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim


def stage(r):
    if r.get("reason") == "ik_fail":
        return "ik"
    if not r.get("pad_contacts"):
        return "no_grip"
    if r.get("lift_only", 0) < 0.05:
        return "no_lift"
    if not r["success"]:
        return "dropped"
    return "ok"


def main(n=4, verbose=False):
    sim = WorkshopSim()
    ik = ArmIK(sim.m)
    res = {t: [] for t in TOOL_NAMES}
    stages = Counter()
    t0 = time.time()
    for seed in range(n):
        for ti, tool in enumerate(TOOL_NAMES):
            rng = np.random.default_rng(1000 * seed + ti)
            sim.reset(rng, target=tool, arm_q=SCAN_Q)
            r = run_grasp(sim, ik, tool, rng)
            res[tool].append(r)
            stages[stage(r)] += 1
            if verbose:
                print(seed, tool, r, flush=True)
    print(f"{time.time() - t0:.0f}s, {n * len(TOOL_NAMES)} episodes")
    for t, rs in res.items():
        st = Counter(stage(r) for r in rs)
        print(f"  {t:12s} {sum(r['success'] for r in rs)}/{len(rs)}   {dict(st)}")
    print("overall", dict(stages), f"success {stages['ok'] / max(sum(stages.values()), 1):.0%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4, "-v" in sys.argv)
