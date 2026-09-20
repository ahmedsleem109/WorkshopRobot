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
from bw.sim.workshop import GRASP_TOOLS, RACK_STATION, TOOL_NAMES
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


def main(n=4, verbose=False, walk=None, tools=None, jitter=False):
    sim = WorkshopSim()
    if walk:
        # legs on the walking policy, base at the rack station -- the mode try_place --walk
        # and the T6 collector run in
        sim.attach_locomotion(walk)
    ik = ArmIK(sim.m)
    tools = tools or GRASP_TOOLS
    res = {t: [] for t in (tools or GRASP_TOOLS)}
    stages = Counter()
    t0 = time.time()
    for seed in range(n):
        # ti is the index in TOOL_NAMES, NOT in GRASP_TOOLS: the seed is 1000*seed + ti, so
        # using the shorter list would reseed every tool and break comparison with every
        # benchmark run before the screwdriver was dropped.
        for tool in tools:
            ti = TOOL_NAMES.index(tool)
            rng = np.random.default_rng(1000 * seed + ti)
            base = RACK_STATION if walk else None
            if walk and jitter:
                # where walk_to() actually stops: ~2-3 cm / 5-8 deg from the station
                j = np.random.default_rng(777 + 1000 * seed + ti)
                base = (RACK_STATION[0] + j.uniform(-0.03, 0.03),
                        RACK_STATION[1] + j.uniform(-0.03, 0.03),
                        RACK_STATION[2] + np.radians(j.uniform(-6, 6)))
            sim.reset(rng, target=tool, arm_q=SCAN_Q, base_pose=base)
            r = run_grasp(sim, ik, tool, rng)
            res[tool].append(r)
            stages[stage(r)] += 1
            if verbose:
                print(seed, tool, r, flush=True)
    print(f"{time.time() - t0:.0f}s, {n * len(tools)} episodes")
    for t, rs in res.items():
        st = Counter(stage(r) for r in rs)
        print(f"  {t:12s} {sum(r['success'] for r in rs)}/{len(rs)}   {dict(st)}")
    print("overall", dict(stages), f"success {stages['ok'] / max(sum(stages.values()), 1):.0%}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="?", type=int, default=4)
    ap.add_argument("--walk", nargs="?", const=str(ROOT / "models/payload_nav_policy.npz"),
                    default=None, help="legs on this walking policy (base at RACK_STATION)")
    ap.add_argument("--tool", action="append", default=None)
    ap.add_argument("--jitter", action="store_true",
                    help="with --walk: base +-3 cm / +-6 deg around the station")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    main(a.seeds, a.v, a.walk, a.tool, a.jitter)
