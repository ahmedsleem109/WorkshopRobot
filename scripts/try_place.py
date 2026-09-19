"""T2.3 benchmark: full two-table transfer -- pick from the rack, move, place in the zone.

    render_venv\\Scripts\\python.exe scripts\\try_place.py [seeds] [--table table_a|table_b]
                                                           [--roll DEG] [-v]

Stages: grasp (the pick failed -- see try_grasp.py) -> place_ik -> outside_zone (released,
but the tool did not end up in the marked zone) -> not_settled (still moving 2 s later) ->
still_held -> ok.

The base move between stations is WorkshopSim.teleport_base, a deliberate stand-in for Layer
3 navigation; this benchmark scores the manipulation, and T5/T9 own the walking.
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.manip import scripted_place as sp
from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp
from bw.manip.scripted_place import run_place
from bw.sim.workshop import (GRASP_TOOLS, PLACE_STATION, PLACE_ZONE, RACK_STATION,
                             TOOL_NAMES)
from bw.sim.workshop_sim import WorkshopSim
from bw.task.spec import Snapshot, Task, evaluate


def stage(g, p):
    if not g["success"]:
        return "grasp"
    if p.get("reason") == "place_ik_fail":
        return "place_ik"
    if p.get("reason") == "not_holding":
        return "dropped_in_transit"
    if p["success"]:
        return "ok"
    return p.get("reason", "fail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="?", type=int, default=8)
    ap.add_argument("--table", default="table_b")
    ap.add_argument("--roll", type=float, default=None, help="PLACE_ROLL in degrees")
    ap.add_argument("--back", type=float, default=None,
                    help="override the station's distance behind the zone centre (m)")
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    if args.roll is not None:
        sp.PLACE_ROLL = np.radians(args.roll)

    station = PLACE_STATION[args.table]
    if args.back is not None:
        zx, zy = PLACE_ZONE[args.table]
        yaw = station[2]
        station = (zx - args.back * np.cos(yaw), zy - args.back * np.sin(yaw), yaw)

    sim = WorkshopSim()
    ik = ArmIK(sim.m)
    stages = Counter()
    per = {t: Counter() for t in GRASP_TOOLS}
    dists = []
    t0 = time.time()
    for seed in range(args.seeds):
        for tool in GRASP_TOOLS:
            ti = TOOL_NAMES.index(tool)
            rng = np.random.default_rng(1000 * seed + ti)
            present = sim.reset(rng, target=tool, arm_q=SCAN_Q, base_pose=RACK_STATION)
            start = Snapshot.take(sim, present)
            g = run_grasp(sim, ik, tool, rng)
            p = {}
            if g["success"]:
                sim.teleport_base(station, carry=tool)
                sim.settle(0.3)
                p = run_place(sim, ik, tool, args.table, rng)
                if p.get("reason") not in ("place_ik_fail", "not_holding"):
                    # the SHARED success definition (bw/task/spec.py), not run_place's own
                    # verdict -- the collector and the evaluator must use the same rule
                    p = {**p, **evaluate(sim, Task("transfer", tool, args.table), start)}
                if p.get("dist_from_centre") is not None:
                    dists.append(p["dist_from_centre"])
            st = stage(g, p)
            stages[st] += 1
            per[tool][st] += 1
            if args.v:
                print(f"  {seed} {tool:12s} {st:12s} {p}", flush=True)

    n = sum(stages.values())
    print(f"\n{time.time() - t0:.0f}s, {n} transfers to {args.table} "
          f"(PLACE_ROLL {'auto' if sp.PLACE_ROLL is None else f'{np.degrees(sp.PLACE_ROLL):.0f} deg'})")
    for t in GRASP_TOOLS:
        print(f"  {t:12s} {per[t]['ok']}/{sum(per[t].values())}   {dict(per[t])}")
    print("overall", dict(stages), f"success {stages['ok'] / max(n, 1):.0%}")
    if dists:
        print(f"placed {np.median(dists) * 1000:.0f} mm from the zone centre (median), "
              f"worst {max(dists) * 1000:.0f} mm")


if __name__ == "__main__":
    main()
