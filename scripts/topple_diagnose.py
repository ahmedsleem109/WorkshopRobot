"""T2.3 diagnosis: where does a released tool TOPPLE to, and is it predictable at release?

    render_venv\Scripts\python.exe scripts\topple_diagnose.py [seeds] [--tool T] [--table X]

Per transfer (teleported base -- this scores the place, not the walk): the tool's xy and its
long axis at the moment the jaws start to open, and where it ends up after the verify hold.
Prints the landing displacement against the release lean, in the zone frame.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp
from bw.manip.scripted_place import run_place
from bw.sim.workshop import PLACE_STATION, PLACE_ZONE, RACK_STATION, TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="?", type=int, default=25)
    ap.add_argument("--tool", default="wrench_13mm", help="a tool name, or 'all'")
    ap.add_argument("--table", default="table_b")
    a = ap.parse_args()
    sim = WorkshopSim()
    ik = ArmIK(sim.m)
    tools = TOOL_NAMES if a.tool == "all" else [a.tool]
    summary = {}
    for tool in tools:
        a.tool = tool
        summary[tool] = one_tool(sim, ik, a)
    print("\nper tool: topple along the approach axis (+ = away from the robot), mm")
    for t, (ok, n, along, across) in summary.items():
        if n:
            print(f"  {t:12s} {ok}/{n} ok   along median {np.median(along):+5.0f}  p10 {np.percentile(along,10):+5.0f}  "
                  f"p90 {np.percentile(along,90):+5.0f}   across sd {np.std(across):4.0f}")


def one_tool(sim, ik, a):
    tb = sim.tool_body[a.tool]
    zone = np.array(PLACE_ZONE[a.table])
    yaw = PLACE_STATION[a.table][2]
    ah = np.array([np.cos(yaw), np.sin(yaw)])           # approach: robot -> table
    rows = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 * seed + TOOL_NAMES.index(a.tool))
        sim.reset(rng, target=a.tool, arm_q=SCAN_Q, base_pose=RACK_STATION)
        g = run_grasp(sim, ik, a.tool, rng)
        if not g["success"]:
            print(a.tool, seed, "grasp failed")
            continue
        sim.teleport_base(PLACE_STATION[a.table], carry=a.tool)
        sim.settle(0.3)
        snap = {}

        def ph(label):
            if label == "release":
                R = sim.d.xmat[tb].reshape(3, 3)
                snap["p"] = sim.gt_tool_pos(a.tool).copy()
                snap["axis"] = R[:, 0].copy()          # tool long axis (x)
                snap["ee"] = sim.ee_pos().copy()
        p = run_place(sim, ik, a.tool, a.table, rng, on_phase=ph)
        if "p" not in snap:
            print(seed, "no release", p.get("reason"))
            continue
        end = sim.gt_tool_pos(a.tool)
        ax = snap["axis"]
        # lean = horizontal direction the tool's UPPER end points (the end it will fall toward
        # is ambiguous; report both the axis tilt and the displacement)
        up_end = ax if ax[2] > 0 else -ax
        lean_xy = up_end[:2]
        disp = end[:2] - snap["p"][:2]
        rows.append(dict(seed=seed, ok=p["success"], reason=p["reason"],
                         rel=snap["p"][:2] - zone, end=end[:2] - zone, disp=disp,
                         lean=lean_xy, tilt=np.degrees(np.arccos(abs(ax[2]))),
                         ee_rel=snap["ee"][:2] - zone, end_z=end[2]))
        r = rows[-1]
        print(f"{a.tool[:8]} {seed:2d} {'OK ' if r['ok'] else 'X  '} {r['reason']:13s} release({1000*r['rel'][0]:+4.0f},{1000*r['rel'][1]:+4.0f}) "
              f"end({1000*r['end'][0]:+4.0f},{1000*r['end'][1]:+4.0f}) disp({1000*disp[0]:+4.0f},{1000*disp[1]:+4.0f}) "
              f"|d|={1000*np.linalg.norm(disp):3.0f}  tilt_from_vertical={r['tilt']:4.1f}deg "
              f"upper_end_xy({lean_xy[0]:+.2f},{lean_xy[1]:+.2f})  ee-tool({1000*(snap['ee'][0]-snap['p'][0]):+.0f},{1000*(snap['ee'][1]-snap['p'][1]):+.0f})",
              flush=True)
    if rows:
        D = np.array([r["disp"] for r in rows])
        L = np.array([r["lean"] for r in rows])
        cos = [float(d @ l / (np.linalg.norm(d) * np.linalg.norm(l) + 1e-9)) for d, l in zip(D, L)]
        print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} ok; displacement median {1000*np.median(np.linalg.norm(D,axis=1)):.0f} mm, "
              f"mean ({1000*D[:,0].mean():+.0f},{1000*D[:,1].mean():+.0f}) mm, sd ({1000*D[:,0].std():.0f},{1000*D[:,1].std():.0f}) mm")
        print(f"cos(displacement, upper-end lean): median {np.median(cos):+.2f}")
        along = [1000 * float(d @ ah) for d in D]
        across = [1000 * float(d @ np.array([-ah[1], ah[0]])) for d in D]
        return sum(r["ok"] for r in rows), len(rows), along, across
    return 0, 0, [], []


if __name__ == "__main__":
    main()
