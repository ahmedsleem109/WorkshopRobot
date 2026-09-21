"""T7 Tier-1 diagnostic: does the SERVING EXECUTION PATH grasp when the actions are PERFECT?

    D:\\hexapod\\render_venv\\Scripts\\python.exe scripts\\vla_exec_check.py [N] [--raw DIR]

No policy and no GPU. It replays a demonstration's OWN recorded actions through the same loop
`bw/policy/vla.py:run_skill` uses -- 10-action chunks at 10 Hz, deltas re-applied to the LIVE joint
position -- into the same scene the demonstration was collected in, and scores the result with the
same `evaluate(Task("pick"))` the VLA eval uses. Perfect actions that fail to grasp mean the loss is
in the execution path, not in the policy, and no amount of training or data can fix it.

Two factors, four cells:

  lock   the legs. Every demonstration was recorded with the legs STAND-LOCKED: `run_grasp` calls
         `sim.lock_stance()` before it plans (scripted_grasp.py:192), and `Locomotion.lock_stance`
         exists because "under the policy the arm's reach and pull push the standing base back
         25-260 mm (it steps away from the load), so the jaws arrive short or the tool is dragged
         against the rack" -- measured, session 5. The VLA path never calls it: it is absent from
         bw/policy/vla.py and from scripts/eval_vla.py. So every rollout that produced the 1/20
         was executed on a base that walks away from the rack while the arm reaches, against demos
         where it could not.

  mode   `abs` applies the recorded action as an absolute joint target; `delta` computes the
         recorded q_cmd - q_state and applies it to the live position, which is exactly what the
         client does for a delta checkpoint. If `delta` fails where `abs` succeeds, the delta
         serving convention is lossy and the representation -- not the data -- is the problem.

Reported per cell: grasp success, how far the BASE moved during the episode, and how far the
gripper ended from where the demonstration's own actions should have put it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.manip.scripted_grasp import SCAN_Q
from bw.sim.workshop import GRASP_TOOLS, RACK_STATION
from bw.sim.workshop_sim import WorkshopSim
from bw.task.spec import Snapshot, Task, evaluate

CHUNK = 10          # actions the server hands over per query (n_action_steps in ops/vla_server.sh)
RATE = 10.0


def collect_scene(sim, seed: int):
    """Rebuild the scene `scripts/collect_demos.py:run` built for this seed -- same rng stream, so
    the same tool, the same clutter, the same base pose and the same scan pose."""
    rng = np.random.default_rng(10_000_019 + seed)
    tool = GRASP_TOOLS[int(rng.integers(len(GRASP_TOOLS)))]
    rng.integers(2)                               # the destination table draw, kept for alignment
    base = (RACK_STATION[0] + rng.uniform(-0.03, 0.03), RACK_STATION[1] + rng.uniform(-0.03, 0.03),
            RACK_STATION[2] + np.radians(rng.uniform(-6, 6)))
    scan = SCAN_Q.copy()
    scan[:6] += rng.uniform(-0.05, 0.05, 6)
    present = sim.reset(rng, target=tool, arm_q=scan, base_pose=base)
    return tool, Snapshot.take(sim, present)


def replay(sim, action, state, mode: str):
    """Execute recorded actions the way bw/policy/vla.py executes predicted ones."""
    for i in range(0, len(action), CHUNK):
        for a, s in zip(action[i:i + CHUNK], state[i:i + CHUNK]):
            if mode == "delta":
                d = a[:6] - s[:6]                      # what to_lerobot.py --delta records
                cmd = np.concatenate([sim.arm_q()[:6] + d, [a[6]]])
            else:
                cmd = np.asarray(a, float)
            sim.move_arm(cmd, 1.0 / RATE, None, RATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=8)
    ap.add_argument("--raw", default="D:/bw_data/raw")
    ap.add_argument("--cells", default="abs:lock,abs:free,delta:lock,delta:free")
    args = ap.parse_args()

    eps = sorted(p for p in Path(args.raw).glob("*_pick") if (p / "data.npz").exists())[:args.n]
    if not eps:
        sys.exit(f"no pick episodes under {args.raw}")
    print(f"{len(eps)} demonstrations, replayed through the serving loop "
          f"(chunk {CHUNK} @ {RATE:.0f} Hz)\n")

    sim = WorkshopSim()
    sim.attach_locomotion()
    rows = {}
    for cell in args.cells.split(","):
        mode, legs = cell.split(":")
        ok = drift = 0.0
        n = 0
        per = []
        for ep in eps:
            seed = int(json.loads((ep / "meta.json").read_text())["seed"])
            d = np.load(ep / "data.npz")
            tool, start = collect_scene(sim, seed)
            if legs == "lock":
                sim.lock_stance()                 # what run_grasp does, and the VLA path does not
            b0 = sim.d.qpos[0:2].copy()
            replay(sim, d["action"], d["state"], mode)
            sim.settle(1.0)
            ev = evaluate(sim, Task("pick", tool), start)
            mm = 1000 * float(np.linalg.norm(sim.d.qpos[0:2] - b0))
            per.append((seed, tool, ev["success"], ev["reason"], round(mm, 1)))
            ok += bool(ev["success"])
            drift += mm
            n += 1
            print(f"  {cell:11s} seed {seed:6d} {tool:12s} "
                  f"{'OK ' if ev['success'] else 'FAIL'} {ev['reason']:14s} base {mm:5.1f} mm",
                  flush=True)
        rows[cell] = {"grasp": f"{int(ok)}/{n}", "base_mm": round(drift / max(n, 1), 1),
                      "per": per}

    print(f"\n{'cell':14s} {'grasp':>8s} {'mean base move':>16s}")
    for cell, r in rows.items():
        print(f"{cell:14s} {r['grasp']:>8s} {r['base_mm']:>13.1f} mm")
    out = ROOT / "runs/eval/vla_exec_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    print("READING IT: a cell that fails with PERFECT actions is an execution-path failure -- the "
          "policy cannot be blamed for it, and no dataset change repairs it.")


if __name__ == "__main__":
    main()
