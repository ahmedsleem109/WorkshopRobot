"""Pick the best SAVED checkpoint of a locomotion run from its own eval log.

    python scripts/pick_best_ckpt.py ~/bringwrench/logs/payload_l5b.log RUN_DIR [--metric reward|succ]

WHY THIS EXISTS (session 7). `payload_l5b` peaked at step 1,310,720 (reward 2,936, succ 0.25) and
then got WORSE: 2,570 / 0.18 by step 3,276,800. Exporting `checkpoints/final`, which is what every
ops script did, would have shipped the worst policy of the run and scored the gate on it -- the
same mistake T7.4 exists to prevent on the VLA side. The eval rows are the only comparable metric
the run produces, so the choice is made from them.

Checkpoints are written on a coarser grid than evals, so the eval row is matched to the newest
checkpoint at or before it. Prints the chosen checkpoint path on stdout and the ranking on stderr.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROW = re.compile(r"^\s*([\d,]+)\s+reward\s+(-?[\d.]+).*?succ\s+([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("run_dir")
    ap.add_argument("--metric", choices=("reward", "succ"), default="reward")
    a = ap.parse_args()

    rows = []
    for line in Path(a.log).read_text(errors="replace").splitlines():
        m = ROW.match(line)
        if m:
            rows.append((int(m.group(1).replace(",", "")), float(m.group(2)), float(m.group(3))))
    if not rows:
        sys.exit(f"no eval rows in {a.log}")

    ck = Path(a.run_dir) / "checkpoints"
    saved = sorted((int(p.name.split("_")[1]), p) for p in ck.glob("step_*"))
    if not saved:
        sys.exit(f"no checkpoints in {ck}")

    best = None
    for step, rew, succ in rows:
        prior = [s for s in saved if s[0] <= step]
        if not prior:
            continue
        val = rew if a.metric == "reward" else succ
        cand = (val, step, prior[-1][1])
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        sys.exit("no eval row has a checkpoint at or before it")

    for step, rew, succ in rows:
        print(f"  step {step:>12,}  reward {rew:9.2f}  succ {succ:.3f}", file=sys.stderr)
    print(f"chosen by {a.metric}: {best[2].name} (eval step {best[1]:,}, value {best[0]})",
          file=sys.stderr)
    print(best[2])


if __name__ == "__main__":
    main()
