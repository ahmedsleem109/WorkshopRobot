"""Choose a seed worth rendering from a suite's scored results.

    python scripts/pick_clip_seed.py runs/eval/s8/nominal.json [--n 1]

Prints one seed per line. Only successful trials are considered, so an unattended render cannot
produce a clip of a failure -- which is the whole risk of rendering from a fixed seed chosen
before the suite ran.

Among the successes it prefers the trial whose wall-clock is closest to the median: the fastest
run is usually the one where everything fell into place and the slowest is usually a near-miss
with retries, and neither is representative of the system it is meant to illustrate.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text())
    ok = [r for r in rows if r.get("success")]
    if not ok:
        sys.exit(f"no successful trial in {args.results} ({len(rows)} scored) -- nothing to render")

    times = sorted(r["wall_s"] for r in ok)
    median = times[len(times) // 2]
    ok.sort(key=lambda r: (abs(r["wall_s"] - median), r["seed"]))
    for r in ok[:args.n]:
        print(r["seed"])


if __name__ == "__main__":
    main()
