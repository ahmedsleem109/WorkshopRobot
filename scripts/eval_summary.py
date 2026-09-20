"""Tabulate the per-suite JSON dumps ops/run_eval_suites.sh writes.

    python scripts/eval_summary.py [runs/eval/s6]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ORDER = ["nominal", "transfer", "missing", "drop", "ambiguous", "retarget", "obstacle"]


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/eval/s6")
    files = {p.stem: p for p in d.glob("*.json")}
    tot_ok = tot_n = 0
    print(f"{'suite':12s} {'success':>9s}   failure causes")
    for s in ORDER + sorted(set(files) - set(ORDER)):
        if s not in files:
            continue
        rows = json.loads(files[s].read_text())
        ok = sum(bool(r["success"]) for r in rows)
        causes = Counter(r["cause"] for r in rows if not r["success"])
        tot_ok, tot_n = tot_ok + ok, tot_n + len(rows)
        print(f"{s:12s} {ok:4d}/{len(rows):<4d}  {dict(causes)}")
    if tot_n:
        print(f"{'TOTAL':12s} {tot_ok:4d}/{tot_n:<4d}  ({100 * tot_ok / tot_n:.0f}%)")


if __name__ == "__main__":
    main()
