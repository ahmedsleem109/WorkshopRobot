#!/bin/bash
# T11: the end-to-end suite, ONE PROCESS PER SUITE.
#
# Trials are not independent inside a process -- the sim and the orchestrator's rng carry over
# between them -- so running several suites in one process makes the numbers depend on the
# order they were listed in (measured 2026-09-20: `transfer,drop` gave drop 6/10, `drop` alone
# gave 9/10, same code, same seeds). One process per suite removes that.
PY=${PY:-"D:/hexapod/render_venv/Scripts/python.exe"}
N=${1:-10}
OUT=${2:-runs/eval/s6}
mkdir -p "$OUT"
for s in nominal transfer missing drop ambiguous retarget obstacle; do
  echo "=== $s ($N seeds) ==="
  "$PY" scripts/eval_suite.py --suite "$s" --n "$N" --out "$OUT/$s.json" | tail -3
done
