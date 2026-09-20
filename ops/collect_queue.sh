#!/bin/bash
# T6 collection queue: chunks of 25 runs, N parallel Windows sim workers (RAM ~0.6 GB each).
# usage: bash ops/collect_queue.sh START END WORKERS
cd /d/bringwrench
S=${1:-0}; E=${2:-600}; N=${3:-4}
seq $S 25 $((E-1)) | xargs -P $N -I{} sh -c \
  'D:/hexapod/render_venv/Scripts/python.exe scripts/collect_demos.py {} $(( {} + 25 )) --out D:/bw_data/raw > D:/bw_data/lane_{}.txt 2>&1'
