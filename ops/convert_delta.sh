#!/bin/bash
# Build the DELTA-action dataset from the same raw episodes (no re-collection).
# 8 shards in parallel then merge, exactly like ops/convert_all.sh, with --delta.
cd /mnt/d/bringwrench
PY=~/bringwrench/.venv-vla/bin/python
ROOT=${1:-$HOME/bringwrench/data/bw_demos_delta}
REPO=local/bw_demos_delta
N=8
for i in $(seq 0 $((N-1))); do
  $PY scripts/to_lerobot.py --raw /mnt/d/bw_data/raw --root $ROOT --repo-id $REPO \
      --delta --shard $i/$N > /tmp/convd_$i.log 2>&1 &
done
wait
grep -h "done:\|skip\|Error" /tmp/convd_*.log | tail -20
$PY scripts/to_lerobot.py --root $ROOT --repo-id $REPO --merge $N 2>&1 | grep -v "^Svt\|^\[mp4\|mov,mp4" | tail -3
