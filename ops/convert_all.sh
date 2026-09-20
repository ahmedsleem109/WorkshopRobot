#!/bin/bash
# T6.4: convert every collected episode to one LeRobotDataset, 8 shards in parallel, then merge.
cd /mnt/d/bringwrench
PY=~/bringwrench/.venv-vla/bin/python
ROOT=${1:-$HOME/bringwrench/data/bw_demos}
N=8
for i in $(seq 0 $((N-1))); do
  $PY scripts/to_lerobot.py --raw /mnt/d/bw_data/raw --root $ROOT --repo-id local/bw_demos --shard $i/$N > /tmp/conv_$i.log 2>&1 &
done
wait
grep -h "done:\|skip\|Error" /tmp/conv_*.log
$PY scripts/to_lerobot.py --root $ROOT --repo-id local/bw_demos --merge $N 2>&1 | grep -v "^Svt\|^\[mp4\|mov,mp4" | tail -3
