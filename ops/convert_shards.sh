#!/bin/bash
# re-run given shards of an 8-way conversion, then merge
cd /mnt/d/bringwrench
PY=~/bringwrench/.venv-vla/bin/python
ROOT=$HOME/bringwrench/data/bw_demos
for i in "$@"; do
  $PY scripts/to_lerobot.py --raw /mnt/d/bw_data/raw --root $ROOT --repo-id local/bw_demos --shard $i/8 > /tmp/conv_$i.log 2>&1 &
done
wait
grep -ah "^done:\|^skip" /tmp/conv_*.log
$PY scripts/to_lerobot.py --root $ROOT --repo-id local/bw_demos --merge 8 2>&1 | grep -av "libx264\|^Svt\|mov,mp4\|^\[mp4" | tail -3
