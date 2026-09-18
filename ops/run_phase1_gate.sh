#!/bin/bash
# Phase 1 gate + push-recovery ablation. First execution of eval_phase1.py -- expect API
# friction (STATUS known bug #5).
PAYLOAD_CKPT=${1:-~/bringwrench/runs/results/2026-09-18_02-47-57-payload/checkpoints/step_33013760}
ORIGINAL_CKPT=${2:-~/go2-stairs/results/2026-08-06_17-17-05-stairs_run7/checkpoints/final}
mkdir -p ~/bringwrench/logs
cd /mnt/d/bringwrench
export PYTHONPATH=/mnt/d/bringwrench:$HOME/go2-stairs
exec ~/go2-stairs/.venv/bin/python -u -W ignore -m bw.locomotion.eval_phase1 \
    --original "$ORIGINAL_CKPT" \
    --payload  "$PAYLOAD_CKPT" \
    --out ~/bringwrench/runs/results/phase1 \
    > ~/bringwrench/logs/phase1.log 2>&1
