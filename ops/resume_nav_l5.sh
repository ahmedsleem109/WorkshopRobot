#!/bin/bash
# Warm-start payload_nav_l5 from where it was stopped. Same pattern as ops/resume_payload_l5.sh:
# train.py has no resume flag, so a resume IS a warm start from the last checkpoint with the
# remaining budget. Stopped at step 3,686,400 of 6,000,000 on 2026-09-21 to give the host's RAM back
# to the desktop -- a SIGSTOPped process still holds its pages, so the VM had to go.
# `[ -e ]` not `[ -d ]`: brax writes a checkpoint as a FILE (this cost a silent no-op once already).
set -e
CKPT=${1:-$(ls -d $HOME/bringwrench/runs/results/*payload_nav_l5/checkpoints/step_3686400 | tail -1)}
STEPS=${2:-2313600}
[ -e "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
mkdir -p ~/bringwrench/runs ~/bringwrench/logs
cd ~/bringwrench/runs
pkill -f 'bw[.]locomotion[.]train_payload' || true
export PYTHONPATH=/mnt/d/bringwrench:$HOME/go2-stairs
exec ~/go2-stairs/.venv/bin/python -u -W ignore -m bw.locomotion.train_payload \
    --config /mnt/d/bringwrench/configs/payload_nav_l5.yaml \
    --run_name payload_nav_l5b --init_params "$CKPT" --num_timesteps "$STEPS" \
    > ~/bringwrench/logs/payload_nav_l5b.log 2>&1
