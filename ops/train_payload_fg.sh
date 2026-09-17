#!/bin/bash
# Foreground training launcher (called from a Windows Scheduled Task via run_payload.bat).
# Args are forwarded to train.py, e.g.  --num_timesteps 3000000 --num_evals 2 --run_name smoke
mkdir -p ~/bringwrench/runs ~/bringwrench/logs
cd ~/bringwrench/runs
pkill -f 'bw[.]locomotion[.]train_payload' || true
export PYTHONPATH=/mnt/d/bringwrench:$HOME/go2-stairs
LOG=~/bringwrench/logs/${LOGNAME_OVERRIDE:-payload}.log
exec ~/go2-stairs/.venv/bin/python -u -W ignore -m bw.locomotion.train_payload \
    --config /mnt/d/bringwrench/configs/payload.yaml "$@" > "$LOG" 2>&1
