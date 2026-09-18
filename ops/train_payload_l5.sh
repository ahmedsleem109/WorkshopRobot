#!/bin/bash
# Targeted L5 continuation -- see configs/payload_l5.yaml for why.
mkdir -p ~/bringwrench/runs ~/bringwrench/logs
cd ~/bringwrench/runs
pkill -f 'bw[.]locomotion[.]train_payload' || true
export PYTHONPATH=/mnt/d/bringwrench:$HOME/go2-stairs
exec ~/go2-stairs/.venv/bin/python -u -W ignore -m bw.locomotion.train_payload \
    --config /mnt/d/bringwrench/configs/payload_l5.yaml "$@" > ~/bringwrench/logs/payload_l5.log 2>&1
