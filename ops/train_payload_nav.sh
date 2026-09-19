#!/bin/bash
# Navigation fine-tune (turn in place / back up / sidestep) -- see configs/payload_nav.yaml.
mkdir -p ~/bringwrench/runs ~/bringwrench/logs
cd ~/bringwrench/runs
pkill -f 'bw[.]locomotion[.]train_payload' || true
export PYTHONPATH=/mnt/d/bringwrench:$HOME/go2-stairs
exec ~/go2-stairs/.venv/bin/python -u -W ignore -m bw.locomotion.train_payload \
    --config /mnt/d/bringwrench/configs/payload_nav.yaml "$@" > ~/bringwrench/logs/payload_nav3.log 2>&1
