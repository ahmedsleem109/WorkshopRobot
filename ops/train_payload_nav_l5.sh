#!/bin/bash
# T3.5, the DEPLOYABLE half: put the 12 cm step inside the NAV lineage's curriculum.
# See configs/payload_nav_l5.yaml for why this is a separate run from payload_l5b.
set -e
STEPS=${1:-6000000}
mkdir -p ~/bringwrench/runs ~/bringwrench/logs
cd ~/bringwrench/runs
pkill -f 'bw[.]locomotion[.]train_payload' || true
export PYTHONPATH=/mnt/d/bringwrench:$HOME/go2-stairs
exec ~/go2-stairs/.venv/bin/python -u -W ignore -m bw.locomotion.train_payload \
    --config /mnt/d/bringwrench/configs/payload_nav_l5.yaml --num_timesteps "$STEPS" \
    > ~/bringwrench/logs/payload_nav_l5.log 2>&1
