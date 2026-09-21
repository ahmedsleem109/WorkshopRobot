#!/bin/bash
# T3.5: resume the L5 step curriculum from where session 5 stopped.
#
# `payload_l5` ran 4,587,520 of its 10M steps and was never evaluated (success 0.25 at level 5,
# reward 2646). train.py has no resume flag: a resume IS a warm start from the last checkpoint,
# with the remaining budget. level_init 5 in configs/payload_l5.yaml puts the curriculum back
# where it was; the success EMA restarts, which only costs a few hundred thousand steps.
set -e
L5=$HOME/bringwrench/runs/results/2026-09-18_12-19-23-payload_l5/checkpoints/step_4587520
STEPS=${1:-5500000}
[ -e "$L5" ] || { echo "no checkpoint at $L5"; exit 1; }
mkdir -p ~/bringwrench/runs ~/bringwrench/logs
cd ~/bringwrench/runs
pkill -f 'bw[.]locomotion[.]train_payload' || true
export PYTHONPATH=/mnt/d/bringwrench:$HOME/go2-stairs
exec ~/go2-stairs/.venv/bin/python -u -W ignore -m bw.locomotion.train_payload \
    --config /mnt/d/bringwrench/configs/payload_l5.yaml \
    --run_name payload_l5b --init_params "$L5" --num_timesteps "$STEPS" \
    > ~/bringwrench/logs/payload_l5b.log 2>&1
