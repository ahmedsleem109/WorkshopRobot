#!/bin/bash
# SmolVLA policy server (keep this process alive: WSL kills children when the session ends)
# usage: bash ops/vla_server.sh CKPT_DIR [cpu|cuda] [N_ACTION_STEPS] [delta]
#
# N_ACTION_STEPS is the OPEN-LOOP HORIZON and it matters more than anything else here. The
# checkpoint's own value is 50, and the client executes a whole chunk before it asks again --
# at 10 Hz that is FIVE SECONDS of blind motion per observation, on episodes that only last
# 9-12 s. Measured 2026-09-20: with the default 50, checkpoint 006000 scored grasp 1/20.
cd /mnt/d/bringwrench
ARGS=(--ckpt "$1" --device "${2:-cuda}")
if [ -n "$3" ]; then ARGS+=(--n-action-steps "$3"); fi
if [ "$4" = "delta" ]; then ARGS+=(--delta); fi
exec ~/bringwrench/.venv-vla/bin/python bw/policy/vla_server.py "${ARGS[@]}"
