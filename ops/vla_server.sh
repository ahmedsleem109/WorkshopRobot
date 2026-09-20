#!/bin/bash
# SmolVLA policy server (keep this process alive: WSL kills children when the session ends)
# usage: bash ops/vla_server.sh CKPT_DIR [cpu|cuda]
cd /mnt/d/bringwrench
exec ~/bringwrench/.venv-vla/bin/python bw/policy/vla_server.py --ckpt "$1" --device "${2:-cuda}"
