"""Payload-aware fine-tune: go2-stairs' train.py, pointed at the Go2+Z1 env.

    cd ~/bringwrench/runs && ~/go2-stairs/.venv/bin/python -u -m bw.locomotion.train_payload \
        --config /mnt/d/bringwrench/configs/payload.yaml

train.py is reused unmodified (PPO settings, checkpointing, curriculum logging, the
jax.device_put_replicated shim, GPU duty cycle). Only its env factory and its
randomization factory are swapped, so every lesson baked into it carries over.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GO2 = Path.home() / "go2-stairs"
for p in (str(REPO), str(GO2)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import train  # noqa: E402  (go2-stairs/train.py)

from bw.locomotion.domain_rand import make_randomization_fn  # noqa: E402
from bw.locomotion.go2_arm_env import ArmStairsConfig, Go2ArmEnv  # noqa: E402


def build_env(cfg):
    return Go2ArmEnv(config=ArmStairsConfig(**cfg.get("env", {})))


train.build_env = build_env
train.make_randomization_fn = make_randomization_fn

if __name__ == "__main__":
    train.main()
