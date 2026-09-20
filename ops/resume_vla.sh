#!/bin/bash
# T7: resume the SmolVLA fine-tune from its last checkpoint.
#
# lerobot 0.6.1 resumes from the checkpoint's OWN train_config.json (`--resume=true`), which
# restores the optimizer, scheduler and rng state as well as the weights -- so a run that was
# killed mid-way costs only the steps since its last save, not the whole run. Every other flag
# on the command line is IGNORED when resuming, by design.
#
# Session 6: the first attempt was killed by the host at step 3,900 of 6,000 (no traceback --
# an external SIGKILL under memory pressure); `checkpoints/003000` had the full training state.
set -e
RUN=${1:-vla_full}
CFG=$HOME/bringwrench/runs/$RUN/checkpoints/last/pretrained_model/train_config.json
[ -f "$CFG" ] || { echo "no checkpoint config at $CFG"; exit 1; }
cd ~/bringwrench
exec .venv-vla/bin/lerobot-train --config_path="$CFG" --resume=true \
  2>&1 | grep --line-buffered -v "^Svt\|Loading weights"
