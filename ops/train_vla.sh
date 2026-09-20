#!/bin/bash
# T7.2: SmolVLA fine-tune on pick + place demos (local 6 GB GPU: frozen vision encoder, action
# expert only -- smolvla_base defaults). Checkpoints every SAVE steps for eval-based selection.
cd ~/bringwrench
NAME=${1:-vla_full}; STEPS=${2:-8000}; BS=${3:-16}; DS=${4:-bw_demos}; SAVE=${5:-2000}
.venv-vla/bin/lerobot-train --policy.path=$HOME/bringwrench/models/smolvla_base \
  --dataset.repo_id=local/$DS --dataset.root=$HOME/bringwrench/data/$DS \
  --batch_size=$BS --steps=$STEPS --save_freq=$SAVE --log_freq=100 \
  --output_dir=$HOME/bringwrench/runs/$NAME --policy.push_to_hub=false --policy.device=cuda \
  --wandb.enable=false --num_workers=6 --seed=0 2>&1 | grep --line-buffered -v "^Svt\|Loading weights"
