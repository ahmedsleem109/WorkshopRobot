#!/bin/bash
# T7.2: SmolVLA fine-tune on pick + place demos (local 6 GB GPU: frozen vision encoder, action
# expert only -- smolvla_base defaults). Checkpoints every SAVE steps for eval-based selection.
#
# num_workers=2, not 6 (2026-09-20). Six pt_data_worker processes held ~0.83 GB EACH -- ~5 GB,
# nearly twice the trainer's own 2.66 GB -- on a host with 15.9 GB total, and that is what put
# the machine into the memory pressure that SIGKILLed a run at step 3,900 of 6,000. They buy
# nothing here: measured over the whole run, data_s is 0.012 s against updt_s 1.05 s, i.e. the
# GPU waits about 1% of a step for data. Two workers keep that margin and give back ~3.3 GB.
cd ~/bringwrench
NAME=${1:-vla_full}; STEPS=${2:-8000}; BS=${3:-16}; DS=${4:-bw_demos}; SAVE=${5:-2000}
.venv-vla/bin/lerobot-train --policy.path=$HOME/bringwrench/models/smolvla_base \
  --dataset.repo_id=local/$DS --dataset.root=$HOME/bringwrench/data/$DS \
  --batch_size=$BS --steps=$STEPS --save_freq=$SAVE --log_freq=100 \
  --output_dir=$HOME/bringwrench/runs/$NAME --policy.push_to_hub=false --policy.device=cuda \
  --wandb.enable=false --num_workers=${WORKERS:-2} --seed=0 2>&1 | grep --line-buffered -v "^Svt\|Loading weights"
