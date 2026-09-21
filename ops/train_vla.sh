#!/bin/bash
# T7.2: SmolVLA fine-tune on pick + place demos (local 6 GB GPU: frozen vision encoder, action
# expert only -- smolvla_base defaults). Checkpoints every SAVE steps for eval-based selection.
#
# num_workers=2, not 6 (2026-09-20). Six pt_data_worker processes held ~0.83 GB EACH -- ~5 GB,
# nearly twice the trainer's own 2.66 GB -- on a host with 15.9 GB total, and that is what put
# the machine into the memory pressure that SIGKILLed a run at step 3,900 of 6,000. They buy
# nothing here: measured over the whole run, data_s is 0.012 s against updt_s 1.05 s, i.e. the
# GPU waits about 1% of a step for data. Two workers keep that margin and give back ~3.3 GB.
# QUANTILE NORMALISATION, not SmolVLA's MEAN_STD default (session 7). The divisor is what this
# project kept getting wrong: measured on the fine phase, joint 1's delta std is 10.3 mrad over pick
# episodes and 149.4 mrad over place ones (p99 756 mrad, max 2,428 -- IK-branch switches in the
# place skill), so under one mean/std scale the pick's median j1 motion of 1.8 mrad is ~1% of the
# normalised range, inside the 4-8% band session 6 measured as unlearnable. pi0.5 defaults
# STATE/ACTION to QUANTILES for exactly this reason: a rarely-used or outlier-heavy dimension gets
# a std that makes normalised values explode. q01/q99 are already in the dataset stats.
# NOTE the CLI form: --policy.normalization_mapping.ACTION=QUANTILES is REJECTED as an
# unrecognized argument; the whole dict must go in as JSON. Verified end to end -- the written
# checkpoint config reads STATE/ACTION QUANTILES. Set NORM=MEAN_STD to reproduce the earlier runs.
cd ~/bringwrench
NAME=${1:-vla_full}; STEPS=${2:-8000}; BS=${3:-16}; DS=${4:-bw_demos}; SAVE=${5:-2000}
NORM=${NORM:-QUANTILES}
.venv-vla/bin/lerobot-train --policy.path=$HOME/bringwrench/models/smolvla_base \
  --policy.normalization_mapping="{\"VISUAL\":\"IDENTITY\",\"STATE\":\"$NORM\",\"ACTION\":\"$NORM\"}" \
  --dataset.repo_id=local/$DS --dataset.root=$HOME/bringwrench/data/$DS \
  --batch_size=$BS --steps=$STEPS --save_freq=$SAVE --log_freq=100 \
  --output_dir=$HOME/bringwrench/runs/$NAME --policy.push_to_hub=false --policy.device=cuda \
  --wandb.enable=false --num_workers=${WORKERS:-2} --seed=0 2>&1 | grep --line-buffered -v "^Svt\|Loading weights"
