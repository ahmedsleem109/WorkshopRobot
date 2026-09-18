#!/bin/bash
# Layer-1 grounding fallback: Qwen3-VL-2B-Instruct.
#
# WHY, 2026-09-18: both Molmo2 candidates are non-viable.
#   * Cycl0/Molmo2-VideoPoint-4B-bnb-4bit ships NO pointing decode path (no
#     extract_image_points, no point tokens among its 303 added tokens) AND fails to load on
#     transformers 5.5.4 with three separate API breaks (processor kwargs, auto class,
#     ROPE_INIT_FUNCTIONS['default']).
#   * reubk/Molmo2-4B-GGUF was wanted for GBNF grammar-constrained output, but Molmo2 emits
#     points as SPECIAL TOKENS decoded with preprocessor metadata, so there is no text grammar
#     to constrain.
# Qwen3-VL is an official repo with first-class transformers support that emits TEXT
# coordinates -- which also restores the grammar-constrained option. 2B over 4B because at run
# time the grounding model shares a 6 GB card with SmolVLA (REMAINING.md T0 model-roles table).
set -e
source ~/bringwrench/.venv-vla/bin/activate
export HF_HUB_DISABLE_XET=1
mkdir -p ~/bringwrench/logs
hf download Qwen/Qwen3-VL-2B-Instruct \
    --local-dir ~/bringwrench/models/qwen3-vl-2b \
    > ~/bringwrench/logs/download_qwen.log 2>&1
du -sh ~/bringwrench/models/qwen3-vl-2b
echo QWEN_DOWNLOAD_DONE
