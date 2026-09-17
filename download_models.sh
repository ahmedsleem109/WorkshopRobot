#!/bin/bash
set -e
source ~/bringwrench/.venv-vla/bin/activate 2>/dev/null || { python3 -m pip install --user -q "huggingface_hub[cli]"; }
export HF_HUB_ENABLE_HF_TRANSFER=0
hf download lerobot/smolvla_base --local-dir ~/bringwrench/models/smolvla_base
# fp32 weights go to C: (WSL disk is on D: with ~32 GB free); only the 4-bit copy lands on ext4.
hf download allenai/Molmo2-ER --local-dir /mnt/c/hf_cache/Molmo2-ER
echo DOWNLOADS_DONE
