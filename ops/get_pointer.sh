#!/bin/bash
# The ONE grounding download (T0.1): ready-made 4-bit pointing model, 3.7 GB.
# Chosen over allenai/Molmo2-ER because that repo ships F32 (19.4 GB for a 4.85B model).
set -e
source ~/bringwrench/.venv-vla/bin/activate
export HF_HUB_DISABLE_XET=1
mkdir -p ~/bringwrench/logs
hf download Cycl0/Molmo2-VideoPoint-4B-bnb-4bit \
    --local-dir ~/bringwrench/models/molmo2-videopoint-4b-4bit \
    > ~/bringwrench/logs/download_pointer.log 2>&1
du -sh ~/bringwrench/models/molmo2-videopoint-4b-4bit
echo POINTER_DOWNLOAD_DONE
