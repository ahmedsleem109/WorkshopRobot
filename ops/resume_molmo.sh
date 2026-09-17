#!/bin/bash
# Resume the Molmo2-ER download (19.4 GB fp32). hf download resumes; xet transport is disabled
# because it failed with "CAS Client Error: Format error: I/O error" at ~16 MB.
# Weights land on C: -- the WSL disk is on D: with ~28 GB free, and only the 4-bit copy
# (~3.5 GB) should ever be written to ext4.
set -e
source ~/bringwrench/.venv-vla/bin/activate
export HF_HUB_DISABLE_XET=1
mkdir -p ~/bringwrench/logs
hf download allenai/Molmo2-ER --local-dir /mnt/c/hf_cache/Molmo2-ER \
    > ~/bringwrench/logs/download_molmo.log 2>&1
du -sh /mnt/c/hf_cache/Molmo2-ER
echo MOLMO_DOWNLOAD_DONE
