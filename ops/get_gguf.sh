#!/bin/bash
# T0.1, second candidate: Molmo2-4B GGUF q4_k_m + mmproj-f16 (~3.6 GB), the llama.cpp path
# that gives GBNF grammar-constrained pointing output (the plan's tool-parse fallback).
# Run AFTER get_pointer.sh finishes — the link is ~0.3 MB/s, parallel downloads only split it.
set -e
source ~/bringwrench/.venv-vla/bin/activate
export HF_HUB_DISABLE_XET=1
mkdir -p ~/bringwrench/logs
# Exact filenames, not --include globs: `--include "a" "b"` makes hf treat the second pattern
# as a positional FILENAME and request it literally (it 404s on the URL-encoded glob).
hf download reubk/Molmo2-4B-GGUF \
    molmo2-4b-q4_k_m.gguf molmo2-4b-mmproj-f16.gguf \
    --local-dir ~/bringwrench/models/molmo2-4b-gguf \
    > ~/bringwrench/logs/download_gguf.log 2>&1
du -sh ~/bringwrench/models/molmo2-4b-gguf
ls -la ~/bringwrench/models/molmo2-4b-gguf
echo GGUF_DOWNLOAD_DONE
