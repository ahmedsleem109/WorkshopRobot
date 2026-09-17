#!/bin/bash
# Torch/LeRobot venv for Layers 1-2 (SmolVLA, Molmo). Separate from the JAX venv on purpose:
# torch's CUDA wheels and jax[cuda12] pin conflicting nvidia-* packages.
set -e
cd ~/bringwrench
export UV_LINK_MODE=copy
uv venv --python 3.12 .venv-vla
source .venv-vla/bin/activate
uv pip install "lerobot[smolvla]" 2>&1 | tail -5
uv pip install bitsandbytes accelerate "huggingface_hub[cli]" mujoco==3.11.0 einops 2>&1 | tail -3
python -c "import torch,lerobot,transformers;print('torch',torch.__version__,torch.cuda.is_available(),'lerobot',lerobot.__version__,'tf',transformers.__version__)"
echo VLA_ENV_DONE
