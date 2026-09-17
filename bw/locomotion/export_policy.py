"""Export a brax PPO checkpoint to a plain .npz so Layer 3 runs without JAX.

    ~/go2-stairs/.venv/bin/python -m bw.locomotion.export_policy <checkpoint> <out.npz>

The full pipeline runs in CPU MuJoCo next to a Torch process (SmolVLA) and a Molmo
server; pulling JAX/XLA into that process for a 512-256-128 MLP would cost ~1 GB of
VRAM and minutes of compile. The network is small enough that numpy runs it at
>2 kHz, far above the 50 Hz it is needed at.
"""
import sys

import numpy as np
from brax.io import model


def main(ckpt: str, out: str, obs_key: str = "privileged_state"):
    norm, policy = model.load_params(ckpt)[:2]
    arrays = {
        "obs_mean": np.asarray(norm.mean[obs_key], np.float32),
        "obs_std": np.asarray(norm.std[obs_key], np.float32),
    }
    layers = sorted(policy["params"], key=lambda k: int(k.split("_")[-1]))
    for i, name in enumerate(layers):
        arrays[f"W{i}"] = np.asarray(policy["params"][name]["kernel"], np.float32)
        arrays[f"b{i}"] = np.asarray(policy["params"][name]["bias"], np.float32)
    arrays["n_layers"] = np.array(len(layers))
    np.savez(out, **arrays)
    print(f"{ckpt} -> {out}: layers {[arrays[f'W{i}'].shape for i in range(len(layers))]}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
