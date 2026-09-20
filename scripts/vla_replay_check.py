"""Does the policy reproduce the demonstrations it was TRAINED on, through the SERVING path?

    ~/bringwrench/.venv-vla/bin/python scripts/vla_replay_check.py CKPT [--episodes 6] [--stride 5]

This is the experiment that separates the two explanations for a policy that trains to a low
loss and then grasps nothing:

  * it never learned the mapping (too little training, or a train/serve mismatch that the
    training loss cannot see) -- then its predictions are wrong even on TRAINING frames, and
  * it learned it, but cannot hold a closed loop in the sim (compounding error, wrong control
    rate, an open-loop horizon that is too long) -- then its predictions on training frames are
    RIGHT and the failure is elsewhere.

Training loss cannot tell these apart: it is computed inside the training pipeline, in
normalised action space. This runs the frames through `bw/policy/vla_server.py`'s own `VLA.act`
-- the exact code the evaluator talks to, images round-tripped to uint8 HWC the way the sim
sends them -- and compares its FIRST predicted action against the action the demonstrator
actually commanded on that frame.

Two reference points are printed with it, because an error in radians means nothing on its own:
  ACTION SPREAD  the std of the recorded actions. An error near this size is no better
                 than predicting the dataset mean.
  COPY-STATE     the error of the trivial predictor "command = current joint position".
                 These actions are commanded TARGETS a tenth of a second ahead, so copying
                 the state is already close; a policy that cannot beat it has learned nothing
                 useful, however small its own error looks.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from bw.policy.vla_server import VLA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--root", default="/home/sleem/bringwrench/data/bw_demos")
    ap.add_argument("--repo-id", default="local/bw_demos")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5, help="sample every Nth frame of an episode")
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--delta", action="store_true",
                    help="the dataset and checkpoint use DELTA arm actions; the trivial "
                         "baseline then becomes 'predict no motion' instead of 'copy the state'")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo_id, root=args.root)
    print(f"dataset: {ds.num_episodes} episodes / {ds.num_frames} frames", flush=True)

    vla = VLA(args.ckpt, args.n_action_steps, args.device, delta=args.delta)
    print(f"policy: {args.ckpt}  n_action_steps={args.n_action_steps}", flush=True)

    # Episode boundaries, so frames are replayed in order within an episode (the policy keeps
    # state between calls; it is reset at the start of each one). lerobot 0.6.1's v3.0 layout
    # carries these on meta.episodes as dataset_from_index / dataset_to_index -- the older
    # `episode_data_index` attribute is gone.
    ep_from = list(ds.meta.episodes["dataset_from_index"])
    ep_to = list(ds.meta.episodes["dataset_to_index"])
    picks = np.linspace(0, len(ep_from) - 1, args.episodes).astype(int)

    err, copy_err, acts = [], [], []
    for e in picks:
        lo, hi = ep_from[e], ep_to[e]
        first = True
        for i in range(lo, hi, args.stride):
            s = ds[i]
            imgs = {}
            for k in ("camera1", "camera2"):
                t = s[f"observation.images.{k}"]                    # CHW float 0..1
                imgs[k] = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            state = s["observation.state"].numpy()
            gt = s["action"].numpy()
            pred = vla.act(imgs, state.tolist(), s["task"], reset=first)[0][0]
            first = False
            err.append(np.abs(pred - gt))
            # The baseline a policy has to beat to be worth anything. Absolute actions: hold
            # the current joint position. Delta actions: command no motion at all (zero).
            copy_err.append(np.abs(gt) if args.delta else np.abs(state - gt))
            acts.append(gt)
        print(f"  episode {e}: {len(range(lo, hi, args.stride))} frames", flush=True)

    err = np.array(err)
    copy_err = np.array(copy_err)
    acts = np.array(acts)
    names = ["j1", "j2", "j3", "j4", "j5", "j6", "grip"]
    base_name = "NO-MOTION" if args.delta else "COPY-STATE"

    print(f"\n{len(err)} frames replayed through the serving path\n")
    print(f"{'joint':6s} {'POLICY err':>11s} {base_name:>11s} {'ACTION SPREAD':>14s}"
          f" {'policy/spread':>14s}")
    for j, n in enumerate(names):
        print(f"{n:6s} {err[:, j].mean():11.4f} {copy_err[:, j].mean():11.4f}"
              f" {acts[:, j].std():14.4f} {err[:, j].mean() / max(acts[:, j].std(), 1e-9):14.2f}")
    print(f"{'ALL':6s} {err.mean():11.4f} {copy_err.mean():11.4f} {acts.std():14.4f}"
          f" {err.mean() / max(acts.std(), 1e-9):14.2f}")

    print("\nREADING IT: policy/spread well under 1 and an error below COPY-STATE means the")
    print("policy did learn the mapping and the serving path is sound -- look at the closed")
    print(f"loop next. At or above {base_name} means it did not, and more closed-loop tuning")
    print("is wasted effort.")


if __name__ == "__main__":
    main()
