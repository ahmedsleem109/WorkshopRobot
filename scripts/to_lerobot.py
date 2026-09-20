"""T6.4 -- convert the collector's episodes (scripts/collect_demos.py) into a LeRobotDataset.

    # WSL, torch venv
    ~/bringwrench/.venv-vla/bin/python scripts/to_lerobot.py \
        --raw /mnt/d/bw_data/raw --root ~/bringwrench/data/bw_demos [--kind pick|place] [--limit N]

lerobot 0.6.1 writes dataset format v3.0 (the plan said v2.0; that is the older layout of the
same library -- v3.0 is what this lerobot loads and trains on).

Features: observation.images.camera1 = wrist, camera2 = mast (video, 256x256x3), observation.state (7: six arm
joints + finger travel), action (7: the commanded target), task = the episode's instruction.

With --delta the six ARM channels of `action` hold q_cmd - q_state (the one-step motion) instead
of the absolute target; the gripper channel stays absolute. A dataset built that way MUST be
served with `vla_server.py --delta`, which reports it on /health so the client cannot mismatch.
Writes <root>/bw_episode_index.json mapping dataset episode index -> raw episode (kind, tool,
table, seed), so a pick-only baseline (T7.1) can select `--dataset.episodes` from it.
"""
import argparse
import json
import shutil
from pathlib import Path

import av
import numpy as np

CAMERAS = ("wrist", "mast")
# Feature keys follow lerobot/smolvla_base's own input names (camera1..3), so fine-tuning needs no
# rename map: camera1 = wrist, camera2 = mast; camera3 is absent (SmolVLA skips missing views).
KEY = {"wrist": "observation.images.camera1", "mast": "observation.images.camera2"}
FPS = 10


def decode(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/mnt/d/bw_data/raw")
    ap.add_argument("--root", default=str(Path.home() / "bringwrench/data/bw_demos"))
    ap.add_argument("--repo-id", default="local/bw_demos")
    ap.add_argument("--kind", default=None, help="only pick or only place episodes")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/N: convert every N-th episode from i")
    ap.add_argument("--merge", type=int, default=None,
                    help="merge <root>_s0..s{N-1} (from --shard runs) into <root>")
    ap.add_argument("--delta", action="store_true",
                    help="record the ARM action as a one-step delta (q_cmd - q_state) instead "
                         "of the absolute target; the gripper channel stays absolute")
    args = ap.parse_args()
    if args.merge:
        return merge(args)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = sorted(p for p in Path(args.raw).iterdir() if p.is_dir() and (p / "meta.json").exists())
    if args.kind:
        eps = [p for p in eps if p.name.endswith("_" + args.kind)]
    if args.limit:
        eps = eps[:args.limit]
    root = Path(args.root)
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        eps = eps[i::n]
        root = Path(f"{args.root}_s{i}")
        args.repo_id = f"{args.repo_id}_s{i}"
    if root.exists():
        shutil.rmtree(root)
    feats = {KEY[c]: {"dtype": "video", "shape": (256, 256, 3),
                                         "names": ["height", "width", "channels"]}
             for c in CAMERAS}
    names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
    feats["observation.state"] = {"dtype": "float32", "shape": (7,), "names": names}
    feats["action"] = {"dtype": "float32", "shape": (7,), "names": names}
    from lerobot.configs.video import RGBEncoderConfig
    # H.264 instead of lerobot's default SVT-AV1: AV1 cost ~16 s per episode (120 episodes in
    # 33 min); keyframe every 2 frames kept for random access during training.
    enc = RGBEncoderConfig(vcodec="h264", crf=20, g=2, preset="veryfast")
    ds = LeRobotDataset.create(args.repo_id, fps=FPS, features=feats, root=root,
                               robot_type="go2_z1", use_videos=True, rgb_encoder=enc)
    index = []
    for k, p in enumerate(eps):
        meta = json.loads((p / "meta.json").read_text())
        arr = np.load(p / "data.npz")
        try:
            frames = {c: decode(p / f"{c}.mp4") for c in CAMERAS}
        except Exception as e:                  # noqa: BLE001 -- a corrupt raw video
            print(f"skip {p.name}: {e!r}", flush=True)
            continue
        n = min([len(arr["action"])] + [len(v) for v in frames.values()])
        if len(arr["action"]) - n > 5:          # a truncated video: drop the episode
            print(f"skip {p.name}: {len(arr['action'])} actions, "
                  f"{ {c: len(v) for c, v in frames.items()} } frames", flush=True)
            continue
        for t in range(n):
            act = np.asarray(arr["action"][t], np.float32).copy()
            if args.delta:
                # DELTA ARM ACTION (2026-09-20). Absolute joint targets make this task nearly
                # unlearnable at this data scale: measured on the 1,128-episode absolute
                # dataset, the demonstrator's per-step motion is ~0.026 rad while the action
                # SPREAD the normaliser divides by is 0.28-0.68 rad per joint, so the quantity
                # the model must resolve is 4-8% of its own normalised scale. The resulting
                # policy predicted training actions to +-0.054 rad -- TWICE the size of the
                # motion it had to produce, and worse than the trivial "hold the current joint
                # position" predictor -- and scored grasp 1/20 (scripts/vla_replay_check.py).
                # As a delta the target IS the motion, normalised to the motion's own scale.
                #
                # The GRIPPER stays absolute: its command is bimodal (open / closed), i.e.
                # already large against its own spread, and it is the one channel the absolute
                # policy beat the copy-state baseline on (0.0016 vs 0.0064 rad).
                act[:6] = act[:6] - np.asarray(arr["state"][t], np.float32)[:6]
            ds.add_frame({KEY["wrist"]: frames["wrist"][t],
                          KEY["mast"]: frames["mast"][t],
                          "observation.state": arr["state"][t],
                          "action": act,
                          "task": meta["instruction"]})
        ds.save_episode()
        index.append({"episode": len(index), "raw": p.name, **{k2: meta[k2] for k2 in
                                                                ("kind", "tool", "table", "seed",
                                                                 "instruction")}})
        if k % 50 == 0:
            print(f"{k + 1}/{len(eps)} {p.name} {n} frames", flush=True)
    ds.finalize()
    (root / "bw_episode_index.json").write_text(json.dumps(index, indent=0))
    print(f"done: {len(index)} episodes -> {root}")


def merge(args):
    """Parallel conversion: shards are written independently (AV1/H.264 encoding and the
    per-frame image writes are the bottleneck, ~6 s per episode single-process), then
    aggregated. The episode index follows the aggregation order."""
    from lerobot.datasets.aggregate import aggregate_datasets
    roots = [Path(f"{args.root}_s{i}") for i in range(args.merge)]
    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    aggregate_datasets([f"{args.repo_id}_s{i}" for i in range(args.merge)], args.repo_id,
                       roots=roots, aggr_root=root)
    index = []
    for r in roots:
        for e in json.loads((r / "bw_episode_index.json").read_text()):
            index.append({**e, "episode": len(index)})
    (root / "bw_episode_index.json").write_text(json.dumps(index, indent=0))
    print(f"merged {len(index)} episodes -> {root}")


if __name__ == "__main__":
    main()
