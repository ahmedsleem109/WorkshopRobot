"""T7 root cause: how far are the LIVE renders the policy is served from the COMPRESSED frames it
was trained on? No GPU, no policy.

    D:\\hexapod\\render_venv\\Scripts\\python.exe scripts\\vla_frame_gap.py [N]

The asymmetry this measures. A demonstration's images are written by `collect_demos.py` with
imageio/libx264 at quality 9 in **yuv420p** -- lossy, and chroma-subsampled, so colour is averaged
over 2x2 pixel blocks -- and `to_lerobot.py` then re-encodes them into the LeRobotDataset's own
video. The policy therefore trained exclusively on twice-encoded frames. At serving,
`bw/policy/vla.py` hands it `sim.render(...)` output: pristine RGB, never through a codec.

That matters here more than it would in most scenes, because the ONLY thing separating the 10 mm
wrench from the 13 mm one is a coloured grip band (T0.3: the model is at chance on size, 92.9% on
the band), and chroma subsampling is precisely the operation that degrades small patches of colour.

`scripts/vla_exec_check.py` established that replaying a demonstration's own absolute actions with
the legs locked reproduces its grasp 8 of 8 times -- so the replay is deterministic and the state at
tick t is the state the demonstration was in at tick t. This script exploits that: it replays, and
at every tick compares the live render against the stored frame for the same tick.

Reported per camera: mean absolute pixel difference, per-channel mean shift, PSNR, and the same
numbers restricted to the ~coloured-band pixels (the saturated ones), which is where a colour codec
hurts a grounding-by-colour task.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np

from bw.sim.workshop_sim import WorkshopSim
from scripts.vla_exec_check import CHUNK, RATE, collect_scene

CAMS = ("wrist", "mast")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def saturated(img: np.ndarray, thresh: int = 60) -> np.ndarray:
    """Pixels whose channels disagree strongly -- the coloured grip bands and handles, as opposed
    to the grey bench, rack and tools."""
    m = img.astype(np.int16)
    return (m.max(2) - m.min(2)) > thresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=4)
    ap.add_argument("--raw", default="D:/bw_data/raw")
    ap.add_argument("--stride", type=int, default=10, help="compare every Nth tick")
    args = ap.parse_args()

    eps = sorted(p for p in Path(args.raw).glob("*_pick") if (p / "data.npz").exists())[:args.n]
    if not eps:
        sys.exit(f"no episodes under {args.raw}")
    sim = WorkshopSim()
    sim.attach_locomotion()

    acc = {c: {"mad": [], "psnr": [], "mad_sat": [], "dch": []} for c in CAMS}
    for ep in eps:
        meta = json.loads((ep / "meta.json").read_text())
        seed = int(meta["seed"])
        d = np.load(ep / "data.npz")
        stored = {c: imageio.mimread(ep / f"{c}.mp4", memtest=False) for c in CAMS}
        tool, _ = collect_scene(sim, seed)
        sim.lock_stance()                       # the regime the demonstration was recorded in
        action = d["action"]
        n = min(len(action), min(len(stored[c]) for c in CAMS))
        for t in range(n):
            if t % args.stride == 0:
                for c in CAMS:
                    live = np.asarray(sim.render(c))[..., :3]
                    ref = np.asarray(stored[c][t])[..., :3]
                    if live.shape != ref.shape:
                        sys.exit(f"shape mismatch {live.shape} vs {ref.shape} -- "
                                 f"render_size and the recorded frames disagree")
                    diff = np.abs(live.astype(np.int16) - ref.astype(np.int16))
                    acc[c]["mad"].append(float(diff.mean()))
                    acc[c]["psnr"].append(psnr(live, ref))
                    m = saturated(ref)
                    acc[c]["mad_sat"].append(float(diff[m].mean()) if m.any() else np.nan)
                    acc[c]["dch"].append((live.reshape(-1, 3).mean(0)
                                          - ref.reshape(-1, 3).mean(0)).tolist())
            sim.move_arm(np.asarray(action[t], float), 1.0 / RATE, None, RATE)
        print(f"  {ep.name} seed {seed:6d} {tool:12s} {n} ticks compared", flush=True)

    print(f"\n{'camera':8s} {'mean|diff|':>11s} {'PSNR dB':>9s} {'mean|diff| on colour':>22s}"
          f" {'channel shift R,G,B':>24s}")
    out = {}
    for c in CAMS:
        mad = float(np.mean(acc[c]["mad"]))
        ps = float(np.mean(acc[c]["psnr"]))
        sat = float(np.nanmean(acc[c]["mad_sat"]))
        dch = np.mean(np.array(acc[c]["dch"]), axis=0)
        out[c] = {"mad": round(mad, 2), "psnr_db": round(ps, 2), "mad_saturated": round(sat, 2),
                  "channel_shift": [round(float(x), 2) for x in dch],
                  "samples": len(acc[c]["mad"])}
        print(f"{c:8s} {mad:11.2f} {ps:9.2f} {sat:22.2f}   "
              f"{dch[0]:+6.2f},{dch[1]:+6.2f},{dch[2]:+6.2f}")
    p = ROOT / "runs/eval/vla_frame_gap.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {p}")
    print("READING IT: the policy trained on the STORED frames and is served the LIVE ones. A few "
          "units of mean difference is ordinary codec noise; a large gap on the coloured pixels "
          "means the thing the task depends on is not the thing the policy was shown.")


if __name__ == "__main__":
    main()
