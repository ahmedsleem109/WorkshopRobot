"""T8: why Qwen3-VL-2B answers "There are none." to "pliers", and which phrasing fixes it.

Runs a set of candidate phrasings over the views of already-rendered seeds and reports, per
phrasing: refusal rate on views where the pliers are actually visible, and how often the
returned point lands on the pliers' own pixels (GT segmentation).

    python scripts/_pliers_probe.py runs/t8_views --n 8 [--phrasings "a;b;c"]

Works against a CPU server (`vlm_server.py --device cpu`) while the GPU is training.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from bw.perception import vlm
from bw.sim.workshop import TOOL_NAMES

TI = TOOL_NAMES.index("pliers")
MIN_PIX = 40          # same visibility threshold as score_locate.py
HIT_R = 3

CANDIDATES = [
    "pliers",                        # the current behaviour, as a control
    "red pliers",
    "pliers with red handles",
    "tool with two red handles",
    "red-handled gripping tool",
]


def on_tool(seg, uv, r=HIT_R):
    h, w = seg.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    patch = seg[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)]
    return TI in set(int(x) for x in np.unique(patch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--phrasings", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--refine", type=int, default=0,
                    help="run the zoom stage too (2x the calls; the refusal is a coarse-stage effect)")
    args = ap.parse_args()
    phrasings = [p.strip() for p in args.phrasings.split(";") if p.strip()] or CANDIDATES

    # Ask for the literal phrase: ALIASES would rewrite every candidate back to "pliers".
    vlm.resolve_query = lambda q: q

    D = Path(args.dir)
    print("vlm server:", vlm.ensure_server(), flush=True)
    seeds = sorted(p.stem for p in D.glob("*.npz"))[args.start:args.start + args.n]

    # The views where the pliers are genuinely visible -- the only ones a refusal is wrong on.
    views = []
    for sd in seeds:
        d = np.load(D / f"{sd}.npz")
        for k in range(len(d["pans"])):
            if int((d["seg"][k] == TI).sum()) >= MIN_PIX:
                views.append((sd, k, d["seg"][k]))
    print(f"{len(views)} views with visible pliers over {len(seeds)} seeds", flush=True)

    rows = []
    for phrase in phrasings:
        refused = hit = 0
        t0 = time.time()
        for sd, k, seg in views:
            img = np.array(Image.open(D / "img" / f"{sd}_{k}.png").convert("RGB"))
            r = vlm.point_ex(img, phrase, refine=bool(args.refine))
            if r["uv"] is None:
                refused += 1
            elif on_tool(seg, r["uv"]):
                hit += 1
            rows.append({"phrase": phrase, "seed": sd, "view": k, "uv": r["uv"],
                         "raw": r["raw"]})
        n = len(views)
        print(f"{phrase:32s} refused {refused}/{n}  on-pliers {hit}/{n}  "
              f"({(time.time() - t0) / max(1, n):.1f} s/view)", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
