"""T8 -- the ABSENT-tool false positive: score candidate rejection signals, offline-scorable.

The open number from session 6: `locate()` returns a point for a tool that is NOT IN THE SCENE
in 14 of 20 cases. Today the missing-tool recovery only passes the suite because
`eval_suite.py` defaults to `--grounding oracle`.

Two rejection signals can be measured WITHOUT any model call, from the replies already stored
in runs/t8_views (see the offline part of this file's companion, scripts/score_absent.py):

    cross-view agreement   present tools cluster across the three scan views; a hallucination
                           does not. Measured on the stored dumps: requiring 2 of 3 views cuts
                           false positives 14/20 -> 2/20 but also drops success 135 -> 99 of
                           180. Too blunt ALONE; the right use is as a confidence tier.
    snap distance          how far the point had to move to land on an object. Separates only
                           weakly (median 0.5 px present vs 2.3 px absent).

So the question is what a SECOND model call adds on top, and only for the answers that the
cheap signals leave ambiguous. This probe records three candidate questions per view+tool on
the already-rendered seeds, so every rule can then be scored offline against ground truth
without re-running the model:

    full    "Is there a {obj} in this image?"            on the whole view
    crop    the same question on the ZOOMx crop around the stored point (what point_ex(verify)
            already does -- measured on 8 seeds it did NOT reject absent tools, 3/4 either
            way; this re-measures it on 32+ seeds so the claim rests on more than 8)
    name    "Name the single tool at the centre of this image."  on the same crop -- an OPEN
            question, so a wrong object is answered with its own name instead of being pushed
            into a yes/no the model is known to answer agreeably

    D:\\hexapod\\render_venv\\Scripts\\python.exe scripts\\_absent_probe.py runs\\t8_views
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

QUERY = {"wrench_10mm": "10mm wrench", "wrench_13mm": "13mm wrench",
         "screwdriver": "screwdriver", "pliers": "pliers", "tape_roll": "tape roll"}
Q_PRESENT = "Is there a {obj} in this image? Answer yes or no."
Q_NAME = "Name the single tool at the centre of this image. Answer with its name only."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(ROOT / "runs/t8_views"))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default=None, help="default <dir>/absent_probe.json")
    args = ap.parse_args()
    D = Path(args.dir)
    out = Path(args.out) if args.out else D / "absent_probe.json"
    print("vlm server:", vlm.ensure_server(), flush=True)
    started_here = True

    seeds = sorted(p.stem for p in D.glob("*.npz"))
    if args.n:
        seeds = seeds[:args.n]
    rows = []
    if out.exists():                                   # resumable: the queue may retry
        rows = json.loads(out.read_text())
        done = {(r["seed"], r["view"], r["tool"]) for r in rows}
        print(f"resuming: {len(rows)} rows already stored", flush=True)
    else:
        done = set()

    t0 = time.time()
    for sd in seeds:
        meta = json.loads((D / f"{sd}.json").read_text())
        if not meta.get("replies"):
            continue
        d = np.load(D / f"{sd}.npz")
        rep = {(r["view"], r["tool"]): r for r in meta["replies"]}
        nv = len(d["pans"])
        for k in range(nv):
            img = None
            for ti, tool in enumerate(TOOL_NAMES):
                if (sd, k, tool) in done:
                    continue
                r = rep.get((k, tool))
                if r is None:
                    continue
                uv = r.get("uv_fine") or r.get("uv_coarse")
                if img is None:
                    img = np.array(Image.open(D / "img" / f"{sd}_{k}.png").convert("RGB"))
                obj = vlm.resolve_query(QUERY[tool])
                row = {"seed": sd, "view": k, "tool": tool, "present": bool(d["present"][ti]),
                       "uv": uv, "full": None, "crop": None, "name": None}
                row["full"] = vlm._post_ask(img, Q_PRESENT.format(obj=obj))["raw"]
                if uv is not None:
                    crop, _ = vlm._crop(img, uv)
                    row["crop"] = vlm._post_ask(crop, Q_PRESENT.format(obj=obj))["raw"]
                    row["name"] = vlm._post_ask(crop, Q_NAME)["raw"]
                rows.append(row)
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"[{sd}] rows {len(rows)}  {time.time() - t0:5.0f}s", flush=True)
    print(f"wrote {out} ({len(rows)} rows)")
    if started_here:                      # never leave the card held: the queue waits for it
        print("vlm server stopped:", vlm.stop_server(), flush=True)


if __name__ == "__main__":
    main()
