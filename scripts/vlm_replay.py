"""Re-query the model process on views already dumped by `bench_locate.py --no-vlm`.

Every stage of vlm.point_ex (coarse / fine / verify) is stored, so score_locate.py can score
each variant from ONE pass of model calls (--stage).

    D:\\hexapod\\render_venv\\Scripts\\python.exe scripts\\vlm_replay.py runs\\t8_views [--n 8]
"""
import argparse
import json
import subprocess
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


def gpu_temp():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                                   "--format=csv,noheader"], capture_output=True, text=True,
                                  timeout=10).stdout.strip())
    except Exception:                                    # noqa: BLE001
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    D = Path(args.dir)
    print("vlm server:", vlm.ensure_server(), flush=True)
    seeds = sorted(p.stem for p in D.glob("*.npz"))[args.start:]
    if args.n:
        seeds = seeds[:args.n]
    for sd in seeds:
        t = gpu_temp()
        if t >= 88:
            print(f"GPU at {t} C -- stopping", flush=True)
            break
        meta = json.loads((D / f"{sd}.json").read_text())
        nv = len(np.load(D / f"{sd}.npz")["pans"])
        t0 = time.time()
        replies = []
        for k in range(nv):
            img = np.array(Image.open(D / "img" / f"{sd}_{k}.png").convert("RGB"))
            for tool in TOOL_NAMES:
                r = vlm.point_ex(img, QUERY[tool])
                replies.append({"view": k, "tool": tool, **r})
        meta["replies"] = replies
        (D / f"{sd}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"[{sd}] {time.time() - t0:5.1f}s  median call "
              f"{np.median([r['roundtrip'] for r in replies]):.2f}s  gpu {t} C", flush=True)


if __name__ == "__main__":
    main()
