"""T0.4 + T8 benchmark, stage 1 (Windows render venv): live sim -> vlm.point() over HTTP.

For every seed: reset the workshop, run locate()'s scan (3 wrist views panned on arm_joint1),
and ask the model process for EVERY tool in EVERY view -- present or not, so the None case
is measured too. Saves, per seed, everything scoring needs (depth, a ground-truth
segmentation map, camera poses, GT tool positions) plus the model's replies. Scoring is
separate and numpy-only (scripts/score_locate.py), so depth/merge variants can be tuned
without re-running the model.

    D:\hexapod\render_venv\Scripts\python.exe scripts\bench_locate.py 30 [--out runs\t8_bench]

Needs the model process: started automatically through wsl.exe if nothing answers on :8765.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from PIL import Image

from bw.manip.scripted_grasp import SCAN_Q
from bw.perception import vlm
from bw.perception.locate import IMG, capture_views
from bw.sim.workshop import TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim

QUERY = {"wrench_10mm": "10mm wrench", "wrench_13mm": "13mm wrench",
         "screwdriver": "screwdriver", "pliers": "pliers", "tape_roll": "tape roll"}


def seg_tools(sim, renderer):
    """HxW int8 map: index into TOOL_NAMES, -1 elsewhere (ground truth, scoring only)."""
    renderer.update_scene(sim.d, camera="wrist", scene_option=sim._vopt)
    seg = renderer.render()
    gid, typ = seg[..., 0], seg[..., 1]
    m = sim.m
    root_to_tool = {sim.tool_body[n]: i for i, n in enumerate(TOOL_NAMES)}
    geom_tool = np.full(m.ngeom + 1, -1, np.int8)
    for g in range(m.ngeom):
        geom_tool[g] = root_to_tool.get(int(m.body_rootid[m.geom_bodyid[g]]), -1)
    out = np.full(gid.shape, -1, np.int8)
    ok = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
    out[ok] = geom_tool[gid[ok]]
    return out


def gpu_temp():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                                   "--format=csv,noheader"], capture_output=True, text=True,
                                  timeout=10).stdout.strip())
    except Exception:                                    # noqa: BLE001
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "runs/t8_bench"))
    ap.add_argument("--no-vlm", action="store_true", help="render + GT only (for offline variants)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    if not args.no_vlm:
        print("vlm server:", vlm.ensure_server(), flush=True)
    sim = WorkshopSim(render_size=IMG)
    segr = mujoco.Renderer(sim.m, *IMG)
    segr.enable_segmentation_rendering()

    for s in range(args.start, args.start + args.n):
        t = gpu_temp()
        if t >= 88:
            print(f"GPU at {t} C -- stopping", flush=True)
            break
        rng = np.random.default_rng(7000 + s)
        target = TOOL_NAMES[s % len(TOOL_NAMES)]
        both = s % 2 == 0          # half the seeds carry every tool; the rest a random subset
        present = sim.reset(rng, target=target, arm_q=SCAN_Q,
                            tools_present=set(TOOL_NAMES) if both else None)
        t0 = time.time()
        views = capture_views(sim, extra=lambda sm: seg_tools(sm, segr))
        segs = [v.extra for v in views]
        replies = []
        for k, vw in enumerate(views):
            Image.fromarray(vw.rgb).save(out / "img" / f"{s:04d}_{k}.png")
            for tool in ([] if args.no_vlm else TOOL_NAMES):
                r = vlm.point_ex(vw.rgb, QUERY[tool])
                replies.append({"view": k, "tool": tool, "uv": r["uv"], "raw": r["raw"],
                                "prompt": r["prompt"], "latency": r["latency"],
                                "roundtrip": r["roundtrip"]})
        np.savez_compressed(
            out / f"{s:04d}.npz",
            depth=np.stack([v.depth for v in views]).astype(np.float32),
            seg=np.stack(segs), K=np.stack([v.K for v in views]),
            cam_pos=np.stack([v.cam_pos for v in views]),
            cam_mat=np.stack([v.cam_mat for v in views]),
            pans=np.array([v.pan for v in views]),
            base_pos=sim.d.qpos[0:3].copy(), base_quat=sim.d.qpos[3:7].copy(),
            gt=np.stack([sim.gt_tool_pos(n) for n in TOOL_NAMES]),
            present=np.array([n in present for n in TOOL_NAMES]))
        (out / f"{s:04d}.json").write_text(json.dumps(
            {"seed": s, "target": target, "present": sorted(present), "replies": replies},
            indent=1), encoding="utf-8")
        lat = np.median([r["roundtrip"] for r in replies]) if replies else 0.0
        print(f"[{s:3d}] present {len(present)}  {time.time() - t0:5.1f}s  "
              f"median call {lat:.2f}s  gpu {t} C", flush=True)


if __name__ == "__main__":
    main()
