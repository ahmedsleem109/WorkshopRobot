"""T0.3 stage 1 (Windows): dump wrist-camera views with GROUND-TRUTH pixel coordinates.

Rendering only works on Windows (EGL fails inside WSL), and the grounding model only runs
in the WSL torch venv, so T0.3 is two-stage like every other render job in this project:
dump here, score there with scripts/qwen_bakeoff.py.

For every view this records, per tool, the pixel the model SHOULD name -- obtained by
projecting the tool's ground-truth position through the wrist camera -- plus whether the
tool is actually visible, which is not the same question. A tool can be in frame and still
be hidden behind the gripper; STATUS.md notes one reply of "There are none." for the pliers,
and blaming the model for that would be wrong if the pliers were occluded. Visibility is
decided by the rendered DEPTH buffer, not by the frustum.

    render_venv\\Scripts\\python.exe scripts\\dump_wrist_views.py [n] [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from bw.manip.scripted_grasp import SCAN_Q
from bw.sim.workshop import TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim

SIZE = 512
OCCLUSION_TOL = 0.02        # m: rendered depth this much nearer than the tool -> occluded


def project(sim, cam, p_world, size):
    """Pixel (u, v) and range for a world point, in MuJoCo's camera convention: the camera
    looks along its own -z, +x right, +y up."""
    cpos, cmat = sim.camera_pose(cam)
    K = sim.camera_intrinsics(cam, size=size)
    pc = cmat.T @ (p_world - cpos)
    depth = -pc[2]
    if depth <= 1e-6:
        return None, depth
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    return (float(cx + f * pc[0] / depth), float(cy - f * pc[1] / depth)), float(depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "runs/t03_views"))
    args = ap.parse_args()

    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    sim = WorkshopSim(render_size=(SIZE, SIZE))
    recs = []
    for i in range(args.n):
        rng = np.random.default_rng(5000 + i)
        target = TOOL_NAMES[i % len(TOOL_NAMES)]
        # Half the views carry EVERY tool, so the 10 mm vs 13 mm discrimination is tested with
        # both wrenches in frame -- the distinction the whole task depends on.
        both = i % 2 == 0
        present = sim.reset(rng, target=target, arm_q=SCAN_Q,
                            tools_present=set(TOOL_NAMES) if both else None)
        rgb = sim.render("wrist", size=(SIZE, SIZE))
        depth = sim.render("wrist", depth=True, size=(SIZE, SIZE))
        name = f"{i:04d}.png"
        Image.fromarray(rgb).save(out / "img" / name)
        tools = {}
        for t in TOOL_NAMES:
            if t not in present:
                continue
            uv, rng_m = project(sim, "wrist", sim.gt_tool_pos(t), (SIZE, SIZE))
            if uv is None:
                continue
            u, v = uv
            inside = 0 <= u < SIZE and 0 <= v < SIZE
            vis = False
            if inside:
                seen = float(depth[int(round(min(v, SIZE - 1))), int(round(min(u, SIZE - 1)))])
                vis = seen > rng_m - OCCLUSION_TOL
            tools[t] = {"u": round(u, 1), "v": round(v, 1), "range": round(rng_m, 4),
                        "in_frame": bool(inside), "visible": bool(vis)}
        recs.append({"image": f"img/{name}", "target": target, "size": SIZE,
                     "all_tools_present": both, "tools": tools})
        vis_n = sum(t["visible"] for t in tools.values())
        print(f"[{i:3d}] target {target:12s} present {len(tools)} visible {vis_n}"
              f"{'  TARGET OCCLUDED' if not tools.get(target, {}).get('visible') else ''}",
              flush=True)

    (out / "views.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
    vis = [r for r in recs if r["tools"].get(r["target"], {}).get("visible")]
    print(f"\n{len(recs)} views -> {out}")
    print(f"target visible in {len(vis)}/{len(recs)} -- a refusal on the rest is CORRECT "
          f"behaviour, not a miss")


if __name__ == "__main__":
    main()
