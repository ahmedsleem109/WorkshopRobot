"""Render scripted-grasp episodes to mp4: bench view with the wrist camera inset.

    render_venv\Scripts\python.exe scripts\make_grasp_video.py wrench_10mm [seed]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np

from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp
from bw.sim.workshop_sim import WorkshopSim

tool = sys.argv[1] if len(sys.argv) > 1 else "wrench_10mm"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
W, H = 960, 540
IW = 240

sim = WorkshopSim()
ik = ArmIK(sim.m)
rng = np.random.default_rng(1000 * seed + list(sim.tool_body).index(tool))
sim.reset(rng, target=tool, arm_q=SCAN_Q)
frames = []


def shot():
    big = sim.render("bench_view", size=(H, W))
    inset = sim.render("wrist", size=(IW, IW))
    out = big.copy()
    out[H - IW - 12:H - 12, W - IW - 12:W - 12] = inset
    out[H - IW - 14:H - IW - 12, W - IW - 14:W - 10] = 255
    out[H - IW - 14:H - 10, W - IW - 14:W - IW - 12] = 255
    frames.append(out)


r = run_grasp(sim, ik, tool, rng, record=lambda q: shot())
for _ in range(8):
    sim.physics_step(10)
    shot()
out = ROOT / f"media/grasp_{tool}.mp4"
imageio.mimwrite(out, frames, fps=12, quality=8, macro_block_size=1)
print(f"{out.name}: {len(frames)} frames, {r['success']=} {r.get('reason')} lifted={r.get('lifted')}")
