"""Render a dumped trajectory to mp4 on Windows (render_venv), tracking the robot."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--traj", default=str(ROOT / "media/loco.npz"))
p.add_argument("--out", default=None)
p.add_argument("--camera", default="track")
p.add_argument("--width", type=int, default=960)
p.add_argument("--height", type=int, default=540)
a = p.parse_args()

d0 = np.load(a.traj)
# the dumped path is a WSL path; the scene lives in this repo either way
scene = str(ROOT / "models" / Path(str(d0["scene"])).name) if "scene" in d0 else str(ROOT / "models/scene_go2z1_step.xml")
m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)
r = mujoco.Renderer(m, a.height, a.width)
vopt = mujoco.MjvOption()
vopt.sitegroup[:] = 0
cam = mujoco.MjvCamera()
if a.camera == "track":
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 2.6, 140, -12
else:
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = m.camera(a.camera).id
frames = []
qpos, mocap, live = d0["qpos"], d0["mocap_pos"], d0["live"]
for i in range(0, len(qpos), 2):          # 50 Hz -> 25 fps
    if live[i] < 0.5:
        break
    d.qpos[:] = qpos[i]
    d.mocap_pos[:] = mocap[i]
    mujoco.mj_forward(m, d)
    if a.camera == "track":
        cam.lookat[:] = d.qpos[:3]
    r.update_scene(d, camera=cam, scene_option=vopt)
    frames.append(r.render())
out = a.out or str(Path(a.traj).with_suffix(".mp4"))
imageio.mimwrite(out, frames, fps=25, quality=8, macro_block_size=1)
print(f"wrote {out}: {len(frames)} frames")
