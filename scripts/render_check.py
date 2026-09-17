"""Render still frames of the embodiment and workshop (run with render_venv on Windows)."""
import sys
from pathlib import Path
import mujoco, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "media"; out.mkdir(exist_ok=True)

def shot(xml, key, camera, name, x=None, w=960, h=540, settle=0):
    m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
    if x is not None:
        d.qpos[0], d.qpos[2] = x
    for _ in range(settle):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, h, w)
    r.update_scene(d, camera=camera)
    Image.fromarray(r.render()).save(out / name)
    print("saved", name)

shot("scene_go2z1_flat.xml", "home", -1, "check_stowed.png", settle=500)
shot("scene_go2z1_flat.xml", "extended", -1, "check_extended.png", settle=500)
shot("workshop.xml", "home", "overview", "check_workshop.png", w=1280, h=720)
shot("workshop.xml", "ready", "bench_view", "check_bench.png", x=(4.1, 0.42), w=1280, h=720)
shot("workshop.xml", "ready", "head", "check_head_cam.png", x=(3.6, 0.42), w=640, h=480)
shot("workshop.xml", "ready", "wrist", "check_wrist_cam.png", x=(4.1, 0.42), w=640, h=480)
