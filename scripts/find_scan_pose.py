"""Joint-space search for the wrist-camera scan pose: camera looks at the tray centre from as
high as possible, image upright (image 'up' points away from the robot). FK only, then render."""
import sys, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import mujoco, numpy as np
from PIL import Image
from bw.sim.workshop_sim import WorkshopSim
from bw.sim.workshop import TRAY_CENTER, BENCH_HEIGHT
sim = WorkshopSim()
sim.reset(np.random.default_rng(0), base_pose=(4.05, 0.0, 0.0), arm_q=np.array([0, 0.9, -1.2, 0.3, 0, 0, -1.2]))
m, d = sim.m, sim.d
scratch = mujoco.MjData(m); scratch.qpos[:] = d.qpos
cid = m.camera("wrist").id
tray = np.array([TRAY_CENTER[0], TRAY_CENTER[1], BENCH_HEIGHT + 0.01])
best = []; stats = []
for j2, j3, j4, j6 in itertools.product(np.linspace(0.2, 2.6, 25), np.linspace(-2.8, -0.2, 27),
                                        np.linspace(-1.5, 1.5, 16), (0.0, np.pi / 2, -np.pi / 2, np.pi)):
    scratch.qpos[19:25] = [0, j2, j3, j4, 0, j6]
    mujoco.mj_kinematics(m, scratch); mujoco.mj_camlight(m, scratch)
    p = scratch.cam_xpos[cid]; R = scratch.cam_xmat[cid].reshape(3, 3)
    look = -R[:, 2]; up = R[:, 1]
    v = tray - p; dist = np.linalg.norm(v)
    ang = np.arccos(np.clip(look @ v / dist, -1, 1))
    height = p[2] - BENCH_HEIGHT
    ee = scratch.site_xpos[sim.ee_site]
    stats.append((ang, height))
    if ang > 0.3 or height < 0.12 or ee[2] < BENCH_HEIGHT + 0.05:
        continue
    upright = up[0]                      # image up along +x (away from robot)
    score = height + 0.3 * upright - 2 * ang
    best.append((score, height, ang, upright, [0, j2, j3, j4, 0, j6]))
best.sort(key=lambda b: -b[0])
print(len(best), "candidates; min ang", min(s[0] for s in stats), "max height", max(s[1] for s in stats))
imgs = []
for sc, h, ang, up, q in best[:4]:
    print(f"score {sc:.3f} height {h:.3f} ang {ang:.3f} upright {up:.2f} q {np.round(q, 3).tolist()}")
    d.qpos[19:25] = q; sim.set_arm_target(np.array(q + [-1.2])); sim.settle(0.5)
    imgs.append(sim.render("wrist", size=(256, 256)))
Image.fromarray(np.concatenate(imgs, 1)).save(ROOT / "media/scan_candidates.png")
