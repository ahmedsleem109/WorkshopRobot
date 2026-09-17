import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(Path.home() / "go2-stairs")]
import jax, jax.numpy as jp, numpy as np, mujoco
from mujoco import mjx
m = mujoco.MjModel.from_xml_path(str(REPO / "models/scene_go2z1_flat.xml"))
print("integrator", m.opt.integrator, "timestep", m.opt.timestep)
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
mx = mjx.put_model(m); dx = mjx.put_data(m, d)
stepj = jax.jit(mjx.step)
for i in range(30):
    dx = stepj(mx, dx)
    q = np.array(dx.qpos)
    if i < 5 or not np.isfinite(q).all():
        print(i, "z", q[2], "arm", np.round(q[19:26], 3), "maxqvel", float(jp.max(jp.abs(dx.qvel))))
    if not np.isfinite(q).all(): break
# same with Euler
m2 = mujoco.MjModel.from_xml_path(str(REPO / "models/scene_go2z1_flat.xml")); m2.opt.integrator = 0
d2 = mujoco.MjData(m2); mujoco.mj_resetDataKeyframe(m2, d2, 0)
mx2 = mjx.put_model(m2); dx2 = mjx.put_data(m2, d2)
for i in range(30):
    dx2 = stepj(mx2, dx2)
print("euler after 30:", np.isfinite(np.array(dx2.qpos)).all(), float(jp.max(jp.abs(dx2.qvel))))
