import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(Path.home() / "go2-stairs")]
import jax, jax.numpy as jp, numpy as np, mujoco, yaml
from mujoco import mjx
m = mujoco.MjModel.from_xml_path(str(REPO / "models/scene_go2z1_flat.xml"))
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
mx = mjx.put_model(m); dx = mjx.put_data(m, d)
def ten(d, _): return mjx.step(mx, d), None
multi = jax.jit(lambda d: jax.lax.scan(ten, d, None, length=10)[0])
for i in range(150):
    dx = multi(dx)
    q = np.array(dx.qpos)
    if not np.isfinite(q).all() or i % 30 == 0:
        print("ctrl step", i, "z", q[2], "finite", np.isfinite(q).all()); 
    if not np.isfinite(q).all(): break
from bw.locomotion.go2_arm_env import ArmStairsConfig, Go2ArmEnv
cfg = yaml.safe_load(open(REPO / "configs/payload.yaml"))
ec = dict(cfg["env"]); ec.update(level_init=4, flat_frac=0.0)
env = Go2ArmEnv(config=ArmStairsConfig(**ec))
st = jax.jit(env.reset)(jax.random.PRNGKey(0))
q = np.array(st.pipeline_state.qpos); print("reset qpos", np.round(q, 3))
print("mocap", np.round(np.array(st.pipeline_state.mocap_pos), 3))
step = jax.jit(env.step)
for i in range(60):
    st = step(st, jp.zeros(12))
    q = np.array(st.pipeline_state.qpos)
    if not np.isfinite(q).all() or i < 3 or i % 20 == 0:
        print("env step", i, "z", q[2], "finite", np.isfinite(q).all(), "done", float(st.done))
    if not np.isfinite(q).all(): break
