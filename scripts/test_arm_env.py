"""Smoke test Go2ArmEnv: shapes, finiteness, arm tracking, run-7 policy rollout, throughput."""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(Path.home() / "go2-stairs")]
import jax, jax.numpy as jp, numpy as np, yaml
from bw.locomotion.go2_arm_env import ArmStairsConfig, Go2ArmEnv
from eval import load_policy

cfg = yaml.safe_load(open(REPO / "configs/payload.yaml"))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
env_cfg = dict(cfg["env"]); env_cfg.update(level_init=4, flat_frac=0.0)
env = Go2ArmEnv(config=ArmStairsConfig(**env_cfg))
print("obs", env.observation_size, "act", env.action_size, "nq", env.mj_model.nq, "nu", env.mj_model.nu)
policy = load_policy(env, cfg, cfg["init_params"])
reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
st = reset(jax.random.split(jax.random.PRNGKey(0), n))
t = time.time(); rng = jax.random.PRNGKey(1)
live = np.ones(n); succ = np.zeros(n); maxs = np.zeros(n); arm_err = []
T = 600
for i in range(T):
    rng, k = jax.random.split(rng)
    a, _ = policy(st.obs, k)
    st = step(st, a)
    if i == 0:
        jax.tree.map(lambda x: x, st.obs); t1 = time.time(); print(f"compile {t1-t:.1f}s")
    d = np.array(st.done)
    succ = np.maximum(succ, np.array(st.metrics["success"]) * live)
    maxs = np.where(live > 0, np.array(st.metrics["max_s"]), maxs)
    arm_err.append(float(jp.mean(jp.abs(st.pipeline_state.qpos[:, 19:25] - st.info["arm_cmd"][:, :6]))))
    live = live * (1 - d)
el = time.time() - t1
print(f"{n} envs x {T-1} steps: {n*(T-1)/el:,.0f} steps/s")
print("alive frac", live.mean(), "success", succ.mean(), "mean max_s", maxs.mean())
print("nan", bool(np.isnan(np.array(st.pipeline_state.qpos)).any()), "arm |q-cmd| mean rad", np.mean(arm_err))
m = st.metrics
for k in ("term_height", "term_attitude", "term_contact", "arm_ext", "rel_height"):
    print(k, float(jp.mean(m[k])))
