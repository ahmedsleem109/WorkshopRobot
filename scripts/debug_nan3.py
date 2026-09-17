import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(Path.home() / "go2-stairs")]
import jax, jax.numpy as jp, numpy as np, yaml
from bw.locomotion.go2_arm_env import ArmStairsConfig, Go2ArmEnv
from eval import load_policy
cfg = yaml.safe_load(open(REPO / "configs/payload.yaml"))
ec = dict(cfg["env"]); ec.update(level_init=4, flat_frac=0.0)
env = Go2ArmEnv(config=ArmStairsConfig(**ec))
policy = load_policy(env, cfg, cfg["init_params"])
n = 16
reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
for mode in ("zero", "policy"):
    st = reset(jax.random.split(jax.random.PRNGKey(0), n)); rng = jax.random.PRNGKey(1)
    for i in range(200):
        rng, k = jax.random.split(rng)
        a = jp.zeros((n, 12)) if mode == "zero" else policy(st.obs, k)[0]
        st = step(st, a)
        q = np.array(st.pipeline_state.qpos)
        bad = ~np.isfinite(q).all(1)
        done = np.array(st.done)
        if bad.any() or done.any():
            j = int(np.argmax(bad | (done > 0)))
            print(mode, "step", i, "nan envs", bad.sum(), "done envs", int(done.sum()),
                  "env", j, "z", q[j, 2], "act max", float(jp.max(jp.abs(a))),
                  "obs finite", bool(jp.isfinite(st.obs["privileged_state"]).all()),
                  {k: float(st.metrics[k][j]) for k in ("term_height","term_attitude","term_contact","term_backward","term_diverged","rel_height")})
            break
    else:
        print(mode, "200 steps clean; z mean", float(jp.mean(st.pipeline_state.qpos[:, 2])), "max_s", np.round(np.array(st.metrics["max_s"]), 2))
