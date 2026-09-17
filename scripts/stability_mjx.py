"""MJX physics stability of the Go2+Z1 under training conditions. Ablates arm motion and DR,
and reports which DOF first exceeds 50 rad/s (or m/s) when an env diverges."""
import os, sys, functools
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(Path.home() / "go2-stairs")]
import jax, jax.numpy as jp, numpy as np, yaml
from bw.locomotion.go2_arm_env import ArmStairsConfig, Go2ArmEnv
from bw.locomotion.domain_rand import make_randomization_fn
from envs.wrappers import wrap_for_stairs
from eval import load_policy
cfg = yaml.safe_load(open(REPO / "configs/payload.yaml"))
n, T = 512, 300
conds = [dict(name="full"), dict(name="arm_stowed", randomize_arm=False)]
for c in conds:
    ec = dict(cfg["env"])
    for k in ("randomize_arm", "flat_frac"):
        if k in c: ec[k] = c[k]
    env = Go2ArmEnv(config=ArmStairsConfig(**ec), generate_scene=False)
    fn = make_randomization_fn(env.mj_model) if c.get("dr", True) else None
    w = wrap_for_stairs(env, episode_length=1000, randomization_fn=functools.partial(fn, rng=jax.random.split(jax.random.PRNGKey(3), n)) if fn else None)
    policy = load_policy(env, cfg, cfg["init_params"])
    reset, step = jax.jit(w.reset), jax.jit(w.step)
    st = reset(jax.random.split(jax.random.PRNGKey(0), n)); rng = jax.random.PRNGKey(1)
    div = dones = 0.0; culprit = np.zeros(env.mj_model.nv)
    for i in range(T):
        rng, k = jax.random.split(rng)
        prev = np.array(st.pipeline_state.qvel)
        a, _ = policy(st.obs, k)
        st = step(st, a)
        dv = np.array(st.metrics["term_diverged"]) > 0
        dones += float(jp.sum(st.done)); div += dv.sum()
        if dv.any():
            q = np.array(st.pipeline_state.qvel)  # already reset for done envs; use prev step's
            for e in np.where(dv)[0]:
                culprit[np.argmax(np.abs(prev[e]))] += 1
    top = np.argsort(-culprit)[:4]
    print(f"[{c['name']:10s}] terminations {dones:.0f} diverged {div:.0f} ({div/max(dones,1):.1%}); top dof before divergence {[(int(t), int(culprit[t])) for t in top if culprit[t] > 0]}", flush=True)
