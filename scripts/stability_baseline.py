"""Same divergence measurement on the UNMODIFIED go2-stairs env + run-7 policy (no arm)."""
import os, sys, functools
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
GO2 = Path.home() / "go2-stairs"; os.chdir(GO2); sys.path.insert(0, str(GO2))
import jax, jax.numpy as jp, numpy as np, yaml
from envs.go2_stairs import Go2StairsEnv, StairsConfig
from envs.domain_rand import make_randomization_fn
from envs.wrappers import wrap_for_stairs
from eval import load_policy
cfg = yaml.safe_load(open("configs/stairs.yaml"))
n, T = 512, 300
for lvl in (1, 4):
    ec = dict(cfg["env"]); ec.update(level_init=lvl, level_min=lvl)
    env = Go2StairsEnv(config=StairsConfig(**ec), generate_scene=False)
    fn = make_randomization_fn(env.mj_model)
    w = wrap_for_stairs(env, episode_length=1000, randomization_fn=functools.partial(fn, rng=jax.random.split(jax.random.PRNGKey(3), n)))
    policy = load_policy(env, cfg, "results/2026-08-06_17-17-05-stairs_run7/checkpoints/final")
    reset, step = jax.jit(w.reset), jax.jit(w.step)
    st = reset(jax.random.split(jax.random.PRNGKey(0), n)); rng = jax.random.PRNGKey(1)
    div = dones = 0.0
    for i in range(T):
        rng, k = jax.random.split(rng)
        a, _ = policy(st.obs, k); st = step(st, a)
        div += float(jp.sum(st.metrics["term_diverged"])); dones += float(jp.sum(st.done))
    print(f"[baseline L{lvl}] terminations {dones:.0f} diverged {div:.0f} ({div/max(dones,1):.1%})", flush=True)
