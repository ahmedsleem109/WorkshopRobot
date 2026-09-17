"""Divergence vs solver iterations for the Go2+Z1 MJX model (the arm adds 33% mass over a
higher CoM; menagerie's go2_mjx.xml runs iterations=1, which the baseline Go2 tolerates)."""
import os, sys, functools, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(Path.home() / "go2-stairs")]
import jax, jax.numpy as jp, numpy as np, yaml
from bw.locomotion.go2_arm_env import ArmStairsConfig, Go2ArmEnv
from bw.locomotion.domain_rand import make_randomization_fn
from envs.wrappers import wrap_for_stairs
from eval import load_policy
from mujoco import mjx
cfg = yaml.safe_load(open(REPO / "configs/payload.yaml"))
n, T = 512, 250
for it, ls in ((1, 5), (2, 8), (4, 10), (8, 12)):
    env = Go2ArmEnv(config=ArmStairsConfig(**cfg["env"]), generate_scene=False)
    env.mj_model.opt.iterations, env.mj_model.opt.ls_iterations = it, ls
    env.mjx_model = mjx.put_model(env.mj_model)
    fn = make_randomization_fn(env.mj_model)
    w = wrap_for_stairs(env, episode_length=1000, randomization_fn=functools.partial(fn, rng=jax.random.split(jax.random.PRNGKey(3), n)))
    policy = load_policy(env, cfg, cfg["init_params"])
    reset, step = jax.jit(w.reset), jax.jit(w.step)
    st = reset(jax.random.split(jax.random.PRNGKey(0), n)); rng = jax.random.PRNGKey(1)
    div = dones = 0.0; t0 = None
    for i in range(T):
        rng, k = jax.random.split(rng)
        a, _ = policy(st.obs, k); st = step(st, a)
        div += float(jp.sum(st.metrics["term_diverged"])); dones += float(jp.sum(st.done))
        if i == 0: t0 = time.time()
    sps = n * (T - 1) / (time.time() - t0)
    print(f"[it={it} ls={ls}] terminations {dones:.0f} diverged {div:.0f} ({div/max(dones,1):.1%})  {sps:,.0f} steps/s", flush=True)
