"""Replay gate 2 exactly and record, per trial, WHY and WHERE it ended.

Gate 2 = cross the 12 cm step with the arm extended, 20 trials, 3 of which fall.
Same env config / seed / command / stair as eval_phase1.gate() so the 3 falls are the
same 3 falls.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import yaml

from bw.locomotion.eval_phase1 import make_env, pin
from eval import load_policy

ap = argparse.ArgumentParser()
ap.add_argument("--payload", required=True)
ap.add_argument("--config", default="/mnt/d/bringwrench/configs/payload.yaml")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
base_cfg = cfg.get("env", {})

N, SEED, STEPS = 20, 101, 1000        # gate() uses seed+1 = 101 for gate 2
env = make_env(base_cfg, flat_frac=0.0, randomize_arm=False, arm_fixed="extended",
               dr_enable=True, cmd_stand_prob=0.0, cmd_resample_s=1e9)
pol = load_policy(env, cfg, a.payload)

reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))
st = pin(reset(jax.random.split(jax.random.PRNGKey(SEED), N)), [0.5, 0.0, 0.0], [0.12, 0.30, 0.0])
print("metric keys:", sorted(st.metrics))
rng = jax.random.PRNGKey(SEED + 1)

alive = np.ones(N, bool)
first_done = np.full(N, -1)
x0 = np.array(st.pipeline_state.qpos[:, 0])
success = np.zeros(N, bool)

# per-trial snapshot at the moment of termination
fall_x = np.full(N, np.nan)
fall_z = np.full(N, np.nan)
fall_upz = np.full(N, np.nan)
fall_vx = np.full(N, np.nan)
fall_cause = [None] * N
# rolling history so we can look at the 20 steps BEFORE the fall
hist_x, hist_z, hist_upz = [], [], []

term_keys = [k for k in st.metrics if k.startswith("term_")]

for i in range(STEPS):
    rng, k = jax.random.split(rng)
    act, _ = pol(st.obs, k)
    prev = st
    st = pin(step(st, act), [0.5, 0.0, 0.0], [0.12, 0.30, 0.0])
    qpos = np.array(st.pipeline_state.qpos)
    qvel = np.array(st.pipeline_state.qvel)
    # world z of the base's up axis, from the free-joint quaternion
    w, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    upz = 1 - 2 * (qx * qx + qy * qy)
    hist_x.append(qpos[:, 0].copy())
    hist_z.append(qpos[:, 2].copy())
    hist_upz.append(upz.copy())

    d = np.array(st.done) > 0
    newly = alive & d
    if newly.any():
        for j in np.flatnonzero(newly):
            first_done[j] = i
            fall_x[j] = qpos[j, 0] - x0[j]
            fall_z[j] = qpos[j, 2]
            fall_upz[j] = upz[j]
            fall_vx[j] = qvel[j, 0]
            causes = {kk: float(np.array(st.metrics[kk])[j]) for kk in term_keys}
            fall_cause[j] = {kk: vv for kk, vv in causes.items() if vv > 0} or causes
    alive &= ~d
    success |= alive & (np.array(st.metrics["success"]) > 0)

hist_x = np.array(hist_x)
hist_z = np.array(hist_z)
hist_upz = np.array(hist_upz)

print(f"\nalive {alive.sum()}/{N}   crossed(success) {success.sum()}/{N}   falls {(~alive).sum()}")
print(f"\n{'trial':>5s} {'done_step':>9s} {'dx_at_end':>9s} {'z':>6s} {'up_z':>6s} {'vx':>6s}  cause")
for j in range(N):
    tag = "FALL" if not alive[j] else ("ok  " if success[j] else "noSucc")
    print(f"{j:>5d} {first_done[j]:>9d} {fall_x[j]:>9.3f} {fall_z[j]:>6.3f} "
          f"{fall_upz[j]:>6.3f} {fall_vx[j]:>6.2f}  {tag} {fall_cause[j] if fall_cause[j] else ''}")

fallen = np.flatnonzero(~alive)
print("\n=== the falls, 25 steps before termination ===")
for j in fallen:
    s = first_done[j]
    lo = max(0, s - 25)
    print(f"\ntrial {j}  fell at step {s}")
    print("   step   dx      z     up_z")
    for t in range(lo, s + 1, 5):
        print(f"  {t:>5d} {hist_x[t, j] - x0[j]:>6.3f} {hist_z[t, j]:>6.3f} {hist_upz[t, j]:>7.3f}")

print("\n=== survivors for comparison: dx at same steps ===")
surv = np.flatnonzero(alive)[:3]
for j in surv:
    print(f"trial {j}: dx@200 {hist_x[200, j]-x0[j]:.2f}  dx@400 {hist_x[400, j]-x0[j]:.2f}  "
          f"dx@600 {hist_x[600, j]-x0[j]:.2f}  final {hist_x[-1, j]-x0[j]:.2f}")
