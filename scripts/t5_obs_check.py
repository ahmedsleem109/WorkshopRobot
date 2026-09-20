"""T5 formal check: numpy Layer 3 (bw/locomotion/controller.py) vs the MJX env + brax inference.

    cd /mnt/d/bringwrench && JAX_PLATFORMS=cpu PYTHONPATH=/mnt/d/bringwrench:$HOME/go2-stairs \
        ~/go2-stairs/.venv/bin/python scripts/t5_obs_check.py [--steps 100]

Three checks, all from the same initial state (Go2ArmEnv.reset, arm stowed, DR/noise/latency/pushes
off, commands scripted: forward, turn R, back, side L, 25 steps each):

0. Weights/constants: the exported .npz equals the checkpoint's normalizer + policy params; the
   controller's default pose / default ctrl / ctrl range / dt on the DEPLOYMENT model
   (models/workshop.xml) equal the MJX env's.
1. Teacher-forced (the element-wise check): each 50 Hz tick, the brax policy acts on the MJX obs
   and the MJX env steps; the MJX post-step qpos/qvel is copied into a CPU MjData and the
   controller builds its own observation history and action from it. Compared per obs block
   (all 5 stacked frames) and per action element. Physics differences cannot leak in here.
2. Closed loop (context only): CPU MuJoCo + controller vs MJX + brax, each on its own physics.
   Differences here are MJX-vs-CPU solver drift, not controller bugs.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(Path.home() / "go2-stairs")]

import json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from bw.locomotion.controller import Locomotion  # noqa: E402
from bw.locomotion.go2_arm_env import ArmStairsConfig, Go2ArmEnv  # noqa: E402
from eval import load_policy  # noqa: E402  (go2-stairs/eval.py)

RUN = Path.home() / "bringwrench/runs/results/2026-09-19_16-27-13-payload_nav3"
BLOCKS = [("lin_vel", 0, 3), ("proj_grav", 3, 6), ("gyro", 6, 9), ("accel", 9, 12),
          ("joint_pos", 12, 24), ("joint_vel", 24, 36), ("last_action", 36, 48), ("command", 48, 51)]
SCHEDULE = [(0.5, 0.0, 0.0), (0.0, 0.0, -0.6), (-0.25, 0.0, 0.0), (0.0, 0.2, 0.0)]


def cmd_at(k, steps):
    return SCHEDULE[min(k * len(SCHEDULE) // steps, len(SCHEDULE) - 1)]


def block_diffs(a, b):
    a = np.asarray(a, np.float64).reshape(-1, 51)
    b = np.asarray(b, np.float64).reshape(-1, 51)
    return {n: float(np.max(np.abs(a[:, i:j] - b[:, i:j]))) for n, i, j in BLOCKS}


def with_command(env, st, cmd, rebuild_obs=False):
    info = dict(st.info)
    info["command"] = jp.array(cmd, jp.float32)
    if rebuild_obs:     # the reset frame was built with the sampled command
        info["obs_history"] = jp.zeros_like(info["obs_history"])
        info["priv_history"] = jp.zeros_like(info["priv_history"])
        obs, hist, priv = env._stacked_obs(st.pipeline_state, info)
        info["obs_history"], info["priv_history"] = hist, priv
        return st.replace(obs=obs, info=info)
    return st.replace(info=info)


def check_constants(env, npz, ckpt):
    from bw.locomotion.export_policy import main as export
    out = {}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.npz")
        export(str(ckpt), p)
        a, b = np.load(p), np.load(npz)
        out["npz_vs_ckpt_max_abs"] = max(float(np.max(np.abs(a[k] - b[k]))) for k in a.files)
    m = mujoco.MjModel.from_xml_path(str(REPO / "models/workshop.xml"))
    L = Locomotion(m, mujoco.MjData(m), npz)
    out["default_pose"] = float(np.max(np.abs(L.default_pose - np.asarray(env._default_pose))))
    out["default_ctrl"] = float(np.max(np.abs(L.default_ctrl - np.asarray(env._default_ctrl))))
    out["ctrl_range"] = float(np.max(np.abs(np.stack([L.ctrl_lo, L.ctrl_hi], 1) - np.asarray(env._ctrl_range))))
    out["dt"] = float(abs(L.dt - env._dt))
    for k in ("scale_gyro", "scale_dof_vel", "scale_accel", "obs_clip", "action_scale", "history_len"):
        out[k] = float(abs(getattr(L, k) - getattr(env.cfg, k)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(REPO / "models/payload_nav_policy.npz"))
    ap.add_argument("--checkpoint", default=str(RUN / "checkpoints/step_8110080"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = json.load(open(RUN / "config.json"))["cfg"]
    ec = dict(cfg["env"])
    ec.update(dr_enable=False, cmd_resample_s=1e9, cmd_stand_prob=0.0, randomize_arm=False,
              arm_fixed="stowed", flat_frac=1.0)
    env = Go2ArmEnv(config=ArmStairsConfig(**ec), generate_scene=False)
    policy = jax.jit(load_policy(env, cfg, a.checkpoint))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    key = jax.random.PRNGKey(0)

    print("== 0. weights / constants (max abs diff)")
    for k, v in check_constants(env, a.npz, a.checkpoint).items():
        print(f"   {k:22s} {v:.3g}")

    st0 = with_command(env, reset(jax.random.PRNGKey(a.seed)), (0.0, 0.0, 0.0), rebuild_obs=True)
    # (zero command in the reset frame: Locomotion.reset() also starts from a zero command)

    # ---------------------------------------------------------------- 1. teacher-forced
    m = env.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(st0.pipeline_state.qpos)
    d.qvel[:] = np.asarray(st0.pipeline_state.qvel)
    mujoco.mj_forward(m, d)
    L = Locomotion(m, d, a.npz)
    L.mirror_when = lambda c: False           # mirroring is a deliberate deployment deviation
    L.reset()
    obs_max = block_diffs(L.hist, st0.info["priv_history"])
    act_max, st = 0.0, st0
    per_step = []
    for k in range(a.steps):
        c = cmd_at(k, a.steps)
        st = with_command(env, st, c)
        L.set_velocity(*c)
        act_b = np.asarray(policy(st.obs, key)[0])
        L.pre_physics()
        da = float(np.max(np.abs(L.action - act_b)))
        act_max = max(act_max, da)
        st = step(st, jp.asarray(act_b))
        d.qpos[:] = np.asarray(st.pipeline_state.qpos)
        d.qvel[:] = np.asarray(st.pipeline_state.qvel)
        L.after_physics()
        bd = block_diffs(L.hist, st.info["priv_history"])
        obs_max = {n: max(obs_max[n], bd[n]) for n in obs_max}
        per_step.append((da, max(bd.values())))
        if float(st.done):
            print(f"   (MJX episode terminated at step {k})")
            break
    print(f"== 1. teacher-forced, {len(per_step)} steps: max abs diff per obs block (5 frames)")
    for n, v in obs_max.items():
        print(f"   {n:12s} {v:.3g}")
    print(f"   {'ACTION':12s} {act_max:.3g}   (|a| max {float(np.max(np.abs(act_b))):.2f})")
    worst = int(np.argmax([p[1] for p in per_step]))
    print(f"   worst obs step {worst}: obs {per_step[worst][1]:.3g}, action {per_step[worst][0]:.3g}")

    # ---------------------------------------------------------------- 2. closed loop
    d2 = mujoco.MjData(m)
    d2.qpos[:] = np.asarray(st0.pipeline_state.qpos)
    d2.qvel[:] = np.asarray(st0.pipeline_state.qvel)
    d2.ctrl[:] = np.asarray(st0.pipeline_state.ctrl)
    d2.mocap_pos[:] = np.asarray(st0.pipeline_state.mocap_pos)
    d2.mocap_quat[:] = np.asarray(st0.pipeline_state.mocap_quat)
    mujoco.mj_forward(m, d2)
    L2 = Locomotion(m, d2, a.npz)
    L2.mirror_when = lambda c: False
    L2.reset()
    st = st0
    rows = []
    for k in range(a.steps):
        c = cmd_at(k, a.steps)
        st = with_command(env, st, c)
        L2.set_velocity(*c)
        act_b = np.asarray(policy(st.obs, key)[0])
        st = step(st, jp.asarray(act_b))
        L2.control_step()
        rows.append((float(np.max(np.abs(L2.action - act_b))),
                     float(np.linalg.norm(d2.qpos[:2] - np.asarray(st.pipeline_state.qpos[:2]))),
                     float(np.max(np.abs(L2.hist[:51] - np.asarray(st.info["priv_history"][:51]))))))
    rows = np.array(rows)
    print(f"== 2. closed loop (own physics each): action diff at steps 1/10/50/{a.steps}: "
          + " / ".join(f"{rows[i, 0]:.2g}" for i in (0, 9, 49, len(rows) - 1)))
    print(f"   base xy divergence at step {len(rows)}: {rows[-1, 1] * 1000:.1f} mm; "
          f"newest-frame obs diff max {rows[:, 2].max():.3g}")


if __name__ == "__main__":
    main()
