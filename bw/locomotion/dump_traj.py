"""Roll out a locomotion policy in the Go2+Z1 MJX env and dump numbers only (no OpenGL).

    PYTHONPATH=/mnt/d/bringwrench:~/go2-stairs ~/go2-stairs/.venv/bin/python -m bw.locomotion.dump_traj \
        --checkpoint <ckpt> --out /mnt/d/bringwrench/media/loco.npz

EGL rendering does not work inside WSL (go2-stairs HANDOFF), so the GPU side writes qpos +
mocap_pos and Windows renders it (scripts/render_traj.py). mocap_pos is dumped because the
step platform lives in mocap bodies -- without it the robot walks on thin air.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(Path.home() / "go2-stairs")]

import jax
import numpy as np
import yaml

from bw.locomotion.go2_arm_env import ArmStairsConfig, Go2ArmEnv
from eval import load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=str(REPO / "configs/payload.yaml"))
    p.add_argument("--out", default=str(REPO / "media/loco.npz"))
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--best_of", type=int, default=8)
    p.add_argument("--rise", type=float, default=0.12)
    p.add_argument("--arm", default="random", choices=["random", "stowed", "extended"])
    p.add_argument("--vx", type=float, default=0.6)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ec = dict(cfg["env"])
    ec.update(flat_frac=0.0, episode_length=args.steps, cmd_stand_prob=0.0,
              cmd_resample_s=1e9, randomize_arm=args.arm == "random",
              arm_fixed="extended" if args.arm == "extended" else "stowed")
    env = Go2ArmEnv(config=ArmStairsConfig(**ec), generate_scene=False)
    policy = load_policy(env, cfg, args.checkpoint)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    n = args.best_of
    st = reset(jax.random.split(jax.random.PRNGKey(0), n))
    import jax.numpy as jp
    info = dict(st.info)
    info["command"] = jp.broadcast_to(jp.array([args.vx, 0.0, 0.0]), info["command"].shape)
    info["stair"] = jp.broadcast_to(jp.array([args.rise, 0.30, 0.0]), info["stair"].shape)
    st = st.replace(info=info)
    rng = jax.random.PRNGKey(1)
    qpos, mocap, live = [], [], []
    alive = np.ones(n)
    for _ in range(args.steps):
        rng, k = jax.random.split(rng)
        a, _ = policy(st.obs, k)
        st = step(st, a)
        i2 = dict(st.info)
        i2["command"] = jp.broadcast_to(jp.array([args.vx, 0.0, 0.0]), i2["command"].shape)
        i2["stair"] = jp.broadcast_to(jp.array([args.rise, 0.30, 0.0]), i2["stair"].shape)
        st = st.replace(info=i2)
        qpos.append(np.array(st.pipeline_state.qpos))
        mocap.append(np.array(st.pipeline_state.mocap_pos))
        live.append(alive.copy())
        alive = alive * (1.0 - np.array(st.done))
    qpos, mocap, live = np.array(qpos), np.array(mocap), np.array(live)
    steps_alive = live.sum(0)
    best = int(np.argmax(steps_alive))
    print(f"steps alive per seed: {steps_alive.astype(int).tolist()}  -> seed {best}")
    np.savez_compressed(args.out, qpos=qpos[:, best], mocap_pos=mocap[:, best],
                        live=live[:, best], scene=str(REPO / "models/scene_go2z1_step.xml"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
