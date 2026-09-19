"""Layer 3 -- locomotion, behind the interface the plan fixes in Week 1:

    locomotion.set_velocity(vx, vy, wz)      # 50 Hz closed loop
    locomotion.get_base_pose() -> SE3
    locomotion.is_stable() -> bool

CPU MuJoCo re-implementation of the MJX env's observation, bit-for-bit in layout and
ordering (bw/locomotion/go2_arm_env.py `_single_obs`, go2-stairs `_stacked_obs`):
51-value privileged frame = [world lin vel (3), projected gravity (3), gyro*0.25 (3),
accelerometer*0.05 (3), leg q - default (12), leg qd*0.05 (12), last action (12),
command (3)], 5 frames stacked newest-first. Sensor noise is NOT added at deployment.

Ordering matters and is the classic silent sim-to-sim bug: in the env, a control step
is physics -> last_action := a_k -> obs(last_lin_vel = v_{k-1}) -> last_lin_vel := v_k.
`after_physics()` reproduces exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

N_LEG = 12
LEG_JOINTS = tuple(f"{l}_{p}_joint" for l in ("FL", "FR", "RL", "RR") for p in ("hip", "thigh", "calf"))


@dataclass
class SE3:
    pos: np.ndarray        # (3,)
    quat: np.ndarray       # (4,) wxyz

    @property
    def yaw(self) -> float:
        w, x, y, z = self.quat
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    @property
    def rot(self) -> np.ndarray:
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, self.quat)
        return m.reshape(3, 3)


class NumpyPolicy:
    def __init__(self, npz: str | Path):
        f = np.load(npz)
        self.mean, self.std = f["obs_mean"], f["obs_std"]
        n = int(f["n_layers"])
        self.W = [f[f"W{i}"] for i in range(n)]
        self.b = [f[f"b{i}"] for i in range(n)]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / self.std
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                # brax make_ppo_networks' default activation is SWISH, not ReLU. With ReLU
                # here the exported policy barely walked (0.19 m in 3 s at vx=0.5).
                x = x / (1.0 + np.exp(-x))
        return np.tanh(x[:N_LEG])


# Left-right mirror of the Go2 (legs FL,FR,RL,RR x hip,thigh,calf; home hips are 0).
_LEG_PERM = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
_LEG_SIGN = np.array([-1.0, 1, 1] * 4)
_FRAME_SIGN = np.concatenate([
    [1, -1, 1],          # world linear velocity (world reflected about its xz-plane)
    [1, -1, 1],          # projected gravity
    [-1, 1, -1],         # gyro: a pseudo-vector
    [1, -1, 1],          # accelerometer
])


def mirror_legs(v: np.ndarray) -> np.ndarray:
    return v[_LEG_PERM] * _LEG_SIGN


def mirror_frame(f: np.ndarray) -> np.ndarray:
    out = f.copy()
    out[:12] = f[:12] * _FRAME_SIGN
    for a in (12, 24, 36):                       # joint pos, joint vel, last action
        out[a:a + 12] = mirror_legs(f[a:a + 12])
    out[48:51] = f[48:51] * [1, -1, -1]          # command (vx, vy, wz)
    return out


class Locomotion:
    """Drives the Go2 legs in a CPU MjData at 50 Hz. The arm is untouched.

    MIRRORING. The nav fine-tune (configs/payload_nav.yaml, run 2) learned turn-in-place to
    the RIGHT (-0.56 of -0.6 rad/s at 5.6M steps) but still stands still when asked to turn
    LEFT. The Go2 is left-right symmetric and the policy does not observe the arm, so a left
    turn is a mirrored right turn: mirror the observation history, ask the policy, mirror its
    action back. This is the deployment-time form of symmetry augmentation. `mirror_when`
    decides per tick; by default it mirrors pure left turns only.
    """

    scale_gyro, scale_dof_vel, scale_accel, obs_clip = 0.25, 0.05, 0.05, 100.0
    action_scale, history_len, frame = 0.5, 5, 51
    # Commands the navigation policy was trained on (configs/payload_nav.yaml: general mode
    # plus the back / sidestep / turn-in-place modes).
    vx_range, vy_range, wz_range = (-0.4, 0.9), (-0.3, 0.3), (-1.0, 1.0)

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, policy_npz: str | Path):
        self.m, self.d = model, data
        self.policy = NumpyPolicy(policy_npz)
        qadr = [model.jnt_qposadr[model.joint(j).id] for j in LEG_JOINTS]
        assert qadr == list(range(7, 19)), qadr
        assert model.nu >= N_LEG and all(model.actuator_trnid[i, 0] == model.joint(j).id
                                          for i, j in enumerate(LEG_JOINTS)), "leg actuators must be 0..11"
        key = model.key("home").id
        self.default_pose = model.key_qpos[key][7:19].copy()
        self.default_ctrl = model.key_ctrl[key][:N_LEG].copy()
        self.ctrl_lo = model.actuator_ctrlrange[:N_LEG, 0]
        self.ctrl_hi = model.actuator_ctrlrange[:N_LEG, 1]
        self.n_substeps = int(round(0.02 / model.opt.timestep))
        self.dt = self.n_substeps * model.opt.timestep
        self.base = model.body("base").id
        self.mirror_when = lambda cmd: cmd[2] >= 0.3 and abs(cmd[0]) < 0.05 and abs(cmd[1]) < 0.05
        self.mirrored = False
        self.reset()

    # -------------------------------------------------------------- interface
    def reset(self):
        self.command = np.zeros(3)
        self.last_action = np.zeros(N_LEG)
        self.last_lin_vel = self.d.qvel[:3].copy()
        self.hist = np.zeros(self.frame * self.history_len, np.float32)
        self._push_frame()
        self.action = np.zeros(N_LEG)

    def set_velocity(self, vx: float, vy: float = 0.0, wz: float = 0.0):
        self.command = np.array([np.clip(vx, *self.vx_range), np.clip(vy, *self.vy_range),
                                 np.clip(wz, *self.wz_range)])

    def get_base_pose(self) -> SE3:
        return SE3(self.d.qpos[:3].copy(), self.d.qpos[3:7].copy())

    def is_stable(self, max_tilt_deg: float = 30.0, min_height: float = 0.15) -> bool:
        g = self._proj_grav()
        tilt = np.degrees(np.arccos(np.clip(-g[2], -1, 1)))
        z_rel = self.d.qpos[2] - self._ground_under()
        return bool(tilt < max_tilt_deg and z_rel > min_height and np.isfinite(self.d.qpos).all())

    # ----------------------------------------------------------- control loop
    def pre_physics(self):
        """Compute a_k from the latest observation and write leg ctrl."""
        self.mirrored = bool(self.mirror_when(self.command))
        if self.mirrored:
            h = self.hist.reshape(self.history_len, self.frame)
            a = mirror_legs(self.policy(np.concatenate([mirror_frame(f) for f in h])))
        else:
            a = self.policy(self.hist)
        self.action = a
        target = np.clip(self.default_ctrl + self.action_scale * a, self.ctrl_lo, self.ctrl_hi)
        self.d.ctrl[:N_LEG] = target

    def after_physics(self):
        self.last_action = self.action
        self._push_frame()
        self.last_lin_vel = self.d.qvel[:3].copy()

    def control_step(self, physics_step=None):
        """One 50 Hz tick: policy -> n substeps -> observation."""
        self.pre_physics()
        for _ in range(self.n_substeps):
            (physics_step or mujoco.mj_step)(self.m, self.d)
        self.after_physics()

    # --------------------------------------------------------------- internals
    def _proj_grav(self):
        w, x, y, z = self.d.qpos[3:7]
        return -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])

    def _rot_inv(self, v):
        R = self.get_base_pose().rot
        return R.T @ v

    def _ground_under(self) -> float:
        # Highest world geom below the base, via a downward ray (walkway = 0.12, floor = 0).
        geomid = np.zeros(1, np.int32)
        pnt = self.d.qpos[:3].copy()
        dist = mujoco.mj_ray(self.m, self.d, pnt, np.array([0, 0, -1.0]), None, 1, self.base, geomid)
        return float(pnt[2] - dist) if dist >= 0 else 0.0

    def _push_frame(self):
        d = self.d
        g = self._proj_grav()
        acc = self._rot_inv((d.qvel[:3] - self.last_lin_vel) / self.dt) - 9.81 * g
        proprio = np.concatenate([
            g, d.qvel[3:6] * self.scale_gyro, acc * self.scale_accel,
            d.qpos[7:19] - self.default_pose, d.qvel[6:18] * self.scale_dof_vel,
            self.last_action, self.command,
        ])
        proprio = np.clip(proprio, -self.obs_clip, self.obs_clip)
        frame = np.concatenate([np.clip(d.qvel[:3], -self.obs_clip, self.obs_clip), proprio])
        frame = np.nan_to_num(frame, nan=0.0, posinf=self.obs_clip, neginf=-self.obs_clip)
        self.hist = np.roll(self.hist, self.frame)
        self.hist[: self.frame] = frame
