"""Damped least-squares IK for the Z1 on a (possibly moving) Go2 base.

Works on any model containing the Go2+Z1 (``go2z1.xml`` or ``workshop.xml``): joints and
the ``ee`` site are resolved by name, and only the six arm joints are solved for -- the
base pose and legs are whatever the MjData currently says. Solving on a scratch copy of
the data keeps IK side-effect free with respect to the live simulation.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = tuple(f"arm_joint{i}" for i in range(1, 7))
GRIPPER_JOINT = "arm_jointGripper"
GRIPPER_OPEN = -1.2
GRIPPER_CLOSED = 0.0


class ArmIK:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        assert self.site >= 0, "site 'ee' missing"
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        assert min(jids) >= 0, "arm joints missing"
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0].copy()
        self.hi = model.jnt_range[jids, 1].copy()
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
        self.grip_qadr = model.jnt_qposadr[g]
        self.scratch = mujoco.MjData(model)

    def ee_pose(self, data: mujoco.MjData):
        return data.site_xpos[self.site].copy(), data.site_xmat[self.site].reshape(3, 3).copy()

    def arm_q(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qadr].copy()

    SEEDS = (
        (0.0, 0.9, -1.2, 0.3, 0.0, 0.0),
        (0.0, 1.4, -0.9, -0.6, 0.0, 0.0),
        (0.0, 1.8, -1.8, 0.0, 0.0, 0.0),
        (0.0, 1.2, -1.6, 1.0, 0.0, 0.0),
    )

    def solve_multi(self, data, target_pos, target_rot=None, q_init=None, **kw):
        """solve() from q_init and from a few canonical seeds; best error wins."""
        inits = ([] if q_init is None else [np.asarray(q_init)]) + [np.array(s) for s in self.SEEDS]
        best = None
        for qi in inits:
            q, ep, er, ok = self.solve(data, target_pos, target_rot, q_init=qi, **kw)
            score = ep + 0.05 * er
            if best is None or (ok and not best[3]) or (ok == best[3] and score < best[1] + 0.05 * best[2]):
                best = (q, ep, er, ok)
            if ok and qi is inits[0]:
                break
        return best

    def solve(self, data: mujoco.MjData, target_pos, target_rot=None, q_init=None,
              iters: int = 200, tol_pos: float = 2e-3, tol_rot: float = 0.03,
              damping: float = 0.05, rot_weight: float = 0.35):
        """Return (q_arm, pos_err, rot_err, ok). target_rot is a 3x3 world rotation of the
        ee site; None solves position only."""
        d = self.scratch
        d.qpos[:] = data.qpos
        d.qvel[:] = 0
        if q_init is not None:
            d.qpos[self.qadr] = q_init
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        target_pos = np.asarray(target_pos, float)
        err_p = err_r = np.inf
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            pos = d.site_xpos[self.site]
            rot = d.site_xmat[self.site].reshape(3, 3)
            ep = target_pos - pos
            err_p = float(np.linalg.norm(ep))
            if target_rot is not None:
                er = _rot_error(rot, np.asarray(target_rot))
                err_r = float(np.linalg.norm(er))
            else:
                er, err_r = np.zeros(3), 0.0
            if err_p < tol_pos and err_r < tol_rot:
                break
            mujoco.mj_jacSite(self.m, d, jacp, jacr, self.site)
            Jp = jacp[:, self.dadr]
            if target_rot is not None:
                J = np.vstack([Jp, rot_weight * jacr[:, self.dadr]])
                e = np.concatenate([ep, rot_weight * er])
            else:
                J, e = Jp, ep
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), e)
            dq = np.clip(dq, -0.25, 0.25)
            q = np.clip(d.qpos[self.qadr] + dq, self.lo, self.hi)
            d.qpos[self.qadr] = q
        ok = err_p < max(tol_pos * 5, 0.01) and err_r < max(tol_rot * 3, 0.1)
        return d.qpos[self.qadr].copy(), err_p, err_r, ok


def _rot_error(R: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """World-frame axis-angle vector rotating R onto R_target."""
    Re = R_target @ R.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Re.reshape(-1))
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    return vel


def topdown_rot(yaw: float, tilt: float = 0.0) -> np.ndarray:
    """ee orientation with the gripper (link06 +x) pointing down, jaws closing across the
    object's long axis at `yaw`. `tilt` (rad) leans the approach back toward the robot,
    which is what makes bench-height grasps reachable for an arm on a quadruped."""
    # link06 frame: x = approach direction, y = jaw axis, z completes.
    x = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    y = np.array([0.0, 1.0, 0.0])
    z = np.cross(x, y)
    R0 = np.stack([x, y, z], axis=1)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ R0
