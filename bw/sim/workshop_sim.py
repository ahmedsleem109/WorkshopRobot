"""CPU MuJoCo wrapper around models/workshop.xml, shared by data collection, VLA eval and the
full pipeline.

Responsibilities: domain randomization (object pose, lighting, clutter, initial arm
config), cameras, gripper-state detection, and ground-truth object poses. Ground truth
is exposed ONLY through `gt_*` methods, which the pipeline never calls -- they exist for
scoring (grounding error, grasp success).

Base modes:
  * "kinematic": base + legs are pinned every physics step (data collection; the arm
    policy is learned independently of the legs, which the plan's Go2+Z1 decoupling
    argument is precisely about).
  * "policy": the Layer-3 locomotion policy owns the legs (full pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from bw.sim.workshop import (BENCH_HEIGHT, RACK_CENTER, RACK_SLOT_Y, STEP_HEIGHT, TOOLS,
                             TOOL_NAMES, rack_floor_z, stand_quat)

ROOT = Path(__file__).resolve().parents[2]
N_LEG, N_ARM = 12, 7                 # command: 6 arm joints + 1 gripper travel
# ctrl layout: 12 legs, 6 arm joints, then the TWO finger servos (same commanded value).
ARM_CTRL = slice(N_LEG, N_LEG + 8)
ARM_J_CTRL = slice(N_LEG, N_LEG + 6)
FING_CTRL = slice(N_LEG + 6, N_LEG + 8)
ARM_J_QPOS = slice(19, 25)           # the six arm joints
FING_A, FING_B = 25, 26              # parallel-jaw finger slides (B mirrors A by equality)
# Parallel jaw: the command is finger travel in metres. 0 = closed (squeezing), 0.038 = open.
GRIPPER_OPEN, GRIPPER_CLOSED = 0.038, -0.008      # closed commands 8 mm past contact
RAMP_CHUNK = 10                  # physics steps per servo-target increment inside a 10 Hz tick


@dataclass
class RandCfg:
    tool_xy_jitter: float = 0.025
    tool_lean: float = 0.018           # small random lean inside the slot (quaternion xyz)
    p_all_tools: float = 0.5          # else a random subset (target always kept)
    light_scale: tuple = (0.55, 1.35)
    light_jitter: float = 0.6
    base_xy: tuple = ((4.00, 4.10), (-0.18, 0.18))
    base_yaw: float = 0.12
    arm_jitter: float = 0.12
    tray_color_jitter: float = 0.08


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


@dataclass
class ToolState:
    name: str
    pos: np.ndarray
    quat: np.ndarray
    present: bool = True


class WorkshopSim:
    def __init__(self, xml: str | Path = ROOT / "models/workshop.xml", render_size=(256, 256),
                 base_mode: str = "kinematic"):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.base_mode = base_mode
        self.h, self.w = render_size
        self._renderers: dict[tuple, mujoco.Renderer] = {}
        self._vopt = mujoco.MjvOption()
        self._vopt.sitegroup[:] = 0          # debug sites never appear in camera images
        self.tool_body = {n: self.m.body(n).id for n in TOOL_NAMES}
        self.tool_qadr = {n: self.m.jnt_qposadr[self.m.joint(f"{n}_free").id] for n in TOOL_NAMES}
        self.tool_dadr = {n: self.m.jnt_dofadr[self.m.joint(f"{n}_free").id] for n in TOOL_NAMES}
        self.ee_site = self.m.site("ee").id
        self.key_home = self.m.key("home").id
        self.key_ready = self.m.key("ready").id
        self.light_ids = [self.m.light(n).id for n in ("key", "fill", "bench_lamp")]
        self.light_diffuse0 = self.m.light_diffuse[self.light_ids].copy()
        self.light_pos0 = self.m.light_pos[self.light_ids].copy()
        self.tray_mat = self.m.material("tray").id
        self.tray_rgba0 = self.m.mat_rgba[self.tray_mat].copy()
        self.mover_body = self.m.body("arm_finger_a").id
        self.stator_body = self.m.body("arm_finger_b").id
        self.finger_bodies = {self.mover_body, self.stator_body}
        self.obstacle_mocap = self.m.body_mocapid[self.m.body("obstacle").id]
        self.stand_z = self._standing_offsets()
        self._hold_qpos = None
        self.arm_target = np.zeros(N_ARM)
        self.time = 0.0
        self.loco = None                 # Layer 3, attached by attach_locomotion()
        self._loco_k = 0

    # ------------------------------------------------------------------ Layer 3
    def attach_locomotion(self, policy_npz: str | Path = ROOT / "models/payload_nav_policy.npz"):
        """Hand the legs to the walking policy for the rest of this sim's life.

        From here on EVERY physics step -- grasp, walk, place -- runs the policy at 50 Hz, so
        the robot stands on its own legs while the arm works instead of being pinned. The
        walk between stations is then real walking (walk_to), not teleport_base.
        """
        from bw.locomotion.controller import Locomotion
        self.base_mode = "policy"
        self.loco = Locomotion(self.m, self.d, policy_npz)
        self._loco_k = 0
        return self.loco

    def _standing_offsets(self) -> dict[str, float]:
        """How high each tool's frame origin must sit above the rack floor when standing.

        Computed from the compiled collision geometry, not by hand: hand-written values put
        the wrenches' open ends 8 mm THROUGH the bench, where the solver wedged them with
        40-70 N and no grasp could ever lift them (scripts/_gripforce.py).
        """
        m = self.m
        out = {}
        for n in TOOL_NAMES:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, np.array(stand_quat(n), float))
            R = R.reshape(3, 3)
            lowest = 0.0
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != self.tool_body[n] or m.geom_contype[g] == 0:
                    continue
                gpos, gquat = m.geom_pos[g], m.geom_quat[g]
                Rg = np.zeros(9)
                mujoco.mju_quat2Mat(Rg, gquat)
                Rg = Rg.reshape(3, 3)
                half = m.geom_aabb[g, 3:]
                centre = m.geom_aabb[g, :3]
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            corner = centre + half * [sx, sy, sz]
                            world = R @ (gpos + Rg @ corner)
                            lowest = min(lowest, world[2])
            out[n] = -lowest + 0.0015           # 1.5 mm of clearance
        return out

    # ------------------------------------------------------------------ reset
    def reset(self, rng: np.random.Generator, target: str | None = None, rand: RandCfg | None = None,
              base_pose=None, arm_q=None, tools_present=None, key="ready"):
        rand = rand or RandCfg()
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
        # base on the walkway in front of the bench
        if base_pose is None:
            bx = rng.uniform(*rand.base_xy[0])
            by = rng.uniform(*rand.base_xy[1])
            byaw = rng.uniform(-rand.base_yaw, rand.base_yaw)
            base_pose = (bx, by, byaw)
        bx, by, byaw = base_pose
        d.qpos[0], d.qpos[1] = bx, by
        d.qpos[2] = STEP_HEIGHT + 0.30 if bx > 1.3 else 0.30
        d.qpos[3:7] = [np.cos(byaw / 2), 0, 0, np.sin(byaw / 2)]
        # arm
        if arm_q is None:
            kq = m.key_qpos[m.key(key).id]
            q_arm = np.concatenate([kq[ARM_J_QPOS], [kq[FING_A]]])
            q_arm[:6] += rng.uniform(-rand.arm_jitter, rand.arm_jitter, 6)
        else:
            q_arm = np.array(arm_q, float)
        lo = np.concatenate([m.actuator_ctrlrange[ARM_J_CTRL, 0], [m.actuator_ctrlrange[FING_CTRL][0, 0]]])
        hi = np.concatenate([m.actuator_ctrlrange[ARM_J_CTRL, 1], [m.actuator_ctrlrange[FING_CTRL][0, 1]]])
        q_arm = np.clip(q_arm, lo, hi)
        d.qpos[ARM_J_QPOS] = q_arm[:6]
        d.qpos[FING_A] = d.qpos[FING_B] = q_arm[6]
        d.ctrl[ARM_J_CTRL] = q_arm[:6]
        d.ctrl[FING_CTRL] = q_arm[6]
        self.arm_target = q_arm.copy()
        # tools: random permutation over tray slots + jitter; absent tools parked under bench
        names = list(TOOL_NAMES)
        if tools_present is None:
            if rng.uniform() < rand.p_all_tools:
                tools_present = set(names)
            else:
                k = rng.integers(2, 5)
                tools_present = set(rng.choice(names, size=k, replace=False))
                if target:
                    tools_present.add(target)
        order = rng.permutation(5)
        z0 = rack_floor_z()
        for i, n in enumerate(names):
            qa = self.tool_qadr[n]
            if n in tools_present:
                # Standing in a rack slot: lateral jitter within the slot, a small lean, and a
                # 180 deg flip about the vertical (ring end up vs open end up for a wrench).
                y = RACK_CENTER[1] + RACK_SLOT_Y[order[i]] + rng.uniform(-1, 1) * rand.tool_xy_jitter * 0.6
                d.qpos[qa:qa + 3] = [RACK_CENTER[0] + rng.uniform(-1, 1) * 0.002, y,
                                     z0 + self.stand_z[n]]
                qt = np.array(stand_quat(n), float)
                flip = rng.uniform() < 0.5
                lean = rng.uniform(-1, 1, 3) * rand.tool_lean
                dq = np.array([1.0, *lean])
                if flip:                       # 180 deg about the tool's own long axis (world -z)
                    dq = _quat_mul(np.array([0.0, 0.0, 0.0, 1.0]), dq)
                q = _quat_mul(dq, qt)
                d.qpos[qa + 3:qa + 7] = q / np.linalg.norm(q)
            else:
                d.qpos[qa:qa + 3] = [4.85, 0.55 + 0.12 * i, 0.03]    # under the bench, out of view
                d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        # lighting / appearance
        s = rng.uniform(*rand.light_scale)
        m.light_diffuse[self.light_ids] = np.clip(self.light_diffuse0 * s * rng.uniform(0.85, 1.15, (3, 1)), 0, 1)
        m.light_pos[self.light_ids] = self.light_pos0 + rng.uniform(-rand.light_jitter, rand.light_jitter, (3, 3)) * [1, 1, 0.3]
        m.mat_rgba[self.tray_mat, :3] = np.clip(self.tray_rgba0[:3] + rng.uniform(-rand.tray_color_jitter, rand.tray_color_jitter, 3), 0, 1)
        m.body_pos[self.m.body("obstacle").id] = [0, -8, 0.2]
        d.mocap_pos[self.obstacle_mocap] = [0, -8, 0.2]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        self._hold_qpos = d.qpos[:19].copy()
        self.time = 0.0
        if self.loco is not None:
            self.loco.reset()                 # fresh observation history for the new state
            self._loco_k = 0
        # Tools must be STATIC before a grasp is planned from their pose -- to a measured
        # velocity threshold, not a fixed wait (see settle_static).
        self.settle_static()
        return tools_present

    # ------------------------------------------------------------------ physics
    def physics_step(self, n: int = 1):
        loco = self.loco if self.base_mode == "policy" else None
        for _ in range(n):
            if self.base_mode == "kinematic":
                self.d.qpos[:19] = self._hold_qpos
                self.d.qvel[:18] = 0
            if loco is not None and self._loco_k == 0:
                loco.pre_physics()                 # 50 Hz: policy -> leg targets
            mujoco.mj_step(self.m, self.d)
            if loco is not None:
                self._loco_k += 1
                if self._loco_k == loco.n_substeps:
                    self._loco_k = 0
                    loco.after_physics()
        self.time += n * self.m.opt.timestep

    def settle(self, seconds: float):
        self.physics_step(int(seconds / self.m.opt.timestep))

    def settle_static(self, max_seconds: float = 4.0, tol: float = 0.002,
                      chunk: float = 0.25) -> float:
        """Settle until no tool is moving faster than `tol` m/s, or `max_seconds`.

        A FIXED settle is not enough: measured 2026-09-18, one reset in ~25 still had a tool
        sliding at 56 mm/s after the old fixed 1.2 s, and it then travelled another 49 mm --
        so the grasp was planned from a pose the tool had already left. Returns the time spent,
        which the caller can log; it is normally ~1.2 s and only occasionally longer.
        """
        t = 0.0
        while t < max_seconds:
            self.settle(chunk)
            t += chunk
            v = max(float(np.linalg.norm(self.d.qvel[self.tool_dadr[n]:self.tool_dadr[n] + 3]))
                    for n in TOOL_NAMES)
            if t >= 1.0 and v < tol:
                break
        return t

    def teleport_base(self, pose, carry: str | None = None):
        """Move the base to (x, y, yaw) as a rigid SE(2) transform, carrying `carry` with it.

        This is the manipulation benchmark's STAND-IN for Layer 3 navigation, and it is a
        stand-in on purpose: the two-table transfer needs the base at a different station for
        the place than for the pick, and walking there is the locomotion policy's job (T5/T9),
        not the grasp demonstrator's. A held tool is a free body, so it must be transformed by
        the same delta or the teleport simply drops it.
        """
        d = self.d
        x0, y0 = float(d.qpos[0]), float(d.qpos[1])
        yaw0 = self.get_base_yaw()
        x1, y1, yaw1 = pose
        dyaw = yaw1 - yaw0
        c, s_ = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        dq = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])

        def xform(p):
            rel = np.asarray(p, float) - np.array([x0, y0, 0.0])
            return R @ rel + np.array([x1, y1, 0.0])

        if carry is not None:
            qa = self.tool_qadr[carry]
            d.qpos[qa:qa + 3] = xform(d.qpos[qa:qa + 3])
            d.qpos[qa + 3:qa + 7] = _quat_mul(dq, d.qpos[qa + 3:qa + 7])
        d.qpos[0], d.qpos[1] = x1, y1
        d.qpos[3:7] = _quat_mul(dq, d.qpos[3:7])
        d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        self._hold_qpos = d.qpos[:19].copy()

    def set_arm_target(self, q: np.ndarray):
        """q is the 7-vector command (6 joints + finger travel); both finger servos get it."""
        m = self.m
        lo = np.concatenate([m.actuator_ctrlrange[ARM_J_CTRL, 0], [m.actuator_ctrlrange[FING_CTRL][0, 0]]])
        hi = np.concatenate([m.actuator_ctrlrange[ARM_J_CTRL, 1], [m.actuator_ctrlrange[FING_CTRL][0, 1]]])
        self.arm_target = np.clip(q, lo, hi)
        self.d.ctrl[ARM_J_CTRL] = self.arm_target[:6]
        self.d.ctrl[FING_CTRL] = self.arm_target[6]

    def move_arm(self, q_goal: np.ndarray, duration: float, record=None, rate_hz: float = 10.0):
        """Linear joint-space interpolation executed by the servos; optional per-tick callback
        at `rate_hz` receives the commanded target (the action a policy would emit)."""
        q0 = self.arm_target.copy()
        steps = max(1, int(round(duration * rate_hz)))
        sub = int(round(1.0 / (rate_hz * self.m.opt.timestep)))
        chunk = max(1, min(RAMP_CHUNK, sub))
        for k in range(1, steps + 1):
            q = q0 + (q_goal - q0) * (k / steps)
            if record is not None:
                record(q)
            # RAMP the servo target across the tick (first-order hold), do not step it. With a
            # stepped target the stiff arm servos lurched to each 10 Hz command and came to a
            # complete stop before the next one -- measured, joint speed peaked at 4.2x its
            # mean every tick and hit exactly zero at every tick's end: a 10 Hz start-stop
            # stutter that reads as the robot vibrating on video. The recorded action is
            # unchanged; it is where the arm should be at the END of the tick.
            q_prev = self.arm_target.copy()
            done = 0
            while done < sub:
                n = min(chunk, sub - done)
                done += n
                self.set_arm_target(q_prev + (q - q_prev) * (done / sub))
                self.physics_step(n)

    def arm_q(self) -> np.ndarray:
        """Six arm joints plus the finger travel -- the 7-vector the VLA acts in."""
        return np.concatenate([self.d.qpos[ARM_J_QPOS], [self.d.qpos[FING_A]]])

    # ------------------------------------------------------------------ sensing
    def render(self, camera: str, depth: bool = False, size=None):
        h, w = size or (self.h, self.w)
        key = (h, w, depth)
        if key not in self._renderers:
            r = mujoco.Renderer(self.m, h, w)
            if depth:
                r.enable_depth_rendering()
            self._renderers[key] = r
        r = self._renderers[key]
        r.update_scene(self.d, camera=camera, scene_option=self._vopt)
        return r.render()

    def camera_intrinsics(self, camera: str, size=None):
        h, w = size or (self.h, self.w)
        fovy = np.radians(self.m.cam_fovy[self.m.camera(camera).id])
        f = 0.5 * h / np.tan(fovy / 2)
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

    def camera_pose(self, camera: str):
        cid = self.m.camera(camera).id
        return self.d.cam_xpos[cid].copy(), self.d.cam_xmat[cid].reshape(3, 3).copy()

    def gripper_contacts(self, body_id: int) -> set[int]:
        """Tool bodies in contact with both jaws."""
        touch = {self.mover_body: set(), self.stator_body: set()}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for jaw in touch:
                if b1 == jaw:
                    touch[jaw].add(b2)
                elif b2 == jaw:
                    touch[jaw].add(b1)
        return touch[self.mover_body] & touch[self.stator_body]

    def gripper_state(self) -> str:
        """open | closed | holding -- from finger travel and bilateral pad contact. No ground
        truth about WHICH object is held; that is what the orchestrator must live with."""
        held = self.gripper_contacts(0) & set(self.tool_body.values())
        commanded_closed = self.arm_target[6] < 0.010
        if held and commanded_closed:
            return "holding"
        return "closed" if commanded_closed else "open"

    def held_tool(self) -> str | None:
        held = self.gripper_contacts(0)
        for n, b in self.tool_body.items():
            if b in held:
                return n
        return None

    # ------------------------------------------------------------------ ground truth (scoring only)
    def gt_tool_pos(self, name: str) -> np.ndarray:
        return self.d.xpos[self.tool_body[name]].copy()

    def gt_tool_yaw(self, name: str) -> float:
        w, x, y, z = self.d.xquat[self.tool_body[name]]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def get_base_yaw(self) -> float:
        w, x, y, z = self.d.qpos[3:7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def ee_pos(self) -> np.ndarray:
        return self.d.site_xpos[self.ee_site].copy()
