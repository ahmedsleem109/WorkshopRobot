"""Payload-aware locomotion: Go2 carrying a Z1 arm, crossing a 12 cm step (MJX / JAX).

Built on the go2-stairs project's ``Go2StairsEnv`` (D:/hexapod, ~/go2-stairs). What
changes, and why each change is forced rather than chosen:

1. **The model has 7 more actuated joints.** Every ``qpos[7:]`` / ``qvel[6:]`` /
   ``actuator_force`` in the parent assumes the Go2 is the whole robot, so ``reset``,
   ``_single_obs`` and ``step`` are re-implemented here with explicit leg slices. The
   policy still outputs 12 leg targets; the arm is driven by its own servos from a
   scripted command stream (below).

2. **The observation layout is frozen** (48 x 5 actor, 51 x 5 critic) so the stairs
   run-7 checkpoint warm-starts directly -- the plan says fine-tune, do not retrain.
   The consequence is that the policy is NOT told where the arm is; it must infer the
   shifting centre of mass from IMU and leg joint states, the same way it already
   infers terrain. Stated as a limitation in the README.

3. **The arm moves during training.** Each env samples an arm goal (stowed / extended /
   uniform over a task-relevant joint box) every 2-6 s and slews toward it at a
   realistic joint speed, so the policy sees both static CoM offsets and the reaction
   torques of an arm in motion. That is the "randomize arm joint configuration during
   training" item, done dynamically instead of per-episode.

4. **Terrain is one step, not a flight.** ``n_steps=1`` plus a 2 m landing gives exactly
   the workshop's raised walkway: step up, cross, step down. Success means crossing the
   whole platform and coming back down, not merely reaching the first tread.

5. **Arm-stability reward.** Roll/pitch rate and tilt are charged extra in proportion to
   how far the gripper is from the mount (0 stowed .. 1 fully out).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

GO2_STAIRS = Path.home() / "go2-stairs"
if str(GO2_STAIRS) not in sys.path:
    sys.path.insert(0, str(GO2_STAIRS))

from envs.go2_flat import State  # noqa: E402
from envs.go2_stairs import Go2StairsEnv, StairsConfig  # noqa: E402
from terrain.stairs import (NUM_LEVELS, StairSpec, box_poses, build_scene_xml,  # noqa: E402
                            stair_frame)

REPO = Path(__file__).resolve().parents[2]
N_LEG = 12
N_ARM = 7                       # command: 6 arm joints + 1 gripper travel
N_ARM_CTRL = 8                  # ctrl: the gripper command drives BOTH finger servos

# Arm joint box sampled for "random" goals: roughly the region SmolVLA's grasps visit.
# Last entry is the parallel jaw's finger travel in metres (0 closed .. 0.038 open).
ARM_LO = jp.array([-1.0, 0.0, -2.4, -1.2, -1.0, -1.5, 0.0])
ARM_HI = jp.array([1.0, 2.2, -0.2, 1.2, 1.0, 1.5, 0.038])


@dataclasses.dataclass
class ArmStairsConfig(StairsConfig):
    n_steps: int = 1
    landing: float = 2.0
    approach: float = 2.0
    spawn_back: float = 2.0
    stair_width: float = 2.5
    # Success = crossed the platform and stepped back down.
    cross_margin: float = 0.4

    # --- arm command stream ---
    arm_p_stowed: float = 0.35
    arm_p_extended: float = 0.25    # remainder: uniform in ARM_LO..ARM_HI
    arm_switch_s: Tuple[float, float] = (2.0, 6.0)
    arm_speed: float = 1.2          # rad/s slew limit on the commanded arm target
    randomize_arm: bool = True      # False = arm fixed at `arm_fixed` (ablation/eval)
    arm_fixed: str = "stowed"       # stowed | extended

    # --- arm-stability reward ---
    w_arm_stab: float = 0.3

    # --- commands for mobile manipulation ---
    # Run 7 only ever saw vx in [0.3, 0.9]: it has never been asked to STAND, and the
    # orchestrator stands still for every grasp and turns in place to face targets.
    # So: a stand probability, a wider yaw range, and commands resampled mid-episode
    # (navigate_to changes the command continuously; per-episode constant commands
    # would never teach the transitions).
    cmd_stand_prob: float = 0.2
    cmd_resample_s: float = 5.0     # mean seconds between resamples
    progress_min_vx: float = 0.2    # progress/height/clearance rewards only when walking


class Go2ArmEnv(Go2StairsEnv):
    def __init__(self, scene_path: str | None = None, config: ArmStairsConfig | None = None,
                 generate_scene: bool = True):
        cfg = config or ArmStairsConfig()
        scene_path = scene_path or str(REPO / "models" / "scene_go2z1_step.xml")
        spec = StairSpec(n_steps=cfg.n_steps, run_max=cfg.run_max, width=cfg.stair_width,
                         approach=cfg.approach, landing=cfg.landing, box_depth=cfg.box_depth)
        if generate_scene:
            xml = build_scene_xml(spec).replace('<include file="go2_mjx.xml"/>',
                                                '<include file="go2z1_mjx.xml"/>')
            Path(scene_path).write_text(xml)
        super().__init__(scene_path=scene_path, config=cfg, generate_scene=False)
        self.cfg: ArmStairsConfig = cfg
        mj = self.mj_model

        leg_names = [f"{l}_{p}_joint" for l in ("FL", "FR", "RL", "RR") for p in ("hip", "thigh", "calf")]
        leg_q = [mj.jnt_qposadr[mj.joint(n).id] for n in leg_names]
        assert leg_q == list(range(7, 19)), f"leg joints not at qpos[7:19]: {leg_q}"
        arm_act = ([mj.actuator(f"arm_motor{i}").id for i in range(1, 7)]
                   + [mj.actuator("arm_motorGripper_a").id, mj.actuator("arm_motorGripper_b").id])
        assert arm_act == list(range(12, 20)), arm_act
        assert mj.nu == N_LEG + N_ARM_CTRL
        # qpos: 7 free + 12 legs + 6 arm joints + 2 finger slides
        assert mj.nq == 7 + N_LEG + 8, mj.nq
        # The arm makes the stock iterations=1 solver diverge (see bw/sim/build_models.py).
        assert mj.opt.iterations >= 4 and mj.opt.ls_iterations >= 10, (
            f"solver too weak for the arm model: iterations={mj.opt.iterations}, "
            f"ls_iterations={mj.opt.ls_iterations}; rebuild with bw.sim.build_models")

        key = mj.key("home").id
        self._nu = N_LEG
        self._default_pose = jp.array(mj.key_qpos[key][7:19])
        self._default_ctrl = jp.array(mj.key_ctrl[key][:N_LEG])
        self._ctrl_range = jp.array(mj.actuator_ctrlrange[:N_LEG])
        self._init_q = jp.array(mj.key_qpos[key])
        self._arm_stowed = self._cmd7(mj.key_ctrl[mj.key("home").id])
        self._arm_extended = self._cmd7(mj.key_ctrl[mj.key("extended").id])
        rng_ = mj.actuator_ctrlrange
        self._arm_ctrl_lo = jp.concatenate([jp.array(rng_[N_LEG:N_LEG + 6, 0]), jp.array(rng_[N_LEG + 6:N_LEG + 7, 0])])
        self._arm_ctrl_hi = jp.concatenate([jp.array(rng_[N_LEG:N_LEG + 6, 1]), jp.array(rng_[N_LEG + 6:N_LEG + 7, 1])])

        jnt_range = jp.array(mj.jnt_range[1:1 + N_LEG])
        mid = 0.5 * (jnt_range[:, 0] + jnt_range[:, 1])
        half = 0.5 * (jnt_range[:, 1] - jnt_range[:, 0]) * cfg.soft_limit_frac
        self._soft_low, self._soft_high = mid - half, mid + half

        self._ee_site = mj.site("ee").id
        self._mount_body = mj.body("arm_link00").id
        self._sw_lo = int(round(cfg.arm_switch_s[0] / self._dt))
        self._sw_hi = int(round(cfg.arm_switch_s[1] / self._dt))

    @staticmethod
    def _cmd7(ctrl) -> jax.Array:
        """ctrl (20) -> the 7-vector arm command (6 joints + one gripper value)."""
        return jp.array(list(ctrl[N_LEG:N_LEG + 6]) + [ctrl[N_LEG + 6]])

    @staticmethod
    def _arm_ctrl(cmd7: jax.Array) -> jax.Array:
        """7-vector command -> 8 arm ctrl values (the gripper value twice)."""
        return jp.concatenate([cmd7[:6], cmd7[6:7], cmd7[6:7]])

    # ------------------------------------------------------------------ terrain
    def _summit_s(self, info):
        _, run, _ = self._stair(info)
        return self.spec.n_steps * run + self.spec.landing + self.cfg.cross_margin

    # ------------------------------------------------------------------ commands
    def _sample_command(self, rng):
        k_cmd, k_stand = jax.random.split(rng)
        cmd = super()._sample_command(k_cmd)
        stand = jax.random.uniform(k_stand, ()) < self.cfg.cmd_stand_prob
        return jp.where(stand, jp.zeros(3), cmd)

    # ------------------------------------------------------------------ arm
    def _sample_arm_goal(self, rng):
        c = self.cfg
        k_mode, k_u = jax.random.split(rng)
        u = jax.random.uniform(k_mode, ())
        rand = jax.random.uniform(k_u, (N_ARM,), minval=ARM_LO, maxval=ARM_HI)
        goal = jp.where(u < c.arm_p_stowed, self._arm_stowed,
                        jp.where(u < c.arm_p_stowed + c.arm_p_extended, self._arm_extended, rand))
        fixed = self._arm_extended if c.arm_fixed == "extended" else self._arm_stowed
        return jp.where(c.randomize_arm, goal, fixed)

    def _arm_extension(self, data):
        """0 with the gripper folded at the mount (~0.25 m), 1 at ~0.6 m out."""
        d = jp.linalg.norm(data.site_xpos[self._ee_site] - data.xpos[self._mount_body])
        return jp.clip((d - 0.25) / 0.35, 0.0, 1.0)

    # --------------------------------------------------------- observation
    def _single_obs(self, data, info):
        quat = data.qpos[3:7]
        c = self.cfg
        proprio = jp.clip(
            jp.concatenate([
                self._projected_gravity(quat),
                data.qvel[3:6] * c.scale_gyro,
                self._accelerometer(data.qvel, quat, info["last_lin_vel"]) * c.scale_accel,
                data.qpos[7:19] - self._default_pose,
                data.qvel[6:18] * c.scale_dof_vel,
                info["last_action"],
                info["command"],
            ]),
            -c.obs_clip, c.obs_clip)
        noise = jax.random.normal(info["noise_rng"], proprio.shape) * self._noise_scale
        noisy = jp.clip(proprio + noise, -c.obs_clip, c.obs_clip)
        privileged = jp.concatenate([jp.clip(data.qvel[:3], -c.obs_clip, c.obs_clip), noisy])
        return noisy, privileged

    # ---------------------------------------------------------------- reset
    def reset(self, rng: jax.Array) -> State:
        cfg = self.cfg
        rng, k_cmd, k_stair, k_qpos, k_vel, k_lat, k_push, k_noise, k_arm, k_sw = jax.random.split(rng, 10)
        level = jp.int32(cfg.level_init)
        stair = self._sample_stair(k_stair, level)

        yaw = stair[2]
        k_j, k_xy = jax.random.split(k_qpos)
        x = self.spec.approach - cfg.spawn_back * jp.cos(yaw)
        y = -cfg.spawn_back * jp.sin(yaw)
        jitter = jax.random.uniform(k_xy, (2,), minval=-0.05, maxval=0.05)
        arm_goal = self._sample_arm_goal(k_arm)
        qpos = self._init_q.at[0].set(x + jitter[0]).at[1].set(y + jitter[1])
        qpos = qpos.at[7:19].add(jax.random.uniform(k_j, (N_LEG,), minval=-0.1, maxval=0.1))
        qpos = qpos.at[19:25].set(arm_goal[:6])
        qpos = qpos.at[25].set(arm_goal[6]).at[26].set(arm_goal[6])
        qvel = jp.zeros(self.mj_model.nv).at[:6].set(
            jax.random.uniform(k_vel, (6,), minval=-0.05, maxval=0.05))
        pos, quat = box_poses(self.spec, stair[0], stair[1], stair[2])
        ctrl = jp.concatenate([self._default_ctrl, self._arm_ctrl(arm_goal)])
        data = mjx.make_data(self.mjx_model).replace(qpos=qpos, qvel=qvel, ctrl=ctrl,
                                                     mocap_pos=pos, mocap_quat=quat)
        data = mjx.kinematics(self.mjx_model, data)
        lift = jp.clip(self._foot_radius + 0.005 - jp.min(data.site_xpos[self._foot_sites, 2]), min=0.0)
        qpos = qpos.at[2].add(lift)
        data = mjx.kinematics(self.mjx_model, data.replace(qpos=qpos))

        s0 = self._along(qpos[:2], {"stair": stair})
        info = {
            "rng": rng, "noise_rng": k_noise,
            "command": self._sample_command(k_cmd),
            "last_action": jp.zeros(N_LEG), "last_joint_vel": jp.zeros(N_LEG),
            "last_lin_vel": qvel[:3],
            "feet_air_time": jp.zeros(4), "peak_clear": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "step": jp.zeros((), dtype=jp.int32),
            "obs_history": jp.zeros(self._obs_size), "priv_history": jp.zeros(self._priv_size),
            "stair": stair, "level": level, "success_ema": jp.float32(cfg.success_ema_init),
            "last_s": s0, "max_s": s0, "last_z": qpos[2],
            "last_foot_xy": data.site_xpos[self._foot_sites, :2],
            "latency": jax.random.uniform(k_lat, (), maxval=cfg.latency_max) * (1.0 if cfg.dr_enable else 0.0),
            "push_step": jax.random.randint(k_push, (), self._push_lo, self._push_hi),
            "arm_goal": arm_goal, "arm_cmd": arm_goal,
            "arm_switch": jax.random.randint(k_sw, (), self._sw_lo, self._sw_hi),
        }
        obs, hist, priv = self._stacked_obs(data, info)
        info["obs_history"], info["priv_history"] = hist, priv
        return State(pipeline_state=data, obs=obs, reward=jp.zeros(()), done=jp.zeros(()),
                     metrics={k: jp.zeros(()) for k in self._metric_keys()}, info=info)

    @staticmethod
    def _metric_keys():
        return Go2StairsEnv._metric_keys() + ("cost_arm_stab", "arm_ext")

    # ----------------------------------------------------------------- step
    def step(self, state: State, action: jax.Array) -> State:
        cfg = self.cfg
        info = dict(state.info)
        rng, k_push, k_interval, k_stair, k_noise, k_arm, k_sw, k_c, k_cr = jax.random.split(info["rng"], 9)
        info["rng"], info["noise_rng"] = rng, k_noise
        resample = jax.random.uniform(k_cr, ()) < self._dt / cfg.cmd_resample_s
        info["command"] = jp.where(resample, self._sample_command(k_c), info["command"])

        action = jp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        applied = (1.0 - info["latency"]) * action + info["latency"] * info["last_action"]
        target = jp.clip(self._default_ctrl + cfg.action_scale * applied,
                         self._ctrl_range[:, 0], self._ctrl_range[:, 1])

        # Arm command stream: new goal on schedule, slewed at arm_speed.
        switch = (info["step"] >= info["arm_switch"]) & cfg.randomize_arm
        info["arm_goal"] = jp.where(switch, self._sample_arm_goal(k_arm), info["arm_goal"])
        info["arm_switch"] = jp.where(
            switch, info["step"] + jax.random.randint(k_sw, (), self._sw_lo, self._sw_hi), info["arm_switch"])
        max_d = cfg.arm_speed * self._dt
        arm_cmd = info["arm_cmd"] + jp.clip(info["arm_goal"] - info["arm_cmd"], -max_d, max_d)
        arm_cmd = jp.clip(arm_cmd, self._arm_ctrl_lo, self._arm_ctrl_hi)
        info["arm_cmd"] = arm_cmd
        ctrl = jp.concatenate([target, self._arm_ctrl(arm_cmd)])

        rise, run, yaw = self._stair(info)
        pos, quat = box_poses(self.spec, rise, run, yaw)
        data0 = state.pipeline_state.replace(mocap_pos=pos, mocap_quat=quat)
        push_now = (info["step"] >= info["push_step"]) & cfg.dr_enable
        kick = jax.random.uniform(k_push, (2,), minval=-cfg.push_vel, maxval=cfg.push_vel)
        data0 = data0.replace(qvel=data0.qvel.at[:2].add(jp.where(push_now, kick, jp.zeros(2))))
        info["push_step"] = jp.where(
            push_now, info["step"] + jax.random.randint(k_interval, (), self._push_lo, self._push_hi),
            info["push_step"])

        def phys(d, _):
            return mjx.step(self.mjx_model, d.replace(ctrl=ctrl)), None

        data, _ = jax.lax.scan(phys, data0, None, length=cfg.n_frames)

        qpos, qvel = data.qpos, data.qvel
        quat_b = qpos[3:7]
        leg_q, joint_vel = qpos[7:19], qvel[6:18]
        proj_grav = self._projected_gravity(quat_b)
        command = info["command"]

        foot_xy = data.site_xpos[self._foot_sites, :2]
        foot_z = data.site_xpos[self._foot_sites, 2]
        foot_ground = jax.vmap(lambda p: self._ground(p, info))(foot_xy)
        clearance = foot_z - foot_ground - self._foot_radius
        foot_vel = jp.linalg.norm((foot_xy - info["last_foot_xy"]) / self._dt, axis=-1)
        rel_h = qpos[2] - jp.mean(foot_ground)
        h_g = self._ground(qpos[:2], info)
        s, _ = stair_frame(self.spec, qpos[:2], yaw)
        s = jp.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        max_s = jp.maximum(jp.nan_to_num(info["max_s"], nan=0.0), s)

        contact = self._feet_contact(data)
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0.0) & contact_filt
        air_time = info["feet_air_time"] + self._dt
        moving = jp.linalg.norm(command[:2]) > cfg.cmd_deadband
        walking = (command[0] > cfg.progress_min_vx).astype(jp.float32)
        air_time_rew = jp.sum(jp.clip(air_time - cfg.air_time_threshold, min=0.0) * first_contact) * moving

        v_b = self._rot_inv(quat_b, qvel[:3])
        w_b = self._rot_inv(quat_b, qvel[3:6])
        lin_err = jp.sum(jp.square(command[:2] - v_b[:2]))
        ang_err = jp.square(command[2] - w_b[2])
        rew_lin = cfg.w_lin_vel * jp.exp(-lin_err / cfg.tracking_sigma)
        rew_ang = cfg.w_ang_vel * jp.exp(-ang_err / cfg.tracking_sigma)
        rew_air = cfg.w_air_time * air_time_rew
        ds = (s - info["last_s"]) / self._dt
        rew_progress = cfg.w_progress * jp.clip(ds, -cfg.progress_cap, cfg.progress_cap)
        dz = (qpos[2] - info["last_z"]) / self._dt
        rew_height = cfg.w_height_gain * jp.clip(dz, -cfg.height_gain_cap, cfg.height_gain_cap)
        # Height gain would pay the climb and charge the step DOWN equally; on a platform
        # that nets to zero, so only the up-step earns (descent is not penalized).
        rew_height = jp.maximum(rew_height, 0.0) * walking
        rew_progress = rew_progress * walking

        cost_orient = cfg.w_orientation * jp.square(proj_grav[1])
        cost_height = cfg.w_base_height * jp.square(rel_h - cfg.target_height)
        cost_torque = cfg.w_torque * jp.sum(jp.square(data.actuator_force[:N_LEG]))
        act_delta = action - info["last_action"]
        cost_arate = cfg.w_action_rate * jp.sum(jp.square(act_delta))
        cost_jvel = cfg.w_joint_vel * jp.sum(jp.square(joint_vel))
        joint_acc = (joint_vel - info["last_joint_vel"]) / self._dt
        cost_jacc = cfg.w_joint_acc * jp.sum(jp.square(joint_acc))
        cost_angdamp = cfg.w_ang_damp * jp.sum(jp.square(w_b[:2]))
        out_of_range = jp.clip(self._soft_low - leg_q, min=0.0) + jp.clip(leg_q - self._soft_high, min=0.0)
        cost_jlimit = cfg.w_joint_limit * jp.sum(out_of_range)
        cost_coll = cfg.w_collision * self._collision_count(data).astype(jp.float32)

        arm_ext = self._arm_extension(data)
        cost_arm = cfg.w_arm_stab * arm_ext * (jp.sum(jp.square(w_b[:2])) + 4.0 * jp.sum(jp.square(proj_grav[:2])))

        target_clear = rise + cfg.clearance_margin
        peak_landed = info["peak_clear"]
        rew_clear = cfg.w_clearance * jp.sum(jp.clip(peak_landed / target_clear, 0.0, 1.0) * first_contact) * moving * walking
        cost_slip = cfg.w_slip * jp.sum(jp.square(foot_vel) * contact)

        bad_height = (rel_h < cfg.min_rel_height) | (rel_h > cfg.max_rel_height)
        pitch = jp.arctan2(-proj_grav[0], -proj_grav[2])
        roll = jp.arctan2(proj_grav[1], -proj_grav[2])
        bad_att = (jp.abs(pitch) > cfg.max_pitch) | (jp.abs(roll) > cfg.max_roll)
        fatal_contact = self._contact_count(data, self._fatal_geoms) > 0
        backward = s < (info["max_s"] - cfg.back_margin)
        nan = ~(jp.isfinite(qpos).all() & jp.isfinite(qvel).all())
        diverged = jp.max(jp.abs(qvel)) > cfg.max_qvel
        blew_up = nan | diverged
        done = (bad_height | bad_att | fatal_contact | backward | blew_up).astype(jp.float32)
        cost_term = cfg.w_termination * done

        reward = (rew_lin + rew_ang + rew_air + rew_progress + rew_height + rew_clear
                  - cost_orient - cost_height - cost_torque - cost_arate - cost_jvel - cost_jacc
                  - cost_jlimit - cost_coll - cost_slip - cost_term - cost_angdamp - cost_arm)
        reward = jp.where(nan, 0.0, reward)

        step_next = info["step"] + 1
        truncating = step_next >= cfg.episode_length
        ending = (done > 0.0) | truncating
        summit = self._summit_s(info)
        success = (max_s >= summit - cfg.success_margin).astype(jp.float32)
        climb_frac = jp.clip(max_s / summit, 0.0, 1.0)
        counts = ending & (rise > 1e-3)
        ema = jp.where(jp.isfinite(info["success_ema"]), info["success_ema"], cfg.success_ema_init)
        ema_next = ema + cfg.success_ema_alpha * (climb_frac - ema)
        ema_next = jp.where(jp.isfinite(ema_next), ema_next, cfg.success_ema_init)
        ema_next = jp.where(counts, ema_next, ema)
        level = info["level"]
        promote = counts & (ema_next > cfg.promote_success)
        demote = counts & (ema_next < cfg.demote_success)
        level_next = jp.clip(level + promote.astype(jp.int32) - demote.astype(jp.int32),
                             cfg.level_min, NUM_LEVELS - 1)
        info["success_ema"] = jp.where(level_next != level, cfg.success_ema_init, ema_next)
        info["level"] = level_next
        info["stair"] = jp.where(ending, self._sample_stair(k_stair, level_next), info["stair"])

        info["last_action"] = action
        info["last_joint_vel"] = joint_vel
        new_last_lin_vel = qvel[:3]
        info["feet_air_time"] = air_time * ~contact_filt
        info["peak_clear"] = jp.maximum(info["peak_clear"], clearance) * ~contact_filt
        info["last_contact"] = contact
        info["step"] = step_next
        info["last_s"], info["max_s"], info["last_z"] = s, max_s, qpos[2]
        info["last_foot_xy"] = foot_xy
        obs, hist, priv = self._stacked_obs(data, info)
        info["obs_history"], info["priv_history"] = hist, priv
        info["last_lin_vel"] = new_last_lin_vel

        metrics = dict(state.metrics)
        metrics.update({
            "rew_lin_vel": rew_lin, "rew_ang_vel": rew_ang, "rew_air_time": rew_air,
            "rew_progress": rew_progress, "rew_height_gain": rew_height, "rew_clearance": rew_clear,
            "peak_clearance": jp.sum(peak_landed * first_contact) / jp.maximum(jp.sum(first_contact), 1.0),
            "cost_orientation": cost_orient, "cost_base_height": cost_height,
            "cost_torque": cost_torque, "cost_action_rate": cost_arate,
            "cost_joint_vel": cost_jvel, "cost_joint_acc": cost_jacc,
            "cost_joint_limit": cost_jlimit, "cost_collision": cost_coll, "cost_slip": cost_slip,
            "cost_termination": cost_term, "cost_ang_damp": cost_angdamp,
            "cost_arm_stab": cost_arm, "arm_ext": arm_ext,
            "ang_rate": jp.linalg.norm(w_b[:2]), "act_rate_raw": jp.linalg.norm(act_delta),
            "lin_vel_error": jp.sqrt(lin_err), "ang_vel_error": jp.sqrt(ang_err),
            "rel_height": rel_h, "total_reward": reward, "nan_steps": nan.astype(jp.float32),
            "level": level_next.astype(jp.float32),
            "success": jp.where(ending, success, 0.0),
            "climb_frac": jp.where(ending, climb_frac, 0.0),
            "max_s": max_s, "terrain_height": h_g,
            "term_diverged": blew_up.astype(jp.float32),
            "term_height": (bad_height & ~blew_up).astype(jp.float32),
            "term_attitude": (bad_att & ~bad_height & ~blew_up).astype(jp.float32),
            "term_contact": (fatal_contact & ~bad_height & ~bad_att & ~blew_up).astype(jp.float32),
            "term_backward": (backward & ~bad_height & ~bad_att & ~fatal_contact & ~blew_up).astype(jp.float32),
            "truncated": jp.where(truncating & (done == 0.0), 1.0, 0.0),
        })
        for i in range(NUM_LEVELS):
            metrics[f"lvl{i}"] = (level_next == i).astype(jp.float32)
        metrics = {k: v if k == "nan_steps" else jp.nan_to_num(v, nan=0.0) for k, v in metrics.items()}
        return State(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)
