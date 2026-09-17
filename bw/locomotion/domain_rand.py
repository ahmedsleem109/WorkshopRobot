"""Model-field domain randomization for the Go2+Z1 (per-env, via brax's DR vmap wrapper).

Differs from go2-stairs' envs/domain_rand.py in three ways:
* PD-gain scaling touches the 12 LEG actuators only -- scaling the Z1's kp=1000 servos
  would randomize the arm command tracking, which is not a locomotion disturbance.
* The trunk payload range drops from 0-3 kg to 0-0.5 kg: the arm is now the payload.
* Carry-load robustness (plan Phase 1 stretch): 0-1 kg added to the gripper finger body,
  i.e. a held object whose mass the policy never observes.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco

FRICTION_RANGE = (0.4, 1.2)
BASE_MASS_FRAC = 0.15
TRUNK_PAYLOAD = (0.0, 0.5)
HELD_MASS = (0.0, 1.0)
GAIN_FRAC = 0.25
N_LEG = 12


def _foot_geoms(m):
    out = []
    for leg in ("FL", "FR", "RL", "RR"):
        calf = m.body(f"{leg}_calf").id
        found = [g for g in range(m.ngeom) if m.geom_bodyid[g] == calf and m.geom_condim[g] == 6]
        assert len(found) == 1
        out.append(found[0])
    return out


def make_randomization_fn(mj_model: mujoco.MjModel, held_mass=HELD_MASS):
    feet = jp.array(_foot_geoms(mj_model))
    base = mj_model.body("base").id
    hand = mj_model.body("arm_gripperMover").id

    def randomize(sys, rng):
        @jax.vmap
        def per_env(key):
            k_f, k_m, k_p, k_h, k_kp, k_kd = jax.random.split(key, 6)
            fric = jax.random.uniform(k_f, (), minval=FRICTION_RANGE[0], maxval=FRICTION_RANGE[1])
            geom_friction = sys.geom_friction.at[feet, 0].set(fric)
            scale = jax.random.uniform(k_m, (), minval=1 - BASE_MASS_FRAC, maxval=1 + BASE_MASS_FRAC)
            payload = jax.random.uniform(k_p, (), minval=TRUNK_PAYLOAD[0], maxval=TRUNK_PAYLOAD[1])
            held = jax.random.uniform(k_h, (), minval=held_mass[0], maxval=held_mass[1])
            body_mass = sys.body_mass.at[base].set(sys.body_mass[base] * scale + payload)
            body_mass = body_mass.at[hand].add(held)
            kp = jax.random.uniform(k_kp, (), minval=1 - GAIN_FRAC, maxval=1 + GAIN_FRAC)
            kd = jax.random.uniform(k_kd, (), minval=1 - GAIN_FRAC, maxval=1 + GAIN_FRAC)
            gainprm = sys.actuator_gainprm.at[:N_LEG, 0].multiply(kp)
            biasprm = sys.actuator_biasprm.at[:N_LEG, 1].multiply(kp)
            biasprm = biasprm.at[:N_LEG, 2].multiply(kd)
            return geom_friction, body_mass, gainprm, biasprm

        geom_friction, body_mass, gainprm, biasprm = per_env(rng)
        in_axes = jax.tree.map(lambda _: None, sys)
        in_axes = in_axes.tree_replace({"geom_friction": 0, "body_mass": 0,
                                        "actuator_gainprm": 0, "actuator_biasprm": 0})
        sys = sys.tree_replace({"geom_friction": geom_friction, "body_mass": body_mass,
                                "actuator_gainprm": gainprm, "actuator_biasprm": biasprm})
        return sys, in_axes

    return randomize
