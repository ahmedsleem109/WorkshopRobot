"""T7 -- noise injection for the scripted demonstrator (the covariate-shift experiment).

Session 6 established, by measurement, that SmolVLA reproduces the demonstrations' state tube
(overfit test: 3.8x better than the no-motion baseline on every joint) and still cannot grasp
in closed loop (1/20). What is left is covariate shift: every demonstration came from a
scripted controller that never made a mistake, so the data holds NO recovery states, and the
policy cannot return to the tube once it leaves it. Testing that is a DATA change.

The mechanism here is DART-style: kick the ARM off the path, leave the LABEL nominal.

    servo command  =  nominal target + offset     (offset non-zero during a kick)
    recorded action = nominal target              (what the expert wants, unchanged)

`WorkshopSim.move_arm` interpolates from `self.arm_target`, which keeps holding the NOMINAL
value, so when a kick ends the controller is still driving the original path -- from wherever
the arm now is. Every frame from the kick's start until the arm is back on the path is a state
off the demonstrated tube paired with the command that corrects it, which is exactly the
recovery data the dataset lacks. `bw/manip/scripted_grasp.cartesian` additionally re-solves IK
from the live state each tick, so its corrections are genuinely closed-loop.

Two things are never perturbed:
  * the GRIPPER (index 6) -- a kick there opens the jaws and drops the tool, which produces no
    recovery data, only a discarded episode (`arm_target[6]` is also the holding gate).
  * the phases where a kick would knock the scene rather than the arm: the descent into the
    slot, the close, the lift and the settle, and the release. `SAFE_PHASES` lists where
    kicks ARE allowed -- the free-space motion, where an imperfect policy actually deviates.

Usage (the collector):

    dz = Disturber(sim, rng)
    with dz.attached():
        run_grasp(sim, ik, tool, rng, record=rec, on_phase=dz.on_phase)
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

# Free-space motion in both skills. Everything else (approach's final descent, close, lift,
# lift_settle, hold, verify, place_descend, release) is left clean on purpose.
SAFE_PHASES = frozenset({"open", "stage", "reach_pre", "retreat",
                         "place_pre", "place_align", "place_retract"})

P_KICK = 0.10            # per 10 Hz tick, in a safe phase: probability a kick STARTS
KICK_TICKS = (2, 5)      # a kick lasts uniformly this many ticks (0.2-0.5 s)
SIGMA = 0.04             # rad, per joint, of the offset (~1-3 cm at the gripper)


class Disturber:
    """Per-tick joint-space kicks applied to the servos, never to the recorded action."""

    def __init__(self, sim, rng: np.random.Generator, p: float = P_KICK,
                 ticks=KICK_TICKS, sigma: float = SIGMA, safe=SAFE_PHASES):
        self.sim, self.rng, self.p, self.ticks, self.sigma = sim, rng, p, ticks, sigma
        self.safe = safe
        self.phase = "plan"
        self.offset = np.zeros(7)
        self.left = 0                     # ticks remaining in the current kick
        self.kicks: list[dict] = []       # audit trail, stored in the episode's meta
        self._orig = None

    # ------------------------------------------------------------------ plumbing
    def attach(self):
        if self._orig is not None:
            return
        sim = self.sim
        self._orig = sim.set_arm_target

        def patched(q):
            self._orig(q)                 # nominal: arm_target and ctrl both set
            if self.left > 0:
                self._apply()

        sim.set_arm_target = patched

    def detach(self):
        if self._orig is None:
            return
        self.sim.set_arm_target = self._orig
        self._orig = None
        self.left = 0
        self.offset[:] = 0.0

    @contextmanager
    def attached(self):
        self.attach()
        try:
            yield self
        finally:
            self.detach()

    def _apply(self):
        """Re-write the actuator command with the offset. `arm_target` keeps the nominal
        value, so the controller's own interpolation is unaffected."""
        from bw.sim.workshop_sim import ARM_J_CTRL
        m, d = self.sim.m, self.sim.d
        lo, hi = m.actuator_ctrlrange[ARM_J_CTRL, 0], m.actuator_ctrlrange[ARM_J_CTRL, 1]
        d.ctrl[ARM_J_CTRL] = np.clip(self.sim.arm_target[:6] + self.offset[:6], lo, hi)

    # ------------------------------------------------------------------ schedule
    def on_phase(self, label: str):
        self.phase = label
        if label not in self.safe:        # a kick never survives into an unsafe phase
            self.left = 0
            self.offset[:] = 0.0

    def tick(self, t: float | None = None):
        """Call once per recorded (10 Hz) frame, BEFORE the tick is executed."""
        if self.left > 0:
            self.left -= 1
            if self.left == 0:
                self.offset[:] = 0.0      # release: the servos pull back to the nominal path
            return
        if self.phase not in self.safe or self.rng.random() >= self.p:
            return
        self.offset[:] = 0.0
        self.offset[:6] = self.rng.normal(0.0, self.sigma, 6)
        self.left = int(self.rng.integers(self.ticks[0], self.ticks[1] + 1))
        self.kicks.append({"phase": self.phase, "ticks": self.left,
                           "offset": [round(float(x), 4) for x in self.offset[:6]]})

    def summary(self) -> dict:
        return {"n_kicks": len(self.kicks), "kicks": self.kicks}
