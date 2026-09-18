"""The residual T1 failures: what touches the target tool BEFORE the jaws close?

Three of the eight failures over 100 episodes have `tool_shift` of 5-19 cm -- the tool has
already been knocked out of position by the time the gripper arrives, so the grasp was never
going to work. This replays exactly those episodes and prints, per phase, the first contact
between the target tool and anything that is not the rack, plus how far the tool has moved.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp
from bw.sim.workshop import TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim

CASES = [(17, "wrench_10mm"), (18, "wrench_10mm"), (21, "wrench_13mm"),
         (11, "tape_roll"), (17, "tape_roll"), (21, "tape_roll"),
         (10, "pliers"), (20, "wrench_10mm")]


class Watch:
    def __init__(self, sim, name):
        self.sim, self.name = sim, name
        self.tb = sim.tool_body[name]
        self.p0 = sim.gt_tool_pos(name).copy()
        self.phase = "init"
        self.events = []          # (phase, geom name, shift mm) -- first hit per (phase, geom)
        self.seen = set()
        self._orig = sim.physics_step
        sim.physics_step = self._stepped

    def close(self):
        self.sim.physics_step = self._orig

    def set_phase(self, label):
        self.phase = label

    def _stepped(self, n=1):
        left = n
        while left > 0:
            k = min(10, left)
            self._orig(k)
            left -= k
            self._check()

    def _check(self):
        sim, d, m = self.sim, self.sim.d, self.sim.m
        shift = 1000 * float(np.linalg.norm(d.xpos[self.tb] - self.p0))
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self.tb not in (b1, b2):
                continue
            g = c.geom2 if b1 == self.tb else c.geom1
            gname = m.geom(g).name or f"geom{g}"
            if gname.startswith(("rack", "bench")):
                continue
            key = (self.phase, gname)
            if key not in self.seen:
                self.seen.add(key)
                self.events.append((self.phase, gname, round(shift, 1)))


def main():
    sim = WorkshopSim()
    ik = ArmIK(sim.m)
    for seed, tool in CASES:
        ti = TOOL_NAMES.index(tool)
        rng = np.random.default_rng(1000 * seed + ti)
        sim.reset(rng, target=tool, arm_q=SCAN_Q)
        w = Watch(sim, tool)
        try:
            r = run_grasp(sim, ik, tool, rng, on_phase=w.set_phase)
        finally:
            w.close()
        moved = 1000 * float(np.linalg.norm(sim.gt_tool_pos(tool) - w.p0))
        print(f"seed {seed:2d} {tool:12s} success={r['success']!s:5s} moved {moved:6.1f} mm")
        for phase, gname, shift in w.events:
            print(f"      {phase:12s} touched {gname:24s} (tool had moved {shift:6.1f} mm)")


if __name__ == "__main__":
    main()
