"""Candidate fixes for the pad-contact creep, scored on the full benchmark.

MECHANISM (scripts/_creep_probe.py): the jaw pads carry solref timeconst 0.002 s while the
scene runs at a 0.002 s timestep. MuJoCo requires the contact time constant to be >= 2 x
timestep; at exactly 1 x it is ill-conditioned, and the consequence is that a gripped tool
slides out of the jaws under its own weight at ~280 mm/s -- with 25 N on each pad and
mu = 2.0, i.e. nowhere near the friction cone. Halving the timestep (making the SAME solref
equal 2 dt) cuts the creep 36x; a pyramidal cone cuts it 21x; turning gravity off stops it
dead. It is not grip force, not grasp height and not squeeze depth.

Each variant is scored with a HOLD-2s criterion, not the old end-of-motion one: the old
benchmark scored the instant the retreat ended, which is exactly when a creeping tool is
still nominally between the pads. That inflated every number in STATUS.md.

    render_venv\\Scripts\\python.exe scripts\\_pad_fix_sweep.py [seeds]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp
from bw.sim.workshop import GRASP_TOOLS, TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim

HOLD_S = 2.0


def pads(m):
    return [m.geom("finger_a_pad").id, m.geom("finger_b_pad").id]


def make_variants(m):
    dt0 = float(m.opt.timestep)
    sr0 = m.geom_solref[pads(m)].copy()
    cone0 = int(m.opt.cone)

    def base():
        m.opt.timestep, m.geom_solref[pads(m)], m.opt.cone = dt0, sr0, cone0

    def apply(dt=None, tc=None, cone=None):
        base()
        if dt is not None:
            m.opt.timestep = dt
        if tc is not None:
            m.geom_solref[pads(m), 0] = tc
        if cone is not None:
            m.opt.cone = cone

    P = mujoco.mjtCone.mjCONE_PYRAMIDAL
    return [
        ("base (dt .002, tc .002, ell)", lambda: apply()),
        ("dt .001", lambda: apply(dt=0.001)),
        ("tc .004", lambda: apply(tc=0.004)),
        ("tc .006", lambda: apply(tc=0.006)),
        ("tc .004 + dt .001", lambda: apply(dt=0.001, tc=0.004)),
        ("pyramidal", lambda: apply(cone=P)),
        ("pyramidal + tc .004", lambda: apply(tc=0.004, cone=P)),
        ("pyramidal + dt .001", lambda: apply(dt=0.001, cone=P)),
    ]


def score(sim, ik, seeds):
    """Returns (n_end_ok, n_hold_ok, n) -- the old criterion and the honest one."""
    end_ok = hold_ok = n = 0
    per = {t: 0 for t in GRASP_TOOLS}
    for seed in range(seeds):
        for tool in GRASP_TOOLS:
            ti = TOOL_NAMES.index(tool)
            rng = np.random.default_rng(1000 * seed + ti)
            sim.reset(rng, target=tool, arm_q=SCAN_Q)
            z0 = sim.gt_tool_pos(tool)[2]
            r = run_grasp(sim, ik, tool, rng)
            end_ok += bool(r["success"])
            sim.settle(HOLD_S)
            still = sim.held_tool() == tool and (sim.gt_tool_pos(tool)[2] - z0) > 0.08
            hold_ok += bool(still)
            per[tool] += bool(still)
            n += 1
    return end_ok, hold_ok, n, per


def main(seeds=4):
    sim = WorkshopSim()
    ik = ArmIK(sim.m)
    print(f"{'variant':30s} {'end-of-motion':>14} {'held 2 s':>9}   per tool")
    for label, apply in make_variants(sim.m):
        apply()
        t0 = time.time()
        e, h, n, per = score(sim, ik, seeds)
        print(f"{label:30s} {e:6d}/{n:<7d} {h:4d}/{n:<4d}   {per}  ({time.time() - t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
