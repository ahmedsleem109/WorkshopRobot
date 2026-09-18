"""Isolate the creep: grasp, then hold the arm PERFECTLY STILL for 3 s and measure how fast
the tool slides through the jaws.

With the arm static the only forces on the tool are gravity (0.45 N for wrench_10mm) and the
pads' 25 N squeeze. Coulomb friction at mu=2.0 gives ~50 N of holding capacity, so a static
hold must show ZERO creep. Whatever makes it non-zero is the bug, and each variant below
isolates one candidate:

  gravity_off   creep driven by the tool's weight (a real friction failure) vs by the squeeze
  friction_x10  creep scales with mu  -> Coulomb-cone slip;  unchanged -> geometric extrusion
  dt_half       creep scales with the timestep -> solver integration drift
  soft_pads     the stiff pad solref (0.002, = 1 timestep) is the destabiliser
  condim3       torsional/rolling friction dimensions are the ones drifting
  cone_pyram    elliptic-cone conditioning
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp
from bw.sim.workshop import TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim

TOOL = "wrench_10mm"
HOLD_S = 3.0


def pads(m):
    return [m.geom("finger_a_pad").id, m.geom("finger_b_pad").id]


def variants(m):
    g0 = m.opt.gravity.copy()
    dt0 = m.opt.timestep
    fr0 = m.geom_friction[pads(m)].copy()
    sr0 = m.geom_solref[pads(m)].copy()
    cd0 = m.geom_condim[pads(m)].copy()
    cone0 = m.opt.cone

    def reset_all():
        m.opt.gravity[:] = g0
        m.opt.timestep = dt0
        m.geom_friction[pads(m)] = fr0
        m.geom_solref[pads(m)] = sr0
        m.geom_condim[pads(m)] = cd0
        m.opt.cone = cone0

    def base():
        reset_all()

    def gravity_off():
        reset_all()
        m.opt.gravity[:] = 0

    def friction_x10():
        reset_all()
        m.geom_friction[pads(m), 0] = 20.0

    def friction_x01():
        reset_all()
        m.geom_friction[pads(m), 0] = 0.2

    def dt_half():
        reset_all()
        m.opt.timestep = dt0 / 2

    def soft_pads():
        reset_all()
        m.geom_solref[pads(m)] = [0.02, 1.0]

    def condim3():
        reset_all()
        m.geom_condim[pads(m)] = 3

    def cone_pyram():
        reset_all()
        m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

    return [base, gravity_off, friction_x10, friction_x01, dt_half, soft_pads, condim3,
            cone_pyram]


def rel(sim, tb):
    p_ee = sim.d.site_xpos[sim.ee_site]
    R_ee = sim.d.site_xmat[sim.ee_site].reshape(3, 3)
    return R_ee.T @ (sim.d.xpos[tb] - p_ee)


def main():
    sim = WorkshopSim()
    ik = ArmIK(sim.m)
    tb = sim.tool_body[TOOL]
    ti = TOOL_NAMES.index(TOOL)
    print(f"{'variant':14s} {'grasped':>8} {'creep mm/s':>11} {'drift mm':>9} {'padF N':>8} "
          f"{'held':>6}")
    for v in variants(sim.m):
        v()
        rng = np.random.default_rng(1000 * 0 + ti)
        sim.reset(rng, target=TOOL, arm_q=SCAN_Q)
        r = run_grasp(sim, ik, TOOL, rng)
        p0 = rel(sim, tb)
        sim.settle(HOLD_S)               # arm command unchanged: a perfectly static hold
        p1 = rel(sim, tb)
        drift = 1000.0 * float(np.linalg.norm(p1 - p0))
        f6 = np.zeros(6)
        fa = 0.0
        for i in range(sim.d.ncon):
            c = sim.d.contact[i]
            b = (sim.m.geom_bodyid[c.geom1], sim.m.geom_bodyid[c.geom2])
            if tb in b and (sim.m.geom_bodyid[c.geom1] in sim.finger_bodies
                            or sim.m.geom_bodyid[c.geom2] in sim.finger_bodies):
                mujoco.mj_contactForce(sim.m, sim.d, i, f6)
                fa = max(fa, abs(float(f6[0])))
        print(f"{v.__name__:14s} {str(r['success']):>8} {drift / HOLD_S:11.2f} {drift:9.2f} "
              f"{fa:8.1f} {str(sim.held_tool()):>6}")


if __name__ == "__main__":
    main()
