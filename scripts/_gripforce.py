import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, mujoco
from bw.manip.ik import ArmIK
import bw.manip.scripted_grasp as sg
from bw.sim.workshop_sim import WorkshopSim, GRIPPER_CLOSED
sim = WorkshopSim(); ik = ArmIK(sim.m); m, d = sim.m, sim.d
tool = sys.argv[1] if len(sys.argv) > 1 else "wrench_10mm"
rng = np.random.default_rng(1); sim.reset(rng, target=tool, arm_q=sg.SCAN_Q)
plan = sg.plan_grasp(sim, ik, tool, rng)
site, Rg, a = plan["site"], plan["Rg"], plan["a"]
op = min(0.038, sg.GRASP_HALF_WIDTH[tool] + 0.012)
sim.move_arm(np.concatenate([plan["q_pre"], [op]]), 2.2)
q = sg.cartesian(sim, ik, plan["q_pre"], site - sg.PRE*a, site, Rg, op, 1.0, None)
sim.move_arm(np.concatenate([q, [op]]), 0.4)
sim.move_arm(np.concatenate([q, [GRIPPER_CLOSED]]), 0.8)
tb = sim.tool_body[tool]
for t in range(6):
    sim.settle(0.15)
    f6 = np.zeros(6); tot = []
    for i in range(d.ncon):
        c = d.contact[i]
        if tb in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
            mujoco.mj_contactForce(m, d, i, f6)
            tot.append((m.geom(c.geom1).name or c.geom1, m.geom(c.geom2).name or c.geom2, round(float(f6[0]),2)))
    print(f"t={t*0.15:.2f} fingers {np.round(sim.arm_q()[6],4)} qfrc {np.round(d.qfrc_actuator[[m.joint('arm_finger_a').dofadr[0], m.joint('arm_finger_b').dofadr[0]]],1)} tool z {d.xpos[tb][2]:.3f} contacts {tot}")
