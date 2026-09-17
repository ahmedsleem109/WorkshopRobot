import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import mujoco, numpy as np
from bw.sim.workshop_sim import WorkshopSim
sim = WorkshopSim()
sim.reset(np.random.default_rng(0), base_pose=(4.05, 0, 0), arm_q=np.array([0, 0.9, -1.2, 0.3, 0, 0, -1.2]))
m, d = sim.m, sim.d
for tgt in (-1.2, -0.6, 0.0):
    q = sim.arm_q(); q[6] = tgt; sim.set_arm_target(q); sim.settle(1.0)
    print("grip target", tgt, "q", round(d.qpos[25], 3), "act force", round(d.actuator_force[m.actuator("arm_motorGripper").id], 2))
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        names = (m.body(b1).name, m.body(b2).name)
        if any("arm" in n for n in names):
            print("   contact", names, m.geom(c.geom1).name, m.geom(c.geom2).name, round(c.dist, 4))
print("gripper joint range", m.jnt_range[m.joint("arm_jointGripper").id], "ctrlrange", m.actuator_ctrlrange[m.actuator("arm_motorGripper").id])
print("gain", m.actuator_gainprm[m.actuator("arm_motorGripper").id][:3], "bias", m.actuator_biasprm[m.actuator("arm_motorGripper").id][:3], "forcerange", m.actuator_forcerange[m.actuator("arm_motorGripper").id])
