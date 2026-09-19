"""Does a harder squeeze stop the residual creep? Grasp, hold the arm still HOLD_S, measure
the tool's slide in the GRIPPER frame, per finger-servo gain (squeeze = kp * overshoot)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp
from bw.sim.workshop import TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim

HOLD_S = 3.0
TOOLS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["pliers", "tape_roll", "wrench_13mm"]
SEEDS = range(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
sim = WorkshopSim()
ik = ArmIK(sim.m)
fa = [sim.m.actuator(f"arm_motorGripper_{t}").id for t in "ab"]


def set_kp(kp):
    for i in fa:
        sim.m.actuator_gainprm[i, 0] = kp
        sim.m.actuator_biasprm[i, 1] = -kp


def rel(tool):
    R = sim.d.site_xmat[sim.ee_site].reshape(3, 3)
    return R.T @ (sim.gt_tool_pos(tool) - sim.ee_pos())


for kp in (1200, 2500, 4000):
    set_kp(kp)
    for tool in TOOLS:
        creep, ok, force = [], 0, []
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * seed + TOOL_NAMES.index(tool))
            sim.reset(rng, target=tool, arm_q=SCAN_Q)
            g = run_grasp(sim, ik, tool, rng)
            if sim.held_tool() != tool:
                continue
            ok += g["success"]
            force.append(float(np.abs(sim.d.actuator_force[fa]).mean()))
            r0 = rel(tool)
            sim.settle(HOLD_S)
            creep.append(1000 * np.linalg.norm(rel(tool) - r0) / HOLD_S)
        print(f"kp {kp:5d} {tool:12s} grasp {ok}/{len(SEEDS)}  squeeze {np.mean(force) if force else 0:5.1f} N  "
              f"creep median {np.median(creep) if creep else float('nan'):5.2f} mm/s  max {max(creep) if creep else float('nan'):5.2f}",
              flush=True)
