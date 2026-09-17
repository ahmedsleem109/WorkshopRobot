import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from bw.manip.ik import ArmIK
import bw.manip.scripted_grasp as sg
from bw.sim.workshop_sim import WorkshopSim, GRIPPER_CLOSED
sim = WorkshopSim(); ik = ArmIK(sim.m)
for tool in ("wrench_10mm", "screwdriver", "pliers"):
    for deeper in (0.0, 0.012, 0.022):
        res = []
        for seed in range(3):
            rng = np.random.default_rng(seed); sim.reset(rng, target=tool, arm_q=sg.SCAN_Q)
            plan = sg.plan_grasp(sim, ik, tool, rng)
            if plan is None:
                res.append("ik"); continue
            site = plan["site"] + deeper * plan["a"]; Rg, a = plan["Rg"], plan["a"]
            op = min(0.038, sg.GRASP_HALF_WIDTH[tool] + 0.012)
            z0 = sim.gt_tool_pos(tool)[2]
            sim.move_arm(np.concatenate([plan["q_pre"], [op]]), 2.2)
            q = sg.cartesian(sim, ik, plan["q_pre"], site - sg.PRE*a, site, Rg, op, 1.0, None)
            sim.move_arm(np.concatenate([q, [op]]), 0.4)
            sim.move_arm(np.concatenate([q, [GRIPPER_CLOSED]]), 0.8); sim.settle(0.4)
            tool_off = float(np.dot(sim.gt_tool_pos(tool) - sim.ee_pos(), a))
            up = site + np.array([0, 0, sg.LIFT])
            q = sg.cartesian(sim, ik, q, site, up, Rg, GRIPPER_CLOSED, 1.2, None)
            sim.settle(1.0)
            hold = sim.gt_tool_pos(tool)[2] - z0
            q = sg.cartesian(sim, ik, q, up, up - 0.22*np.array([a[0], a[1], 0.0]), Rg, GRIPPER_CLOSED, 1.6, None)
            res.append(f"off{tool_off:+.3f} hold{hold:.2f} ret{sim.gt_tool_pos(tool)[2]-z0:.2f}")
        print(f"{tool:12s} deeper {deeper:.3f}: {res}", flush=True)
