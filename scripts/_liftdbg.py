import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, mujoco
from PIL import Image
from bw.manip.ik import ArmIK
import bw.manip.scripted_grasp as sg
from bw.sim.workshop_sim import WorkshopSim
tool = sys.argv[1] if len(sys.argv) > 1 else "wrench_13mm"
sim = WorkshopSim(); ik = ArmIK(sim.m)
for seed in range(10):
    rng = np.random.default_rng(seed); sim.reset(rng, target=tool, arm_q=sg.SCAN_Q)
    plan = sg.plan_grasp(sim, ik, tool, rng)
    if plan: break
print("seed", seed, "site", np.round(plan["site"],3), "gp", np.round(sg.grasp_point(sim, tool),3), "tool", np.round(sim.gt_tool_pos(tool),3))
r = mujoco.Renderer(sim.m, 300, 400); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
shots = []
def snap(tag):
    cam.lookat[:] = sim.ee_pos(); cam.distance = 0.4; cam.azimuth = 150; cam.elevation = -15
    r.update_scene(sim.d, camera=cam); shots.append(r.render())
    print(tag, "ee", np.round(sim.ee_pos(),3), "tool", np.round(sim.gt_tool_pos(tool),3), "grip", round(sim.d.qpos[25],3), "pads", sorted(sg._pad_contacts(sim, tool)))
op = -0.6
sim.move_arm(np.concatenate([plan["q_pre"], [op]]), 2.0); snap("pre")
q = sg.cartesian(sim, ik, plan["q_pre"], plan["site"] - sg.PRE*plan["a"], plan["site"], plan["Rg"], op, 1.0, None)
sim.move_arm(np.concatenate([q, [op]]), 0.4); snap("at grasp")
sim.move_arm(np.concatenate([q, [0.0]]), 0.8); sim.settle(0.3); snap("closed")
for k in range(1, 5):
    q = sg.cartesian(sim, ik, q, plan["site"] + [0,0,0.03*(k-1)], plan["site"] + [0,0,0.03*k], plan["Rg"], 0.0, 0.4, None)
    snap(f"lift {0.03*k:.2f}")
Image.fromarray(np.concatenate([np.concatenate(shots[:4],1), np.concatenate(shots[4:8],1)],0)).save(ROOT/"media/lift_dbg.png")
