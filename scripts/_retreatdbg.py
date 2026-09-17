import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, mujoco
from PIL import Image
from bw.manip.ik import ArmIK
import bw.manip.scripted_grasp as sg
from bw.sim.workshop_sim import WorkshopSim, GRIPPER_CLOSED
tool = sys.argv[1] if len(sys.argv) > 1 else "wrench_10mm"
sim = WorkshopSim(); ik = ArmIK(sim.m)
rng = np.random.default_rng(0); sim.reset(rng, target=tool, arm_q=sg.SCAN_Q)
plan = sg.plan_grasp(sim, ik, tool, rng)
r = mujoco.Renderer(sim.m, 320, 420); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
shots = []
def snap(tag):
    cam.lookat[:] = sim.ee_pos(); cam.distance = 0.5; cam.azimuth = 160; cam.elevation = -12
    r.update_scene(sim.d, camera=cam); shots.append(r.render())
    print(f"{tag:12s} ee {np.round(sim.ee_pos(),3)} tool {np.round(sim.gt_tool_pos(tool),3)} fingers {np.round(sim.arm_q()[6],4)} pads {sorted(sg._pad_contacts(sim, tool))}")
site, Rg, a = plan["site"], plan["Rg"], plan["a"]
op = 0.02
sim.move_arm(np.concatenate([plan["q_pre"], [op]]), 2.2); snap("pre")
q = sg.cartesian(sim, ik, plan["q_pre"], site - sg.PRE*a, site, Rg, op, 1.0, None); sim.move_arm(np.concatenate([q,[op]]), 0.4); snap("at grasp")
sim.move_arm(np.concatenate([q, [GRIPPER_CLOSED]]), 0.8); sim.settle(0.3); snap("closed")
up = site + np.array([0,0,sg.LIFT])
q = sg.cartesian(sim, ik, q, site, up, Rg, GRIPPER_CLOSED, 1.2, None); snap("lifted")
for k in (1, 2, 3):
    tgt = up - 0.073*k*np.array([a[0],a[1],0.0])
    q = sg.cartesian(sim, ik, q, up - 0.073*(k-1)*np.array([a[0],a[1],0.0]), tgt, Rg, GRIPPER_CLOSED, 0.6, None)
    snap(f"retreat {0.073*k:.2f}")
Image.fromarray(np.concatenate([np.concatenate(shots[:4],1), np.concatenate(shots[4:7]+[shots[-1]],1)],0)).save(ROOT/"media/retreat_dbg.png")
