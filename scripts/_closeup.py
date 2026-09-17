import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import mujoco, numpy as np
from PIL import Image
from bw.manip.ik import ArmIK
import bw.manip.scripted_grasp as sg
from bw.sim.workshop_sim import WorkshopSim
tool = sys.argv[1] if len(sys.argv) > 1 else "wrench_13mm"
sim = WorkshopSim(); ik = ArmIK(sim.m); rng = np.random.default_rng(1)
sim.reset(rng, target=tool, arm_q=sg.SCAN_Q)
m, d = sim.m, sim.d
r = mujoco.Renderer(m, 360, 480)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
shots = []
def snap(tag):
    cam.lookat[:] = sim.gt_tool_pos(tool); cam.distance = 0.35; cam.azimuth = 200; cam.elevation = -25
    r.update_scene(d, camera=cam); a = r.render()
    cam.azimuth = 110; r.update_scene(d, camera=cam); b = r.render()
    shots.append(np.concatenate([a, b], 1))
    pads = {m.geom(g).name or g: np.round(d.geom_xpos[g], 3) for g in range(m.ngeom) if m.geom_bodyid[g] in (sim.mover_body, sim.stator_body) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX}
    print(tag, "q err", np.round(sim.arm_target[:6] - sim.arm_q()[:6], 3), "target", np.round(sim.arm_target[:6], 3))
    print("   qfrc_act", np.round(d.qfrc_actuator[18:24], 2), "constraint", np.round(d.qfrc_constraint[18:24], 2), "bias", np.round(d.qfrc_bias[18:24], 2), "passive", np.round(d.qfrc_passive[18:24], 2), "qvel", np.round(d.qvel[18:24], 3))
    fk = ik.solve(d, plan["site"], plan["Rg"], q_init=sim.arm_target[:6], iters=0)
    print("   site err of target q (FK):", round(ik.solve(d, plan["site"], plan["Rg"], q_init=sim.arm_target[:6], iters=1)[1], 4))
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
        if "arm" in b1 or "arm" in b2: print("   contact", b1, m.geom(c.geom1).name or c.geom1, "<->", b2, m.geom(c.geom2).name or c.geom2, round(c.dist, 4))
    print(tag, "ee", np.round(sim.ee_pos(), 3), "tool", np.round(sim.gt_tool_pos(tool), 3), "grip q", round(d.qpos[25], 3), "pads", pads)
n = [0]
orig_move = sim.move_arm
def rec(q):
    n[0] += 1
plan = None
for seed in range(20):
    rng = np.random.default_rng(seed); sim.reset(rng, target=tool, arm_q=sg.SCAN_Q)
    plan = sg.plan_grasp(sim, ik, tool, rng)
    if plan is not None:
        break
print("planned site", np.round(plan["site"], 3), "a", np.round(plan["a"], 2), "jaw", np.round(plan["Rg"][:, 2], 2))
op = -1.2
sim.move_arm(np.concatenate([plan["q_hover"], [op]]), 2.0)
hover = plan["site"] + [0, 0, sg.HOVER]; pre = plan["site"] - sg.PRE * plan["a"]
q = sg.cartesian(sim, ik, plan["q_hover"], hover, pre, plan["Rg"], op, 1.0, None)
q = sg.cartesian(sim, ik, q, pre, plan["site"], plan["Rg"], op, 1.0, None)
sim.settle(0.3); snap("at grasp (open)")
sim.move_arm(np.concatenate([q, [0.15]]), 0.8); sim.settle(0.3); snap("closed")
q = sg.cartesian(sim, ik, q, plan["site"], plan["site"] + np.array([0, 0, 0.15]), plan["Rg"], 0.15, 1.0, None)
sim.settle(0.3); snap("lifted")
print("held", sim.held_tool(), "gripper contacts", sim.gripper_contacts(0), "tool body", sim.tool_body[tool])
Image.fromarray(np.concatenate(shots, 0)).save(ROOT / "media/closeup.png")
