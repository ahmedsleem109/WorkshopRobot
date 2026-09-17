import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp, plan_grasp
from bw.sim.workshop_sim import WorkshopSim
tool = sys.argv[1] if len(sys.argv) > 1 else "wrench_13mm"
sim = WorkshopSim(); ik = ArmIK(sim.m)
rng = np.random.default_rng(1)
sim.reset(rng, target=tool, arm_q=SCAN_Q)
print("tool pos", np.round(sim.gt_tool_pos(tool), 3), "ee", np.round(sim.ee_pos(), 3))
frames = []
def rec(q):
    frames.append(np.concatenate([sim.render("bench_view", size=(270, 480)),
                                  np.pad(sim.render("wrist", size=(270, 270)), ((0, 0), (0, 0), (0, 0)))], 1))
    if len(frames) in (36, 40, 44):
        for i in range(sim.d.ncon):
            c = sim.d.contact[i]
            n1, n2 = sim.m.geom(c.geom1).name or f"g{c.geom1}", sim.m.geom(c.geom2).name or f"g{c.geom2}"
            b1, b2 = sim.m.body(sim.m.geom_bodyid[c.geom1]).name, sim.m.body(sim.m.geom_bodyid[c.geom2]).name
            if "arm" in b1 or "arm" in b2:
                print("   contact", b1, n1, "<->", b2, n2, round(c.dist, 4))
    if len(frames) % 5 == 0:
        print(len(frames), "ee", np.round(sim.ee_pos(), 3), "target q", np.round(q, 2).tolist(), "tool", np.round(sim.gt_tool_pos(tool), 3))
r = run_grasp(sim, ik, tool, rng, record=rec)
print(r)
g = sim.m.actuator("arm_motorGripper").id
print("gripper q", sim.d.qpos[25], "ctrl", sim.d.ctrl[g], "contacts with tool:", sim.gripper_contacts(0), "tool body", sim.tool_body[tool])
imgs = [Image.fromarray(f) for f in frames]
imgs[0].save(ROOT / f"media/grasp_{tool}.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)
for i in (20, 32, 38, 48, 60): 
    if i < len(frames): Image.fromarray(frames[i]).save(ROOT / f"media/grasp_f{i}.png")
