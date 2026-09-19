"""Render one full two-table transfer with its language command and live action readout.

    render_venv\\Scripts\\python.exe scripts\\make_transfer_video.py [tool] [table] [seed]

What is on screen, and what it is NOT: no VLA has been trained yet (T6 -> T7). The command
is the instruction this episode is LABELLED with (bw/task/language.py), and the action is the
SCRIPTED DEMONSTRATOR's 10 Hz arm command -- 6 joint targets + gripper travel -- i.e. exactly
the (instruction, action) pair SmolVLA will be trained to reproduce. The base move between
stations is WorkshopSim.teleport_base, a stand-in for Layer 3 walking: a KINEMATIC slide on
the wide shot (legs frozen), captioned as such.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp
from bw.manip.scripted_place import run_place
from bw.sim.workshop import PLACE_STATION, RACK_STATION, TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim
from bw.task.language import instruction
from bw.task.spec import Snapshot, Task, evaluate

TOOL = sys.argv[1] if len(sys.argv) > 1 else "wrench_10mm"
TABLE = sys.argv[2] if len(sys.argv) > 2 else "table_b"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
W, H, IW = 1280, 720, 300
FPS = 25
FONT = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 26)
SMALL = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 19)
JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6", "grip")

sim = WorkshopSim()
ik = ArmIK(sim.m)
rng = np.random.default_rng(1000 * SEED + TOOL_NAMES.index(TOOL))
task = Task("transfer", TOOL, TABLE)
command = instruction(task, np.random.default_rng(SEED))
present = sim.reset(rng, target=TOOL, arm_q=SCAN_Q, base_pose=RACK_STATION)
start = Snapshot.take(sim, present)

state = {"phase": "start", "action": sim.arm_target.copy(), "t_sim": 0.0, "stage": "PICK"}
frames = []
steps_per_frame = int(round(1.0 / (FPS * sim.m.opt.timestep)))


def camera():
    if state.get("cam"):
        return state["cam"]
    return "table_b_view" if state["stage"] == "PLACE" and TABLE == "table_b" else "bench_view"


def compose(title=None):
    big = sim.render(camera(), size=(H, W))
    inset = sim.render("wrist", size=(IW, IW))
    img = Image.fromarray(big)
    img.paste(Image.fromarray(inset), (W - IW - 16, H - IW - 16))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([W - IW - 18, H - IW - 18, W - 14, H - 14], outline=(255, 255, 255, 255), width=2)
    d.text((W - IW - 16, H - IW - 44), "wrist camera (policy input)", font=SMALL,
           fill=(255, 255, 255, 255))
    # command banner
    d.rectangle([0, 0, W, 58], fill=(10, 12, 16, 200))
    d.text((18, 12), f'COMMAND:  "{command}"', font=FONT, fill=(255, 225, 120, 255))
    # action readout
    a = state["action"]
    d.rectangle([0, H - 150, 560, H], fill=(10, 12, 16, 200))
    d.text((14, H - 144), f"{state['stage']} · {state['phase']}", font=FONT,
           fill=(140, 220, 255, 255))
    d.text((14, H - 104), "ACTION (10 Hz, 7-D command):", font=SMALL, fill=(200, 200, 200, 255))
    row1 = "  ".join(f"{n}={v:+.2f}" for n, v in zip(JOINTS[:4], a[:4]))
    row2 = "  ".join(f"{n}={v:+.2f}" for n, v in zip(JOINTS[4:6], a[4:6])) + \
        f"  grip={1000 * a[6]:+.0f}mm"
    d.text((14, H - 78), row1, font=SMALL, fill=(255, 255, 255, 255))
    d.text((14, H - 52), row2, font=SMALL, fill=(255, 255, 255, 255))
    d.text((14, H - 24), f"t = {sim.time:5.1f} s   (scripted demonstrator, not yet a VLA)",
           font=SMALL, fill=(160, 160, 160, 255))
    if title:
        d.rectangle([W // 2 - 420, H // 2 - 50, W // 2 + 420, H // 2 + 50], fill=(0, 0, 0, 190))
        d.text((W // 2 - 400, H // 2 - 20), title, font=FONT, fill=(255, 255, 255, 255))
    return np.asarray(img)


orig_step = sim.physics_step
acc = {"n": 0}


def stepped(n=1):
    left = n
    while left > 0:
        k = min(steps_per_frame - acc["n"], left)
        orig_step(k)
        left -= k
        acc["n"] += k
        if acc["n"] >= steps_per_frame:
            acc["n"] = 0
            frames.append(compose())


def record(q):
    state["action"] = np.array(q, float)


def phase(label):
    state["phase"] = label


sim.physics_step = stepped
# establishing shot: BOTH tables in frame before anything moves
state["cam"] = "scene_wide"
for _ in range(int(2.5 * FPS)):
    frames.append(compose("table A: workbench + rack (left / far)     table B: side table (right / near)"))
state["cam"] = None
for _ in range(int(0.6 * FPS)):
    frames.append(compose())
g = run_grasp(sim, ik, TOOL, rng, record=record, on_phase=phase)
result = {"success": False, "reason": "grasp_" + g.get("reason", "fail")}
if g["success"]:
    # Move the base between the tables on the wide shot, so both tables and the carry are
    # visible. KINEMATIC slide, legs frozen: the walking policy (Layer 3) is not wired into
    # this scene yet, and the caption says so.
    sim.physics_step = orig_step
    state["cam"] = "scene_wide"
    state["stage"], state["phase"] = "MOVE", "carry to " + TABLE.replace("_", " ")
    x0, y0 = float(sim.d.qpos[0]), float(sim.d.qpos[1])
    yaw0 = sim.get_base_yaw()
    x1, y1, yaw1 = PLACE_STATION[TABLE]
    n_move = int(3.5 * FPS)
    for i in range(1, n_move + 1):
        u = 0.5 - 0.5 * np.cos(np.pi * i / n_move)          # ease in / out
        sim.teleport_base((x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, yaw0 + (yaw1 - yaw0) * u),
                          carry=TOOL)
        frames.append(compose("base moved kinematically -- walking policy not wired in yet"))
    state["cam"] = None
    sim.settle(0.3)
    sim.physics_step = stepped
    state["stage"] = "PLACE"
    p = run_place(sim, ik, TOOL, TABLE, rng, record=record, on_phase=phase)
    result = evaluate(sim, task, start) if p.get("reason") not in (
        "place_ik_fail", "not_holding") else p
sim.physics_step = orig_step
verdict = "SUCCESS" if result["success"] else f"FAILED: {result['reason']}"
frames.extend([compose(f"{verdict}  --  scored by bw/task/spec.py")] * int(2.0 * FPS))

out = ROOT / f"media/transfer_{TOOL}_{TABLE}.mp4"
imageio.mimwrite(out, frames, fps=FPS, quality=8, macro_block_size=1)
print(f"{out.name}: {len(frames)} frames ({len(frames) / FPS:.0f} s)  command={command!r}  {verdict}")
