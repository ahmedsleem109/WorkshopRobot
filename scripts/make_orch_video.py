"""Render one full orchestrator run: typed command -> state machine -> delivery.

    render_venv\\Scripts\\python.exe scripts\\make_orch_video.py SEED OUT.mp4
        [--suite nominal|drop|ambiguous|missing|transfer] [--backend scripted|vla]
        [--grounding oracle|vlm] [--camera track]

The command is NOT an argument -- the suite and seed generate it, through the same trial
setup eval_suite.py uses, so a rendered clip is the same trial the suite scored under
that seed.

Main view: the robot's tracking camera; inset: the wrist camera (what grounding and the VLA
see). Caption: the command, the orchestrator's current state, and anything asked of the human.
Frames stream straight to the encoder (a 70 s run is ~1,700 frames).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import scripts.eval_suite as es
from bw.orchestrator import Orchestrator
from bw.sim.workshop_sim import WorkshopSim

W, H, FPS = 1280, 720, 25
FONT = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 28)
SMALL = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--suite", default="nominal")
    ap.add_argument("--backend", default="scripted")
    ap.add_argument("--grounding", default="oracle")
    ap.add_argument("--camera", default="track")
    args = ap.parse_args()
    sim = WorkshopSim()
    sim.attach_locomotion()
    orch = Orchestrator(sim, backend=args.backend, grounding=args.grounding)
    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8,
                                macro_block_size=1)
    cap = {"state": "", "said": "", "cmd": ""}
    steps = int(round(1.0 / (FPS * sim.m.opt.timestep)))
    orig_step = sim.physics_step
    n = {"k": 0}

    def frame():
        main = sim.render(args.camera, size=(H, W))
        wrist = sim.render("wrist", size=(240, 240))
        img = Image.fromarray(main)
        img.paste(Image.fromarray(wrist), (W - 250, 10))
        d = ImageDraw.Draw(img)
        d.rectangle([0, H - 118, W, H], fill=(0, 0, 0))
        d.text((20, H - 110), f'"{cap["cmd"]}"', font=FONT, fill=(255, 255, 255))
        d.text((20, H - 70), f"state: {cap['state']}", font=SMALL, fill=(120, 220, 255))
        if cap["said"]:
            d.text((20, H - 40), cap["said"], font=SMALL, fill=(255, 210, 90))
        d.text((W - 250, 255), "wrist camera", font=SMALL, fill=(255, 255, 255))
        writer.append_data(np.asarray(img))

    def step(k=1):
        for _ in range(k):
            orig_step(1)
            n["k"] += 1
            if n["k"] % steps == 0:
                frame()

    orig_log = orch._log

    def log(res, state, **kw):
        orig_log(res, state, **kw)
        cap["state"] = state + ("" if state not in ("NAV", "LOCATE", "GRASP", "PLACE", "HANDOFF")
                                else "  " + ", ".join(f"{k}={v}" for k, v in kw.items()
                                                      if k in ("to", "tool", "found", "holding",
                                                               "table", "success")))
        if state == "ASK_HUMAN":
            cap["said"] = f'robot: "{kw["q"]}"   human: "{kw["a"]}"'
    orch._log = log

    # reuse the evaluation's scene + scenario setup, with our frame hook installed
    orig_run = orch.run

    def run(cmd):
        cap["cmd"] = cmd
        sim.physics_step = step
        return orig_run(cmd)
    orch.run = run
    row = es.trial(sim, orch, args.suite, args.seed)
    sim.physics_step = step
    cap["state"] = "DONE: " + ("success" if row["success"] else row["cause"])
    for _ in range(FPS * 2):
        sim.settle(1.0 / FPS)
    writer.close()
    print(row)


if __name__ == "__main__":
    main()
