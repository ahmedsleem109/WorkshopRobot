"""T0.3: can Qwen3-VL-2B point at our own wrist-camera renders, and how accurately?

Unlike the Molmo2 mirrors this needs NO trust_remote_code -- Qwen3VLForConditionalGeneration
ships inside transformers 5.5.4. Points come back as TEXT, so we can read them directly.

Scored against sim ground truth: WorkshopSim knows exactly where every tool is, and
camera_intrinsics/camera_pose let us project that to the pixel the model should have named.

    ~/bringwrench/.venv-vla/bin/python /mnt/d/bringwrench/scripts/test_qwen_point.py [n_images]
"""
import re
import sys
import time
from pathlib import Path

MODEL = Path.home() / "bringwrench/models/qwen3-vl-2b"
REPO = Path("/mnt/d/bringwrench")
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from PIL import Image

from bw.manip.scripted_grasp import SCAN_Q
from bw.sim.workshop import TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LABEL = {"wrench_10mm": "10mm wrench", "wrench_13mm": "13mm wrench",
         "screwdriver": "screwdriver", "pliers": "pliers", "tape_roll": "roll of tape"}

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

print("loading", MODEL)
t0 = time.time()
proc = AutoProcessor.from_pretrained(MODEL)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
print(f"loaded in {time.time() - t0:.0f}s  vram {torch.cuda.memory_allocated()/1e9:.2f} GB")

sim = WorkshopSim(render_size=(512, 512))
NUM = re.compile(r"-?\d+\.?\d*")

rows = []
for i in range(N):
    tool = TOOL_NAMES[i % len(TOOL_NAMES)]
    sim.reset(np.random.default_rng(100 + i), target=tool, arm_q=SCAN_Q)
    img = sim.render("wrist")
    pil = Image.fromarray(img)
    H, W = img.shape[:2]

    q = (f"Point to the {LABEL[tool]} in the image. "
         f"Reply with only its pixel coordinates as (x, y).")
    msgs = [{"role": "user", "content": [{"type": "image", "image": pil},
                                         {"type": "text", "text": q}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt").to("cuda:0")
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    txt = proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    dt = time.time() - t0

    nums = [float(x) for x in NUM.findall(txt)]
    uv = (nums[0], nums[1]) if len(nums) >= 2 else None
    rows.append((tool, txt, uv, dt))
    print(f"\n[{i}] {tool:12s} ({dt:.1f}s)  raw: {txt[:120]!r}")
    print(f"     parsed: {uv}   image {W}x{H}")

ok = sum(1 for _, _, uv, _ in rows if uv is not None)
lat = np.mean([r[3] for r in rows])
print(f"\nparsed a coordinate in {ok}/{len(rows)} replies, mean latency {lat:.1f}s")
print("NOTE: this measures FORMAT only. Pixel accuracy vs ground truth is the next step;")
print("it needs the depth lookup + camera projection, which is T0.3 proper.")
