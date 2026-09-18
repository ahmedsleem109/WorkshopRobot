"""T0.3 smoke test: can the downloaded grounding model produce a decodable POINT at all?

This is deliberately NOT the bake-off. T0.2 established that Molmo2 emits points as special
tokens decoded by `extract_image_points` with preprocessor metadata -- and that the 4-bit
mirror we downloaded ships NO pointing code (0 point functions, no point tokens among its 303
added tokens). So the question before "how accurate is it" is "does it point at all".

Run (WSL, torch venv, needs the GPU free):
    ~/bringwrench/.venv-vla/bin/python /mnt/d/bringwrench/scripts/test_pointer.py
"""
import sys
import time
from pathlib import Path

MODEL = Path.home() / "bringwrench/models/molmo2-videopoint-4b-4bit"
REPO = Path("/mnt/d/bringwrench")
sys.path.insert(0, str(REPO))

import numpy as np
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available())

# ---- 1. a real wrist-camera render from our own scene, not a stock photo -------------------
from bw.sim.workshop_sim import WorkshopSim
from bw.manip.scripted_grasp import SCAN_Q

sim = WorkshopSim(render_size=(512, 512))
sim.reset(np.random.default_rng(0), target="wrench_10mm", arm_q=SCAN_Q)
img = sim.render("wrist")
print("render", img.shape, img.dtype)
out_png = REPO / "media/pointer_test_input.png"
try:
    import imageio.v2 as imageio
    imageio.imwrite(out_png, img)
    print("wrote", out_png)
except Exception as e:                                   # noqa: BLE001
    print("could not write png:", e)

from PIL import Image
pil = Image.fromarray(img)

# ---- 2. load the model --------------------------------------------------------------------
from transformers import AutoModelForImageTextToText, AutoProcessor

t0 = time.time()
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype="auto", device_map="cuda:0")
print(f"loaded in {time.time() - t0:.0f}s")

print("processor has extract_image_points:", hasattr(proc, "extract_image_points"))
print("processor attrs with 'point':", [a for a in dir(proc) if "point" in a.lower()])

# ---- 3. ask it to point -------------------------------------------------------------------
for prompt in ("Point to the 10mm wrench.",
               "point to the wrench",
               "Locate the wrench and give its pixel coordinates."):
    print("\n" + "=" * 70)
    print("PROMPT:", prompt)
    try:
        inputs = proc.process(images=[pil], text=prompt)
        inputs = {k: (v.to("cuda:0").unsqueeze(0) if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate_from_batch(
                inputs, dict(max_new_tokens=128, stop_strings="<|endoftext|>"),
                tokenizer=proc.tokenizer)
        gen = out[0, inputs["input_ids"].size(1):]
        text = proc.tokenizer.decode(gen, skip_special_tokens=False)
        print(f"latency {time.time() - t0:.1f}s")
        print("RAW OUTPUT:", repr(text[:500]))
    except Exception as e:                               # noqa: BLE001
        print("FAILED:", type(e).__name__, e)
