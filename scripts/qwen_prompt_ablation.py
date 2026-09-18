"""T0.3 follow-up: is Qwen3-VL-2B's failure on 10 mm vs 13 mm a PROMPT failure or a MODEL one?

The default harness asks for "pixel coordinates" and the model answers in its own 0-1000
normalised frame, so before concluding anything we try the phrasings that could plausibly
change the answer:

  point_px      what the bake-off used
  point_norm    same, but asking explicitly for 0-1000 normalised coordinates
  grounding     Qwen's OWN grounding format (JSON bbox_2d), centre of the returned box
  colour_band   names the distinguishing FEATURE instead of the size ("blue grip band" /
                "red grip band"). Our wrenches differ mainly in size plus a coloured band
                (STATUS.md known bug 9), so this tests whether the model can see the band at
                all -- which decides between T0.4's option (a), give the wrenches a clearer
                feature, and T8.6, LoRA on sim point labels.

Scored only on views where BOTH wrenches are visible, which is the case the task turns on.

    ~/bringwrench/.venv-vla/bin/python /mnt/d/bringwrench/scripts/qwen_prompt_ablation.py
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = Path.home() / "bringwrench/models/qwen3-vl-2b"
VIEWS = Path("/mnt/d/bringwrench/runs/t03_views")
NUM = re.compile(r"-?\d+\.?\d*")
S = 512

SIZE_LABEL = {"wrench_10mm": "10mm wrench", "wrench_13mm": "13mm wrench"}
BAND_LABEL = {"wrench_10mm": "wrench with the blue grip band",
              "wrench_13mm": "wrench with the red grip band"}

PROMPTS = {
    "point_px": lambda lab: f"Point to the {lab} in the image. "
                            f"Reply with only its pixel coordinates as (x, y).",
    "point_norm": lambda lab: f"Point to the {lab}. Reply with only (x, y), where x and y "
                              f"are normalised to 0-1000 across the image.",
    "grounding": lambda lab: f"Locate the {lab} in this image and output its bbox "
                             f"coordinates in JSON format.",
}


def ask(model, proc, pil, text):
    msgs = [{"role": "user", "content": [{"type": "image", "image": pil},
                                         {"type": "text", "text": text}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    return proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def to_point(txt, kind):
    nums = [float(x) for x in NUM.findall(txt)]
    if kind == "grounding":
        if len(nums) < 4:
            return None
        x0, y0, x1, y1 = nums[:4]
        pt = ((x0 + x1) / 2, (y0 + y1) / 2)
    else:
        if len(nums) < 2:
            return None
        pt = (nums[0], nums[1])
    return (pt[0] / 1000 * S, pt[1] / 1000 * S)      # confirmed convention (see bake-off)


def main():
    recs = json.loads((VIEWS / "views.json").read_text())
    both = [r for r in recs
            if r["tools"].get("wrench_10mm", {}).get("visible")
            and r["tools"].get("wrench_13mm", {}).get("visible")]
    print(f"{len(both)} views with BOTH wrenches visible", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()

    variants = [(k, v, SIZE_LABEL) for k, v in PROMPTS.items()]
    variants.append(("colour_band", PROMPTS["point_px"], BAND_LABEL))

    print(f"\n{'variant':14s} {'parsed':>10} {'median err':>11} {'correct wrench':>16}")
    results = {}
    for vname, tmpl, labels in variants:
        errs, right, n, parsed = [], 0, 0, 0
        for r in both:
            for tool in ("wrench_10mm", "wrench_13mm"):
                # ask for each wrench in turn, so a model that always names the same one
                # scores 50% rather than looking good on one side
                if not r["tools"][tool]["visible"]:
                    continue
                n += 1
                pil = Image.open(VIEWS / r["image"]).convert("RGB")
                txt = ask(model, proc, pil, tmpl(labels[tool]))
                pt = to_point(txt, vname)
                if pt is None:
                    continue
                parsed += 1
                other = "wrench_13mm" if tool == "wrench_10mm" else "wrench_10mm"
                g = np.array([r["tools"][tool]["u"], r["tools"][tool]["v"]])
                o = np.array([r["tools"][other]["u"], r["tools"][other]["v"]])
                p = np.array(pt)
                errs.append(float(np.linalg.norm(p - g)))
                right += np.linalg.norm(p - g) < np.linalg.norm(p - o)
        med = np.median(errs) if errs else float("nan")
        acc = right / parsed if parsed else float("nan")
        results[vname] = {"parsed": parsed, "n": n, "median_err": med, "correct": acc}
        print(f"{vname:14s} {parsed:4d}/{n:<5d} {med:11.1f} {right:6d}/{parsed:<4d} "
              f"({100 * acc:4.0f}%)", flush=True)

    (VIEWS / "prompt_ablation.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\nchance is 50%; the plan's correct-wrench gate is 80%.")


if __name__ == "__main__":
    main()
