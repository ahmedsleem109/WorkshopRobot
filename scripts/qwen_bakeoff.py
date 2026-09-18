"""T0.3 stage 2 (WSL, torch venv): score Qwen3-VL-2B's pointing against sim ground truth.

    ~/bringwrench/.venv-vla/bin/python /mnt/d/bringwrench/scripts/qwen_bakeoff.py [--n N]

Reads the views dumped by scripts/dump_wrist_views.py (Windows) and answers the four things
T0.3 asks for:

  1. THE COORDINATE SCALE. Session 2 saw `(800, 455)` and `(844, 500)` on a 512x512 image and
     inferred Qwen's 0-1000 normalised convention. That is a guess until it is scored: this
     computes the median pixel error under BOTH readings (raw pixels, and x/1000*W) over the
     whole set. Whichever is dramatically smaller IS the convention -- and if neither is
     small, the model is not pointing at all and no rescaling will save it.
  2. MISS RATE, split honestly. A refusal on a view where the target is OCCLUDED is correct
     behaviour, and the dump records visibility from the depth buffer, so those are reported
     separately instead of being charged to the model.
  3. LATENCY per call.
  4. 10 mm vs 13 mm DISCRIMINATION -- scored only on views where BOTH wrenches are visible,
     by asking for one and checking which wrench's ground-truth pixel the reply lands nearer.
     This is the distinction the whole task depends on and the one most likely to fail.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = Path.home() / "bringwrench/models/qwen3-vl-2b"
VIEWS = Path("/mnt/d/bringwrench/runs/t03_views")
LABEL = {"wrench_10mm": "10mm wrench", "wrench_13mm": "13mm wrench",
         "screwdriver": "screwdriver", "pliers": "pliers", "tape_roll": "roll of tape"}
NUM = re.compile(r"-?\d+\.?\d*")


def ask(model, proc, pil, text):
    msgs = [{"role": "user", "content": [{"type": "image", "image": pil},
                                         {"type": "text", "text": text}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    txt = proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return txt, time.time() - t0


def parse(txt):
    nums = [float(x) for x in NUM.findall(txt)]
    return (nums[0], nums[1]) if len(nums) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = all views")
    ap.add_argument("--out", default=str(VIEWS / "qwen_scores.json"))
    args = ap.parse_args()

    recs = json.loads((VIEWS / "views.json").read_text())
    if args.n:
        recs = recs[:args.n]

    print("loading", MODEL, flush=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
    print(f"loaded in {time.time() - t0:.0f}s  vram {torch.cuda.memory_allocated()/1e9:.2f} GB",
          flush=True)

    rows = []
    for i, r in enumerate(recs):
        S = r["size"]
        gt = r["tools"].get(r["target"])
        pil = Image.open(VIEWS / r["image"]).convert("RGB")
        q = (f"Point to the {LABEL[r['target']]} in the image. "
             f"Reply with only its pixel coordinates as (x, y).")
        txt, dt = ask(model, proc, pil, q)
        uv = parse(txt)
        row = {"i": i, "target": r["target"], "raw": txt[:120], "uv": uv, "latency": dt,
               "visible": bool(gt and gt["visible"]),
               "gt": [gt["u"], gt["v"]] if gt else None,
               "both_wrenches": bool(r["tools"].get("wrench_10mm", {}).get("visible")
                                     and r["tools"].get("wrench_13mm", {}).get("visible")),
               "other_wrench": None}
        if uv and gt:
            g = np.array([gt["u"], gt["v"]])
            row["err_px"] = float(np.linalg.norm(np.array(uv) - g))
            row["err_norm"] = float(np.linalg.norm(
                np.array([uv[0] / 1000 * S, uv[1] / 1000 * S]) - g))
        other = "wrench_13mm" if r["target"] == "wrench_10mm" else (
            "wrench_10mm" if r["target"] == "wrench_13mm" else None)
        if other and uv and gt and r["tools"].get(other, {}).get("visible"):
            o = r["tools"][other]
            row["other_wrench"] = [o["u"], o["v"]]
        rows.append(row)
        if i % 10 == 0:
            print(f"[{i:3d}/{len(recs)}] {r['target']:12s} {txt[:48]!r}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    report(rows)


def report(rows):
    S = 512
    vis = [r for r in rows if r["visible"]]
    occ = [r for r in rows if not r["visible"]]
    got = [r for r in vis if r["uv"] is not None]
    print("\n" + "=" * 72)
    print(f"views {len(rows)}   target visible {len(vis)}   target occluded {len(occ)}")
    print(f"latency  median {np.median([r['latency'] for r in rows]):.2f}s  "
          f"mean {np.mean([r['latency'] for r in rows]):.2f}s")
    print(f"replied with a coordinate: {len(got)}/{len(vis)} visible  "
          f"({sum(r['uv'] is not None for r in occ)}/{len(occ)} when OCCLUDED -- "
          f"a refusal there is correct)")

    print("\n-- 1. COORDINATE SCALE (median pixel error on visible targets) --")
    for key, name in (("err_px", "read as raw PIXELS"),
                      ("err_norm", "read as 0-1000 NORMALISED (x/1000*W)")):
        e = [r[key] for r in got if key in r]
        if e:
            print(f"   {name:42s} median {np.median(e):6.1f} px   "
                  f"p25 {np.percentile(e, 25):5.1f}  p75 {np.percentile(e, 75):5.1f}")
    ep, en = [np.median([r[k] for r in got if k in r]) for k in ("err_px", "err_norm")]
    best = "0-1000 NORMALISED" if en < ep else "raw PIXELS"
    print(f"   -> convention is {best}")
    e = [r["err_norm" if en < ep else "err_px"] for r in got]
    print(f"   -> median error {np.median(e):.1f} px on a {S}px image "
          f"({100 * np.median(e) / S:.1f}% of the frame); "
          f"within 25 px: {100 * np.mean(np.array(e) < 25):.0f}%")

    print("\n-- 2. 10 mm vs 13 mm DISCRIMINATION (both wrenches visible) --")
    key = "err_norm" if en < ep else "err_px"
    disc = [r for r in got if r["other_wrench"] is not None]
    if not disc:
        print("   no view had both wrenches visible with a parsed reply")
    else:
        right = 0
        for r in disc:
            uv = np.array(r["uv"])
            if key == "err_norm":
                uv = uv / 1000 * S
            d_t = np.linalg.norm(uv - np.array(r["gt"]))
            d_o = np.linalg.norm(uv - np.array(r["other_wrench"]))
            right += d_t < d_o
        print(f"   named the wrench it was ASKED for: {right}/{len(disc)} "
              f"({100 * right / len(disc):.0f}%)   [chance = 50%]")
        print(f"   plan's gate for correct-wrench selection is 80%: "
              f"{'PASS' if right / len(disc) >= 0.8 else 'FAIL'}")

    print("\n-- 3. PER TOOL (median error px, parsed rate) --")
    for t in sorted({r["target"] for r in rows}):
        tv = [r for r in vis if r["target"] == t]
        tg = [r for r in tv if r["uv"] is not None and key in r]
        med = np.median([r[key] for r in tg]) if tg else float("nan")
        print(f"   {t:12s} {len(tg):3d}/{len(tv):<3d} parsed   median {med:6.1f} px")


if __name__ == "__main__":
    main()
