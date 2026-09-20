"""The grounding model's process: Qwen3-VL-2B behind a tiny HTTP/JSON endpoint (T0.4).

Runs ONLY in the WSL torch venv. Everything else in the project talks to it through
`bw.perception.vlm.point()`, so the sim's Python (Windows render_venv, WSL JAX venv) never
imports torch, and swapping the model is a change to this one file.

    ~/bringwrench/.venv-vla/bin/python /mnt/d/bringwrench/bw/perception/vlm_server.py \
        [--port 8765] [--quant bf16|nf4]

    GET  /health  -> {"ok": true, "model": ..., "vram_gb": ...}
    POST /point   {"shape": [H, W, 3], "rgb_b64": <raw uint8 bytes>, "prompt": "<phrase>"}
                  -> {"xy_norm": [x, y] | null, "raw": "...", "latency": s}

The server receives an already-resolved object PHRASE (e.g. "wrench with the blue grip
band"); the size -> colour lookup lives in the client, next to the scene knowledge. The reply
is Qwen's 0-1000 normalised convention (T0.3: 44.5 px vs 278.4 px median error when read as
raw pixels); the client converts to pixels.
"""
import argparse
import base64
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = Path.home() / "bringwrench/models/qwen3-vl-2b"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
# T0.3 / prompt ablation: this phrasing with the colour-band label scored 92.9% correct wrench
TEMPLATE = ("Point to the {obj} in the image. If there is no {obj} in the image, reply "
            "'none'. Otherwise reply with only its pixel coordinates as (x, y).")
TEMPLATE_PLAIN = "Point to the {obj} in the image. Reply with only its pixel coordinates as (x, y)."
REFUSAL = re.compile(r"\b(none|no |not |cannot|can't|there is no|there are no)", re.I)


def parse_reply(txt: str):
    """(x, y) in 0-1000 normalised coordinates, or None for a refusal / unparseable reply."""
    if REFUSAL.search(txt) and len(NUM.findall(txt)) < 2:
        return None
    nums = [float(x) for x in NUM.findall(txt)]
    if len(nums) < 2:
        return None
    x, y = nums[0], nums[1]
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return [x, y]


class Model:
    def __init__(self, quant: str, template: str, device: str = "cuda:0"):
        self.proc = AutoProcessor.from_pretrained(MODEL)
        kw = dict(dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                  device_map=device)
        if quant == "nf4":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, **kw).eval()
        self.template = template
        self.lock = threading.Lock()
        self.quant = quant

    def generate(self, rgb: np.ndarray, text: str, max_new_tokens: int = 32):
        pil = Image.fromarray(rgb)
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil},
                                             {"type": "text", "text": text}]}]
        with self.lock:
            t0 = time.time()
            inputs = self.proc.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False)
            txt = self.proc.decode(out[0, inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
            dt = time.time() - t0
        return txt, dt

    def point(self, rgb: np.ndarray, obj: str, style: str | None = None):
        tmpl = {"plain": TEMPLATE_PLAIN, "none": TEMPLATE}.get(style, self.template)
        txt, dt = self.generate(rgb, tmpl.format(obj=obj))
        return parse_reply(txt), txt, dt

    def ask(self, rgb: np.ndarray, question: str):
        return self.generate(rgb, question, max_new_tokens=8)


def make_handler(model: Model):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):          # quiet
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "model": str(MODEL.name), "quant": model.quant,
                                 "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/point", "/ask"):
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                h, w, c = req["shape"]
                rgb = np.frombuffer(base64.b64decode(req["rgb_b64"]), np.uint8).reshape(h, w, c)
                rgb = rgb[..., :3].copy()
                if self.path == "/ask":
                    raw, dt = model.ask(rgb, req["question"])
                    return self._send(200, {"raw": raw[:200], "latency": dt})
                xy, raw, dt = model.point(rgb, req["prompt"], req.get("style"))
                self._send(200, {"xy_norm": xy, "raw": raw[:200], "latency": dt})
            except Exception as e:                       # noqa: BLE001
                self._send(500, {"error": repr(e)})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--quant", choices=("bf16", "nf4"), default="bf16")
    ap.add_argument("--device", default="cuda:0",
                    help="cpu: query the model while the GPU is training (slow, ~30 s/call)")
    ap.add_argument("--none-clause", action="store_true",
                    help="default prompt invites a 'none' reply (measured: kills recall)")
    args = ap.parse_args()
    t0 = time.time()
    model = Model(args.quant, TEMPLATE if args.none_clause else TEMPLATE_PLAIN, args.device)
    print(f"[vlm_server] loaded {MODEL.name} ({args.quant}) in {time.time() - t0:.0f}s, "
          f"vram {torch.cuda.memory_allocated() / 1e9:.2f} GB, listening :{args.port}",
          flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(model)).serve_forever()


if __name__ == "__main__":
    main()
