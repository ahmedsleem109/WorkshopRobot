"""SmolVLA inference server (WSL, torch venv) -- the VLA half of the grasp/place skills.

    ~/bringwrench/.venv-vla/bin/python bw/policy/vla_server.py --ckpt DIR [--port 8766]

POST /act  {"images": {"camera1": b64, "camera2": b64}, "shape": [h, w, 3], "state": [7],
            "task": str, "reset": bool}  ->  {"actions": [[7], ...], "latency": s}
Returns the policy's whole remaining action CHUNK (n_action_steps); the client executes it at
10 Hz and asks again. `reset` clears the policy's internal action queue for a new episode.
Loads the checkpoint's own pre/post-processors, so normalisation matches training exactly.
"""
import argparse
import base64
import os
import sys

if "--device" in sys.argv and sys.argv[sys.argv.index("--device") + 1] == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # the GPU is in exclusive use by training
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import numpy as np
import torch


class VLA:
    def __init__(self, ckpt: str, n_action_steps: int | None, device: str = "cuda"):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.configs.policies import PreTrainedConfig
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.device = device                  # load straight onto the target device
        self.policy = SmolVLAPolicy.from_pretrained(ckpt, config=cfg)
        if n_action_steps:
            self.policy.config.n_action_steps = n_action_steps
        self.policy.config.device = device
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": device}})
        self.lock = Lock()

    @torch.no_grad()
    def act(self, images: dict, state, task: str, reset: bool):
        with self.lock:
            if reset:
                self.policy.reset()
            batch = {f"observation.images.{k}":
                     torch.from_numpy(v).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
                     for k, v in images.items()}
            batch["observation.state"] = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            batch["task"] = [task]
            obs = self.pre(batch)
            t0 = time.time()
            # one chunk: pop select_action until the queue refills on the next call
            acts = [self.post(self.policy.select_action(obs)).squeeze(0).cpu().numpy()]
            while len(self.policy._queues["action"]) > 0:
                acts.append(self.post(self.policy.select_action(obs)).squeeze(0).cpu().numpy())
            return np.stack(acts), time.time() - t0


def make_handler(vla: VLA):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"ok": True}) if self.path == "/health" else self._send(404, {})

        def do_POST(self):
            if self.path != "/act":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                h, w, c = req["shape"]
                imgs = {k: np.frombuffer(base64.b64decode(v), np.uint8).reshape(h, w, c)[..., :3].copy()
                        for k, v in req["images"].items()}
                a, dt = vla.act(imgs, req["state"], req["task"], bool(req.get("reset")))
                self._send(200, {"actions": a.tolist(), "latency": dt})
            except Exception as e:                       # noqa: BLE001
                import traceback
                self._send(500, {"error": repr(e), "tb": traceback.format_exc()[-1500:]})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--device", default="cuda", help="cpu: evaluate while the GPU trains")
    args = ap.parse_args()
    t0 = time.time()
    if args.device == "cpu":
        torch.set_num_threads(8)
    vla = VLA(args.ckpt, args.n_action_steps, args.device)
    print(f"[vla_server] {args.ckpt} loaded in {time.time() - t0:.0f}s, "
          f"{args.device}, :{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(vla)).serve_forever()


if __name__ == "__main__":
    main()
