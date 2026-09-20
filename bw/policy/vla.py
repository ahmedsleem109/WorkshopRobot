"""SmolVLA skill client (stdlib + numpy): runs a grasp or place episode in the sim by querying
bw/policy/vla_server.py at 10 Hz -- the same rate, cameras and state the demonstrator recorded
(scripts/collect_demos.py). The policy sees ONLY wrist + mast RGB, the 7-D arm state and the
instruction; no ground truth reaches it.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

import numpy as np

URL = os.environ.get("BW_VLA_URL", "http://127.0.0.1:8766")
CAMS = {"camera1": "wrist", "camera2": "mast"}


def _post(path, payload, timeout=60.0):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    if "error" in out:
        raise RuntimeError(out["error"] + "\n" + out.get("tb", ""))
    return out


def health() -> bool:
    try:
        with urllib.request.urlopen(URL + "/health", timeout=2.0) as r:
            return json.loads(r.read()).get("ok", False)
    except Exception:                                   # noqa: BLE001
        return False


def query(sim, task: str, reset: bool) -> np.ndarray:
    imgs = {k: np.ascontiguousarray(sim.render(cam)[..., :3]) for k, cam in CAMS.items()}
    h, w = next(iter(imgs.values())).shape[:2]
    r = _post("/act", {"shape": [h, w, 3], "state": sim.arm_q().tolist(), "task": task,
                       "reset": reset,
                       "images": {k: base64.b64encode(v.tobytes()).decode() for k, v in imgs.items()}})
    return np.asarray(r["actions"], float)


def run_skill(sim, task: str, max_s: float, done=None, rate_hz: float = 10.0,
              record=None) -> dict:
    """Execute the policy for up to `max_s` seconds of sim time. `done(sim)` -> bool ends the
    episode early (e.g. the tool is lifted and held). Returns {"steps", "stopped_early"}."""
    steps, first = 0, True
    n_max = int(max_s * rate_hz)
    while steps < n_max:
        chunk = query(sim, task, reset=first)
        first = False
        for a in chunk:
            sim.move_arm(np.asarray(a, float), 1.0 / rate_hz, record, rate_hz)
            steps += 1
            if steps >= n_max:
                break
        if done is not None and done(sim):
            return {"steps": steps, "stopped_early": True}
    return {"steps": steps, "stopped_early": False}
