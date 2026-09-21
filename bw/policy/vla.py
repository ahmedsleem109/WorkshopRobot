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


_delta = None


def is_delta() -> bool:
    """Does the served checkpoint predict DELTA arm actions? Asked of the server, once.

    The convention lives with the checkpoint, not with this client, because getting it wrong is
    silent: a delta policy whose output is applied as an absolute target drives the arm to
    roughly zero joint angles, and an absolute policy applied as a delta barely moves it. Both
    look like "the policy is bad" rather than "the client is wrong".
    """
    global _delta
    if _delta is None:
        with urllib.request.urlopen(URL + "/health", timeout=5.0) as r:
            _delta = bool(json.loads(r.read()).get("delta", False))
    return _delta


def query(sim, task: str, reset: bool) -> np.ndarray:
    imgs = {k: np.ascontiguousarray(sim.render(cam)[..., :3]) for k, cam in CAMS.items()}
    h, w = next(iter(imgs.values())).shape[:2]
    r = _post("/act", {"shape": [h, w, 3], "state": sim.arm_q().tolist(), "task": task,
                       "reset": reset,
                       "images": {k: base64.b64encode(v.tobytes()).decode() for k, v in imgs.items()}})
    return np.asarray(r["actions"], float)


def run_skill(sim, task: str, max_s: float, done=None, rate_hz: float = 10.0,
              record=None, lock_stance: bool = True) -> dict:
    """Execute the policy for up to `max_s` seconds of sim time. `done(sim)` -> bool ends the
    episode early (e.g. the tool is lifted and held). Returns {"steps", "stopped_early"}.

    THE LEGS ARE STAND-LOCKED, like every demonstration this policy was trained on (session 7).
    `run_grasp` and `run_place` call `sim.lock_stance()` before they plan, and
    `Locomotion.lock_stance` exists because, measured, "under the policy the arm's reach and pull
    push the standing base back 25-260 mm (it steps away from the load), so the jaws arrive short or
    the tool is dragged against the rack". This function did not call it and neither did
    scripts/eval_vla.py, so every rollout behind the 1/20 was executed on a base that walks away
    from the rack while the arm reaches -- a regime absent from the training data, and one the
    orchestrator never actually uses, since it locks the stance itself before calling a skill.
    Measured with PERFECT actions replayed through this loop (scripts/vla_exec_check.py): locked,
    the demonstrations' own actions grasp; free, the base moves ~36 mm and grasps start failing."""
    if lock_stance:
        sim.lock_stance()
    steps, first = 0, True
    delta = is_delta()
    n_max = int(max_s * rate_hz)
    while steps < n_max:
        chunk = query(sim, task, reset=first)
        first = False
        for a in chunk:
            a = np.asarray(a, float)
            if delta:
                # Each delta is applied to the LIVE joint position at the moment it executes,
                # not to the state that was sent with the query. Within one chunk the arm has
                # already moved, so chaining from the query-time state would accumulate the
                # very drift the delta form is meant to avoid.
                a = np.concatenate([sim.arm_q()[:6] + a[:6], [a[6]]])
            sim.move_arm(a, 1.0 / rate_hz, record, rate_hz)
            steps += 1
            if steps >= n_max:
                break
        if done is not None and done(sim):
            return {"steps": steps, "stopped_early": True}
    return {"steps": steps, "stopped_early": False}
