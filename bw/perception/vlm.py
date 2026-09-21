"""T0.4 -- the Layer 1 grounding contract:  point(image_rgb, query) -> (u, v) | None.

The model (Qwen3-VL-2B) runs in its OWN process (`bw/perception/vlm_server.py`, WSL torch
venv); this client is stdlib + numpy only, so it imports from the Windows render venv and the
WSL JAX venv alike. Swapping the grounding model is a change to the server file alone.

Size -> colour lookup (the T0.3 decision). Qwen3-VL-2B is at CHANCE separating the 10 mm from
the 13 mm wrench by size (50 / 52 / 50% over three phrasings) and at 92.9% when the query names
the grip band. So a size in the query is rewritten to the band colour before the model sees
it. This is a stated limitation: the size discrimination is carried by the scene's colour
coding plus this lookup, not by the vision model.

The server is started on demand (`ensure_server()`): from Windows through wsl.exe, from WSL
directly. Loading takes ~15-30 s; after that a call is ~0.3-1 s.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

URL = os.environ.get("BW_VLM_URL", "http://127.0.0.1:8765")
WSL_PY = "~/bringwrench/.venv-vla/bin/python"
WSL_SERVER = "/mnt/d/bringwrench/bw/perception/vlm_server.py"

# Scene fact (bw/sim/workshop.py: grip rgba): 10 mm = blue band, 13 mm = red band.
SIZE_TO_BAND = {10: "blue", 13: "red"}
# "pliers" alone is REFUSED -- Qwen3-VL-2B answers "There are none." to 11 of the 17 views in
# which the pliers are plainly visible. Naming their colour is what fixes it, the same pattern
# as the wrenches' grip bands. Measured 2026-09-20, seeds 0-7, coarse stage
# (scripts/_pliers_probe.py), refusals on those 17 views:
#     pliers                       11/17 refused,  0/17 on the tool   <- the control
#     red pliers                    2/17 refused,  3/17 on the tool   <- chosen
#     red-handled gripping tool     1/17 refused,  2/17 on the tool
#     pliers with red handles       6/17 refused,  1/17 on the tool
#     tool with two red handles     8/17 refused,  1/17 on the tool
# "red pliers" keeps the fewest refusals AND the most hits; "red-handled gripping tool" refuses
# least but points worse, and a phrase that keeps the noun is the safer one to generalise from.
ALIASES = {"tape": "roll of tape", "tape roll": "roll of tape", "tape_roll": "roll of tape",
           "roll of tape": "roll of tape", "screwdriver": "screwdriver",
           "pliers": "red pliers"}
_SIZE = re.compile(r"(\d+)\s*(?:mm|millimet)", re.I)


def resolve_query(query: str) -> str:
    """User wording -> the phrase the model is actually asked about."""
    q = query.lower().replace("_", " ").strip()
    q = re.sub(r"^(the|a|an)\s+", "", q)
    m = _SIZE.search(q)
    if m and ("wrench" in q or "spanner" in q):
        band = SIZE_TO_BAND.get(int(m.group(1)))
        if band is None:
            return q                     # a size we have no band for: ask as-is
        return f"wrench with the {band} grip band"
    for k, v in ALIASES.items():
        if k in q:
            return v
    return q


def _request(path: str, payload: dict | None = None, timeout: float = 60.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def health() -> dict | None:
    try:
        return _request("/health", timeout=2.0)
    except (urllib.error.URLError, ConnectionError, OSError):
        return None


_proc = None


def ensure_server(quant: str = "bf16", wait_s: float = 180.0) -> dict:
    """Start the model process if nothing answers at URL; block until it is healthy."""
    global _proc
    h = health()
    if h:
        return h
    cmd = f"{WSL_PY} {WSL_SERVER} --quant {quant}"
    if sys.platform == "win32":
        _proc = subprocess.Popen(["wsl.exe", "-e", "bash", "-lc", cmd],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        _proc = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(2.0)
        h = health()
        if h:
            return h
        if _proc.poll() is not None:
            raise RuntimeError(f"vlm_server exited with {_proc.returncode}: {cmd}")
    raise TimeoutError(f"vlm_server not healthy after {wait_s}s")


def stop_server(timeout: float = 15.0) -> bool:
    """Stop the model process and CONFIRM the port has gone quiet. A script that starts the server
    must call this: job 310 (session 7) left it holding 4.85 GB after it finished, and the queue
    runner -- which waits for a free card by design -- stalled behind it until the process was
    killed by hand. Same failure the VLA server had, and the same fix."""
    global _proc
    if sys.platform == "win32":
        subprocess.run(["wsl.exe", "-e", "bash", "-lc", "pkill -f 'vlm_server[.]py'"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(["bash", "-lc", "pkill -f 'vlm_server[.]py'"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if health() is None:
            _proc = None
            return True
        time.sleep(1.0)
    return False


CROP = 160          # px (of a 512 image): refinement window around the first answer
ZOOM = 3            # nearest-neighbour upscale of the crop before re-asking
VERIFY_Q = "Is there a {obj} in this image? Answer yes or no."
# The ABSENT-tool tier (T8, session 7) asks an OPEN question instead: the model answers a yes/no
# agreeably (crop verify left 12 of 20 false positives standing, naming left 4), so the rejection
# rule reads a name and compares it to what was asked for. See bw/perception/locate.name_verifier.
NAME_Q = "Name the single tool at the centre of this image. Answer with its name only."


def _post_point(img: np.ndarray, obj: str) -> dict:
    h, w = img.shape[:2]
    return _request("/point", {"shape": [h, w, 3], "prompt": obj, "style": "plain",
                               "rgb_b64": base64.b64encode(img.tobytes()).decode()})


def _post_ask(img: np.ndarray, question: str) -> dict:
    h, w = img.shape[:2]
    return _request("/ask", {"shape": [h, w, 3], "question": question,
                             "rgb_b64": base64.b64encode(img.tobytes()).decode()})


def _crop(img: np.ndarray, uv, size: int = CROP):
    h, w = img.shape[:2]
    u0 = int(np.clip(round(uv[0] - size / 2), 0, max(0, w - size)))
    v0 = int(np.clip(round(uv[1] - size / 2), 0, max(0, h - size)))
    c = img[v0:v0 + size, u0:u0 + size]
    return np.ascontiguousarray(c.repeat(ZOOM, 0).repeat(ZOOM, 1)), (u0, v0)


def point_ex(image_rgb: np.ndarray, query: str, refine: bool = True,
             verify: bool = False) -> dict:
    """point() with every stage exposed:
        coarse  the model's answer on the full image
        fine    its answer again on a ZOOMx crop around `coarse` (None: keep coarse)
        verify  yes/no on that crop -- "no" turns the answer into None. OFF by default:
                measured (8 seeds) it cost recall (false None 23% -> 32%) and did not
                reject hallucinated points for ABSENT tools (false positives 3/4 either way)
    Returns {"uv", "uv_coarse", "uv_fine", "verify", "raw", "prompt", "latency", "roundtrip"}."""
    img = np.ascontiguousarray(np.asarray(image_rgb, np.uint8)[..., :3])
    h, w = img.shape[:2]
    obj = resolve_query(query)
    t0 = time.time()
    out = {"prompt": obj, "uv_coarse": None, "uv_fine": None, "verify": None, "uv": None}
    r = _post_point(img, obj)
    if "error" in r:
        raise RuntimeError(r["error"])
    lat = r["latency"]
    raw = [r["raw"]]
    if r["xy_norm"] is not None:
        uv = (r["xy_norm"][0] / 1000.0 * w, r["xy_norm"][1] / 1000.0 * h)
        out["uv_coarse"] = out["uv"] = uv
        if refine or verify:
            crop, (u0, v0) = _crop(img, uv)
            ch, cw = crop.shape[:2]
            if refine:
                r2 = _post_point(crop, obj)
                lat += r2["latency"]
                raw.append(r2["raw"])
                if r2.get("xy_norm") is not None:
                    fu = u0 + r2["xy_norm"][0] / 1000.0 * cw / ZOOM
                    fv = v0 + r2["xy_norm"][1] / 1000.0 * ch / ZOOM
                    out["uv_fine"] = out["uv"] = (fu, fv)
            if verify:
                r3 = _post_ask(crop, VERIFY_Q.format(obj=obj))
                lat += r3["latency"]
                raw.append(r3["raw"])
                out["verify"] = not r3["raw"].strip().lower().startswith("no")
                if not out["verify"]:
                    out["uv"] = None
    out.update(raw=" | ".join(raw), latency=lat, roundtrip=time.time() - t0)
    return out


def point(image_rgb: np.ndarray, query: str) -> tuple[float, float] | None:
    """Pixel (u, v) -- u right, v down, origin top-left -- of `query` in the image, or None
    if the model says it is not there."""
    return point_ex(image_rgb, query)["uv"]
