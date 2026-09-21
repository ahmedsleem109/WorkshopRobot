"""T8 -- locate(description) -> 3D position | None.

    point (vlm.point, 2D pixel)  ->  wrist depth at that pixel  ->  camera frame
    ->  world frame  ->  base frame;   repeated over SCAN views and merged.

Scan strategy: the wrist camera at the scan pose sees only part of the tool rack (in T0.3,
half of the "occluded" targets were simply OUT OF FRAME -- the rack is wider than the view)
and the gripper hides the lower middle of the image. So locate() pans `arm_joint1` across
SCAN_PANS and asks the model in every view.

Depth: the VLM's pixel error (~40 px median at 512 px, T0.3) is several times a tool's width,
so the depth AT the returned pixel often belongs to the bench behind the tool, which would put
the point 10-30 cm too far along the ray. `pixel_to_world` therefore takes a FOREGROUND
depth from a window around the pixel (a low percentile, ignoring the gripper which sits much
nearer the lens). No ground truth enters this module; the robot's own FK gives the camera pose.

Merge: points from different views are clustered (MERGE_TOL); the largest cluster wins and
its median is returned. No point from any view -> None (drives the floor-sweep recovery).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

SCAN_PANS = (0.0, 0.35, -0.35)       # arm_joint1 offsets from SCAN_Q (rad)
IMG = (512, 512)
WIN = 10                             # px half-width of the depth window around the point
FG_PCT = 20                          # foreground percentile of the window's depths
NEAR_CLIP = 0.16                     # m: nearer is the gripper (median 0.105 m in the view). Was 0.25, which
                                     # also rejected tools at the rack ends: 0.22 m from the lens (session 5)
MERGE_TOL = 0.05                     # m: views agreeing within this form one cluster
AGREE = 2                            # views behind the winning cluster that make a point "supported"
SNAP_R = 14                          # px: search radius for the nearest foreground pixel
BG_R = 40                            # px: window that estimates the local background depth
FG_MARGIN = 0.03                     # m: this much nearer than the background = an object
DEPTH_MODE = "snap"


@dataclass
class View:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    cam_pos: np.ndarray
    cam_mat: np.ndarray
    pan: float = 0.0
    extra: object = None


@dataclass
class Located:
    world: np.ndarray                 # (3,) world frame
    base: np.ndarray                  # (3,) base frame (x forward, y left, z up)
    n_views: int                      # views whose point joined the winning cluster
    n_pointed: int                    # views where the model returned a point at all
    per_view: list = field(default_factory=list)


def foreground_depth(depth: np.ndarray, uv, win: int = WIN, pct: float = FG_PCT,
                     near: float = NEAR_CLIP) -> float | None:
    h, w = depth.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    if u0 >= u1 or v0 >= v1:
        return None
    d = depth[v0:v1, u0:u1].ravel()
    d = d[(d > near) & np.isfinite(d)]
    if d.size == 0:
        return None
    return float(np.percentile(d, pct))


def snap_foreground(depth: np.ndarray, uv, r: int = SNAP_R, bg_r: int = BG_R,
                    margin: float = FG_MARGIN, near: float = NEAR_CLIP):
    """Move `uv` to the nearest pixel that stands OUT of the local background (an object, not
    the bench behind it) and return (uv, depth there). A pointing miss of a few pixels beside
    a thin tool otherwise reads the bench's depth and lands 10-30 cm too far along the ray.
    Falls back to the depth at uv when nothing within `r` stands out (a tool lying flat)."""
    h, w = depth.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < w and 0 <= v < h):
        return None
    big = depth[max(0, v - bg_r):v + bg_r + 1, max(0, u - bg_r):u + bg_r + 1]
    big = big[(big > near) & np.isfinite(big)]
    if big.size == 0:
        return None
    bg = float(np.percentile(big, 80))
    v0, u0 = max(0, v - r), max(0, u - r)
    win = depth[v0:v + r + 1, u0:u + r + 1]
    fg = (win > near) & (win < bg - margin)
    if not fg.any():
        z = float(depth[v, u])
        return ((uv[0], uv[1]), z) if z > near else None
    vv, uu = np.nonzero(fg)
    i = int(np.argmin((vv + v0 - uv[1]) ** 2 + (uu + u0 - uv[0]) ** 2))
    su, sv = uu[i] + u0, vv[i] + v0
    return (float(su), float(sv)), float(depth[sv, su])


def pixel_to_world(uv, z: float, K: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray):
    """MuJoCo camera convention: looks along -z, +x right, +y up; depth is along the axis."""
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    pc = np.array([(uv[0] - cx) * z / f, -(uv[1] - cy) * z / f, -z])
    return cam_pos + cam_mat @ pc


def view_point(view: View, uv, mode: str = DEPTH_MODE, info: dict | None = None,
               **kw) -> np.ndarray | None:
    if uv is None:
        return None
    if mode == "snap":
        s = snap_foreground(view.depth, uv, **kw)
        if s is None:
            return None
        if info is not None:        # how far the answer had to move to land on an object
            info["snap_px"] = float(np.hypot(s[0][0] - uv[0], s[0][1] - uv[1]))
        uv, z = s
    else:
        z = foreground_depth(view.depth, uv, **kw)
        if z is None:
            return None
    return pixel_to_world(uv, z, view.K, view.cam_pos, view.cam_mat)


def merge(points: list, tol: float = MERGE_TOL, cost: list | None = None):
    """Largest cluster; ties -> lowest mean `cost` (the snap distance: an answer that already
    sat ON an object beats one that had to be moved), then smallest spread.
    Returns (median, member indices into the non-None points)."""
    keep = [i for i, p in enumerate(points) if p is not None]
    if not keep:
        return None, []
    P = np.array([points[i] for i in keep])
    C = np.array([0.0 if cost is None or cost[i] is None else cost[i] for i in keep])
    best = None
    for i in range(len(P)):
        mem = np.where(np.linalg.norm(P - P[i], axis=1) < tol)[0]
        spread = np.linalg.norm(P[mem] - P[mem].mean(0), axis=1).sum()
        key = (-len(mem), float(C[mem].mean()), spread)
        if best is None or key < best[0]:
            best = (key, mem)
    mem = best[1]
    return np.median(P[mem], axis=0), list(mem)


def world_to_base(p_world: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray):
    w, x, y, z = base_quat
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ (np.asarray(p_world) - np.asarray(base_pos))


def locate_in_views(views: list[View], description: str, point_fn, base_pos, base_quat,
                    tol: float = MERGE_TOL, verify_fn=None, agree: int = AGREE, **kw) -> Located | None:
    """The whole pipeline on already-captured views (used by locate() and the benchmark).

    `verify_fn(rgb, description, uv) -> bool` is the ABSENT-TOOL tier (T8, session 7). It is called
    ONLY when the winning cluster has fewer than `agree` views behind it, and a view whose point it
    rejects is dropped before the merge is redone. The split matters: measured on 40 stored seeds,
    a point corroborated by a second view is a false positive 2 times in 20 against 14 in 20 for an
    uncorroborated one, so a blanket rule spends its recall on answers that were already right.

    Measured, 40 seeds / 180 present / 20 absent (scripts/score_absent.py):
        no verify                       FP 14/20 (70%)  miss   4/180  success 135/180 (75%)
        blanket "2 of 3 views agree"    FP  2/20 (10%)  miss  71/180  success 100/180 (56%)
        tier + "name the tool" on crop  FP  4/20 (20%)  miss  37/180  success 122/180 (68%)
        tier + "is there a X" (full)    FP 11/20 (55%)  miss   6/180  success 137/180 (76%)
    The name tier is the one that makes a REFUSAL trustworthy, which is what the missing-tool
    recovery needs; it costs 13 of 180 successes and ~79 extra model calls per 200 queries. A
    refusal costs a recovery (floor sweep, ask the human), a false positive costs a confident grasp
    at nothing -- which is why the trade is taken where the answer is thin and nowhere else."""
    per_view, pts, cost = [], [], []
    for vw in views:
        uv = point_fn(vw.rgb, description)
        info = {}
        p = view_point(vw, uv, info=info, **kw)
        per_view.append({"pan": vw.pan, "uv": None if uv is None else [float(uv[0]), float(uv[1])],
                         "world": None if p is None else p.tolist(), **info})
        pts.append(p)
        cost.append(info.get("snap_px"))
    med, mem = merge(pts, tol, cost)
    if med is None:
        return None
    if verify_fn is not None and len(mem) < agree:
        kept = [i for i, p in enumerate(pts) if p is None
                or verify_fn(views[i].rgb, description, per_view[i]["uv"])]
        for i, p in enumerate(pts):
            if i not in kept:
                pts[i] = None
                per_view[i]["verify"] = False
        med, mem = merge(pts, tol, cost)
        if med is None:
            return None
    return Located(world=med, base=world_to_base(med, base_pos, base_quat), n_views=len(mem),
                   n_pointed=sum(p is not None for p in pts), per_view=per_view)


def capture_views(sim, pans=SCAN_PANS, size=IMG, move_s: float = 0.8, extra=None) -> list[View]:
    """Pan the wrist camera across the scan fan and grab RGB + depth + the camera pose (FK).
    `extra(sim)`, if given, is called at every view and its result stored in View.extra."""
    from bw.manip.scripted_grasp import SCAN_Q
    views = []
    for pan in pans:
        q = SCAN_Q.copy()
        q[0] += pan
        sim.move_arm(q, move_s)
        sim.settle(0.2)
        cpos, cmat = sim.camera_pose("wrist")
        views.append(View(rgb=sim.render("wrist", size=size),
                          depth=sim.render("wrist", depth=True, size=size),
                          K=sim.camera_intrinsics("wrist", size=size),
                          cam_pos=cpos, cam_mat=cmat, pan=pan,
                          extra=None if extra is None else extra(sim)))
    q = SCAN_Q.copy()
    sim.move_arm(q, move_s)
    return views


def name_verifier(point_fn=None):
    """The tier that made a refusal trustworthy: ask the model to NAME what is at the point, on the
    zoomed crop, and keep the point only if the name is the queried tool. Open question rather than
    yes/no, because the model answers "yes" agreeably (crop verify: FP 12/20 vs 4/20 for this)."""
    from bw.perception import vlm
    import numpy as _np

    def verify(rgb, description, uv):
        if rgb is None or uv is None:
            return True                      # nothing to look at: do not invent a rejection
        crop, _ = vlm._crop(_np.asarray(rgb), uv)
        raw = vlm._post_ask(crop, vlm.NAME_Q)["raw"].strip().lower()
        want = vlm.resolve_query(description).lower()
        head = [w for w in re.split(r"[^a-z]+", want) if len(w) > 3]
        return any(w in raw for w in head) if head else True
    return verify


def locate(sim, description: str, point_fn=None, pans=SCAN_PANS, verify: bool = False
           ) -> Located | None:
    """Scan, point, back-project, merge. `point_fn` defaults to vlm.point (the model process).
    `verify=True` adds the absent-tool tier (see locate_in_views): fewer false positives on a tool
    that is not there, at the cost of some recall."""
    if point_fn is None:
        from bw.perception.vlm import point as point_fn
    views = capture_views(sim, pans)
    return locate_in_views(views, description, point_fn, sim.d.qpos[0:3].copy(),
                           sim.d.qpos[3:7].copy(),
                           verify_fn=name_verifier() if verify else None)
