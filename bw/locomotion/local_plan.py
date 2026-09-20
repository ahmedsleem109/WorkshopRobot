"""T10 #3: a local planner for walking AROUND an unseen blockage.

The robot has no obstacle perception. What it has is the evidence of a walk that ran out of
time with the base still far from its station: the pose it stalled in, and the direction it
was being commanded in at that moment. This module turns those into a geometric plan:

    blocked_direction(stall)           the world direction the base was being pushed in
    estimate_blockage(stall)           a disc AHEAD metres along that direction
    plan_detour(start, goal, blocked)  waypoints, best first, whose two straight legs
                                       (start -> way -> goal) stay on the standable surface
                                       and give the discs as wide a berth as the map allows

"Standable" is the raised walkway minus a base-width margin, plus the landing strip under
table B's near side -- everything else is a 12 cm step DOWN (STEP_HEIGHT), which is what the
base falls off. The previous recovery offset a fixed 0.9 m from the midpoint of the original
line and checked nothing: on the rack -> table B route that lands at (4.06, -1.37), hard
against the walkway's right edge in the bench corner, and the walk there fell.
"""
from __future__ import annotations

import numpy as np

BASE_R = 0.30        # m: the Go2's footprint radius with a margin -- the clearance a leg needs
# What stopped the walk is not necessarily in front of the base: the scenario box is dropped
# BEHIND the robot and stops it during the route's opening backup, so a disc projected along
# the heading lands on clear floor and sends the detour the wrong way. The disc is projected
# along the DIRECTION THE BASE WAS BEING COMMANDED instead (walk_to reports it), one base
# half-length plus one box half-width out, which is where a box's centre sits once the body
# has come to rest against its face.
BOX_R = 0.50         # m: radius assumed for an unseen blockage (the scenario box is 0.6 m square)
AHEAD = 0.70         # m: how far along the blocked direction its centre is assumed to sit

# Standable rectangles ((x0, x1), (y0, y1)), already shrunk by the base margin. See
# WALKWAY_X / WALKWAY_Y / WALKWAY_SPUR_B in bw/sim/workshop.py.
FREE = (((1.60, 4.15), (-1.30, 1.30)),
        ((2.45, 2.95), (-1.95, -1.25)))

# Candidate waypoints are sampled as (fraction along the original line, lateral offset).
ALONG = (0.25, 0.35, 0.5, 0.65, 0.8)
OFFSETS = (0.5, 0.75, 1.0, 1.25)
MIN_CLEAR = 0.30     # m from the blockage's assumed surface before a route counts as roomy
CLEAR_CAP = 0.80     # m past which more clearance does not justify more walking
CLEAR_W = 2.0        # how many metres of detour one metre of clearance is worth


def in_free(p) -> bool:
    x, y = float(p[0]), float(p[1])
    return any(x0 <= x <= x1 and y0 <= y <= y1 for (x0, x1), (y0, y1) in FREE)


def seg_standable(a, b, step: float = 0.06) -> bool:
    """Is every point of the straight segment a -> b on the raised surface?"""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    return all(in_free(a + t * (b - a)) for t in np.linspace(0.0, 1.0, n))


def seg_clearance(a, b, blocked=(), step: float = 0.06) -> float:
    """Smallest distance from the segment a -> b to any blocked disc's SURFACE."""
    if not blocked:
        return float("inf")
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    pts = np.array([a + t * (b - a) for t in np.linspace(0.0, 1.0, n)])
    return min(float(np.linalg.norm(pts - np.asarray(c, float)[:2], axis=1).min() - r)
               for c, r in blocked)


def blocked_direction(stall: dict) -> np.ndarray:
    """World-frame unit vector the base was being pushed in when it stalled.

    A pure in-place turn has no linear command; there the heading is the best guess left.
    """
    yaw = float(stall["yaw"])
    vx, vy = float(stall["cmd"][0]), float(stall["cmd"][1])
    c, s = np.cos(yaw), np.sin(yaw)
    v = np.array([c * vx - s * vy, s * vx + c * vy])
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.array([c, s])


def estimate_blockage(stall: dict, ahead: float = AHEAD, radius: float = BOX_R):
    """The disc a stalled walk is assumed to have run into, from walk_to's `stall` record."""
    p = np.asarray(stall["xy"], float)[:2] + ahead * blocked_direction(stall)
    return (p, radius)


def plan_detour(start, goal, blocked=(), n: int = 3, min_clear: float = MIN_CLEAR):
    """Waypoints (x, y, yaw) that route start -> way -> goal around `blocked`, best first.

    Standability is a HARD constraint -- the map is known exactly, and stepping off the
    walkway is a 12 cm drop. Clearance is a SOFT one: where the blockage is, is a guess from
    where the walk stalled, so a route is scored by how wide a berth it gives (capped at
    CLEAR_CAP, past which more clearance is not worth more walking) against how far it adds:

        cost = length - CLEAR_W * min(clearance, CLEAR_CAP)

    Routes with less than `min_clear` of clearance are dropped; if none survive, the corridor
    is genuinely tight and the best-clearance standable route is returned anyway, since
    walking a tight corridor beats standing still. yaw is the bearing from the waypoint to
    the goal, so the leg after the detour needs no large in-place turn.
    """
    start = np.asarray(start, float)[:2]
    goal = np.asarray(goal, float)[:2]
    d = goal - start
    L = float(np.linalg.norm(d))
    if L < 1e-3:
        return []
    d = d / L
    perp = np.array([-d[1], d[0]])

    cands = []
    for s in ALONG:
        for off in OFFSETS:
            for side in (1.0, -1.0):
                cands.append(start + s * L * d + side * off * perp)
    # Also hug each blockage: step off to the side of the disc itself, which is the shortest
    # way past a box that sits right on the line.
    for c, r in blocked:
        c = np.asarray(c, float)[:2]
        for side in (1.0, -1.0):
            cands.append(c + side * (r + BASE_R + 0.25) * perp)

    feasible = []
    for w in cands:
        if not in_free(w):
            continue
        if not (seg_standable(start, w) and seg_standable(w, goal)):
            continue
        clear = min(seg_clearance(start, w, blocked), seg_clearance(w, goal, blocked))
        length = float(np.linalg.norm(w - start) + np.linalg.norm(goal - w))
        feasible.append((length - CLEAR_W * min(clear, CLEAR_CAP), clear, w))
    roomy = [f for f in feasible if f[1] >= min_clear]
    if roomy:
        scored = sorted(roomy, key=lambda t: t[0])
    else:                       # tight corridor: take the widest berth that exists
        scored = sorted(feasible, key=lambda t: -t[1])
    scored = [(c, w) for c, _, w in scored]

    out, kept = [], []
    for _, w in scored:
        if any(np.linalg.norm(w - k) < 0.35 for k in kept):   # near-duplicates add nothing
            continue
        kept.append(w)
        g = goal - w
        out.append((float(w[0]), float(w[1]), float(np.arctan2(g[1], g[0]))))
        if len(out) >= n:
            break
    return out
