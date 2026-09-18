"""Tool roll about its own long axis AT THE MOMENT THE JAWS CLOSE (not at reset).

An earlier check measured roll after settling at reset (+-8 deg) and concluded the jaws were
not gripping the handle's diagonal. That measured the wrong instant: the jaws close ~10 s
later, and the screwdriver's shaft is a CAPSULE, so the tool is free to spin about its long
axis. Asymmetric pad contact during the squeeze can rotate it.

Presented width of a square section of side w rolled by theta is w*(|cos|+|sin|):
  0 deg -> 25.0 mm (flat face)   45 deg -> 35.4 mm (corner to corner)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from bw.manip.ik import ArmIK
from bw.manip import scripted_grasp as SG
from bw.sim.workshop import TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim

TOOL = sys.argv[1] if len(sys.argv) > 1 else "screwdriver"
ti = TOOL_NAMES.index(TOOL)
sim = WorkshopSim()
ik = ArmIK(sim.m)
m, d = sim.m, sim.d

ja = m.joint("arm_finger_a").id
jb = m.joint("arm_finger_b").id
pad_a = m.geom("finger_a_pad").id


def jaw_gap():
    return float(d.qpos[m.jnt_qposadr[ja]] + d.qpos[m.jnt_qposadr[jb]])


def roll_vs_jaw():
    """Angle between the tool's cross-section axes and the jaw closing axis."""
    Rt = d.xmat[sim.tool_body[TOOL]].reshape(3, 3)
    axis = Rt[:, 0]                      # tool long axis
    jaw = d.geom_xmat[pad_a].reshape(3, 3)[:, 2]   # pad normal = closing direction
    jaw_p = jaw - np.dot(jaw, axis) * axis
    n = np.linalg.norm(jaw_p)
    if n < 1e-9:
        return float("nan")
    jaw_p /= n
    y = Rt[:, 1] - np.dot(Rt[:, 1], axis) * axis
    y /= np.linalg.norm(y) + 1e-12
    c = float(np.clip(abs(np.dot(jaw_p, y)), 0, 1))
    return float(np.degrees(np.arccos(c)))     # 0 = face-on, 45 = corner-on


orig = SG._pad_contacts
seen = {"n": 0}
rows = []


def spy(sim_, name):
    seen["n"] += 1
    if seen["n"] == 1:
        a = roll_vs_jaw()
        g = jaw_gap()
        w = 25.0 * (abs(np.cos(np.deg2rad(a))) + abs(np.sin(np.deg2rad(a))))
        rows.append((seed, a, g * 1000, w))
    return orig(sim_, name)


SG._pad_contacts = spy

print(f"{'seed':>4s} {'roll_vs_jaw_deg':>15s} {'jaw_gap_mm':>11s} {'predicted_width_mm':>19s}")
for seed in range(8):
    seen["n"] = 0
    rng = np.random.default_rng(1000 * seed + ti)
    sim.reset(rng, target=TOOL, arm_q=SG.SCAN_Q)
    SG.run_grasp(sim, ik, TOOL, rng)

for seed, a, g, w in rows:
    print(f"{seed:>4d} {a:>15.1f} {g:>11.1f} {w:>19.1f}")
if rows:
    A = np.array([r[1] for r in rows])
    G = np.array([r[2] for r in rows])
    W = np.array([r[3] for r in rows])
    print(f"\nmean roll {A.mean():.1f} deg   mean gap {G.mean():.1f} mm   "
          f"mean predicted width {W.mean():.1f} mm")
    print(f"gap - predicted width: {(G - W).mean():+.1f} mm "
          "(near 0 => the jaws are meeting the rolled cross-section)")
