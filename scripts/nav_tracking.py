"""Command-tracking test of a Layer 3 policy in CPU MuJoCo: can it do what navigation asks?

    render_venv\\Scripts\\python.exe scripts\\nav_tracking.py models\\payload_nav_policy.npz [--scene flat|workshop]

For each command: 1 s stand, 1 s ramp-in on the command, then 4 s measured. Reports achieved
body-frame velocity and yaw rate against the command, over 4 seeds (random initial joint
jitter). Written 2026-09-19 because the 33M payload policy passed the Phase 1 gate (walk
forward, cross the step) and still could not turn in place, back up or sidestep -- which
nothing had measured.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from bw.locomotion.controller import Locomotion

CMDS = [("stand", (0, 0, 0)), ("forward", (0.5, 0, 0)), ("slow", (0.15, 0, 0)),
        ("back", (-0.25, 0, 0)), ("turn L", (0, 0, 0.6)), ("turn R", (0, 0, -0.6)),
        ("side L", (0, 0.2, 0)), ("side R", (0, -0.2, 0)),
        ("arc L", (0.25, 0, 0.5)), ("arc R", (0.25, 0, -0.5))]


def measure(m, npz, cmd, seed, scene):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    if scene == "workshop":
        d.qpos[:3] = [2.9, 0.0, 0.42]           # open walkway, clear of both tables
    rng = np.random.default_rng(seed)
    d.qpos[7:19] += rng.uniform(-0.05, 0.05, 12)
    mujoco.mj_forward(m, d)
    L = Locomotion(m, d, npz)
    L.set_velocity(0, 0, 0)
    for _ in range(50):
        L.control_step()
    L.set_velocity(*cmd)
    for _ in range(50):
        L.control_step()
    p0 = L.get_base_pose()
    yaw_prev, dyaw, v_body = p0.yaw, 0.0, []
    for _ in range(200):
        L.control_step()
        p = L.get_base_pose()
        dyaw += np.angle(np.exp(1j * (p.yaw - yaw_prev)))
        yaw_prev = p.yaw
        v_body.append(p.rot.T @ d.qvel[:3])
        if not L.is_stable():
            return None
    v = np.mean(v_body, axis=0)
    return v[0], v[1], dyaw / 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--scene", default="flat", choices=["flat", "workshop"])
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    xml = ROOT / ("models/scene_go2z1_flat.xml" if a.scene == "flat" else "models/workshop.xml")
    m = mujoco.MjModel.from_xml_path(str(xml))
    print(f"{Path(a.npz).name} on {a.scene}   (achieved, mean over {a.seeds} seeds; 4 s each)")
    print(f"  {'command':8s} {'vx':>6s} {'vy':>6s} {'wz':>6s}  ->  {'vx':>6s} {'vy':>6s} {'wz':>6s}  falls")
    for name, cmd in CMDS:
        rs = [measure(m, a.npz, cmd, s, a.scene) for s in range(a.seeds)]
        ok = [r for r in rs if r is not None]
        mean = np.mean(ok, axis=0) if ok else (np.nan,) * 3
        print(f"  {name:8s} {cmd[0]:+6.2f} {cmd[1]:+6.2f} {cmd[2]:+6.2f}  ->  "
              f"{mean[0]:+6.2f} {mean[1]:+6.2f} {mean[2]:+6.2f}  {len(rs) - len(ok)}")


if __name__ == "__main__":
    main()
