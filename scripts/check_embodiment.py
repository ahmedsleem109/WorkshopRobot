"""Sanity checks on the Go2+Z1 embodiment and the workshop, CPU MuJoCo only.

  1. mass budget: arm mass, total mass, CoM shift stowed vs extended
  2. passive stand: legs held at home by the PD servos for 3 s, arm stowed and extended
  3. reach: can the ee reach every tray slot on the bench from the walkway, top-down-ish?
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bw.manip.ik import ArmIK, topdown_rot  # noqa: E402
from bw.sim.workshop import (TOOLS, default_tool_poses, WALKWAY_X, STEP_HEIGHT)  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"


def com(m, d, bodies):
    ms = np.array([m.body_mass[b] for b in bodies])
    xs = np.array([d.xipos[b] for b in bodies])
    return (ms[:, None] * xs).sum(0) / ms.sum(), ms.sum()


def main():
    m = mujoco.MjModel.from_xml_path(str(MODELS / "scene_go2z1_flat.xml"))
    d = mujoco.MjData(m)
    arm = [b for b in range(m.nbody) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).startswith("arm_")]
    robot = list(range(1, m.nbody))
    for key in ("home", "extended", "ready"):
        mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
        mujoco.mj_forward(m, d)
        c_arm, m_arm = com(m, d, arm)
        c_all, m_all = com(m, d, robot)
        base = d.xpos[1]
        print(f"[{key:8s}] arm {m_arm:.2f} kg  total {m_all:.2f} kg  arm CoM rel base "
              f"{np.round(c_arm - base, 3)}  robot CoM rel base {np.round(c_all - base, 3)}  "
              f"ee {np.round(d.site_xpos[m.site('ee').id] - base, 3)}")

    # 2. passive stand
    for key in ("home", "extended"):
        mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
        for _ in range(1500):
            mujoco.mj_step(m, d)
        z = d.qpos[2]
        up = d.xmat[1].reshape(3, 3)[2, 2]
        print(f"stand [{key}] after 3 s: base z {z:.3f}  up.z {up:.3f}  finite {np.isfinite(d.qpos).all()}")

    # 3. reach from the walkway edge
    w = mujoco.MjModel.from_xml_path(str(MODELS / "workshop.xml"))
    wd = mujoco.MjData(w)
    ik = ArmIK(w)
    best = None
    for stand_x in (4.05, 4.1, 4.15, 4.2):
        mujoco.mj_resetDataKeyframe(w, wd, w.key("ready").id)
        wd.qpos[0] = stand_x
        wd.qpos[2] = 0.30 + STEP_HEIGHT
        mujoco.mj_forward(w, wd)
        oks = []
        for t, p in zip(TOOLS, default_tool_poses().values()):
            res = None
            for tilt in (0.0, 0.3, 0.6, 0.9):
                q, ep, er, ok = ik.solve(wd, np.array(p) + [0, 0, 0.005], topdown_rot(0.0, tilt),
                                         q_init=wd.qpos[ik.qadr])
                if ok:
                    res = tilt
                    break
            oks.append(res)
        n = sum(r is not None for r in oks)
        print(f"base x {stand_x:.2f}: reachable {n}/5  tilts {oks}")
        if best is None or n > best[1]:
            best = (stand_x, n)
    print("best stand x", best)


if __name__ == "__main__":
    main()
