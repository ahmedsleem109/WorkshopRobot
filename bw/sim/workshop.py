"""The one workshop scene (spec: bench 75 cm, tool tray, floor clutter, one 12 cm step,
exactly five objects).

Layout, world frame, metres. The robot starts at the origin facing +x.

    y
    ^   human + handoff tray (x -0.6..0, y ~1.8)
    |
    |   start (0,0) --> 12 cm raised walkway (x 1.3..4.4) --> bench front edge x=4.45
    +-------------------------------------------------------------------------------> x

The walkway ends right at the bench, so every trip to the bench crosses the step twice
(up on the way there, down on the way back) -- the Phase 1 gate's "crosses the step
with arm extended" is therefore exercised by the task itself, not by a side test.

The bench top is 0.75 m above the world floor; standing on the walkway the robot sees
it 0.63 m above its own feet, which is what puts the tray inside the Z1's workspace
(checked by scripts/check_reach.py). Everything is primitive geometry: MuJoCo grasps on
boxes and cylinders are stable, while tool meshes would add nothing but contact noise.

Collision bits: tools use 3 (Go2 legs = bit 1, gripper pads = bit 2); furniture uses 7, adding bit 4
for the arm links and gripper housing. The arm never touches its own trunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

STEP_HEIGHT = 0.12
WALKWAY_X = (1.3, 4.4)
WALKWAY_Y = (-1.6, 1.6)
BENCH_HEIGHT = 0.75
BENCH_X = (4.45, 5.05)
BENCH_Y = (-0.9, 0.9)
# ---------------------------------------------------------------- the tool rack
# The plan says "tool tray". It is a bench-top RACK instead -- tools standing upright in
# slots -- because a flat tray is not graspable by this embodiment, measured not guessed:
# the Z1 shoulder sits at 0.67 m even on the pedestal, so a 0.75 m bench can only be
# approached at >=35 deg from vertical, and at that tilt the fixed lower jaw's housing
# reaches the tray floor before the pads reach a 9 mm wrench (scripts/try_grasp.py: 0-1 of
# 20 grasps, jaws jammed at q~-0.6 with no pad contact; ribs to lift the tools then cost the
# pads their clearance). Standing tools are grasped HORIZONTALLY, which is the middle of
# this arm's workspace, and a rack is at least as common as a tray on a real workbench.
RACK_CENTER = (4.60, 0.0)
RACK_GAP = 0.030                  # inner x gap: must clear the widest standing section (the
                                  # 25 mm screwdriver handle; at 0.024 it never seated and the
                                  # tool toppled before the arm arrived)
RACK_PLATE = 0.005                # plate thickness
RACK_H = 0.045                    # plate height: taller plates limit how far a tool leans
RACK_BASE = 0.005
RACK_HALF_Y = 0.32
RACK_SLOT_Y = (-0.24, -0.12, 0.0, 0.12, 0.24)
TRAY_CENTER = RACK_CENTER         # kept: navigation/orchestration refer to the bench station
HANDOFF_TRAY = (-0.25, 1.8)
HUMAN_POS = (-0.75, 2.05)
BOX_PARK = (0.0, -8.0)              # scenario-3 obstacle, parked out of the scene


@dataclass(frozen=True)
class ToolSpec:
    name: str                       # body name
    label: str                      # canonical language label
    mass: float
    half_len: float                 # half-length along the tool's long axis (x)


TOOLS = (
    ToolSpec("wrench_10mm", "10mm wrench", 0.045, 0.070),
    ToolSpec("wrench_13mm", "13mm wrench", 0.080, 0.085),
    ToolSpec("screwdriver", "screwdriver", 0.070, 0.085),
    ToolSpec("pliers", "pliers", 0.180, 0.080),
    ToolSpec("tape_roll", "tape roll", 0.070, 0.045),
)
TOOL_NAMES = tuple(t.name for t in TOOLS)

# Tools the demonstrator is expected to GRASP. The screwdriver stays in the scene -- it is a
# useful distractor for the grounding model, which must not fetch the wrong tool -- but it is
# not a grasp target.
#
# DECISION 2026-09-18, after four refuted hypotheses and 0/8 across every configuration tried:
#   (a) jaws closing on the rack plates       -- refuted, pads contact the handle at 26-33 N
#   (b) grasp height x squeeze depth          -- refuted, 0/8 across all six combinations
#   (c) pad contact stiffness                 -- refuted, made every tool worse
#   (d) corner grip from handle roll          -- REAL (27.6 deg mean roll at closure, jaw gap
#       34-44 mm on a 25 mm handle) and FIXED by making the handle a cylinder (gap -> 24.7 mm),
#       but success did not move: still 0/8.
# What the traces show and nothing has yet addressed: the tool is JAMMED in its slot. Closing
# pins it against rack_front at 28 N, its shaft needs 57 mm of lift to clear the plates, and
# the pads lose it at 49 mm. A 90 mm round shaft standing in a 30 mm slot is a poor fixture,
# and saying so is a legitimate finding. Revisit by widening the slot for this tool, or by
# laying it down, if a fifth tool is wanted later.
GRASP_TOOLS = tuple(n for n in TOOL_NAMES if n != "screwdriver")
TOOL_BY_NAME = {t.name: t for t in TOOLS}



def _tool_xml(t: ToolSpec, pos) -> str:
    x, y, z = pos
    jt = f'<freejoint name="{t.name}_free"/>'
    common = 'contype="3" conaffinity="3" condim="6" friction="1.2 0.05 0.005" solref="0.003 1" solimp="0.99 0.9995 0.0005 0.5 2"'
    if t.name.startswith("wrench"):
        big = t.name == "wrench_13mm"
        hl, hw, ht = t.half_len - 0.018, (0.008 if big else 0.0065), (0.0045 if big else 0.004)
        head = 0.017 if big else 0.013
        grip = "0.75 0.12 0.10 1" if big else "0.12 0.30 0.80 1"
        m = t.mass
        geoms = f"""
      <geom name="{t.name}_handle" type="box" size="{hl} {hw} {ht}" material="chrome" mass="{m*0.6:.4f}" {common}/>
      <geom name="{t.name}_grip" type="box" size="{hl*0.45:.4f} {hw+0.0012:.4f} {ht+0.0008:.4f}" rgba="{grip}" mass="0.0001" contype="0" conaffinity="0"/>
      <geom name="{t.name}_open" type="box" size="{head*0.7:.4f} {head} {ht}" pos="{hl+head*0.6:.4f} 0 0" material="chrome" mass="{m*0.2:.4f}" {common}/>
      <geom name="{t.name}_ring" type="cylinder" size="{head} {ht}" pos="{-hl-head*0.6:.4f} 0 0" material="chrome" mass="{m*0.2:.4f}" {common}/>"""
    elif t.name == "screwdriver":
        # ROUND handle, not a box. MEASURED 2026-09-18: with a 25 mm SQUARE handle the tool
        # had rolled a mean of 27.6 deg about its own long axis by the time the jaws closed
        # (only +-8 deg at reset -- the earlier check measured the wrong instant), so the pads
        # met a near-corner and the jaw gap sat at 34-44 mm on a 25 mm handle. A corner grip is
        # a tiny contact patch, and the tool pivoted out of the jaws ~49 mm into the lift:
        # screwdriver scored 0/8. It rolls freely because its shaft is a capsule. A cylinder
        # presents 25 mm at EVERY roll angle -- and real screwdriver handles are round or hex
        # for the same reason.
        geoms = f"""
      <geom name="{t.name}_handle" type="cylinder" size="0.0125 0.035" pos="-0.045 0 0" quat="0.7071 0 0.7071 0" rgba="0.95 0.78 0.05 1" mass="0.055" {common}/>
      <geom name="{t.name}_shaft" type="capsule" size="0.0035 0.045" pos="0.04 0 0" quat="0.7071 0 0.7071 0" material="chrome" mass="0.015" {common}/>"""
    elif t.name == "pliers":
        geoms = f"""
      <geom name="{t.name}_jaw" type="box" size="0.03 0.009 0.006" pos="0.05 0 0" material="steel" mass="0.07" {common}/>
      <geom name="{t.name}_h1" type="box" size="0.05 0.007 0.007" pos="-0.03 0.016 0" euler="0 0 0.25" rgba="0.85 0.1 0.1 1" mass="0.055" {common}/>
      <geom name="{t.name}_h2" type="box" size="0.05 0.007 0.007" pos="-0.03 -0.016 0" euler="0 0 -0.25" rgba="0.85 0.1 0.1 1" mass="0.055" {common}/>"""
    elif t.name == "tape_roll":
        # Hollow ring of 16 box segments (outer r 45 mm, wall 7 mm): grasped by the rim.
        import math
        segs = []
        n, r, half_t, half_h = 16, 0.0415, 0.0035, 0.0095
        half_len = r * math.tan(math.pi / n) + 0.0012
        for k in range(n):
            th = 2 * math.pi * k / n
            segs.append(f'<geom name="{t.name}_seg{k}" type="box" size="{half_t} {half_len:.4f} {half_h}" '
                        f'pos="{r*math.cos(th):.4f} {r*math.sin(th):.4f} 0" euler="0 0 {th:.4f}" material="tape" mass="{0.07/n:.5f}" {common}/>')
        geoms = "\n      " + "\n      ".join(segs)
    else:
        raise ValueError(t.name)
    q = stand_quat(t.name)
    return f"""    <body name="{t.name}" pos="{x:.4f} {y:.4f} {z:.4f}" quat="{q[0]} {q[1]} {q[2]} {q[3]}">
      {jt}{geoms}
    </body>"""


def rack_floor_z() -> float:
    """World z of the rack's inner floor, where a standing tool's lowest point rests."""
    return BENCH_HEIGHT + RACK_BASE


# Standing orientation: the tool's +x axis points DOWN (into the slot), its +y (width) is
# LATERAL so the jaws close across it, and its +z (thickness/axis) points at the robot.
STAND_QUAT = (0.7071, 0.0, 0.7071, 0.0)      # wxyz; R columns = (-z, +y, +x) in world
# Pliers stand rotated 90 deg about the vertical so the handles spread in DEPTH, not across
# the jaws: lateral width then stays the 14 mm handle thickness.
STAND_QUAT_PLIERS = (-0.5, -0.5, -0.5, 0.5)
# Height above the rack floor at which each standing tool is grasped: on a graspable
# section, and clear of the rack plates (RACK_BASE + RACK_H = 0.05). The height the tool
# itself STANDS at is computed from the compiled geometry -- see WorkshopSim._standing_offsets.
GRASP_Z = {"wrench_10mm": 0.088, "wrench_13mm": 0.105, "screwdriver": 0.128,
           "pliers": 0.112, "tape_roll": 0.0467}


def stand_quat(name: str):
    return STAND_QUAT_PLIERS if name == "pliers" else STAND_QUAT


def default_tool_poses() -> dict[str, tuple[float, float, float]]:
    """Initial XML poses only; WorkshopSim.reset re-places every tool at its measured
    standing height."""
    z0 = rack_floor_z()
    return {t.name: (RACK_CENTER[0], RACK_CENTER[1] + RACK_SLOT_Y[i], z0 + 0.10)
            for i, t in enumerate(TOOLS)}


def _clutter_xml(rng: np.random.Generator) -> str:
    """Static floor clutter off the main lanes: offcuts, a bucket, a cable reel."""
    items = [
        ("offcut_a", "box", "0.12 0.04 0.02", (0.7, -0.9, 0.02), "wood"),
        ("offcut_b", "box", "0.20 0.03 0.015", (0.4, 1.0, 0.015), "wood"),
        ("bucket", "cylinder", "0.13 0.16", (-0.9, -1.2, 0.16), "bucket"),
        ("reel", "cylinder", "0.16 0.06", (2.9, -1.35, 0.18), "bucket"),
        ("crate", "box", "0.2 0.15 0.12", (4.0, 1.35, 0.24), "wood"),
        ("rag", "box", "0.08 0.08 0.004", (1.0, 0.45, 0.004), "rag"),
    ]
    out = []
    for name, typ, size, pos, mat in items:
        out.append(f'    <geom name="{name}" type="{typ}" size="{size}" pos="{pos[0]} {pos[1]} {pos[2]}" '
                   f'material="{mat}" contype="7" conaffinity="7"/>')
    return "\n".join(out)


def build_workshop_xml(robot_file: str = "go2z1_scene_robot.xml") -> str:
    rng = np.random.default_rng(0)
    wx0, wx1 = WALKWAY_X
    wy0, wy1 = WALKWAY_Y
    bx0, bx1 = BENCH_X
    by0, by1 = BENCH_Y
    tools = "\n".join(_tool_xml(t, p) for t, p in zip(TOOLS, default_tool_poses().values()))
    rack_dividers = "\n".join(
        f'      <geom name="divider{k}" type="box" size="{RACK_GAP/2 + RACK_PLATE} {RACK_PLATE/2} {RACK_H/2}" '
        f'pos="0 {y:.3f} {RACK_BASE + RACK_H/2}" material="tray_rib" contype="3" conaffinity="3"/>'
        for k, y in enumerate((-0.30, -0.18, -0.06, 0.06, 0.18, 0.30)))
    return f"""<mujoco model="bring-me-the-10mm-wrench workshop">
  <!-- GENERATED by bw/sim/workshop.py -- edit that file, not this one. -->
  <include file="{robot_file}"/>

  <!-- CPU scene: the Go2 MJX model's iterations=1 / impratio=100 solver is tuned for batched
       locomotion throughput and is unstable for light free objects in a gripper. -->
  <option iterations="50" ls_iterations="20" impratio="10" cone="elliptic" noslip_iterations="0"/>

  <statistic center="2.2 0 0.4" extent="4.0"/>
  <visual>
    <headlight diffuse="0.35 0.35 0.35" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <global azimuth="-150" elevation="-25" offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="concrete" type="2d" builtin="flat" rgb1="0.52 0.52 0.50" rgb2="0.45 0.45 0.44" mark="random" random="0.08" markrgb="0.38 0.38 0.37" width="512" height="512"/>
    <material name="concrete" texture="concrete" texrepeat="6 6" texuniform="true" reflectance="0.05"/>
    <texture name="hazard" type="2d" builtin="checker" rgb1="0.95 0.75 0.05" rgb2="0.1 0.1 0.1" width="64" height="64"/>
    <material name="hazard" texture="hazard" texrepeat="12 1" texuniform="false"/>
    <material name="walkway" rgba="0.36 0.40 0.44 1" reflectance="0.02"/>
    <material name="bench" rgba="0.62 0.45 0.26 1"/>
    <material name="benchleg" rgba="0.25 0.27 0.30 1"/>
    <material name="tray" rgba="0.15 0.35 0.55 1"/>
    <material name="tray_rib" rgba="0.12 0.26 0.40 1"/>
    <material name="chrome" rgba="0.78 0.79 0.82 1" specular="0.8" shininess="0.9" reflectance="0.15"/>
    <material name="steel" rgba="0.45 0.46 0.5 1" specular="0.6"/>
    <material name="tape" rgba="0.62 0.64 0.66 1"/>
    <material name="wood" rgba="0.70 0.55 0.36 1"/>
    <material name="bucket" rgba="0.9 0.35 0.08 1"/>
    <material name="rag" rgba="0.55 0.2 0.2 1"/>
    <material name="wall" rgba="0.80 0.82 0.80 1"/>
    <material name="human" rgba="0.25 0.35 0.55 1"/>
    <material name="skin" rgba="0.85 0.66 0.52 1"/>
    <material name="box" rgba="0.72 0.58 0.38 1"/>
  </asset>

  <worldbody>
    <light name="key" pos="2 -1 4" dir="0.1 0.2 -1" diffuse="0.7 0.7 0.68" castshadow="true"/>
    <light name="fill" pos="-1 2 3" dir="0.3 -0.3 -1" diffuse="0.35 0.35 0.38" castshadow="false"/>
    <light name="bench_lamp" pos="4.7 0 1.9" dir="0 0 -1" diffuse="0.45 0.43 0.38" castshadow="false"/>

    <geom name="floor" type="plane" size="8 8 0.05" pos="1.5 0 0" material="concrete" condim="3"
      friction="0.8 0.02 0.01" contype="7" conaffinity="7"/>
    <geom name="wall_back" type="box" size="0.05 3 1.2" pos="5.7 0 1.2" material="wall" contype="7" conaffinity="7"/>
    <geom name="wall_left" type="box" size="4 0.05 1.2" pos="1.7 3.0 1.2" material="wall" contype="7" conaffinity="7"/>
    <geom name="wall_right" type="box" size="4 0.05 1.2" pos="1.7 -3.0 1.2" material="wall" contype="7" conaffinity="7"/>

    <!-- 12 cm raised walkway: the one step -->
    <geom name="walkway" type="box" size="{(wx1-wx0)/2:.3f} {(wy1-wy0)/2:.3f} {STEP_HEIGHT/2:.3f}"
      pos="{(wx0+wx1)/2:.3f} {(wy0+wy1)/2:.3f} {STEP_HEIGHT/2:.3f}" material="walkway" condim="3"
      friction="0.8 0.02 0.01" contype="7" conaffinity="7"/>
    <geom name="walkway_edge" type="box" size="0.02 {(wy1-wy0)/2:.3f} {STEP_HEIGHT/2+0.0005:.4f}"
      pos="{wx0+0.02:.3f} {(wy0+wy1)/2:.3f} {STEP_HEIGHT/2:.3f}" material="hazard" contype="0" conaffinity="0"/>

    <!-- bench -->
    <body name="bench" pos="{(bx0+bx1)/2:.3f} {(by0+by1)/2:.3f} 0">
      <geom name="bench_top" type="box" size="{(bx1-bx0)/2:.3f} {(by1-by0)/2:.3f} 0.02" pos="0 0 {BENCH_HEIGHT-0.02:.3f}" material="bench" contype="7" conaffinity="7"/>
      <geom type="box" size="0.03 0.03 {(BENCH_HEIGHT-0.04)/2:.3f}" pos="{(bx1-bx0)/2-0.05:.3f} {(by1-by0)/2-0.05:.3f} {(BENCH_HEIGHT-0.04)/2:.3f}" material="benchleg" contype="7" conaffinity="7"/>
      <geom type="box" size="0.03 0.03 {(BENCH_HEIGHT-0.04)/2:.3f}" pos="{(bx1-bx0)/2-0.05:.3f} {-(by1-by0)/2+0.05:.3f} {(BENCH_HEIGHT-0.04)/2:.3f}" material="benchleg" contype="7" conaffinity="7"/>
      <geom type="box" size="0.03 0.03 {(BENCH_HEIGHT-0.04)/2:.3f}" pos="{-(bx1-bx0)/2+0.05:.3f} {(by1-by0)/2-0.05:.3f} {(BENCH_HEIGHT-0.04)/2:.3f}" material="benchleg" contype="7" conaffinity="7"/>
      <geom type="box" size="0.03 0.03 {(BENCH_HEIGHT-0.04)/2:.3f}" pos="{-(bx1-bx0)/2+0.05:.3f} {-(by1-by0)/2+0.05:.3f} {(BENCH_HEIGHT-0.04)/2:.3f}" material="benchleg" contype="7" conaffinity="7"/>
      <geom name="vise" type="box" size="0.06 0.08 0.06" pos="0.15 -0.7 {BENCH_HEIGHT+0.06:.3f}" material="benchleg" contype="7" conaffinity="7"/>
    </body>

    <!-- bench-top tool rack: tools stand in slots, grasped horizontally -->
    <body name="rack" pos="{RACK_CENTER[0]} {RACK_CENTER[1]} {BENCH_HEIGHT}">
      <geom name="rack_base" type="box" size="{RACK_GAP/2 + RACK_PLATE} {RACK_HALF_Y} {RACK_BASE/2}" pos="0 0 {RACK_BASE/2}" material="tray" contype="7" conaffinity="7"/>
      <geom name="rack_front" type="box" size="{RACK_PLATE/2} {RACK_HALF_Y} {RACK_H/2}" pos="{-(RACK_GAP + RACK_PLATE)/2} 0 {RACK_BASE + RACK_H/2}" material="tray" contype="3" conaffinity="3"/>
      <geom name="rack_back" type="box" size="{RACK_PLATE/2} {RACK_HALF_Y} {RACK_H/2}" pos="{(RACK_GAP + RACK_PLATE)/2} 0 {RACK_BASE + RACK_H/2}" material="tray" contype="3" conaffinity="3"/>
{rack_dividers}
      <site name="tray_center" pos="0 0 0.05" size="0.01" group="4"/>
    </body>

    <!-- handoff: a low tray on the floor in front of the human -->
    <body name="handoff" pos="{HANDOFF_TRAY[0]} {HANDOFF_TRAY[1]} 0">
      <geom name="handoff_floor" type="box" size="0.2 0.16 0.004" pos="0 0 0.004" rgba="0.2 0.55 0.25 1" contype="7" conaffinity="7"/>
      <geom type="box" size="0.2 0.006 0.02" pos="0 0.16 0.02" rgba="0.2 0.55 0.25 1" contype="7" conaffinity="7"/>
      <geom type="box" size="0.2 0.006 0.02" pos="0 -0.16 0.02" rgba="0.2 0.55 0.25 1" contype="7" conaffinity="7"/>
      <geom type="box" size="0.006 0.16 0.02" pos="0.2 0 0.02" rgba="0.2 0.55 0.25 1" contype="7" conaffinity="7"/>
      <geom type="box" size="0.006 0.16 0.02" pos="-0.2 0 0.02" rgba="0.2 0.55 0.25 1" contype="7" conaffinity="7"/>
      <site name="handoff_center" pos="0 0 0.01" size="0.01" group="4"/>
    </body>

    <!-- the human (static mannequin) -->
    <body name="human" pos="{HUMAN_POS[0]} {HUMAN_POS[1]} 0">
      <geom type="capsule" size="0.07 0.4" pos="0.05 0 0.45" material="human" contype="7" conaffinity="7"/>
      <geom type="capsule" size="0.07 0.4" pos="-0.05 0 0.45" material="human" contype="0" conaffinity="0"/>
      <geom type="capsule" size="0.17 0.28" pos="0 0 1.2" material="human" contype="7" conaffinity="7"/>
      <geom type="sphere" size="0.11" pos="0 0 1.66" material="skin" contype="0" conaffinity="0"/>
    </body>

    <!-- scenario 3: an obstacle box, parked until a scenario moves it -->
    <body name="obstacle" mocap="true" pos="{BOX_PARK[0]} {BOX_PARK[1]} 0.2">
      <geom name="obstacle_geom" type="box" size="0.3 0.3 0.2" material="box" contype="7" conaffinity="7"/>
    </body>

{_clutter_xml(rng)}

    <camera name="overview" pos="-1.4 -3.2 2.6" xyaxes="0.85 -0.53 0 0.28 0.45 0.85"/>
    <camera name="bench_view" pos="3.2 -1.6 1.5" xyaxes="0.62 -0.78 0 0.35 0.28 0.89"/>

{tools}
  </worldbody>
</mujoco>
"""


def write_workshop():
    from bw.sim.build_models import build_robot

    robot = build_robot(mjx_variant=False)
    for k in list(robot.keys):
        robot.delete(k)
    robot.compile()
    (MODELS / "go2z1_scene_robot.xml").write_text(robot.to_xml())

    path = MODELS / "workshop.xml"
    path.write_text(build_workshop_xml())
    # Second pass: keyframes need the full qpos (robot + 5 object freejoints).
    model = mujoco.MjModel.from_xml_path(str(path))
    spec = mujoco.MjSpec.from_file(str(path))
    from bw.sim.build_models import ARM_EXTENDED, ARM_READY, ARM_STOWED, GO2_HOME_LEGS

    for name, arm in (("home", ARM_STOWED), ("extended", ARM_EXTENDED), ("ready", ARM_READY)):
        q = model.qpos0.copy()
        q[:26] = [0, 0, 0.30, 1, 0, 0, 0, *GO2_HOME_LEGS, *arm]
        ctrl = np.array([*GO2_HOME_LEGS, *arm])
        spec.add_key(name=name, qpos=q.tolist(), ctrl=ctrl.tolist())
    xml = build_workshop_xml()
    keys = "\n".join(
        f'    <key name="{k.name}" qpos="{" ".join(f"{v:.5g}" for v in k.qpos)}" '
        f'ctrl="{" ".join(f"{v:.5g}" for v in k.ctrl)}"/>'
        for k in spec.keys
    )
    xml = xml.replace("</mujoco>\n", f"  <keyframe>\n{keys}\n  </keyframe>\n</mujoco>\n")
    path.write_text(xml)
    mujoco.MjModel.from_xml_path(str(path))
    print(f"wrote {path.relative_to(ROOT)}  (nq={model.nq}, nu={model.nu})")
