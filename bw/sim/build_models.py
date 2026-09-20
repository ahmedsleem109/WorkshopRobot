"""Build the Go2 + Z1 embodiment and the workshop scene from the upstream Menagerie models.

    python -m bw.sim.build_models            # writes models/go2z1_mjx.xml, go2z1.xml, workshop.xml

Two robot variants come out of one attachment, because the two consumers need different
things from the arm:

* ``go2z1_mjx.xml`` -- locomotion training under MJX. Every arm geom is contact-free.
  The arm only matters to the legs through its mass, inertia and the reaction torques
  of its motors; arm-terrain contact only ever happens after a fall. Mesh collisions
  are also the slowest thing MJX can be asked to do, and 18 convex gripper hulls would
  multiply the contact array across 4096 envs.
* ``go2z1.xml`` -- CPU MuJoCo for manipulation, data collection and the full pipeline.
  Gripper pads and hulls collide; arm links do not collide with the trunk they sit on.

Both switch the integrator to ``implicitfast``. The Go2 MJX model ships with Euler and
eulerdamp disabled, which is fine for its kp=50/kd=0.5 legs but NOT for the Z1's
kp=1000/kd=100 position servos: kd*dt/I on the wrist links is ~10, far past explicit
stability. implicitfast integrates actuator damping implicitly and is supported by MJX.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

# Mount: slightly forward of the trunk CoM, on a 12 cm aluminium pedestal.
# Without the pedestal the Z1 shoulder sits at 0.55 m world height when the Go2 stands on
# the 12 cm walkway -- 20 cm BELOW the 75 cm bench top -- and the only reachable tray poses
# approach almost horizontally, which cannot pick a flat wrench off a tray floor
# (scripts/find_scan_pose.py: 7/48 scan targets reachable, all at 63 deg tilt). The trunk
# collision box half-height is 0.057 m.
PEDESTAL_H = 0.12
PEDESTAL_MASS = 0.4
ARM_MOUNT = (0.06, 0.0, 0.057 + PEDESTAL_H)

# Joint configurations (joint1..6, gripper). Found by FK in scripts/check_embodiment.py:
# STOWED folds link02 back along the trunk and link03 forward over it, keeping the arm's
# CoM within ~4 cm of the mount; EXTENDED reaches ~0.55 m forward at trunk height.
# Gripper command is now the finger travel of a PARALLEL jaw: 0 = closed, FINGER_TRAVEL = open.
FINGER_TRAVEL = 0.038
# Finger servo stiffness, N/m; the squeeze is GRIP_KP x how far the command overshoots
# contact. 1200 gave ~20 N per pad and the 13 mm wrench still crept through the jaws at a
# median 55 mm/s (one in four slid out entirely) with the arm held perfectly still. 4000
# gives ~60 N -- ordinary for an off-the-shelf parallel gripper -- and 3.4 mm/s (max 7);
# tape roll 1.6 -> 0.5 mm/s; every grasp still closes (scripts/_grip_force_probe.py,
# 2026-09-19). The pliers' ~8 mm/s does not respond to force at all: a separate mechanism.
GRIP_KP = 4000.0
GRIP_OVERSHOOT = 0.008
# Jaw-pad contact time constant, = 1*model timestep (workshop.xml runs the MuJoCo default
# dt=0.002). MEASURED 2026-09-18: raising this to 0.004 (the usual ">= 2*dt" guidance) did NOT
# fix the "Nan/Inf in QACC at DOF 29-48" warnings -- they recurred at the IDENTICAL episode
# times -- and cost 7 points of grasp success (57% -> 50%, tape_roll 6/8 -> 3/8). The pad
# contact is therefore NOT the instability source; do not re-try this without new evidence.
PAD_SOLREF_T = 0.002
ARM_STOWED = (0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0)
ARM_EXTENDED = (0.0, 1.35, -0.45, -0.9, 0.0, 0.0, 0.03)
ARM_READY = (0.0, 0.9, -1.2, 0.3, 0.0, 0.0, 0.035)

GO2_HOME_LEGS = (0.0, 0.9, -1.8) * 4


def _base_spec() -> mujoco.MjSpec:
    go2 = mujoco.MjSpec.from_file(str(MODELS / "go2_mjx_upstream.xml"))
    z1 = mujoco.MjSpec.from_file(str(MODELS / "z1_gripper_upstream.xml"))
    # Keyframes are rebuilt for the combined model below; child keys would not fit its qpos.
    for k in list(z1.keys):
        z1.delete(k)
    for k in list(go2.keys):
        go2.delete(k)

    # Integrator: Go2's own Euler, but with eulerdamp ENABLED (menagerie's MJX model disables
    # it). implicitfast was tried first and diverged 26% of MJX episodes in the LEGS even with
    # the arm stowed and DR off (scripts/stability_mjx.py). Euler + implicit joint damping keeps
    # the legs on the dynamics run 7 was trained on, and the arm's derivative gain is moved
    # from the actuator into joint damping below so eulerdamp integrates it implicitly.
    go2.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    z1.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    go2.option.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    base = go2.body("base")
    ped = base.add_body(name="arm_pedestal", pos=[ARM_MOUNT[0], 0.0, 0.057 + PEDESTAL_H / 2])
    ped.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.045, PEDESTAL_H / 2],
                 rgba=[0.6, 0.62, 0.66, 1], contype=0, conaffinity=0, group=2, mass=PEDESTAL_MASS)
    frame = base.add_frame(pos=list(ARM_MOUNT))
    frame.attach_body(z1.body("link00"), "arm_", "")

    go2.compiler.meshdir = "assets"

    # Reflected rotor inertia of the Z1's geared joints (menagerie omits it). Without it the
    # gripper servo (kp=1000, kd=100 on a 0.3 g*m^2 jaw) saturates its 30 N force clamp, loses
    # implicit damping, and sits in a +-30 N limit cycle at ~20 rad/s: the jaw never closes.
    # Gravity compensation on the arm links, as the Z1's own controller does. With servo gains
    # low enough to be stable (below) the uncompensated sag is ~0.04 rad at joint 3, which
    # left the gripper 2 cm above every grasp target.
    for b in go2.bodies:
        if b.name.startswith("arm_link") or b.name == "arm_gripperMover":
            b.gravcomp = 1.0
    for j in go2.joints:
        if j.name.startswith("arm_joint"):
            j.armature = 0.08
    # Arm servos: kd*dt/I must stay < ~0.5 so a servo that saturates its force clamp (and so
    # drops out of the implicit integrator's damping) is still explicitly stable. Menagerie's
    # kd=100-150 gives kd*dt/I ~2-3 and diverged 16% of MJX training episodes
    # (scripts/stability_mjx.py). kp is lowered with it to keep the loop well damped.
    for i in range(1, 7):
        act = go2.actuator(f"arm_motor{i}")
        kp, kd = (2000.0, 30.0) if i == 2 else (1500.0, 20.0)
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biasprm[2] = 0.0
        go2.joint(f"arm_joint{i}").damping = [kd, 0.0, 0.0]

    # ---- parallel-jaw gripper -------------------------------------------------------------
    # The Z1 ships a single-DOF ROTARY jaw: one fixed lower jaw plus one jaw swinging on an
    # arc, so its pads are never parallel. Measured consequence on standing/lying tools: the
    # pads catch an edge, the swinging jaw bottoms out on the fixture before it closes
    # (q ~ -0.6 with no pad contact), and a gripped tool is levered out of the jaws by the
    # wedge as it squeezes (scripts/try_grasp.py, ~30 configurations). Replaced with a
    # symmetric parallel-jaw hand -- an off-the-shelf swap on a real Z1 too. One actuator
    # drives finger A; finger B mirrors it through a JOINT equality, so the gripper is still
    # a single command and the action space is unchanged (12 legs + 6 arm + 1 gripper).
    go2.delete(go2.body("arm_gripperMover"))     # removes its joint and actuator with it
    link06 = go2.body("arm_link06")
    for sign, tag in ((1.0, "a"), (-1.0, "b")):
        f = link06.add_body(name=f"arm_finger_{tag}", pos=[0.10, 0.0, 0.0], mass=0.09,
                            ipos=[0.045, 0.0, 0.0], inertia=[4e-5, 4e-5, 4e-5])
        j = f.add_joint(name=f"arm_finger_{tag}", type=mujoco.mjtJoint.mjJNT_SLIDE,
                        axis=[0, 0, sign], range=[0.0, FINGER_TRAVEL], damping=[8.0, 0, 0],
                        armature=0.004)
        f.add_geom(name=f"finger_{tag}_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.026, 0.013, 0.005], pos=[0.045, 0.0, sign * 0.005],
                   rgba=[0.15, 0.15, 0.17, 1], condim=6, friction=[2.0, 0.25, 0.02],
                   # Stiff, barely-penetrating contact. With MuJoCo's default softness the
                   # pads sank 3.5 mm into a 13 mm handle under 25 N and extruded the tool
                   # out of the jaws during the retreat.
                   solref=[PAD_SOLREF_T, 1.0], solimp=[0.995, 0.9995, 0.0002, 0.5, 2.0],
                   contype=2, conaffinity=2, group=2)
        f.add_geom(name=f"finger_{tag}_back", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.012, 0.013, 0.012], pos=[0.008, 0.0, sign * 0.012],
                   rgba=[0.55, 0.57, 0.6, 1], contype=0, conaffinity=0, group=2)
    # One position servo PER finger, both driven by the same gripper command. A single
    # actuator plus a joint equality was tried first: the driven finger's grip force then
    # arrives through the constraint solver, which is compliant, and gripped tools were
    # measured slipping out during the retreat (lift 0.14 m, then dropped).
    for tag in ("a", "b"):
        go2.add_actuator(name=f"arm_motorGripper_{tag}", target=f"arm_finger_{tag}",
                         trntype=mujoco.mjtTrn.mjTRN_JOINT,
                         gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                         biastype=mujoco.mjtBias.mjBIAS_AFFINE, gainprm=[GRIP_KP] + [0] * 9,
                         biasprm=[0, -GRIP_KP, -25] + [0] * 7, forcerange=[-150, 150],
                         # The command may go 15 mm PAST contact, as a real gripper's
                         # position/force command does: the servo then squeezes at
                         # kp * overshoot (~27 N) instead of the ~9 N it produces when the
                         # target is exactly zero, which was letting tools rotate and squirt
                         # out of the pads during the retreat.
                         ctrlrange=[-GRIP_OVERSHOOT, FINGER_TRAVEL], ctrllimited=True)
    # ee: centred between the pads, 1.5 cm behind the fingertips.
    link06.add_site(name="ee", pos=[0.145, 0.0, 0.0], size=[0.01, 0, 0], group=1,
                    rgba=[1, 0.3, 0.3, 1])
    # Wrist RGB for SmolVLA: above the gripper, looking along the fingers.
    pitch = math.radians(10)
    link06.add_camera(name="wrist", pos=[0.0, 0.0, 0.11], fovy=75,
                      xyaxes=[0, -1, 0, math.sin(pitch), 0, math.cos(pitch)])
    # Head RGB-D for grounding and navigation: on the Go2 nose, pitched 20 deg down.
    pitch = math.radians(20)
    base.add_camera(name="head", pos=[0.33, 0.0, 0.04], fovy=75,
                    xyaxes=[0, -1, 0, math.sin(pitch), 0, math.cos(pitch)])
    base.add_site(name="head_cam_site", pos=[0.33, 0.0, 0.04], size=[0.005, 0, 0], group=4)
    # Mast RGB for SmolVLA (T6): the head camera sits at bench-panel height and sees only the
    # bench front / table legs while manipulating, so the VLA's second view is a mast camera
    # over the arm, pitched down at the bench top (rack and place zones ~0.5 m ahead).
    # Offset to the right of the arm and yawed in, so the upper arm does not fill the view.
    pitch, yaw = math.radians(27), math.radians(11)
    d = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)]
    xa = [d[1], -d[0], 0.0]
    n = math.hypot(xa[0], xa[1])
    xa = [xa[0] / n, xa[1] / n, 0.0]
    ya = [xa[1] * d[2] - xa[2] * d[1], xa[2] * d[0] - xa[0] * d[2], xa[0] * d[1] - xa[1] * d[0]]
    base.add_camera(name="mast", pos=[-0.10, -0.12, 0.75], fovy=70, xyaxes=xa + ya)
    return go2


def _arm_geoms(spec: mujoco.MjSpec):
    for g in spec.geoms:
        body = g.parent
        if body.name.startswith("arm_"):
            yield g


def _add_keys(spec: mujoco.MjSpec, model: mujoco.MjModel):
    qpos0 = [0, 0, 0.30, 1, 0, 0, 0, *GO2_HOME_LEGS]
    for name, arm in (("home", ARM_STOWED), ("extended", ARM_EXTENDED), ("ready", ARM_READY)):
        # qpos carries BOTH finger slides (the second mirrors the first); ctrl carries one.
        # qpos and ctrl both carry BOTH finger slides; the 7-vector arm command duplicates
        # the gripper value across them.
        spec.add_key(name=name, qpos=qpos0 + list(arm) + [arm[6]],
                     ctrl=list(GO2_HOME_LEGS) + list(arm) + [arm[6]])


def build_robot(mjx_variant: bool) -> mujoco.MjSpec:
    spec = _base_spec()
    if mjx_variant:
        # MEASURED (scripts/solver_sweep.py): menagerie ships iterations=1, which is fine for a
        # 15 kg Go2 but diverged 24.7% of MJX training terminations once the arm put 19.9 kg over
        # a higher CoM. it=4 / ls=10 gives 0.0% at 2,030 steps/s; it=2 still leaves 5.2%.
        spec.option.iterations = 4
        spec.option.ls_iterations = 10
    for g in _arm_geoms(spec):
        if mjx_variant or g.classname.name.endswith("visual"):
            g.contype = 0
            g.conaffinity = 0
        else:
            # Gripper and link collisions on, but in their own bit so the arm never
            # self-collides with the Go2 trunk/legs (bit 1) -- only with the world.
            # Jaw pads (boxes) grip tools: bit 2. Housing hulls and link cylinders: bit 4, which
            # only static furniture carries -- they stop the arm going through the bench/tray,
            # but the bulky stator hull no longer lands on a tool before the pads reach it.
            is_pad = (g.name or "").endswith("_pad")
            is_hull = g.type == mujoco.mjtGeom.mjGEOM_MESH
            # Gripper housing hulls collide with nothing: measured, they jam the moving jaw on
            # the tray at q ~ -0.7 before the pads reach the tool (scripts/try_grasp.py diag).
            is_struct = (g.name or "").endswith("_back")
            g.contype = 2 if is_pad else (0 if (is_hull or is_struct) else 4)
            g.conaffinity = 2 if is_pad else (0 if (is_hull or is_struct) else 4)
            g.condim = 6 if is_pad else 3
            g.friction = [2.0, 0.25, 0.02] if is_pad else [0.8, 0.02, 0.002]   # rubber jaw pads
            if is_pad:
                g.solref = [PAD_SOLREF_T, 1.0]
                g.solimp = [0.995, 0.9995, 0.0002, 0.5, 2.0]
    model = spec.compile()
    _add_keys(spec, model)
    return spec


def write(spec: mujoco.MjSpec, path: Path):
    spec.compile()
    path.write_text(spec.to_xml())
    print(f"wrote {path.relative_to(ROOT)}")


def main():
    write(build_robot(mjx_variant=True), MODELS / "go2z1_mjx.xml")
    write(build_robot(mjx_variant=False), MODELS / "go2z1.xml")
    from bw.sim.workshop import write_workshop

    write_workshop()


if __name__ == "__main__":
    main()
