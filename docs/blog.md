# Five bugs between "it works" and it working

*Building a language-commanded mobile manipulator in simulation on one 6 GB laptop GPU.*

The demo is one sentence: you type **"bring me the 10mm wrench"**, and a Unitree Go2 with a Z1 arm
walks to a workbench, picks the right wrench out of a rack, carries it across a 12 cm step and
hands it over. Three layers do the work — an RL walking policy, a fine-tuned SmolVLA for the arm,
and a hand-written state machine that calls a VLM to answer *where is it?*

None of that is the interesting part. The interesting part is that **every single time a number
was bad, the cause was geometry or bookkeeping, and never the thing I expected.** Here are five.

## 1. The success metric was lying (69% → 31%)

The first grasp benchmark scored the tool at the instant the arm stopped moving. It read 69%.
Holding the arm still for two more seconds dropped it to 31%: the tools were *creeping* through
the jaws the whole time, a millimetre at a time. The fix was in the contact model — a pad-contact
time constant of exactly one timestep — but the lesson is the benchmark: **score after a hold, not
at the end of the motion.** Every grasp number from before that change is uncomparable and is
marked as such in the repo.

## 2. The robot was standing on a number, not on its legs

For a long time the manipulation benchmarks ran with the base *pinned* — a kinematic stand-in for
the locomotion policy. When the legs were finally handed to the policy for the whole episode, the
tape roll's grasp fell from 21/25 to 8/25.

Two measurements explained it. First, a standing robot is **not a fixed frame**: reaching into the
rack pushes the base back 25–35 mm, so the jaws close short. Second — and this was the real bug —
the ring was being gripped 45° up its rim with a sideways-closing jaw, which puts the *corner of
the 26 mm pad* into the ring's crown. The pad tip hit the tape at **310 N during the approach**,
which a pinned base absorbs silently and a standing robot does not: it gets shoved backwards.

Grasping the ring **radially at its crown** — one pad inside the hole, both pads flat on the 7 mm
wall — took it to 25/25, and removed the swing on lift as a free bonus: held at the top, the ring
already hangs under its grip point.

The attempted fix that did *not* work is worth recording: a closed-loop re-servo of the gripper
before closing. With a compliant base it chases the contact it is itself creating — the robot
walked itself 150–250 mm backwards trying to reach a target that kept receding.

## 3. The screwdriver was being stood on its tip

Placement had one bad tool: the screwdriver, 16/25 into the zone, toppling 10–16 cm in a random
direction every time. The place skill rolls the wrist so the held tool "hangs down". The
screwdriver is gripped by its handle and stands handle-down in the rack — so "hang down" meant
**flipping it 180° and standing it on the tip of its shaft.** Placing it upright, the way it
stood in the rack, gives 24/25 and 25/25 on the two tables.

## 4. The hand-off was a 50 cm drop

The scene put the human's hand-off tray on the floor. With a level grip the arm cannot get below
~0.5 m over a tray 0.55 m ahead, so the tool was released from half a metre up: the screwdriver
bounced out, and the tape roll stayed hooked on a finger. Raising the tray onto a 45 cm stand — a
hand-off height, which is what a person would actually offer — fixed it.

A second bug hid underneath: the Cartesian move to the tray sent the incremental IK onto a
different solution branch, and the arm *came back toward the body*, dropping the tool 30–40 cm
short. Moving in joint space to a multi-start IK solution fixed that.

## 5. The perception module was rejecting its own targets

`locate()` back-projects the VLM's pixel through the wrist depth buffer, and it discarded anything
nearer than 0.25 m as "that's the gripper". Tools at the ends of the rack sit **0.22 m** from the
lens. The gripper itself reads 0.105 m. One constant, and a class of targets that could never be
found.

## What the final numbers say

| | |
|---|---|
| Grasp, legs on the walking policy | 125/125 |
| Two-table transfer, walking between stations | 120/125 and 124/125 |
| End-to-end task suite (5 scenarios, 130 trials) | 114/130 (88%) |
| Dominant failure | the base falling while stepping DOWN off the walkway with a tool held (13 of 16) |

That last row is the honest headline. After all the manipulation work, **what limits the system is
locomotion** — a 12 cm step down, taken with a 2 kg arm extended and a tool in the jaws, which the
walking policy's curriculum never trained on. The demonstrator succeeds; the robot falls over on
the way to the person.

## What is still open

- The grounding model does not recognise the pliers (0/7) — the primitive red-V geometry does not
  read as pliers to Qwen3-VL-2B. It is asked about a *visible feature* for the wrenches (a coloured
  grip band) and needs the same treatment here.
- The 10 mm vs 13 mm distinction is carried by a size→colour lookup in our code, because the model
  is at chance on the size itself and 92.9% on the band. Stated, measured, not hidden.
- It has never run on hardware.

Everything above is reproducible from the repo, on fixed seeds, with the failures counted.
