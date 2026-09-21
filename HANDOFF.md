# HANDOFF — "Bring me the 10mm wrench"

A complete account of the project as of **2026-09-21, end of session 8**: what exists, what was
built, what failed and why, and what to do next.

This document is written for someone with no prior context. `STATUS.md` is the live working
state and is more detailed on current numbers; `docs/STATUS_archive.md` holds the session
narratives; `REMAINING.md` is the dependency-ordered task list. **This file is the overview that
ties them together.**

---

## 1. What the project is

A quadruped (Unitree Go2) with a 6-DoF arm (Unitree Z1), in MuJoCo, takes a spoken-style command
— *"bring me the 10mm wrench"* — walks to a tool rack, grounds the named tool with a
vision-language model, grasps it, walks to a destination, and places it or hands it to a person.
It recovers from a missing tool, a dropped tool, an ambiguous name, a blocked route and a change
of mind.

**Constraint that shaped everything:** one 6 GB consumer GPU (RTX 3060 laptop), no API keys, no
cloud, no cluster. Every model runs locally. This is why the VLM is a 2B model, why only one GPU
job runs at a time, and why there is a job-queue runner instead of a scheduler.

**Scope honesty:** simulation only. Nothing has run on hardware and no claim is made that it
would transfer.

---

## 2. State at a glance

| layer | state | evidence |
|---|---|---|
| Grasp, on legs, 5 tools | **125/125** | `scripts/try_grasp.py 25 --walk` |
| Place, on legs | **120/125** (table A), **124/125** (table B) | `scripts/try_place.py` |
| Grounding `locate()`, oracle-scored | miss **2.8%**, median xy **10.7 mm** | `scripts/bench_locate.py` |
| Grounding, absent tool, real VLM | **9/10** — but nominal drops to **3/10** | session 8 |
| End to end, 7 suites × 10 seeds | **60/70 (86%)** | `ops/run_eval_suites.sh` |
| Learned policy (SmolVLA), held out | **grasp 18/20, correct object 20/20** | `ops/run_eval_vla.bat` |
| Locomotion gate 2 (0.12 m step, arm extended) | **FAIL** on the deployable policy | `ops/run_phase1_gate.sh` |
| Demonstration data | 1,128 episodes / 104k frames, LeRobot v3.0 | `scripts/collect_demos.py` |
| Media | 146 s montage, 5 tools + 5 recovery modes | `media/montage.mp4` |

**The single bottleneck:** the base falls on the 12 cm step DOWN off the walkway while carrying a
tool. 8 of the 10 end-to-end failures. Nothing else loses a trial to its own skill.

---

## 3. Architecture

```
"bring me the 10mm wrench"
        |
        v
  orchestrator (bw/orchestrator.py) — a state machine on OBSERVABLE GATES
        |   gates: base at station? point returned? gripper holding? gripper empty?
        +--> navigate_to   Layer 3: RL walking policy (numpy, 50 Hz)       bw/locomotion/
        +--> locate        Layer 1: Qwen3-VL-2B point -> wrist depth -> 3D bw/perception/
        +--> grasp/place   Layer 2: SmolVLA, or the scripted demonstrator  bw/manip/, bw/policy/
        +--> ask_human     when the command is ambiguous or the tool is gone
```

The central design decision: **the orchestrator never advances on "the script probably worked".**
Every transition is gated on something observable in the simulator — the base's distance to the
station, whether `locate()` returned a point, whether the gripper reports contact. This is why
recovery behaviours were cheap to add later: a dropped tool is just "gripper empty when it should
not be".

Four layers, one contract each. Layers are swappable: Layer 2 accepts either the scripted
demonstrator or the learned policy behind the same interface, which is what made the VLA
measurable against a known-good baseline.

---

## 4. What has been built

### Layer 0 — simulation and scene
One workshop scene: a bench at 75 cm, a tool rack with five tools, two tables with named place
zones, floor clutter, a human with a handoff tray, and **one 12 cm raised walkway** (x 1.3–4.4)
that every trip to the bench crosses twice — up on the way out, down on the way back with the
tool. That step is deliberate, and it became the project's hardest problem.

Physics settled by measurement, not by default: timestep 0.001 and a PYRAMIDAL friction cone,
both chosen after a solver sweep (`scripts/_creep_probe.py`).

### Layer 1 — grounding
`point()` returns a pixel from Qwen3-VL-2B; `locate()` lifts it to 3D through the wrist depth
buffer. The model runs in its own process (a server on port 8765) because a 2B VLM and a physics
sim cannot share 6 GB comfortably.

**T0.3, answered by measurement:** the VLM is at **chance** distinguishing a 10 mm from a 13 mm
wrench by size, and **92.9%** by a coloured grip band. The project therefore carries the
distinction with a colour band plus a size→colour lookup. This is a stated limitation, not a
hidden one.

### Layer 2 — manipulation
A scripted demonstrator that grasps all five tools 125/125 on legs. Three non-obvious things were
required, each found by tracing failures:
- **the creep** — a slow final approach; without it, "dropped" was 8 of 10 failures
- **radial grasp for the tape roll** — grip the crown, not the side
- **stand-lock** — the legs hold still during the grasp

### Layer 3 — locomotion
A walking policy trained in MJX/brax, exported to a **numpy controller** that runs at 50 Hz with
no JAX dependency at inference. Verified element-wise against the MJX environment: action
difference **1.8e-6**. A navigator drives it to stations, with stall detection, a local planner,
and a `via` waypoint mode.

### The orchestrator and recovery
Five recovery scenarios, each a scored suite rather than a demo: missing tool (10/10), mid-carry
drop (9/10), ambiguous name (8/10), blocked route (10/10), retarget mid-task (6/10).

### Data and the learned policy
1,128 demonstration episodes / 104k frames in LeRobot v3.0 format, with instructions paraphrased
per episode. SmolVLA fine-tuned on them — see §5.3, which is the most instructive part of the
project.

### Operations
A **job-queue runner** (`ops/queue_runner.ps1`) that exists because an agent's background shell
gets reaped under memory pressure — it killed the same fine-tune twice at steps 3,900 and 4,700
of 6,000. The rule that came out of it: *the agent must never own a long job.* It enqueues by
writing a file; a runner started once by a human executes them one at a time, waiting for a cool
and free GPU.

---

## 5. What failed, and why

This is the section worth reading. Failures are grouped by whether they were fixed, remain open,
or were failures of *process* rather than code.

### 5.1 Failures that were found and fixed

| failure | mechanism | fix |
|---|---|---|
| Grasp 8/25 on legs | the arm reached at full speed and knocked tools | **the creep** — slow final approach |
| tape_roll 3/8 | grasp point was 5 mm **below the rack plate tops** — a geometry bug, not control | raise the grasp point; grip radially at the crown |
| screwdriver 0/8 | needs **57 mm** of lift to clear the plates, escaped at **49 mm** — 8 mm short | rack/lift geometry |
| Obstacle recovery 0/2 | three separate defects | stall-direction estimate, local planner, `via` mode, burst cap → **10/10** |
| pliers grounding 0/7 | the VLM could not name them | the phrase **"red pliers"** → 32/36; miss rate 11% → **2.8%** |
| Suite numbers depended on suite ORDER | trials share a process; the sim and rng carried over. `transfer,drop` scored drop 6/10 where `drop` alone scored 9/10 — same code, same seeds | **one process and one reseeded rng per trial** |

Three hypotheses about the screwdriver were tested and **refuted** — jaws closing on the rack
plates, handle roll, and grasp height/squeeze depth (a full sweep gave 0/8 on all six cells).
They are recorded so nobody repeats them.

### 5.2 The locomotion failure — open, and the project's main blocker

The 12 cm step down is the bottleneck. It **is** solvable — in a policy that cannot be deployed.

| policy | gate 2 @ 0.12 m | turn / back / sidestep | end to end |
|---|---|---|---|
| original (`stairs_run7`) | 19 falls / 20 | yes | 60/70 |
| `payload_l5b` (forward-only curriculum) | **0 falls / 20** | **no** | cannot ship |
| `payload_nav_l5b` (curriculum + full command set) | 12 falls / 20 | yes, within 8% | **28/70 — reverted** |

Step-height sweep, falls per 20 — the most publishable table in the project:

| height | 0.06 | 0.08 | 0.10 | 0.11 | 0.12 | 0.13 |
|---|---|---|---|---|---|---|
| original | 0 | 3 | 14 | 18 | **19** | 20 |
| `payload_l5b` | 0 | 0 | 0 | 0 | **0** | 5 |
| `payload_nav_l5b` | 0 | 0 | 1 | 8 | **12** | 12 |

**Why it failed:** training the same level-5 curriculum under the *full* command distribution
(turn, back up, sidestep) bought the **best disturbance rejection in the project** — 320 N push
recovery with the arm extended, against the original's 200 N — and did **not** buy the step.
Disturbance rejection and step descent are separable capabilities. The forward-only lineage
learned the step because nearly all of its experience was crossing it.

**What was tried and must not be repeated:** `stow_arm()` during transit (needed, kept), a
via-waypoint past the step (17/20, vs 16/24 crossing at full speed), tucking the tool further in
(7/12), clockwise-only turns (needed), and now training the step curriculum directly in the nav
lineage (28/70).

### 5.3 The VLA — a negative result, diagnosed until it turned positive

SmolVLA fine-tuned on 1,128 demonstrations scored **1/20**, then **0/20**. Rather than spend GPU
hours, each hypothesis was eliminated by a measurement:

| hypothesis | test | verdict |
|---|---|---|
| Not enough capacity / broken pipeline | overfit control: 40 episodes seen 11.7× | **ruled out** — 3.8× better than baseline |
| Wrong action representation | `--delta` records `q_cmd − q_state` | real, not sufficient |
| Covariate shift / generalisation | evaluated on *training* scenes | **ruled out** — still 0/20 |
| Image domain gap | PSNR 39 dB train vs eval | ruled out |
| Chunk horizon | swept `n_action_steps` | ruled out — 10 is correct |
| Legs free during rollout | no demonstration ever had them free | real, fixed |
| **Action scale** | per-joint motion measured against the normaliser | **this was the cause** |

**Root cause, two independent parts.** The transport swing accounted for **91% of joint 1's
squared motion** (27–28 mrad/tick, against 3–5 in the approach that actually decides the grasp),
drowning the signal. And pick and place shared one normaliser while place carries **2.4 rad IK
jumps** in a single 10 Hz tick (j1 delta std 149 mrad vs 10.3 for pick), so place set the divisor
for both.

**Three corrections, all needed:** train on the **fine phase** only; **pick only**; **quantile**
normalisation instead of mean/std.

```
                      before      after
grasp, held out        0/20       18/20
correct object          —         20/20
j1 error           0.0289 rad   0.0067 rad     (~15 mm -> ~3.5 mm at the gripper)
gripper channel    0.0095       0.0011         (9x better than "do nothing")
```

**Two traps inside this result.**
- **Training loss is blind here.** It fell 0.438 → 0.065 in the run that worked, and looked the
  same in runs that scored 1/20. `scripts/vla_replay_check.py` scores the policy on its own
  training frames through the serving path against a "command no motion" baseline — that is what
  loss does not tell you.
- **The ratio-to-baseline metric is the wrong yardstick.** j1 sat at parity with "do nothing"
  while the policy grasped 5/10, because what changed was the *absolute* error. **Read millimetres
  against the task tolerance, not ratios.**

**What the VLA still does not do:** transport is scripted (the policy is handed the pre-grasp pose
and does approach/close/lift with 7 s of control); place is 0/20 **by construction**, excluded
from training; and the **second grasp after a tool swap is 0/20** while the first is 18/20 in the
same trials — unexplained.

### 5.4 Grounding — a trade-off, not a fix

Adding an absent-tool verification tier fixed the false positives and created false negatives:

| | oracle | real VLM |
|---|---|---|
| missing tool reported | 10/10 | **9/10** |
| nominal delivery | 10/10 | **3/10** |

All seven nominal failures are `locate()` calling a **present** tool absent. A ~70%
false-positive rate on absent tools became a ~70% false-negative rate on present ones. Median
`locate` latency rose to 9.3 s on nominal against 5.4 s on missing — the model exhausts its scan
views before giving up. **The operating point needs moving; the mechanism is sound.**

### 5.5 Failures of process — the most transferable lessons

**`exit 0` means nothing on this machine.** Five separate things reported success while doing
nothing or the wrong thing:
1. `eval_suite.py` had not parsed since a commit that dedented a block out from under an `if` —
   one job ran **0 of 20 trials** and was filed `done exit=0`. It would have taken the seven
   end-to-end suites with it.
2. Two locomotion jobs were no-ops filed the same way; one was silently re-run 90 minutes later.
3. The montage's closing card had been **silently dropping its subtitle** for every cut ever
   made, because `drawtext` expands `%` and discards the whole string on failure ("2.8% miss").
   Its title also overflowed the frame and was clipped at both edges.
4. `run_eval_vla.bat`'s 40 s server wait **never waited** — GNU `timeout` shadows Windows' on
   PATH, *and* Windows' own refuses to run under redirected stdin. It passed only because
   PowerShell's startup cost accidentally supplied the delay.
5. The queue runner filed a job's deliberate `exit /b 1` as `done exit=0`.

**Read the log, never the exit code.** Fixing the runner's exit-code capture is the single
highest-value ops task available.

**A cause label can hide two causes.** `nav_dest_failed` is written by two branches of the
orchestrator — one guarded by `not loco.is_stable()` (a fall), one for a walk that ends stable but
short. Both store the same string. For five sessions the project's headline claim rested on that
label. It turned out to be **right**, but it could not have been *known* from the label; an
interim analysis in session 8 argued the opposite from base positions and was wrong. The trial row
now records `stable`, `fell_any` and the per-leg `nav` log. **When a claim matters, instrument it.**

**A training metric can score a different task than the gate.** The success EMA of a stairs run
scores the level-5 task (0.130 m); the gate is 0.120 m. Reading the EMA produced a wrong
"negative" call on a run that had actually passed.

**A plausible story is not evidence.** `payload_nav_l5b` failed gate 2 and was swapped in anyway,
on the reasoning that the gate tests the arm extended while the pipeline stows it. End-to-end
collapsed 60/70 → 28/70. **The gate said FAIL and the gate was right.**

---

## 6. Next enhancements, in priority order

### P0 — the step-down (unblocks everything)
This is 86% → mid-90s. The step is confirmed as the binding failure and it is already solved in
`payload_l5b`. The task is getting that robustness into a policy that can also turn. Three
candidates, none tried:
1. **Warm-start the nav curriculum from `payload_l5b`'s weights** rather than from nav3.
2. **Two policies, switched per phase** — the orchestrator already switches per phase, and
   reverting a swap is one file copy. Use `payload_l5b` for the `via_step` leg only.
3. **Skew the command distribution** toward forward motion at level 5 instead of sampling the
   full set uniformly.

### P1 — the VLA's second grasp
18/20 on the first grasp and 0/20 on the second, in the same trials, on the same tools. Not
explained by pick-only training. This is the thread that leads somewhere.

### P1 — the grounding operating point
You have a measured trade-off curve, not a fix. Tune the absent-tool threshold so nominal
recovers from 3/10 without losing missing 9/10. Cheap: the 32 rendered seeds in `runs/t8_views`
re-score it without the GPU.

### P2 — ops hardening
Make the queue runner's exit codes trustworthy (`|| exit /b 1` per line; have the runner trust
`%ERRORLEVEL%`). Everything downstream depends on being able to believe the job log.

### P2 — the tail
- 1 transfer in 10 places the tool outside the zone; the residual is the wrench's post-release
  travel with a walked base. Dropping the 45 mm aim shift was A/B'd and made it **worse**.
- `wrench_13mm` is the only tool with **no** nominal success — seeds 1 and 6 are the two nominal
  failures and both are that tool, reproduced independently twice.
- Retarget is 6/10; 3 of those are the step again.
- ~20 GB of VLA checkpoints in `runs/vla_delta`, `vla_overfit`, `vla_delta_long` should be pruned
  to the evaluated ones. **Deliberately not done by an agent — it is a destructive call.**

### P3 — presentation, which is where the return is
The engineering is stronger than its presentation. In rough order of impact:
- **A narrated 90-second video.** The montage is silent. Talk over the VLA diagnosis — "the loss
  curve was perfect and the robot did nothing; here is how I found out why."
- **A blog post titled around the negative result.** `docs/blog.md` is most of the way there.
- **Make one number reproducible by a stranger in five minutes** — a container or a Colab that
  runs the grasp benchmark. It currently needs three environments and a 6 GB GPU.
- **Contribute the `n_action_steps` finding upstream to LeRobot** (it matches their issue #4614).

### Explicitly NOT worth doing
`ops/queue/hold/` holds three jobs (460/470/480) that would test whether noise-free scripted
demonstrations were what stopped the VLA learning. **That question is answered.** The same dataset
went 0/20 → 18/20 once the action scale was fixed, without touching the data. They are answered,
not deferred — delete them.

---

## 7. How to run it

Three environments, because they cannot be one (`docs/ARCHITECTURE.md` explains why):

| environment | purpose |
|---|---|
| `D:\hexapod\render_venv` (Windows) | CPU sim + **all** rendering — EGL fails inside WSL |
| `~/go2-stairs/.venv` (WSL) | MJX/brax locomotion training + the Phase 1 gate |
| `~/bringwrench/.venv-vla` (WSL) | SmolVLA fine-tune + policy server |

```bash
pip install -e .                                    # makes `bw` importable
bash download_models.sh                             # the trained policies (*.npz are gitignored)
python -m bw.sim.build_models                       # regenerate the four model XMLs

python scripts/try_grasp.py 25 --walk               # grasp benchmark: expect 125/125
python scripts/eval_suite.py --suite all --n 10     # the seven end-to-end suites
python scripts/make_orch_video.py 3 out.mp4 --suite nominal   # render a run
```

Long GPU jobs go through the queue, never a shell:

```
powershell -NoProfile -ExecutionPolicy Bypass -File ops\queue_runner.ps1
# then drop a .cmd into ops/queue/pending/ — filename order decides when it runs
```

**Note on filename order:** the runner sorts with PowerShell's culture-aware comparison, which
**ignores hyphens** — `455b-gate` sorts *before* `455-eval`. Check the ordering, do not reason
about it.

---

## 8. Traps — do not re-derive these

* The VLA's training loss is blind. Use `vla_replay_check.py`, and read **millimetres**, not ratios.
* `n_action_steps` stays at 10. 50 costs ~20 points; 1 and 10 are equivalent.
* One process per suite — trials are not independent inside a process.
* The stairs success EMA scores the level-5 task (0.130 m), not the gate (0.120 m).
* Windows `timeout /t` cannot be used in a queued job. Use `ping -n N+1 127.0.0.1`.
* `brax` writes a checkpoint as a **file**, not a directory — guard with `[ -e ]`, not `[ -d ]`.
* A run's `checkpoints/final` is often **not** its best policy. `scripts/pick_best_ckpt.py`
  chooses from the run's own eval rows. `payload_l5b` peaked at step 1.3M and declined for the
  next 4.6M steps.
* The `## Known bugs` section of `STATUS.md` is **partly historical** — the screwdriver and
  tape_roll entries describe bugs that were later fixed (grasp is now 125/125). Read it as a
  record of investigations, not a current defect list.

---

## 9. Repository layout

```
bw/sim/             the workshop scene and the MuJoCo wrapper (WorkshopSim)
bw/locomotion/      Layer 3: MJX/brax training, the numpy controller, the navigator
bw/manip/           scripted grasp / place / handoff demonstrator + IK
bw/perception/      Layer 1: vlm.point() (model in its own process) and locate()
bw/policy/          SmolVLA inference server + the skill client
bw/task/            the one success spec, and the instruction paraphrases
bw/orchestrator.py  the state machine
scripts/            benchmarks, collectors, evaluation suites, video renderers
ops/                run scripts + the job queue runner
docs/               ARCHITECTURE.md, blog.md, STATUS_archive.md, REPO_LAYOUT_PLAN.md
media/              rendered clips and the montage
```

Branch convention: one branch per feature, merged into a `session-N` branch. Measurements go in
the commit message — the git log is part of the record, not just the code.
