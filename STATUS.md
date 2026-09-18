# STATUS — "Bring me the 10mm wrench"

Last updated **2026-09-18, end of session 2**. Read this, then `REMAINING.md` for what to do
next (tasks are ordered by dependency there, not by phase).

---

## START HERE — next session

**Done this session:** T3 (payload fine-tune, 33M steps) and T4 (gate + ablation) are COMPLETE.
T0 is all but settled. 2 of 13 tasks done, both locomotion.

**Two decisions taken 2026-09-18, do not relitigate:**
1. **Screwdriver dropped from the grasp set** (`GRASP_TOOLS` in `bw/sim/workshop.py`). It stays
   in the scene as a distractor for the grounding model. Four hypotheses refuted, 0/8
   throughout; the rack jam is the untested explanation and the fixture is a poor one.
   Benchmark is now **69%** over 4 tools (was 57% over 5).
2. **Qwen3-VL-2B is the Layer-1 grounding model** (`~/bringwrench/models/qwen3-vl-2b`, 4.0 GB).
   Both Molmo2 mirrors are dead (Known bug #6). Qwen needs NO `trust_remote_code` — it is built
   into transformers 5.5.4 — loads in bf16, and emits TEXT coordinates at ~1.2 s per call.

**Next tasks, in order:**

| # | task | why now |
|---|---|---|
| **T0.3 finish** | Confirm Qwen's coordinate SCALE and score accuracy vs sim ground truth | It replied `(800, 455)` and `(844, 500)` on a **512x512** image — that is Qwen's **0-1000 normalised** convention, NOT pixels. Convert `x_px = x/1000*W` and VERIFY. Also 1 of 5 replies refused ("There are none." for pliers) — check whether the tool was actually in the wrist view before blaming the model. `scripts/test_qwen_point.py` is the harness. |
| **T1** | Grasp 69% -> >=90% per tool | **`dropped` is now 8 of 10 failures** (wrench_13mm 3, pliers 4, wrench_10mm 1). One failure mode, one place to look: the tool leaves the jaws during the lift/retreat. This is the critical path — it blocks T2 -> T6 -> T7 (the SmolVLA fine-tune, the centrepiece). |
| **T5** | Validate `controller.py` against MJX | Small, self-contained, blocks T9, and independent of T1. Verify the observation bit-for-bit against `Go2ArmEnv._single_obs` — note the actor reads the **255-dim `privileged_state`** (see decision 6). |
| **T8** | `locate()` behind `vlm.point()` | Unblocked as soon as T0.3 closes. |

**Unfinished business worth knowing:**
- `media/loco_payload.npz` was dumped but **never rendered** — run `scripts/render_traj.py` on
  Windows for the money shot (the policy crossing the step with the arm extended).
- The push-ablation anomaly (original stowed 160 N < extended 200 N) is REPRODUCIBLE and
  unexplained. **Do not publish that table until it is.**
- `QACC` NaN warnings still appear in CPU grasp runs. They are NOT independent of the jaw
  pads after all: raising `impratio` to 50-100 makes them much more frequent, and that is the
  same conditioning problem as the creep. Worth re-checking now that the timestep is halved.
- The GPU clock cap (`nvidia-smi -lgc 300,1100`, Administrator) does NOT survive a driver
  re-init and lapsed mid-run once. `nvidia-smi -rgc` to release it. `ops/thermal_guard.sh`
  stops training at 88 C as a backstop.

Plan of record: `bring-me-the-10mm-wrench-plan (1).md`. Upstream locomotion project:
`D:\hexapod` on Windows, `~/go2-stairs` in WSL (its `HANDOFF.md` and `plan.md` still apply to
everything about the locomotion policy).

---

## State in one line

Phase 1 is **measured and done** — the payload fine-tune trained (attitude terminations
0.20 -> 0.05) and the gate/ablation ran, with gate 2 failing at 12 cm for a geometric reason
that is understood and quantified; Phase 2's demonstrator is at **69% over 4 tools** and is now
the project's critical path; Phase 3 does not exist yet.

---

## What exists

| Component | File(s) | State |
|---|---|---|
| Go2 + Z1 embodiment builder | `bw/sim/build_models.py` | works; regenerates all 4 model XMLs |
| Workshop scene generator | `bw/sim/workshop.py` | works |
| CPU sim wrapper (randomization, cameras, gripper state, ground truth) | `bw/sim/workshop_sim.py` | works |
| Arm IK (damped least squares, multi-seed) | `bw/manip/ik.py` | works, 0.6 mm typical residual |
| Scripted grasp demonstrator | `bw/manip/scripted_grasp.py` | **50% — must improve** |
| Payload-aware MJX locomotion env | `bw/locomotion/go2_arm_env.py` | works; not yet trained |
| Leg-only domain randomization + carry load | `bw/locomotion/domain_rand.py` | works |
| Training entrypoint (reuses go2-stairs `train.py`) | `bw/locomotion/train_payload.py` | works; launches and learns (see "Session 2" below) |
| Phase 1 gate + push-recovery ablation | `bw/locomotion/eval_phase1.py` | written, **never run** |
| Checkpoint → numpy export | `bw/locomotion/export_policy.py` | written, not run |
| Layer 3 interface (`set_velocity` / `get_base_pose` / `is_stable`) | `bw/locomotion/controller.py` | written, **not validated against MJX** |
| Trajectory dump (WSL) + render (Windows) | `bw/locomotion/dump_traj.py`, `scripts/render_traj.py` | works |
| Grasp benchmark with failure-stage breakdown | `scripts/try_grasp.py` | works |
| Slip/force instrumentation (T1.1), per-sample CSV | `scripts/grasp_diagnose.py` | works; found the creep |

Generated models (all from `python -m bw.sim.build_models`):
`models/go2z1_mjx.xml` (MJX training), `models/go2z1.xml` (CPU), `models/workshop.xml`,
`models/scene_go2z1_flat.xml`, `models/scene_go2z1_step.xml` (written by the env),
`models/go2z1_scene_robot.xml` (robot without keyframes, for the scene include).

Media: `media/grasp_wrench_10mm.mp4` (success), `media/grasp_wrench_13mm.mp4` (failure —
lifts then drops), `media/loco_run7_arm.mp4` (run-7 policy carrying the arm, pre-fine-tune),
`media/check_workshop.png`, `media/rack_check.png`.

---

## Measured numbers (do not re-derive these)

### Simulation stability — the two Phase 1 blockers, both found by measurement

| Question | Measurement | Conclusion |
|---|---|---|
| MJX solver iterations | `scripts/solver_sweep.py`: it=1 → **24.7%** of terminations diverge @4,589 sps; it=2 → 5.2% @3,025; **it=4 ls=10 → 0.0% @2,030 sps**; it=8 → 0% @1,172 | **use iterations=4, ls_iterations=10** for the arm model |
| Is it the arm or the env? | `scripts/stability_baseline.py`: unmodified go2-stairs env + run 7 → **0% divergence** at L1 and L4 | the arm causes it: +33% mass (19.9 kg) over a higher CoM, and menagerie's `go2_mjx.xml` ships `iterations=1` |
| Integrator | `implicitfast` → 1,386 terminations/300 steps; **Euler + `eulerdamp` enabled → 355** | keep Go2's Euler, enable eulerdamp, put the arm's derivative gain in joint damping so it stays implicit |
| Arm servo tuning | menagerie kd=100–150 on ~0.1 kg·m² gives kd·dt/I ≈ 2–3; the gripper limit-cycled at ±30 N / 22 rad/s and never closed | arm joints: armature 0.08, kp 1500 (joint 2: 2000), kd → joint damping 20/30; gravcomp on all arm links |

### Embodiment

- Arm 4.69 kg, robot total **19.90 kg**. Arm CoM relative to base: stowed (−0.02, 0, 0.18), extended (0.13, 0, 0.30).
- Z1 shoulder height standing on the walkway: **0.67 m** (with the 12 cm pedestal; 0.55 m without).
- Reach: all 5 rack slots reachable horizontally from base x ∈ [4.00, 4.12].
- Passive stand holds: base z 0.22–0.24, up·z ≥ 0.987 after 3 s, arm stowed and extended.

### Scripted grasp demonstrator -- 100-episode benchmark, seeds 0-24 (`scripts/try_grasp.py 25`)

**SUCCESS IS NOW SCORED AFTER A 2-SECOND STATIC HOLD** (`HOLD_VERIFY` in `scripted_grasp.py`),
not at the instant the retreat ends. Every number below and in the change log is on that
criterion; numbers from before 2026-09-18 session 3 are NOT comparable, because the old
criterion scored the tool while it was still sliding out of the jaws (see "the creep" below).

| Tool | Success | Failure stages |
|---|---|---|
| wrench_10mm | **24/25** | no_grip 1 |
| wrench_13mm | **25/25** | -- |
| pliers | **24/25** | dropped 1 |
| tape_roll | **24/25** | dropped 1 |
| **overall** | **97/100** | dropped 2, no_grip 1 |

Against the same criterion the session-2 configuration scored **10/32 (31%)**, not the 69% in
the old table. T1's acceptance is >=90% PER TOOL and >=92% overall: **both pass, T1 is closed.**
The three residual failures are one per mechanism and none is systematic.

Four things got it there, in order of size:

| fix | what it was | effect |
|---|---|---|
| timestep 0.001 + pyramidal cone | the pad-contact creep, below | 10/32 -> 26/32 held-2s |
| `TAPE_PHI` 45 deg | tape grasped below the rack plate tops | tape_roll 2/8 -> 8/8 |
| staging waypoint above the pre-grasp | the swing from the scan pose swept the rack and knocked a NEIGHBOURING tool into the target -- 50-68 mm before the jaws arrived (`scripts/_knock_probe.py`) | 92/100 -> 97/100 |
| `settle_static` instead of a fixed 1.2 s | one reset in ~25 still had a tool sliding at 56 mm/s when the grasp was planned, and it moved another 49 mm | (same run) |

### THE CREEP -- why "dropped" was 8 of 10 failures (found 2026-09-18, session 3)

The jaw pads carry `solref` timeconst **0.002 s** while the CPU scene ran at MuJoCo's default
**0.002 s** timestep. MuJoCo requires a contact time constant of **at least 2 x timestep**; at
exactly 1x the contact is ill-conditioned, and the symptom is not a visible blow-up but a
silent one: **a gripped tool slides out of the jaws under its own weight at ~280 mm/s** with
25 N on each pad and pad friction 2.0 -- about 50 N of Coulomb capacity against a 0.45 N
wrench. `scripts/grasp_diagnose.py` shows the tool creeping at a constant ~12 mm/s in the
GRIPPER FRAME even while the arm is completely stationary, which no Coulomb contact can do.

`scripts/_creep_probe.py` isolates it by holding a grasped tool still for 3 s:

| variant | creep | reading |
|---|---|---|
| base | **282 mm/s** (tool on the floor in 3 s) | -- |
| gravity off | 0.00 mm/s | the creep is load-driven: a genuine friction failure |
| pad friction x10 | 48 mm/s | scales with mu -> cone slip, not geometry |
| **timestep halved** | **7.8 mm/s** | 36x better: conditioning, exactly as predicted |
| pyramidal cone | 13.5 mm/s | 21x better |
| pad solref 0.02 (softer) | grasp fails outright | do not soften the pads instead |

Fix, now the default in `bw/sim/workshop.py`: **`timestep="0.001"` and `cone="pyramidal"`**.
Held-2s over the four tools, 32 episodes: **10/32 -> 26/32**. Both are needed (timestep alone
9/16, pyramidal alone 5/16). **Raising `impratio`, the usual internet advice for this symptom,
makes it strictly worse** -- 0/16 at 50 and at 100, and it brings back the QACC warnings.
The MJX locomotion model is untouched, so no Phase 1 number is affected.

### Tape roll: the grasp point was below the rack plates

The ring was grasped at its equator, which is the ring's own centre -- 45 mm above the rack
floor, **5 mm BELOW the 50 mm plate tops**. The jaws reached the rack before the tape:
`site_err` 8-19 mm against ~1 mm for every other tool. Grasping `TAPE_PHI` up the rim instead
(`scripts/_tape_sweep.py`, 8 seeds, held-2s): 0 deg **2/8**, 15 deg 6/8, 30 deg 7/8,
**45 deg 8/8** (site_err 1.2 mm), 60 deg 5/8, 70 deg 1/8. Clearance above the plates buys the
left half of the curve; the wall's apparent width across a laterally-closing jaw (7 mm/cos phi)
costs the right half. `TAPE_PHI = 45 deg`.

### Locomotion, before any fine-tuning

Run-7 stairs policy on the Go2+Z1 with the arm moving, 12 cm step, 500-step episodes:
steps alive per seed **[146, 370, 162, 500, 142, 153, 152, 342]**. It walks and then loses
attitude — exactly the payload problem Phase 1 exists to fix.

---

## Decisions and deviations from the plan

Every one of these was forced by a measurement, and each is a talking point rather than a
compromise. They must appear in the README and the write-up.

1. **Tool tray → bench-top tool rack (tools standing in slots).** The Z1 shoulder sits ~8 cm
   below a 75 cm bench, so a tray can only be approached ≥35° off vertical, and at that tilt
   the gripper housing reaches the tray floor before the pads reach a 9 mm wrench. Measured:
   0–1 of 20 grasps, jaws jammed at q ≈ −0.6 with no pad contact. Standing tools are grasped
   **horizontally**, in the middle of this arm's workspace. The bench is still 75 cm.
2. **Z1 rotary gripper → parallel-jaw gripper** (two sliding fingers, one servo each, driven by
   one command; `FINGER_TRAVEL = 0.038`, squeeze commands 8 mm past contact). The stock Z1 jaw
   swings on an arc with non-parallel pads: it wedged tools out of the jaws under squeeze and
   bottomed out on fixtures. An off-the-shelf swap on real hardware too.
3. **A 12 cm pedestal under the arm mount.** Without it the arm cannot reach bench height at
   any usable orientation. It also raises the CoM, which makes the Phase 1 payload problem
   harder and more honest.
4. **`locate()` on the bench uses the WRIST camera in a scan pose**, not the head camera: the
   head camera sits at 0.46 m and can never see a 0.75 m surface. The head camera keeps
   navigation and floor search (recovery scenario 1). Scan pose found by joint-space search
   (`scripts/find_scan_pose.py`): `[0, 1.5, −2.1, 1.5, 0, 0]`.
5. **Contact stiffness on the jaw pads** (`solref 0.002 1`, `solimp 0.995 0.9995 0.0002 0.5 2`).
   With MuJoCo's defaults the pads sank 3.5 mm into a 13 mm handle under 25 N and extruded the
   tool during the retreat. Pad friction 2.0 with torsional 0.25. Contact *priority* was tried
   and made things worse (10% vs 30% at the time) — do not re-add it.
6. **Observation layout frozen** to Phase 1's (51×5 = 255, the `privileged_state`, read by BOTH
   actor and critic — `actor_privileged: true` in payload.yaml *and* in run 7's stairs.yaml, so
   the warm start is consistent; the 48×5 `state` vector exists but nothing reads it) so run 7
   warm-starts:
   the policy is NOT told where the arm is and must infer the shifting CoM from IMU and leg
   states. Must be stated as a limitation.
7. **Commands widened for mobile manipulation**: vx [0, 0.9], vy ±0.2, wz ±0.6, 20% stand
   probability, resampled every ~5 s mid-episode; progress/height/clearance rewards gated on
   vx > 0.2 so a stand command does not pay for walking. Run 7 had only ever seen vx ∈ [0.3, 0.9]
   and had never been asked to stand still — which the orchestrator does for every grasp.
8. **Base is held kinematically during manipulation** (`WorkshopSim(base_mode="kinematic")`).
   The full pipeline must switch to `base_mode="policy"`; that path is written but unvalidated.

---

## Environment and operations

- **Repo**: `D:\bringwrench` (this directory). Code runs from `/mnt/d/bringwrench` in WSL; run
  outputs go to `~/bringwrench/runs` on ext4 (never write results to `/mnt/d` — ~10× slower).
- **JAX/MJX venv** (locomotion): `~/go2-stairs/.venv` — jax 0.10 + cuda12, mujoco 3.11, brax
  0.14.2. Do not upgrade JAX (brax 0.14.2 needs the `device_put_replicated` shim in `train.py`).
- **Torch venv** (SmolVLA, Molmo): `~/bringwrench/.venv-vla` — torch 2.11.0+cu130, lerobot
  0.6.1, transformers 5.5.4, bitsandbytes, accelerate. Built by `setup_vla_env.sh`.
- **Windows render venv**: `D:\hexapod\render_venv\Scripts\python.exe` (Python 3.11, mujoco
  3.11, imageio + imageio-ffmpeg, pillow). **All rendering happens here** — EGL/OpenGL fails
  inside WSL (go2-stairs HANDOFF §Known-broken). Pattern: roll out in WSL → dump `.npz` →
  render on Windows.
- **Models**: `lerobot/smolvla_base` → `~/bringwrench/models/smolvla_base` (873 MB, **done**).
  `allenai/Molmo2-ER` → `/mnt/c/hf_cache/Molmo2-ER`, **abandoned on purpose** (it had resumed and
  reached 15 of 19.4 GB before being stopped on 2026-09-18; the partial is still on C:):
  every Ai2 Molmo2 repo ships F32, so that download carries a 4.85B model that needs ~3.2 GB in
  4-bit, and ER's edge is embodied reasoning — which this architecture replaces with the
  hand-written state machine. `REMAINING.md` T0 lists the replacements
  (`Cycl0/Molmo2-VideoPoint-4B-bnb-4bit` 3.7 GB, `reubk/Molmo2-4B-GGUF` q4 3.6 GB,
  `Qwen/Qwen3-VL-4B-Instruct` 8.9 GB bf16) and the ground-truth bake-off that picks one.
- **GPU**: RTX 3060 Laptop, 6 GB, board power limit locked. 4096 envs with the arm model fits
  (~4.9 GB). Never stack GPU jobs. `rest_every_s: 900 / rest_seconds: 120` duty cycle is on in
  `configs/payload.yaml` (was 7200/300 — the first cooldown arrived ~100 min AFTER the card was
  already over its limit). Watch temperature: stop at ≥88 °C. **This is currently the blocker on
  T3 — see "Session 2" above for the measurements and the three options.** Clock capping needs
  an Administrator shell: `nvidia-smi -lgc 300,1100`, undo with `nvidia-smi -rgc`.
- **Long jobs**: `nohup`/`setsid`/agent background commands all die when no Windows session holds
  WSL open. Use a Windows Scheduled Task — `ops/run_payload.bat` + `ops/train_payload_fg.sh` are
  ready:
  ```
  schtasks /create /tn BwTrain /tr "D:\bringwrench\ops\run_payload.bat" /sc once /st 23:59 /f
  schtasks /run /tn BwTrain
  schtasks /delete /tn BwTrain /f      ALWAYS delete after starting, or it re-fires nightly
  ```
- **`pkill` trap**: match the module, not the script name — `pkill -f 'bw[.]locomotion[.]train_payload'`.
  A pattern matching the launcher kills the launching shell (this cost one smoke run).

---

## Session 2 (2026-09-18) — what changed

**Three defects were blocking T3; all three are fixed, two are proven.**

1. **MJX solver divergence (fixed, PROVEN).** `iterations=1` → `4`, `ls_iterations=5` → `10`
   in `build_models.py` + regenerated `go2z1_mjx.xml`, plus an assert in `go2_arm_env.py`.
   Step-0 eval, before → after: reward **−262,894 → +2,306**, ep_len **136.5 → 584.5**,
   dive terminations **0.20 → 0.00**, attitude **0.67 → 0.18**. Reproduced twice (2,306 / 2,257).
2. **`domain_rand.py` crash (fixed).** It looked up body `arm_gripperMover`, which
   `build_models.py:114` DELETES in the parallel-jaw swap → `KeyError` at startup, so T3 could
   never launch at all. Carry-load mass now goes on `arm_link06` (the gripper base), not a
   0.09 kg slider finger, so it does not randomize finger servo dynamics.
3. **Fake all-zero log rows (fixed, in `~/go2-stairs/train.py`; backup `.bak-20260918`).**
   See "Known bugs" #8 — this one cost a wrong "training died" diagnosis.

**Thermals — the open blocker. T3 is NOT safe to launch yet.**

| condition | result |
|---|---|
| no external fan, `rest_every_s 7200 / rest_seconds 300` | **88 °C in 20 min** at 4096 envs, 100% util |
| external fan, `rest 900/120` | idle 51→42 °C, rest pulls card to ~52 °C, but **back to 86 °C in ~4 min** of sustained load |

The thermal time constant is ~4 min, so holding under 84 °C by duty cycling alone needs a rest
every ~3 min (≈60% duty) → the 60M run exceeds 20 h. **Clock capping (`nvidia-smi -lgc`) is the
right lever but needs an ADMINISTRATOR shell** — it fails with "current user does not have
permission" from both WSL and a normal Windows shell. `power.default_limit` is 80 W and
`power.limit` reads N/A, confirming the board limit is locked. Unresolved options: (A) capped
clock ~1100 MHz, keeps 4096 envs and run-7 comparability, ~16–18 h; (B) `num_envs` 2048, cooler
but changes effective batch and weakens the T7 comparison; (C) 60% duty cycle, >20 h.

**Throughput:** ~1,520 sps sustained at 4096 envs (the "sps" column is a cumulative average and
reads low early because it includes the ~9 min compile; there is a SECOND compile of
`jit_generate_eval_unroll` after the step-0 eval). 60M steps ≈ 11 h of pure compute.

**Models:** `Cycl0/Molmo2-VideoPoint-4B-bnb-4bit` **downloaded and verified** (3.5 GB,
`model.safetensors` 3,732,535,100 bytes, no `.incomplete` left). GGUF q4 pair downloading.
Link speed measured at **~280 KB/s** — budget downloads in hours, not minutes.

---

## Phase 1 gate + ablation — RUN 2026-09-18, gate 2 FAILS

Policy: `runs/results/2026-09-18_02-47-57-payload/checkpoints/step_33013760` (the 60M run was
stopped at 33M; reward had plateaued ~3,050-3,200 since step 20M). Results in
`runs/results/phase1/` (`phase1_eval.json`, `push_ablation.md`). The script ran clean on first
execution — the "expect API friction" warning was wrong.

| gate criterion | original (run 7) | payload-aware | |
|---|---|---|---|
| 1. walk 5 m flat, arm stowed, 20/20 | 20/20 | 20/20 | PASS (both — does NOT discriminate) |
| 2. cross 12 cm step, arm extended, **0 falls**/20 | 3 crossed, 17 falls | **19 crossed, 1 fall** | **FAIL** |

### Step-height sweep — THE result to publish (added 2026-09-18, gate re-run)

Crossed / falls out of 20. Solver divergences are excluded from `falls` and counted separately
(see `rollout`): a diverged episode is upright at normal ride height and is a simulator
artifact, not the robot falling.

| rise | payload-aware | original (run 7) |
|---|---|---|
| 0.06 m | **20 / 0** | 19 / 1 |
| 0.08 m | **20 / 0** | 15 / 5 |
| 0.10 m | **19 / 0** (1 diverged) | 2 / 18 |
| 0.11 m | **19 / 0** (1 diverged) | 1 / 19 |
| **0.12 m** | **19 / 1** | 3 / 17 |
| 0.13 m | 9 / 11 | 3 / 17 |

Zero falls through 0.11 m, one at 0.12 m, cliff at 0.13 m — against an original policy already
collapsing at 0.10 m. **The cliff lands exactly where the trunk geometry predicts** (front
sphere rides at ~0.118 m under this gait's ~13 deg pitch; see "the 3 gate-2 falls" below), so
mechanism and measurement agree. That is a far stronger claim than a single pass/fail.

NOTE the single-height number is NOISY: the same seed gave 3 falls on the first run and 1 on
the second (MJX run-to-run nondeterminism). The SWEEP is the trustworthy artifact, not the
0.12 m cell. The ablation, by contrast, reproduced all four cells exactly.

**UNEXPLAINED, and reproducible:** original stowed 160 N < original extended 200 N. It repeated
identically across both runs, so it is NOT grid noise — a policy that falls 17/20 crossing a
step with the arm extended should not resist lateral pushes BETTER in that pose. Do not publish
the ablation table until this is explained; the diagnostic is to record per-force recovery
rates rather than just the monotone-envelope maximum.

| max recoverable push | arm stowed | arm extended |
|---|---|---|
| original (run 7) | 160 N | 200 N |
| payload-aware | **280 N** | **280 N** |

**Best result of the project so far:** the payload policy is **invariant to arm configuration**
(280 N either way) while the original swings with it — exactly what Phase 1 set out to prove.
Step-crossing went 1/20 -> 17/20. Training: attitude terminations 0.20 -> 0.05 over 33M steps.

**CAVEAT on the ablation:** original stowed 160 N < original extended 200 N is BACKWARDS and
unexplained. One grid step (sweep is 0/40/.../200/240) at 40 trials per force, so probably
noise — but it is in the headline table and a reviewer will ask. Re-run that row on a finer
grid before publishing.

### The 3 gate-2 falls, diagnosed (`scripts/_diag_gate2.py` replays gate 2 per-trial)

| trial | step | dx at end | base z | up_z | cause |
|---|---|---|---|---|---|
| 4 | 167 | **1.923** | 0.290 | 0.975 | `term_contact` |
| 14 | 164 | **1.891** | 0.288 | 0.976 | `term_contact` |
| 2 | 391 | 4.177 | 0.378 | **0.999** | `term_diverged` |

Trials 4 and 14 are the SAME failure: same place (3 cm apart), same moment, same cause, same
posture. The robot spawns at x=0 and the riser is at x=`approach`=2.0, so they strike the step
face 8-11 cm before it. `fatal_contact` is TRUNK contact only (calf contact is deliberately not
fatal), so these are genuine falls — the torso hits the riser.

Trial 2 is NOT a locomotion failure: `term_diverged` at up_z 0.999 (perfectly upright) and a
normal height for the raised platform. Residual MJX instability, the class the solver fix
reduced but did not eliminate. Do not count it against the policy.

### TRAP — the curriculum could never reach the terrain the gate tests

`~/go2-stairs/terrain/stairs.py` LEVELS: **L4 = (0.110 m rise, 0.32 m run)**, L5 = (0.130, 0.30).
Gate 2 tests **(0.12, 0.30)** — taller rise AND shorter tread than L4. The 60M run trained
EXCLUSIVELY at L4 (`lvl` read exactly 4.00 in all 13 evals) because promotion needs a success
EMA > `promote_success: 0.65` and the run plateaued at **0.37-0.40**. It was therefore
STRUCTURALLY INCAPABLE of ever training on a 0.12 m step — at 60M steps or 600M. **More
training would not have fixed gate 2.** Check the `lvl` column against the eval geometry before
spending GPU-hours on any curriculum run.

Fix in flight: `configs/payload_l5.yaml` — `level_init: 5`, `level_min: 4`, warm-started from
step_33013760, 10M steps (~1.5-3 h), so the gate geometry is inside the training distribution.

---

## Known bugs and open questions

1. **wrench_13mm drops after the pick** (visible in `media/grasp_wrench_13mm.mp4`): lifts
   ~8.9 cm, then the tool leaves the jaws during the lift/retreat. Highest-priority grasp bug.
2. **screwdriver 0/8 — DIAGNOSED to the lift, not yet fixed (2026-09-18).**
   The re-point fix from session 1 IS in and benchmarked: it helped tape_roll (3/8 -> 6/8) and
   pliers (3/8 -> 4/8) but did nothing for the screwdriver. What the traces show:
   - `site_err` ~1 mm, so pointing/IK are NOT the problem.
   - The jaws DO grip the handle: both pads at 26-33 N, ~0 penetration. With pad friction 2.0
     that is ~120 N of hold against a 0.7 N tool, so it cannot slip under its own weight.
   - During the lift the hand rises 68 mm while the tool rises only 49 mm (~19 mm of slip),
     `finger_b_pad` registers TWO contact points (the tool rocks), then both pads vanish at
     once and the tool falls.
   - The shaft needs **57 mm** of lift to clear the plates (shaft bottom 0.743, plate top
     0.800). It escapes at **49 mm** — 8 mm short. Closing also pins the tool against
     `rack_front` at 28 N.
   - NOTE the `no_lift` threshold is 0.05 and these reach 0.049: they are LATE DROPS
     mislabelled as `no_lift`. The stage breakdown for this tool is partly an artifact.
   **Three hypotheses tested and REFUTED — do not repeat:** (a) jaws closing on the rack
   plates (they contact `screwdriver_handle` directly); (b) square handle rolled ~45 deg so
   the jaws meet its 35 mm diagonal (measured roll is only +-8 deg, effective width 26-28 mm;
   wrenches/pliers sit at +-2 deg); (c) grasp height / squeeze depth (full sweep of
   GRASP_Z 0.128/0.140/0.150 x squeeze -0.008/-0.016 gave 0/8 on ALL six, and raising the
   grasp point collapses max lift from 160 mm to 2 mm).
   One seed DOES lift it 160 mm clear before losing it on the retreat, so extraction is
   possible and the grasp is simply not repeatable. Next candidates: rack/slot geometry for
   this tool, or the shaft's 45 mm of engagement between the plates.
3. **pliers / tape_roll -- FIXED 2026-09-18 (session 3).** pliers 24/25 (the creep); tape_roll
 22/25 (the creep, plus a grasp point 5 mm below the rack plate tops). The 8 residual
 failures over 100 episodes are scattered across `no_lift` 4, `dropped` 2, `no_grip` 2 with
 no dominant mechanism -- that is what the remaining T1 work has to attack.
4. `controller.py` (numpy Layer 3) has never been run against the MJX env — the observation
   must be verified bit-for-bit against `Go2ArmEnv._single_obs` before trusting it.
5. `eval_phase1.py` has never been executed; expect API friction on first run.
6. **T0.2 ANSWERED, and both downloaded candidates are DEAD (2026-09-18).** Molmo2 emits points
   as **special tokens**, not text — decoded by `extract_image_points` / `extract_video_points`
   using preprocessor metadata, returning `(object_id, {image_num|timestamps}, pixel_x,
   pixel_y)` (Ai2 `MOLMO_POINT_README.md`).
   * `Cycl0/Molmo2-VideoPoint-4B-bnb-4bit`: ships **no pointing code at all** (0 point
     functions across its 5 .py files; none of its 303 added tokens are point tokens), AND
     fails to load on transformers 5.5.4 with three separate API breaks (processor kwargs
     `image_use_col_tokens`; `AutoModelForCausalLM` does not accept `Molmo2Config` — the
     auto_map says `AutoModelForImageTextToText`; `ROPE_INIT_FUNCTIONS['default']` KeyError).
     NOTE its `processing_molmo2.py` carries a LOCAL PATCH for the first break and the backup
     copy silently failed — re-download the file if a pristine copy is needed.
   * `reubk/Molmo2-4B-GGUF`: premise invalid. It was wanted for GBNF grammar-constrained
     output, but there is no text to constrain if points are special tokens.
   Both are single-uploader community mirrors, so this does NOT condemn Molmo2 itself —
   Ai2's official repos are untested, and `allenai/Molmo2-ER` sits at 15 of 19.4 GB in
   `/mnt/c/hf_cache` if an official Molmo2 is wanted. Current fallback: `Qwen3-VL-2B-Instruct`
   (official, bf16, TEXT coordinates, so the grammar option returns), downloading to
   `~/bringwrench/models/qwen3-vl-2b`. The plan's own cut line (classical CV on sim renders)
   remains available.
8. **TRAP — all-zero rows in a training log are NOT divergence.** brax calls the SAME
   `progress_fn` from two producers: the evaluator (`eval/*` keys) and, when
   `log_training_metrics: true`, `EpisodeMetricsLogger` (`episode/*` keys only, every ~54 s).
   The latter fell through every `metrics.get("eval/...", 0.0)` and printed a row of exact
   zeros identical to a dead run. Genuine evals are only the `num_evals` rows. Fixed by an
   early return in `~/go2-stairs/train.py` — but note `_maybe_rest()` is the LAST statement of
   `progress()`, so the guard must call it before returning or the GPU duty cycle silently
   degrades from ~54 s to ~26 min granularity. The `NaN or Inf found in input tensor` warnings
   from tensorboardX are a separate, minor issue in the training-metric payload.
9. Tool geometry is primitive (boxes/capsules/a 16-segment ring). Fine for physics, but the
   10 mm vs 13 mm distinction leans on a coloured grip band plus size — check that Molmo and
   SmolVLA can actually separate them before trusting the "correct-wrench ≥80%" criterion.

---

## Reproducing the current state

```bash
# rebuild every model (WSL, JAX venv not needed)
cd /mnt/d/bringwrench && ~/go2-stairs/.venv/bin/python -W ignore -m bw.sim.build_models

# grasp benchmark (Windows, 40 episodes, ~60 s)
D:\hexapod\render_venv\Scripts\python.exe scripts\try_grasp.py 8

# simulation stability + solver sweep (WSL, GPU, ~10 min each)
~/go2-stairs/.venv/bin/python -W ignore scripts/stability_mjx.py
~/go2-stairs/.venv/bin/python -W ignore scripts/solver_sweep.py

# locomotion clip: roll out in WSL, render on Windows
PYTHONPATH=/mnt/d/bringwrench:$HOME/go2-stairs ~/go2-stairs/.venv/bin/python -m bw.locomotion.dump_traj \
    --checkpoint ~/go2-stairs/results/2026-08-06_17-17-05-stairs_run7/checkpoints/final --arm random
D:\hexapod\render_venv\Scripts\python.exe scripts\render_traj.py --traj media\loco.npz
```
