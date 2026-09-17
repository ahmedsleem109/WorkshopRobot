# STATUS — "Bring me the 10mm wrench"

Last updated **2026-09-17, end of session 1**. Read this, then `REMAINING.md` for what to do
next (tasks are ordered by dependency there, not by phase).

Plan of record: `bring-me-the-10mm-wrench-plan (1).md`. Upstream locomotion project:
`D:\hexapod` on Windows, `~/go2-stairs` in WSL (its `HANDOFF.md` and `plan.md` still apply to
everything about the locomotion policy).

---

## State in one line

Phase 1's **embodiment, scene and simulation are built and stable**; the Phase 1 payload
fine-tune has **not been launched** (its two blocking simulation bugs were found and fixed);
Phase 2's scripted demonstrator **works but only at 50%**, which is not acceptable and is the
first thing to fix next session; nothing of Phase 3 exists yet.

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

### Scripted grasp demonstrator — 40-episode benchmark, seeds 0–7 (`scripts/try_grasp.py 8`)

| Tool | Success | Failure stages |
|---|---|---|
| wrench_10mm | **8/8** | — |
| wrench_13mm | 6/8 | dropped 2 |
| tape_roll | 3/8 | dropped 3, no_lift 2 |
| pliers | 3/8 | dropped 3, no_lift 2 |
| screwdriver | **0/8** | no_grip 3, no_lift 4, dropped 1 |
| **overall** | **50%** | dropped 9, no_lift 8, no_grip 3 |

Stage meanings: `ik` no reachable plan · `no_grip` pads never closed on the tool ·
`no_lift` gripped but never left the rack · `dropped` lifted, then lost during the retreat.

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

## Known bugs and open questions

1. **wrench_13mm drops after the pick** (visible in `media/grasp_wrench_13mm.mp4`): lifts
   ~8.9 cm, then the tool leaves the jaws during the lift/retreat. Highest-priority grasp bug.
2. **screwdriver 0/8.** Mixed `no_grip` and `no_lift`. Its box handle settles ~1 cm deeper into
   the slot over the first seconds, so a plan made at reset goes stale; a re-point from the
   pre-grasp pose was written but **not yet benchmarked** (it was the change in flight when the
   session ended — verify `grasp_point` is re-evaluated before the descent).
3. **pliers / tape_roll ~3/8**, both mostly `dropped`.
4. `controller.py` (numpy Layer 3) has never been run against the MJX env — the observation
   must be verified bit-for-bit against `Go2ArmEnv._single_obs` before trusting it.
5. `eval_phase1.py` has never been executed; expect API friction on first run.
6. The Molmo pointing output format (how points are encoded, coordinate scale) is **not yet
   confirmed** — the HF model card does not document it; check `allenai/molmo2` on GitHub. The
   grounding model itself is not settled: `REMAINING.md` T0 holds the candidates and the
   bake-off that decides, scored against sim ground truth.
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
