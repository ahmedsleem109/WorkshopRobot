# REMAINING — dependency-ordered task list

Written at the end of session 1. Ordering is by **dependency**, not by plan phase: each task
lists what it blocks and what must be true before it starts. Read `STATUS.md` first for the
measured numbers and the traps.

Standing rules for next session:
- **50% grasp success is rejected.** The demonstrator target is **≥90% per tool** on a fixed
  100-episode benchmark before any data is collected (T1).
- **Fine-tuning must be done properly**, not launched and hoped for: every run gets a stated
  hypothesis, a fixed eval protocol, and a comparison against the previous checkpoint (T3, T7).
- New scope this session: **pick from one table and place on another, commanded in language at
  run time, executed by the VLA** (T2, then T6/T7/T9).


> **Session 3 (2026-09-19):** the per-task DONE / IN PROGRESS / NOT STARTED board is at the top of
> `STATUS.md`. Some checkboxes below (T3, T4) were finished in session 2 and never ticked; the
> board is authoritative.
---

## Dependency graph

```
T0 choose + verify grounding model ────┐
T1 grasp reliability >=90% ──┬── T2 two-table pick&place ──┬── T6 data collection ── T7 SmolVLA fine-tune ──┐
                             │                             │                                                │
T3 payload locomotion run ───┴── T4 Phase 1 eval/ablation ─┘                                                │
                                                                                                            │
T5 Layer 3 numpy controller validation ─────────────────────────────────────────────────────────────────────┤
T0 ──> T8 grounding: locate()      ─────────────────────────────────────────────────────────────────────────┤
                                                                                                            ▼
                                                                          T9 orchestrator (nav/pick/place/recover)
                                                                                     │
                                                                          T10 recovery scenarios
                                                                                     │
                                                                          T11 evaluation suite (50 trials)
                                                                                     │
                                                                          T12 video + README + blog + outreach
```

T3 uses the GPU for ~8 h and depends on nothing that is not already done — **launch it first**,
then work on T1/T2 on CPU while it trains.

---

## Model roles — settled, do not relitigate

Two distinct slots, two different models. Asked in session 1 whether Qwen3-VL-4B should be the
fine-tuned model; the answer differs per slot.

| Slot | Model | Fine-tuned? | Why |
|---|---|---|---|
| Layer 2 — manipulation (VLA) | **SmolVLA** (`SmolVLM2-500M` backbone + action expert, `chunk_size=50`, frozen vision encoder, `train_expert_only=true`) | **yes** — this is the centerpiece | ~0.9 GB bf16 so it can be co-resident with the grounding model on a 6 GB card; emits 50 actions per forward = 5 s of motion at 10 Hz, so no model sits inside the control loop; documented 50–200 demo range matches our data scale; 1–2 h fine-tune |
| Layer 1 — grounding (VLM) | a 4-bit pointing model (T0) | **no**, by default | called 4–6 times per episode through `vlm.point()` only |

**Why NOT a 4B VLM as the VLA** (e.g. Qwen3-VL-4B): (a) VRAM — grounding at 4-bit is
~3.2–3.7 GB and a 4B VLA at 4-bit adds ~3.2 GB, over the 6 GB budget before MuJoCo's OpenGL
and the desktop compositor; (b) it is a VLM, not a VLA — an action head, action tokenization
and the chunked control loop would all have to be built and validated, which SmolVLA ships;
(c) a 4B backbone on ~1k demos likely underperforms the 450M specialist at 10x the compute,
and the compute budget is Kaggle's free 16 GB tier.

**Where Qwen3-VL IS appropriate:** as the Layer-1 grounding model, and — as a conditional
stretch — LoRA-fine-tuned on sim-generated point labels. Sim gives unlimited EXACT labels
(ground-truth poses per render), which is the most reliable route to the 10 mm vs 13 mm
discrimination. Conditions: only if T0.3's zero-shot bake-off fails that test, and prefer
Qwen3-VL-**2B** (~1.7 GB at 4-bit) over the 4B, because at run time the grounding model shares
the card with SmolVLA. The plan itself files "fine-tune the VLM for your scene" under [LATER].

---

## T0 — Pick and verify the grounding model `[blocks T8]` · ~2 h

**Decision (session 1, from HF API file sizes): do NOT finish the 19.4 GB Molmo2-ER download.**
Every Ai2 Molmo2 repo ships **F32**, so 19.4 GB carries a 4.85B model that needs ~9.7 GB in
bf16 and ~3.2 GB in 4-bit; half the download is discarded at quantization, against ~28 GB free
on D:. And Molmo2-ER's advantage is embodied *reasoning* + tool orchestration, which this
architecture deliberately does not use — the plan replaces it with the hand-written state
machine and calls the VLM only through `vlm.point()`.

| Candidate | Download | ~VRAM 4-bit | Note |
|---|---|---|---|
| `Cycl0/Molmo2-VideoPoint-4B-bnb-4bit` | **3.7 GB** | ~3.7 GB | already 4-bit (bitsandbytes, CUDA-only — fine here), pointing specialist, ships `modeling_molmo2.py` |
| `reubk/Molmo2-4B-GGUF` (`q4_k_m` + `mmproj-f16`) | **3.6 GB** | ~3.5 GB | llama.cpp path -> GBNF grammar-constrained output, the plan's tool-parse fallback |
| `Qwen/Qwen3-VL-4B-Instruct` | 8.9 GB bf16 | ~3 GB | general-purpose baseline; the better default for OTHER projects |
| `allenai/Molmo2-ER` | 19.4 GB F32 | ~3.2 GB | only if the ER-vs-state-machine comparison (plan's Phase 3 stretch) is actually attempted |
| `allenai/MolmoPoint-Vid-4B` | 19.5 GB F32 | ~3.2 GB | Ai2's grounding-token pointing architecture, same F32 tax |

Both small candidates are **community mirrors** (single uploader, a few hundred downloads), so
they are unverified — cheap to settle here, because sim hands us exact ground-truth poses.

- [x] **T0.1 DONE 2026-09-18** - both downloaded, and both DEAD. See `STATUS.md` Known bug #6.
      Molmo2 emits points as SPECIAL TOKENS (not text); the 4-bit mirror ships no pointing code
      and will not load on transformers 5.5.4; the GGUF's grammar-constrained-text premise is
      therefore void. **DECISION: Layer 1 is `Qwen/Qwen3-VL-2B-Instruct`** - 4.0 GB, official,
      built into transformers (no trust_remote_code), TEXT coordinates, ~1.2 s per call.
      The partial `allenai/Molmo2-ER` (15 of 19.4 GB) is still on C: if an OFFICIAL Molmo2 is
      ever wanted; otherwise it can be deleted.
- [ ] **T0.2** Confirm the pointing output format from `github.com/allenai/molmo2`
      (`MOLMO_POINT_README.md`): how points are encoded in the text, and the coordinate scale.
      The HF model card does not document it. Record it in `STATUS.md`.
- [x] **T0.3 DONE 2026-09-18 (session 3).** 200 views, scored against ground truth; full table
      in `STATUS.md`. Scale is **0-1000 normalised** (44.5 px vs 278.4 px median), latency
      1.12 s, 4.26 GB. **The model is at CHANCE (50/52/50%) on 10 mm vs 13 mm across three
      phrasings including Qwen's own grounding format, and at 92.9% when the query names the
      coloured grip band.** Original: Bake-off on our own renders, scored against ground truth.
      For each candidate: 200 wrist-camera views from `WorkshopSim` with randomized pose,
      lighting and clutter; report **median pixel error**, **median 3D error after the depth
      lookup**, **miss rate**, **latency per call**, and above all **10 mm vs 13 mm wrench
      discrimination** — the one distinction the whole task depends on, and the hardest for
      any of these models, since the two wrenches differ mainly in size plus a coloured band.
- [ ] **T0.4 DECIDED, not yet implemented.** Option **(a)** is taken: keep Qwen3-VL-2B and let
      `vlm.point()` map the size in the instruction to the grip-band colour ("10mm" -> blue,
      "13mm" -> red), which clears the 80% correct-wrench gate at 92.9%. Still to do: put it
      behind `vlm.point(image, description) -> (u, v) | None` in a separate process, so
      swapping stays a one-file change (the plan's Layer 1 contract). The limitation must be
      stated in the write-up: the size discrimination is carried by colour coding plus a
      lookup in our code, not by the vision model.

`ops/resume_molmo.sh` / `ops/run_molmo_download.bat` remain if ER is ever wanted
(`HF_HUB_DISABLE_XET=1` is set in them; the xet transport failed at ~16 MB).

**For other projects (noted while we were here):** `Qwen3-VL-4B-Instruct` is the better
general-purpose default — bf16 weights (half the download of an F32 Molmo) and much wider
tooling (vLLM, quantization, fine-tuning recipes). Keep a Molmo pointing model only for
grounding-specific work.

## T1 — Grasp demonstrator to ≥90% per tool `[blocks T2, T6]` · the first real work

Current: 50% overall; wrench_10mm 8/8, wrench_13mm 6/8, tape 3/8, pliers 3/8, screwdriver 0/8.
Failure stages: `dropped` 9, `no_lift` 8, `no_grip` 3 (see `STATUS.md` for definitions).

**Acceptance:** `scripts/try_grasp.py 20` (100 episodes, fixed seeds) shows **≥90% per tool and
≥92% overall**, and the wrench_13mm drop seen in `media/grasp_wrench_13mm.mp4` is gone.

**CLOSED 2026-09-18 (session 3): 97/100, per tool 96/100/96/96, on the held-for-2s criterion.**
See STATUS.md for the mechanism (a pad-contact time constant of exactly 1 x the timestep) and
for why every grasp number written before session 3 is not comparable. T1.2/T1.3/T1.5/T1.6
were not needed to clear the bar and are left unticked on purpose.

**UPDATED 2026-09-18.** The screwdriver is **dropped from the grasp set** (`GRASP_TOOLS` in
`bw/sim/workshop.py`); it stays in the scene as a distractor for the grounding model. Current:
**69%** over 4 tools - wrench_10mm 7/8, wrench_13mm 5/8, pliers 4/8, tape_roll 6/8, so the
acceptance bar is now >=90% per tool over FOUR tools. **`dropped` is 8 of the 10 failures**, so
there is ONE failure mode to attack: the tool leaves the jaws during the lift/retreat. Start
there, not with a survey.

Ordered sub-tasks — the first two are diagnosis, do not skip them:

- [x] **T1.1 DONE 2026-09-18 (session 3).** `scripts/grasp_diagnose.py`. It found the creep
      (STATUS.md), which was the whole of the `dropped` mode. Instrument the slip. Log, per episode: contact normal force per pad, tool
      pose in the *gripper frame* at close / after lift / after retreat, and the frame at which
      the tool's position in the gripper frame first moves >2 mm. That one number separates
      "never gripped", "slipped on lift", "extruded under squeeze" and "knocked during
      approach", and every fix below should be judged by it. Write it as
      `scripts/grasp_diagnose.py` producing a CSV + a one-page summary.
- [ ] **T1.2 Verify the re-point fix that was in flight.** `run_grasp` should re-evaluate
      `grasp_point(sim, name)` at the pre-grasp pose before descending (tools settle ~1 cm into
      their slots over the first seconds, and `WorkshopSim.reset` must settle ≥1.2 s so the
      scene is static before planning). Benchmark with and without.
- [ ] **T1.3 Fix `no_lift` (8/40).** Hypotheses in order: (a) the tool is still touching the
      rack plates/dividers when the lift starts — raise `GRASP_Z` per tool so the gripped
      section is ≥2 cm above the plate tops, or lower the plates for the slots that need it;
      (b) the lift is not vertical in the tool's own frame when it leans — lift along the tool's
      long axis instead of world +z; (c) friction against the plates — reduce the slot gap for
      thin tools using per-slot gaps rather than one `RACK_GAP`.
- [x] **T1.4 DONE 2026-09-18 (session 3)** -- but by NONE of the hypotheses below. `dropped`
      was solver conditioning in the pad contacts, not squeeze force, contact softness,
      retreat jerk or grasp height. wrench_13mm is 25/25. Original text kept for the record:
      Fix `dropped` (9/40), including wrench_13mm. Hypotheses: (a) squeeze force too
      low for the heavier tools — scale `GRIP_OVERSHOOT` with tool mass, and verify the pad
      normal force from T1.1 is ≥15 N for 180 g pliers; (b) contact softness still allowing
      slow extrusion — tune `solimp` width, not stiffness; (c) the retreat's jerk — profile the
      Cartesian path with a cosine ease instead of linear steps; (d) grip closer to the tool's
      CoM so it does not pendulum (per-tool `GRASP_Z` again).
- [ ] **T1.5 Force-closure check + regrasp in the demonstrator.** After closing, require both
      pads in contact AND a tool-frame pose within tolerance; otherwise open, re-point, retry
      (max 2). A retrying demonstrator is also what the orchestrator's `on_failure` does, so
      this is not throwaway code.
- [ ] **T1.6 Widen the benchmark** to 20 seeds × 5 tools with the same seeds every run, and
      print a per-tool table plus the stage histogram. Keep every config change in a short
      log at the bottom of this file so the improvements are attributable.

Do **not** spend the session hand-tuning single seeds: fix the mechanism T1.1 points at, then
re-run the 100-episode benchmark.

---

## T2 — Two-table pick-and-place `[needs T1; blocks T6, T9]` · new scope

Language-commanded *transfer*: "put the 10mm wrench on the right table", "move the pliers to
the far bench". The VLA must execute both halves, so the demonstrations must contain both.

- [x] **T2.1 DONE (session 3).** Table B off the walkway's right edge, painted zones on both
      tables, cameras `table_b_view` + `scene_wide`; names agree in both schemes (A = left/far,
      B = right/near). Original: **T2.1 Scene.** Add a second table (table B) to `bw/sim/workshop.py`: same 75 cm height,
      placed so both tables are reachable from a standing pose after a short base move, with a
      marked **place zone** on each (a shallow tray or a painted rectangle) plus a slot rack on
      table A only. Name the tables in world terms the language can refer to ("left"/"right"
      from the robot's start pose, and "near"/"far"). Keep the 12 cm step on the route.
- [x] **T2.2 DONE (session 3).** `scripts/reach_audit.py`: the place roll must be about the
      APPROACH axis (joint 6, free); about the jaw axis is unreachable past ~45 deg. Stations
      0.48 m behind each zone, from a `try_place.py --back` sweep. Original: **T2.2 Reachability audit.** Re-run the reach search (`scripts/find_scan_pose.py` pattern)
      for both tables and record which base poses serve which table; the orchestrator needs
      those as navigation goals.
- [ ] **T2.3 BUILT, below the bar (session 3):** see the STATUS.md header for numbers; the
      dominant failure is the tool toppling out of the zone after release. Original: **T2.3 `place` skill** in `bw/manip/scripted_place.py`: approach above the place zone,
      descend to contact (or a fixed clearance), open the jaws, retract, verify the tool is
      resting in the zone and the gripper is empty. Acceptance: **≥90%** placement success for
      each tool on 100 episodes, tool inside the zone and stable for 2 s.
- [x] **T2.4 DONE (session 3).** `bw/task/spec.py` (`Task`, `Snapshot`, `evaluate`), already
      used by `try_place.py`. Original: **T2.4 Task spec + success criteria** for transfer: correct object, correct destination
      table, tool stable and inside the zone, nothing else knocked off. Encode it as one
      function both the data collector and the evaluator call, so training and scoring cannot
      drift apart.
- [x] **T2.5 DONE (session 3).** `bw/task/language.py`: template x synonym x table alias,
      `check_diversity` guard (242 distinct strings / 300 draws). Original: **T2.5 Language templates** for transfer commands (destination phrased as left/right,
      near/far, and "the other table"), with paraphrases generated the same way as the pick
      instructions (T6.2).

---

## T3 — Payload-aware locomotion fine-tune `[independent; blocks T4]` · ~8 h GPU, launch first

Everything needed is in place; **the two bugs that would have wasted the run are fixed**
(solver iterations, integrator — see `STATUS.md`).

- [ ] **T3.1** Set `iterations: 4`, `ls_iterations: 10` on the MJX model used for training
      (currently only proven via `scripts/solver_sweep.py`; make it the model default in
      `bw/sim/build_models.py` for the MJX variant, and assert it in `Go2ArmEnv.__init__`).
- [ ] **T3.2** 3M-step smoke run: `succ` must not collapse, `nan_steps` must stay 0,
      `term_diverged` must stay ~0. Compare against the smoke numbers in this file's log.
- [ ] **T3.3** Full run, 60M steps, warm-started from run 7 (`configs/payload.yaml` already
      points at it), via Scheduled Task. Expect ~2,000 steps/s → ~8.5 h.
- [ ] **T3.4** Score with `bw/locomotion/eval_phase1.py` (first execution — expect friction).
- [ ] **T3.5 If it is not good enough, make the fine-tune better** rather than longer: the
      levers, in order — (a) let the CRITIC see the arm state (asymmetric critic only; this
      breaks the frozen-obs warm start for the value net, so use brax's `restore_value_fn=False`
      and check the normalizer shapes); (b) curriculum on arm aggressiveness (`arm_speed`,
      probability of extended poses) instead of terrain; (c) raise `w_arm_stab` only after
      confirming attitude is still the binding termination; (d) longer horizon for the
      stand-still commands. One change per run, and always compare at a fixed step count.

---

## T4 — Phase 1 gate + push ablation `[needs T3]`

- [ ] Gate 1: 5 m flat walk, arm stowed, 20/20.
- [ ] Gate 2: cross the 12 cm step, arm extended, 0 falls in 20.
- [ ] The ablation table (max recoverable lateral push, {run 7, payload} × {stowed, extended}),
      DR off, 40 trials per force, monotone envelope — already implemented in `eval_phase1.py`.
- [ ] Sanity-check the silent-eval guards that are already coded: arm configuration asserted
      from `qpos`, push verified via the measured Δv, identical seeds across policies.

---

## T5 — Validate the numpy Layer 3 controller `[blocks T9]`

- [ ] Roll out the same checkpoint in MJX (`Go2ArmEnv`) and in `controller.py` on CPU from an
      identical state and assert the observation vectors match element-wise (they must, the
      layout is copied by hand) and that the actions agree to ~1e-5 for 100 steps.
- [ ] Then validate `base_mode="policy"` in `WorkshopSim`: the robot must stand still under a
      zero command while the arm executes a grasp, and stay standing (this is the mode the whole
      pipeline runs in, and it has never been exercised).

---

## T6 — Demonstration data in LeRobot v2.0 format `[needs T1, T2]`

- [ ] **T6.1** Collector: run pick and pick-and-place episodes on Windows (rendering), write
      wrist RGB (256×256) + arm state (7) + action (7, the 10 Hz commanded target) + language
      instruction per episode; keep successes only; randomize object pose, lighting, clutter,
      initial arm configuration, base pose.
- [ ] **T6.2 Instruction paraphrases — the known VLA failure mode.** Every episode gets a
      different phrasing of the same intent ("grab the 10mm", "hand me the small spanner",
      "put the pliers on the far table"). Generate with a local LLM or template × synonym
      expansion; assert at collection time that no object is ever paired with a single fixed
      string. If phrasing does not vary, the policy learns the object from vision and ignores
      language entirely.
- [ ] **T6.3** Target volume: ~600 pick + ~600 transfer episodes attempted; with T1/T2 at ≥90%
      that yields ≥1,000 clean demos (the plan's documented range is 50–200 per task, so this
      is headroom for the two-skill dataset).
- [ ] **T6.4** Convert to `LeRobotDataset` v2.0 in the torch venv, verify it loads, videos
      encode, and the instruction is attached per episode.

---

## T7 — Fine-tune SmolVLA properly `[needs T6]`

Backbone choice is settled — see "Model roles" above. SmolVLA, not a 4B VLM.

- [ ] **T7.1** Baseline run from `lerobot/smolvla_base` on the pick-only subset, to establish the
      pipeline and a reference number.
- [ ] **T7.2** Full run on pick + transfer. Kaggle's free tier (~30 GPU h/week, 16 GB) is the
      intended compute; local 6 GB is the fallback with LoRA + bf16 + grad accumulation.
- [ ] **T7.3** A real eval protocol, fixed before training: held-out seeds, 100 rollouts,
      reporting **grasp success**, **correct-object selection with both wrenches present**, and
      **transfer success (right object AND right table)** as three separate numbers. Assert the
      instruction string that actually reaches the policy each episode — a stale cache produced
      a fake result in the previous project.
- [ ] **T7.4** Checkpoint selection on the eval metric, not on training loss; keep the eval
      rollouts on a fixed seed set so checkpoints are comparable.
- [ ] **T7.5** Language-sensitivity test: swap the instruction with the scene fixed and measure
      how much success drops. If it barely drops, the policy is ignoring language — that is a
      finding, and it must be fixed (more paraphrases, harder distractors) before Phase 3.
- [ ] **T7.6** Data-scaling curve (50/100/200/400/800 demos) if time allows — cheap, and it
      answers "how much data does this need?".

Targets: grasp ≥70%, correct-wrench ≥80% (plan's gate), transfer ≥60% as the new-feature gate.

---

## T8 — Grounding: `locate()` `[needs T0]`

- [ ] `vlm.point(image, description) -> (u, v) | None` behind one function, separate process,
      4-bit, **never inside a control loop** (2–5 s per call). Model chosen in T0 — a 4-bit
      pointing specialist, not Molmo2-ER, unless T0.3 says otherwise.
- [ ] `locate(description) -> Pose3D | None`: point → wrist-camera depth lookup → camera frame →
      base frame. The **`None` case is required** (it drives the floor-sweep recovery).
- [ ] Score grounding error against ground truth (sim gives it free): report median error and
      miss rate. Do not feed ground truth into the pipeline.
- [ ] Scan strategy: the gripper occludes the middle of the wrist view, so `locate` takes 2–3
      views at different `arm_joint1` offsets and merges.
- [ ] **T8.6 (conditional stretch, condition NOT triggered) fine-tune the grounding model** on sim-generated point
      labels — LoRA on Qwen3-VL-2B/4B or the chosen Molmo pointer, using the unlimited exact
      labels sim provides. Do this ONLY if T0.3 shows no zero-shot candidate separates the
      10 mm from the 13 mm wrench. Overfitting to sim is acceptable here (the project is
      simulation-only and says so), and it is a second fine-tuning result for the write-up.
      **T0.3 (session 3) means this is now OPTIONAL:** zero-shot fails the wrench distinction
      on size but clears the gate on colour, so the LoRA is an upside, not a rescue.

---

## T9 — Orchestrator `[needs T5, T7, T8]`

- [ ] Tools: `navigate_to`, `locate`, `grasp`, `place`, `release`, `stow_arm`, `ask_human`.
- [ ] State machine with explicit gate checks (base at pose? gripper holding? point returned?),
      timeouts, and an `on_failure` transition per state.
- [ ] `stow_arm` needs its own tuning: folding the arm with a tool in the jaws dropped it in
      testing, so stow must keep the wrist orientation and move in Cartesian space.
- [ ] Nominal decompositions: `navigate_to(bench) → locate → grasp → navigate_to(human) → release`
      and `navigate_to(table A) → locate → grasp → navigate_to(table B) → place`.

---

## T10 — Recovery scenarios `[needs T9]`

Keep 1, 2 and 4 if time is short (the plan's own cut line):
1. tool on the floor → `locate` returns `None` → floor sweep (head camera).
2. mid-carry drop → gripper state transition → return and re-grasp.
3. obstacle blocks the route after planning → replan (the `obstacle` mocap body already exists,
   parked at (0, −8)).
4. both wrenches present, command says only "wrench" → `ask_human`.
5. "actually the 13mm" typed mid-walk → abandon and re-target.

---

## T11 — Evaluation suite `[needs T9, T10]`

- [ ] 50 nominal trials (5 objects × 10) + the transfer task + 5 scenarios × 5 trials.
- [ ] Every failure tagged with a cause; the failure-cause table is the headline result.
- [ ] Latency budget per layer (afternoon's work, systems people love it).

## T12 — Publish `[needs T11]`

- [ ] 90 s video: nominal, transfer, recovery montage, push-recovery ablation; captions.
- [ ] README leading with "runs entirely on one 6 GB consumer GPU, no API keys"; architecture
      diagram; `docker compose up` repro; eval logs committed; the deviations from `STATUS.md`
      stated honestly with their measurements.
- [ ] Blog post: the two simulation bugs and how they were found, the gripper swap, the payload
      ablation, grounding error stats, state-machine design.
- [ ] Post and send to the five target companies.

---

## Change log (append one line per config change, with its benchmark number)

| When | Change | Benchmark |
|---|---|---|
| session 1 | parallel-jaw gripper replaces the Z1 rotary jaw | grasp 10% → 30% |
| session 1 | standing heights computed from geometry (tools were wedged 2 cm into the bench) | 30% → 33% |
| session 1 | stiff pad contacts (`solref 0.002`, tight `solimp`) | 33% → 47% |
| session 1 | retreat 0.22 → 0.13 m, squeeze 12 → 8 mm past contact | 47% → 48% |
| session 1 | rack slot gap 24 → 30 mm (screwdriver handle never seated) | 48% → **50%** |
| session 1 | re-point at pre-grasp + 1.2 s settle | **not yet benchmarked (T1.2)** |
| session 3 | success scored after a 2 s static hold, not at end of motion | 69% -> **31%** (the old number was an artifact) |
| session 3 | CPU scene `timestep` 0.002 -> 0.001 and `cone` elliptic -> pyramidal (the pad-contact creep) | 10/32 -> 26/32 held-2s |
| session 3 | tape roll grasped 45 deg up the rim instead of at the equator (below the plate tops) | tape_roll 2/8 -> 8/8 |
| session 3 | staging waypoint above the pre-grasp (the scan->pre-grasp swing swept the rack) | 92/100 -> 97/100 |
| session 3 | `settle_static` (velocity threshold) replaces the fixed 1.2 s settle | (same run) |
| session 3 | **100-episode benchmark** | **97/100** (96/100/96/96 per tool) -- **T1 acceptance met** |
| session 3 | pliers GRASP_Z 0.112 -> 0.080 (grip-to-centre 92 -> 53 mm) | pliers held-5s 0/8 -> 8/8 |
| session 3 | HOLD_VERIFY 2 -> 4 s (2 s hid a drop at ~2.3 s) | 95/100 at 4 s (4 tools) |
| session 3 | screwdriver stands handle-down; back in GRASP_TOOLS, GRASP_Z 0.064 | spontaneous falls 9/40 -> 0/40; screwdriver 0/8 -> 12/12 |
| session 3 | servo target ramped across each 10 Hz tick (the "vibration") | see STATUS header |
| session 1 | grounding model: 4-bit pointing model (3.7 GB) chosen over Molmo2-ER (19.4 GB F32) | see T0 |
