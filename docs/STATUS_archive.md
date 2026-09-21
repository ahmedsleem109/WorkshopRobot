# STATUS archive -- superseded session narratives

Moved out of `STATUS.md` on 2026-09-20 (session 6) to keep the live document readable.
Nothing here is wrong; it is simply no longer the current state. Every measurement that is
still load-bearing was carried forward into `STATUS.md` -- these are the narratives around
them, kept because they record WHY things are the way they are.

## SESSION 6 PLAN (kept for its measurements; what came of it is in "SESSION 6" below)

**D: was FULL at the start of session 6 (68 MB free of 318 GB); ~39 GB has since been freed.** That is what killed the first full
SmolVLA run: WSL's `D:\WSL\Ubuntu\ext4.vhdx` (141 GB) could not grow, the ext4 inside went
READ-ONLY mid-training, and the distro then refused to start until space was freed. Nothing in
this repo is the cause -- the big consumers on D: are `D:\WSL` 141 GB, `Adas` 43 GB,
`humanoid` 30 GB, `openarm_data` 22 GB, `hand` 15 GB. Free **at least ~20 GB on D:** before
running anything long. (Session 5 freed ~4 GB INSIDE the WSL disk -- the shard datasets, the
smoke run and the baseline checkpoints -- which is why it runs at all now.)

### 1. Finish T7 -- the only thing on the critical path
The data is done and the pipeline is proven end to end (collect -> convert -> train -> serve ->
roll out). What is missing is a trained checkpoint and its numbers.
```
bash ops/train_vla.sh vla_full 6000 16 bw_demos 3000    # ~1.8 h at 1.07 s/step, bs 16, 4.5 GB VRAM
bash ops/vla_server.sh ~/bringwrench/runs/vla_full/checkpoints/last/pretrained_model
D:\hexapod\render_venv\Scripts\python.exe scripts\eval_vla.py 20 --transfer --swap --out runs\eval\vla_full.json
```
- Dataset: `~/bringwrench/data/bw_demos`, **1,128 episodes / ~104k frames**, LeRobot v3.0,
  feature keys `observation.images.camera1` (wrist) + `camera2` (mast) so `smolvla_base` needs no
  rename map. Raw episodes are in `D:\bw_data\raw` (1 GB) if it must be re-converted
  (`bash ops/convert_all.sh`, 8 shards, ~25 min).
- Reference so far: **T7.1 pick-only baseline** (120 episodes, 2,000 steps, loss 0.67 -> 0.060)
  scored **grasp 1/10** on held-out seeds -- the pipeline works; the data scale was the point.
  The full run reached loss ~0.19 at step 2,000 before the disk died; two runs were lost to it.
- The eval protocol is FIXED and written (`scripts/eval_vla.py`): held-out seeds (the collector
  used `rng(10_000_019 + seed)`, the eval uses `1000*seed + tool`), all five tools present, and
  grasp / correct-object / transfer / language-swap reported separately. T7.4 (checkpoint
  selection on the eval metric) needs two checkpoints scored on the same seeds -- note the GPU
  cannot train and serve at once (exclusive mode) and CPU inference is ~50 s per action chunk,
  so plan on alternating.
- Then run the end-to-end suite with the VLA in place of the demonstrator:
  `scripts\eval_suite.py --suite all --n 10 --backend vla`, and compare with the scripted numbers.

### 2. The dominant failure is LOCOMOTION, not manipulation -- resume T3.5
13 of the 16 end-to-end failures are the base falling, almost all on the 12 cm step DOWN off the
walkway while carrying a tool (bring-me 25/30). The step curriculum run `payload_l5` stopped at
4.59M of 10M steps and was never evaluated. Resume it, re-run the height sweep, then re-measure
the bring-me suite. Already tried and measured, do not repeat: `stow_arm()` (needed, kept), a
via-waypoint past the step so it is not crossed at creep speed (17/20, vs 16/24 crossing at full
speed), tucking the tool further in (0.15, 0, 0.52) -> 7/12, clockwise-only turns (needed).

### 3. Grounding: the pliers hole
`locate()` is at 13.4 mm median xy error but **0/7 on the pliers** -- Qwen3-VL-2B answers "There
are none." to "pliers" in 11 of 17 visible views. Same fix pattern as the wrenches: describe a
visible feature in `ALIASES` ("red pliers"), then re-score. 32 already-rendered seeds are waiting
in `runs/t8_views` (`vlm_replay.py runs\t8_views --start 8`, then `score_locate.py`). The server
now takes `--device cpu` when the GPU is busy. Also re-time nf4 (1.66 GB) on an idle machine: the
full pipeline needs the VLM and SmolVLA co-resident on 6 GB.

### 4. Smaller open items
- **S3 obstacle recovery does not work** (0/2): the blockage is detected and a detour planned, but
  the navigator cannot free the base from contact with the box. Needs a local planner.
- S5 retarget is 16/20; the 3 losses are the returned tool or its neighbour knocked over in the
  rack by `return_to_rack()`, after which it lies below the plate tops and `locate` cannot see it.
- 3 transfers in 30 put the tool outside the zone (the tail of the place distribution).
- ~12 raw episodes are corrupt (collector workers killed mid-write); conversion skips them.
- T12: `README.md` and `docs/blog.md` are drafted and need the VLA numbers; the 90 s montage is
  not cut yet (clips: `media/bring_10mm_wrench_small.mp4`, `recover_drop_small.mp4`,
  `transfer_side_table_small.mp4`).


## SESSION 5 (2026-09-20) -- everything on legs; T6 collected; orchestrator + eval suite

### 1. Grasp on legs: 8/25 -> 125/125 (the tape roll was a GEOMETRY bug, not a control bug)
With the legs on the walking policy the standing base is NOT a fixed frame. Two measurements:
- Reaching in pushes the base back **25-35 mm** (`grasp_diagnose.py --walk` logs base xy now).
- The tape roll's failures started BEFORE the close: gripping the rim at 45 deg above the
  equator (TAPE_PHI) with a laterally-closing jaw puts the pad's upper corner into the ring's
  crown, so the **pad TIP hit the ring at ~310 N during the approach**, knocking the ring over
  or shoving the whole robot back.
**Fix:** grasp the ring RADIALLY at its crown -- jaw axis along the ring's radius, one pad
inside the hole, pads flat on the 7 mm wall (`tape_radial()`), with a LEVEL approach only
(a pitched approach tilts the pads off the flat crown; pinned 7/12 with pitch, 12/12 without).
Held at the crown the ring also hangs under its grip point, so the lift no longer swings it.
- A closed-loop ee re-servo before the close was tried and REJECTED: with a compliant base it
  chases the contact it is making (base pushed back 150-250 mm, 3/6).
- A joint-PD "stand lock" for the legs during manipulation (`Locomotion.lock_stance`, what a
  real Go2's balance-stand does) is IN and used by grasp/place; `walk_to` unlocks.

| grasp, 25 seeds/tool, legs on the policy (`try_grasp.py 25 --walk`) | |
|---|---|
| wrench_10mm / wrench_13mm / screwdriver / pliers | 25/25 each |
| tape_roll | 25/25 (also 25/25 with +-3 cm / +-6 deg base jitter, the walk's own stopping spread) |

### 2. Place on legs: every tool >= 92%, both tables
The screwdriver was 16/25 on table A. Cause, from the phase trace: `place_rolls()` rolls the
wrist so the tool "hangs down", which for a handle-gripped screwdriver means **flipping it 180
deg and standing it on its SHAFT TIP** -- it toppled 100-160 mm every time. Fixes:
- `STAND_UPRIGHT = {"screwdriver"}`: place it upright, handle down, the way it stood in the rack;
  no topple-aim shift. (Laying it flat is out of IK reach at the place station; a 45 deg tilt
  left the handle leaning on a pad and the back-off dragged it ~190 mm out of the zone.)
- For those tools the retract goes UP first, then back.

| transfer on legs (`try_place.py 25 --walk --table X`) | table A | table B |
|---|---|---|
| wrench_10mm | 25/25 | 25/25 |
| wrench_13mm | 23/25 | 25/25 |
| screwdriver | 24/25 | 25/25 |
| pliers | 23/25 | 24/25 |
| tape_roll | 25/25 | 25/25 |
| **total** | **120/125 (96%)** | **124/125 (99%)** |

### 3. T6 -- 1,128 demonstration episodes (LeRobot **v3.0**, not v2.0)
`scripts/collect_demos.py` runs the full transfer on legs and saves up to two episodes per run:
`pick` (instruction from PICK_TEMPLATES) and `place` (TRANSFER_TEMPLATES), successes only, judged
by `bw/task/spec.py`. Per 10 Hz tick: **wrist + mast RGB 256x256**, arm state (7) and the
commanded target (7). Randomised: tool subset/slots/lean/flip, lighting, rack colour, base pose
(+-3 cm / +-6 deg), scan pose (+-0.05 rad), destination table, phrasing.
- A **mast camera** was added to the robot (`build_models.py`): the head camera sits at bench-panel
  height and sees only the bench front / table legs while manipulating.
- 600 runs, 3 parallel Windows sims, ~28 s per run. 598 picks + 540 places kept; ~12 raw videos
  were corrupted by killed workers and are skipped at conversion.
- `scripts/to_lerobot.py` converts (8 parallel shards + `aggregate_datasets`; H.264 instead of
  lerobot's SVT-AV1 default, which cost ~16 s per episode). Result: **1,128 episodes / ~104k
  frames**, features named `observation.images.camera1|camera2` so `lerobot/smolvla_base` needs
  no rename map. lerobot 0.6.1 writes dataset **v3.0** -- the plan's "v2.0" is the older layout.

### 4. T9 orchestrator + T10 recovery + T11 suite
`bw/orchestrator.py`: parse -> NAV_RACK -> LOCATE -> GRASP -> [STOW] -> NAV_DEST -> PLACE|HANDOFF.
Every transition is on an OBSERVABLE gate (walk_to's own verdict, locate returned a point,
`gripper_state() == "holding"`, gripper empty after release); bounded retries; skills are
pluggable (`backend="scripted"|"vla"`, `grounding="vlm"|"oracle"`). Grounding picks the tool:
the scripted grasp is aimed at whichever tool the located point is nearest, so a grounding error
shows up as a wrong-object failure, as it would on hardware.
- **Handoff**: the tray is now on a **0.45 m stand** (`HANDOFF_Z`), not the floor -- with a level
  grip the arm cannot get below ~0.5 m over a tray 0.55 m ahead, so a floor tray meant a 0.5 m
  drop (screwdriver bounced out, tape stayed hooked on a finger). The move to the tray is
  JOINT-space to a multi-start IK solution: a Cartesian line from the stow pose sent the
  incremental IK onto another branch and pulled the arm back 0.3-0.4 m.
- **Route to the human** crosses the 12 cm step DOWN off the walkway. It needs `stow_arm()`
  (Cartesian pull-in, `STOW_B`) and a via-waypoint at x=0.3 so the step is not crossed at creep
  speed: 17/20 walks reach the human; crossing at full speed (via x=-0.3) is worse, 16/24, and
  tucking the tool further in (0.15, 0, 0.52) is worse again, 7/12. Left turns at the rack fell
  3/3 (mirrored policy), so the route turns clockwise only (`turn_sign=-1`).

**T11, scripted skills + oracle grounding (`scripts/eval_suite.py`), 130 trials:**
| suite | success | failure causes |
|---|---|---|
| transfer (pick -> walk -> place in the named zone) | **27/30 (90%)** | 3 placed outside the zone |
| bring-me (pick -> walk to the human -> handoff) | **25/30 (83%)** | 5 falls on the step down |
| missing tool (S1: absent, must report and deliver nothing) | **20/20 (100%)** | -- |
| mid-carry drop (S2: jaws opened mid-walk, human returns it) | **26/30 (87%)** | 4 falls on the outbound walk (3 were tagged `nav_rack_failed`: the robot had already fallen, so the recovery started from a fallen base -- the attribution is fixed in `bw/orchestrator.py`, the outcome is not) |
| ambiguous "wrench" (S4: must ask, then deliver the answer) | **16/20 (80%)** | 4 falls on the step down |
| retarget (S5: "actually, the 13mm" after the first tool is in the jaws) | **16/20 (80%)** | 3 lost the second tool after the return-to-rack, 1 fall |
| obstacle (S3: a box appears on the route to table B) | **0/2 -- does NOT work** | the blockage IS detected and a detour is planned, but the navigator cannot extract the base from contact with the box |
| **total (the five working suites)** | **130/150 (87%)** | walking 14, placement 3, lost tool 3 |

**S5 (retarget)** needed a skill the plan did not list: `return_to_rack()`, the grasp played
backwards, because the robot has to get rid of the tool it is holding before fetching another
one, and it cannot walk anywhere useful to put it down -- the station-to-station hop is 0.6 m
sideways and the policy's sidestep only manages ~0.1 m of it (measured). Putting the tool back in
its own slot works 16/20; the 3 losses are the returned tool or its neighbour being knocked over,
after which it lies below the rack plates and `locate` cannot see it.

**S3 (obstacle) is an honest failure.** `navigate_to_replan()` detects the stall, backs off and
plans a detour around the straight line, and the robot still cannot get free: it ends wedged
against the box or falls. A local planner (or any obstacle perception at all) is missing; the
code path is in `bw/orchestrator.py` and the scenario is in the suite, both marked as not working.

Median wall clock per layer: locate 2.2 s, grasp 6.3 s, walk 5-10 s, place 4.2 s; median 54 s of
simulated time per trial. **The dominant end-to-end failure is locomotion, not manipulation**:
13 of the 16 failures are the base falling, almost all of them on the step DOWN off the walkway
with a tool held. That is exactly what the unfinished T3.5 step curriculum is for.

### 5. Other fixes
- `locate()`'s `NEAR_CLIP` was 0.25 m, which also rejected tools at the rack ends -- they sit
  0.22 m from the wrist lens. Now 0.16 m (the gripper reads 0.105 m median).
- `try_place.py --walk` now scores with the shared `evaluate()`, like the teleport path.
- `try_grasp.py` gained `--walk`, `--tool`, `--jitter`; `grasp_diagnose.py` gained `--walk` and
  logs the base pose.


## SESSION 4 (2026-09-19) — the base WALKS between tables; T2.3 at 95-96%

**Why:** the transfer clip's base move was `teleport_base`, a kinematic slide with frozen legs.
That is replaced: the Layer 3 policy now owns the legs for the whole episode, and
`bw/locomotion/navigate.walk_to()` drives it with velocity commands only.

**What it took (each measured):**
1. `controller.py` ran the policy with **ReLU; brax trains SWISH**. 0.19 m -> 1.70 m in 3 s at
   vx 0.5; MJX gives 1.76 m. T5's sim-to-sim check is effectively done (`scripts/nav_tracking.py`).
2. The 33M payload policy **only walks forward >= 0.2 m/s**: pure turn, back-up, sidestep and
   slow all STAND (CPU and MJX agree). The reward made standing optimal. Nav fine-tune
   (`configs/payload_nav.yaml`, header has the history):
   run 1 sigma 0.25 -> nothing learned in 5M; run 2 sigma 0.1 -> turn R (4M) then turn L (7M);
   run 3 back-up weighted 35% -> **back -0.16/-0.25, side +0.14/-0.10 at 1.8M**. Run 3 is still
   TRAINING (8M total, ~2.5 h from 16:27): `~/bringwrench/runs/results/*payload_nav3`.
   Export: `JAX_PLATFORMS=cpu ... export_policy <ckpt> models/X.npz` (CPU, or it grabs GPU memory).
   `models/nav3_b.npz` = run 3 @ 1.8M, the one everything below used.
3. Left turns can also be MIRRORED (`Locomotion.mirror_when`): the Go2 is symmetric and the
   policy does not see the arm.
4. Station B's front feet were **16 cm past the walkway edge** (hidden by pinned legs): walkway
   now runs to the bench (x 4.45) + `WALKWAY_SPUR_B` under station B. Table heights unchanged.
5. Navigator shaped to the policy: deadband (turns >= 0.45 rad/s, 0.25 m/s creep), in-place
   turns DRIFT ~3.5 cm/s (big turn at a far waypoint; at the rack the drift pinned the robot
   against the bench -> must BACK UP first), lateral error fixed by sidestep, not arcs.

**Walking transfer, FINAL (`try_place.py 25 --table X --walk`, default policy =
`models/payload_nav_policy.npz` = run 3 final @ 8.1M; back -0.25/-0.25, turns +-0.63, side
+0.18/-0.14):**
| | walks reaching station | full transfer | failures |
|---|---|---|---|
| table B | **107/107** (median 19 mm / 8.5 deg, 16.5 s) | 100/125 (80%) | tape grasp 17, outside_zone 6 (pliers 5), drop 1, grasp 1 |
| table A | **107/107** (median 28 mm / 5.6 deg, 16.0 s) | 93/125 (74%) | tape grasp 17, outside_zone 11 (pliers 6, screwdriver 3), not_settled 2, drop 1 |
Walking itself is solved (214/214, no falls). Excluding the tape roll: 93/100 (B), 88/100 (A).
The navigator needed (all measured, all in navigate.py): stall kick (the policy slips into its
stand state mid-creep and will not restart at 0.25 m/s), stop-then-correct fine alignment,
yaw trimmed FIRST only (a yaw trim slides the base ~10 cm sideways, or toward the bench).
Earlier numbers with the 1.8M checkpoint: B 20/25, A 17/25 (5 seeds).

**T2.3 (teleported base, `try_place.py 25`): table B 78% -> 96%, table A 80% -> 95%.**
- The topple after release has a direction: aim `PLACE_AIM_SHIFT` 45 mm past centre, re-aimed
  from the tool's measured lean before descent (`scripts/topple_diagnose.py`).
- Grip force PER TOOL (`GRIP_FORCE`, `GRIP_KP` 4000): 20 N lets the 13 mm wrench creep out
  (55 mm/s), 60 N extrudes the 10 mm wrench and wedges the tape roll out.
- Grasp 119/125; every tool >= 96% except **tape_roll (grasp 21/25; transfer 22/25 B, 20/25 A)**:
  its failures start with the ring pinched against a rack plate at 75-120 N when the jaws close.

**Next, in order:** (a) TAPE ROLL grasp on legs: 8/25 -- the standing base drifts ~27 mm during
the grasp (pinned: 21/25); the close already pinches the ring against a rack plate; (b) place on
legs, pliers 19-20/25 outside_zone (pinned 24-25/25) -- standing sway during the release;
(c) then T6 data collection runs in THIS mode (legs on the policy), not the pinned one.
Clip: `media/transfer_screwdriver_table_b.mp4` (walking, SUCCESS).

---


## START HERE — next session

**Done in session 3:** T1 (grasp) closed, T0.3 answered (with the T0.4 decision), and T2 mostly
built: the second table, the place skill, the shared success spec, and the language templates.
Session 2 had already finished T3/T4 (locomotion).

**Current numbers (5 tools, 25 seeds each, `try_grasp.py 25` / `try_place.py 25 --table X`):**

| | wrench_10mm | wrench_13mm | screwdriver | pliers | tape_roll | overall |
|---|---|---|---|---|---|---|
| **grasp**, held 4 s | 25/25 | 23/25 | 25/25 | 25/25 | **21/25** | **119/125 (95%)** |
| **transfer → table B** | 23/25 | **13/25** | 24/25 | 21/25 | **16/25** | **97/125 (78%)** |
| **transfer → table A** | 24/25 | **13/25** | 24/25 | 19/25 | 20/25 | **100/125 (80%)** |

The grasp passes the overall bar, but tape_roll (84%) is under the 90% per-tool bar at the
stricter 4 s hold. Transfer is **below T2.3's 90% bar**. 42 of its 53 failures are
`outside_zone`, and wrench_13mm alone accounts for 24: it is the longest tool, and it topples
furthest after release. The tape roll's transfer fell from 21/25 to 16/25 on table B after the
servo-ramp change and has not been diagnosed yet.

**Decisions taken in session 3, do not relitigate:**
1. **Grasp success is scored after a 4 s static hold** (`HOLD_VERIFY`). Scoring at the end of
   the motion inflated everything: 69% read as 31% once the tool had to stay put for 2 s. 2 s
   was still too lenient, because the pliers passed it and dropped at ~2.3 s.
2. **CPU scene: `timestep 0.001`, `cone="pyramidal"`.** This fixed the pad-contact creep, where
   the pad time constant was exactly 1× the timestep (see "THE CREEP"). `impratio`,
   `noslip_iterations` and a 0.5 ms timestep were all tried and do not help further.
3. **The servo target ramps across each 10 Hz tick** (`RAMP_CHUNK`, `WorkshopSim.move_arm`).
   A stepped target made the arm stop dead 10 times a second; on video that is the robot
   "vibrating". Joint-speed peak/mean went from 4.2× to 1.05×. The recorded action is unchanged.
4. **The screwdriver stands HANDLE-DOWN and is a grasp tool again.** Shaft-down, it fell over on
   its own in 9/40 untouched scenes. Handle-down it never falls, and it grasps 12/12 on the
   handle (`GRASP_Z` 0.064).
5. **Layer 1 = Qwen3-VL-2B, with size mapped to grip-band colour.** It is at CHANCE on 10 mm
   vs 13 mm by size and 92.9% by colour (T0.3 section below). This is a stated limitation.
6. **Two tables:** A = the workbench with the rack (left / far), B = the side table (right /
   near). Each has a painted green place zone. `PLACE_ZONE` / `PLACE_STATION` in
   `bw/sim/workshop.py`, measured by `scripts/reach_audit.py` and the `--back` sweep.
7. **Base moves between stations with `WorkshopSim.teleport_base`**, a rigid kinematic stand-in
   for Layer 3 walking. The clips caption it as such. Wiring the real policy in is T5 → T9.

**Next tasks, in order:**

| # | task | why now |
|---|---|---|
| **T2.3 finish** | Transfer ≥90% per tool | The dominant failure is `outside_zone`: the tool is released hanging and TOPPLES as it lands, and long tools (wrench_13mm) travel furthest. Refuted so far: aim jitter (moves the median, not the tail) and releasing laid flat (worse, 58 mm median). Untried: lower the release so the topple starts from contact rather than a drop, tip the tool over with the gripper still closed, or aim the release so the topple direction points INTO the zone (it is predictable from the lean). |
| **T6** | Data collection | Blocked on T2.3. `bw/task/spec.py` and `bw/task/language.py` are ready for it; `run_grasp`/`run_place` take `record` and `on_phase` callbacks. |
| **T5** | Validate `controller.py` against MJX | Small, self-contained, blocks T9. |
| **T8** | `locate()` behind `vlm.point()` | T0.4 decided. Must use 2–3 views: the target is occluded in 27.5% of scan-pose views. |

**Clips:** `media/transfer_screwdriver_table_b.mp4` (the current one: both tables, command +
live action, no vibration), `media/transfer_wrench_10mm_table_b.mp4` (made before the vibration
fix), `media/grasp_*.mp4`. Render with `scripts/make_transfer_video.py TOOL TABLE SEED`.

**Unfinished business worth knowing:**
- The pad creep is 36× smaller but NOT zero: ~7 mm/s during lift and retreat on the 13 mm
  wrench. Tools usually survive because their head catches on the pads. It is the root of the
  remaining grasp drops.
- `media/loco_payload.npz` was dumped but never rendered (`scripts/render_traj.py`, Windows).
- The push-ablation anomaly (original stowed 160 N < extended 200 N) is reproducible and
  unexplained. Do not publish that table until it is.
- The GPU clock cap (`nvidia-smi -lgc 300,1100`, Administrator) does not survive a driver
  re-init. `nvidia-smi -rgc` releases it; `ops/thermal_guard.sh` stops training at 88 °C.

Plan of record: `bring-me-the-10mm-wrench-plan (1).md`. Upstream locomotion project:
`D:\hexapod` on Windows, `~/go2-stairs` in WSL.

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


## T5 (session 5)

**T5 part 1 PASSES: `controller.py` matches the MJX env + brax inference element-wise.** No
changes to `controller.py`. Script: `scripts/t5_obs_check.py` (WSL, `JAX_PLATFORMS=cpu`, ~1.5 min),
`models/payload_nav_policy.npz` vs its checkpoint `payload_nav3/checkpoints/step_8110080`,
DR/noise/latency/pushes off, arm stowed, 100 steps of forward 0.5 / turn R 0.6 / back 0.25 / side L 0.2.
- Weights: re-export of the checkpoint == the .npz (diff 0). Default pose / default ctrl / ctrl
  range on `workshop.xml` vs the MJX model: <= 5e-8; dt, obs scales, clip, action scale, history equal.
- Teacher-forced (MJX state copied into CPU MjData each tick; controller builds its own 5-frame
  history and action), max abs diff over 100 steps x 5 frames: lin_vel 0, proj_grav 7e-9, gyro 0,
  accel 1.2e-7, joint_pos 6e-8, joint_vel 1.5e-8, last_action 1.8e-6, command 0;
  **action 1.8e-6** (bar 1e-5). Float32 rounding only; ordering (last_lin_vel lag) is correct.
- Closed loop (each on its own physics, same start): action diff 3e-7 at step 1, 9e-4 at 10,
  ~1e-2 at 50-100; base xy 1.7 mm apart after 2 s. That is MJX-vs-CPU solver drift, not the controller.
- Caveats: `Locomotion.reset()` always starts from a zero command (the check aligns the MJX reset
  frame to that); left-turn mirroring is a deliberate deployment deviation and was disabled here.
- Still open in T5: part 2 (`base_mode="policy"` standing still under zero command during a grasp).

---


## T0.4 + T8 (session 5)

**Files (new, uncommitted):** `bw/perception/vlm_server.py` (the model process, WSL torch venv),
`bw/perception/vlm.py` (client + `point()` contract, stdlib+numpy only), `bw/perception/locate.py`
(`locate()`), `scripts/bench_locate.py` (Windows: live sim -> scan -> model), `scripts/vlm_replay.py`
(re-query stored views), `scripts/score_locate.py` (numpy-only scorer, all variants offline).
Data: `runs/t8_views` (40 seeds rendered, **8 scored** with bf16), `runs/t8_views_nf4` (2 seeds, nf4),
`runs/t8_smoke` (2 seeds, the rejected "reply none" prompt).

**Contract.** `vlm.point(image_rgb, query) -> (u, v) | None`. The model runs in its own process
(HTTP on :8765, raw RGB bytes in JSON); `vlm.ensure_server()` starts it via `wsl.exe` from Windows
or directly in WSL; Windows -> WSL localhost forwarding works. `point()` does:
1. `resolve_query`: "10mm wrench" -> "wrench with the blue grip band", 13mm -> red (T0.3 decision).
2. coarse point on the full 512 image (T0.3 prompt, 0-1000 normalised).
3. **refine**: 160 px crop around it, 3x upscaled, ask again, map back. Median distance to the
   target's pixels 11.8 px -> **2.7 px**; locate success 28% -> 50% (before the merge fix below).
4. (optional `verify=True`: yes/no on the crop -- measured, it COSTS recall (false None 23% -> 32%)
   and did not reject any hallucination on absent tools, so it is off.)

`locate(sim, "10mm wrench")` pans `arm_joint1` over (0, +0.35, -0.35) from `SCAN_Q`, points in each
view, back-projects through the wrist depth using the robot's own FK camera pose, and merges.
Two things mattered, both measured on the same 8 seeds:
- **Depth snap to foreground.** A pixel beside a thin tool reads the BENCH depth and lands 10-30 cm
  too far along the ray. `snap_foreground` moves the answer to the nearest pixel >=3 cm in front of
  the local background (80th pct depth in 81x81 px). xy error median 67.8 mm (window-percentile
  depth) -> 25.9 mm.
- **Merge tie-break.** When the 3 views disagree (all clusters of 1), prefer the view whose answer
  already sat ON an object (smallest snap distance). xy median 25.9 -> **13.4 mm**, success 50 -> 64%.

**Results, bf16, 8 seeds x 3 views x 5 queries = 120 queries (every tool asked in every view, present
or not), 36 present tool instances.** Run under heavy CPU/GPU contention (the other agent's 12
render processes), so latency is pessimistic.

| T0.4 `point()` per view | |
|---|---|
| lands on the target (<=3 px of its GT segmentation) | 37/93 visible (39.8%); median px distance to target **2.7 px**, p75 20.5 |
| lands on a different tool | 6/93 (6.5%) |
| false None on a visible target | 21/93 (22.6%) -- almost all **pliers (11/17) and tape (7/21)** |
| 10 vs 13 mm when both visible and it landed on a wrench | 14/18 (77.8%) |
| correct None on absent tools | 6/12 (50%) -- the model hallucinates a point half the time |
| latency | 2 model calls; **uncontended ~0.7-0.8 s per call** (smoke run), contended median 2.9 s / point() |
| VRAM | bf16 **4.27 GB**; nf4 **1.66 GB** |

| T8 `locate()` per (seed, tool) | |
|---|---|
| miss rate (present -> None) | **4/36 (11.1%)** (3 pliers, 1 tape) |
| horizontal (xy) error vs GT body | **median 13.4 mm**, p75 52.9, p90 345 |
| distance to the target's visible surface | **median 1.8 mm**, p75 20.1 |
| 3D error vs GT body origin | median 79 mm -- mostly Z: the model points at the TOP of a standing tool, the body origin is mid-height. Use xy + known rack/table height for grasping |
| found, nearest tool is the target, xy < 30 mm | **23/36 (63.9%)** |
| per tool | wrench_10mm 6/7 right, 9.9 mm; wrench_13mm 6/7, 13.2 mm; screwdriver 6/7, 2.6 mm; tape 7/8 found, 25.7 mm (xy vs the RING CENTRE; surface error ~1 mm); **pliers 0/7 right** |
| false positive (absent tool -> a point) | 3/4 |

**nf4 (2 seeds only, same views; the coordinator asked for it):** 1.66 GB VRAM; locate xy median
43.6 mm vs 32.8 mm bf16 on the SAME 2 seeds, success 4/9 both, false-None 3/22 vs 6/22. Too few to
separate accuracy; **latency is ~4x worse** (single call 9.5-10.6 s vs 2-5 s for bf16 under the same
contention; 16.9 s median per point()). If VRAM forces nf4 next to SmolVLA, re-measure latency on an
idle machine before deciding; 8-bit is the untested middle.

**Open problems (ordered):**
1. **Pliers are not recognised** ("There are none." on 11/17 visible; 0/7 located). The primitive
   red V geometry does not read as pliers. Same fix pattern as the wrenches: a visible-feature
   description in `ALIASES` (e.g. "red pliers") -- an ablation was started and not finished.
2. **"red grip band" vs the red pliers**: the 13 mm query sometimes lands on the pliers (both red).
   Try "silver wrench with a red band on the handle".
3. **Absent-tool hallucination** (3/4 false positives in locate, 50% per view). Verify-on-crop did
   not fix it. Candidates: require >=2 agreeing views before returning a point; or a colour check.
4. Only **8 seeds** scored for bf16 (the GPU was shared; stopped on request). 32 more seeds are
   already rendered in `runs/t8_views` -- `vlm_replay.py runs/t8_views --start 8` then re-score.

**How to run**
```
# model process (WSL; or let vlm.ensure_server() start it from Windows)
~/bringwrench/.venv-vla/bin/python /mnt/d/bringwrench/bw/perception/vlm_server.py [--quant nf4]
# render + GT (Windows), then query the model, then score (any venv)
D:\hexapod\render_venv\Scripts\python.exe scripts\bench_locate.py 40 --no-vlm --out runs\t8_views
D:\hexapod\render_venv\Scripts\python.exe scripts\vlm_replay.py runs\t8_views [--n 8]
D:\hexapod\render_venv\Scripts\python.exe scripts\score_locate.py runs\t8_views [--stage coarse|fine] [--no-verify] [--mode snap|pct]
# or all-in-one live: scripts\bench_locate.py N  (renders, calls point_ex, saves replies)
```
In code: `from bw.perception.locate import locate; L = locate(sim, "10mm wrench")` ->
`L.world`, `L.base` (base frame), `L.n_views`, or `None`. STOP the server when done
(`pkill -f 'perception/vlm_server[.]py'`) -- it holds 4.3 GB (bf16).

---

## SESSION 6 (2026-09-20) -- obstacle recovery fixed; T7 re-run; the step is the last blocker

### 1. T10 #3 obstacle recovery: 0/2 -> 9/10 (three separate defects, all measured)
The scenario drops a 0.6 m box on the route to table B the moment the tool is in the jaws.
Session 5 left this at 0/2 with "the navigator cannot free the base from contact with the box".
It was not one bug:

1. **The box lands BEHIND the robot**, so the walk stalls in `walk_to`'s opening *backup* --
   and the old recovery drove backwards (`back_off`), i.e. further into it, then estimated the
   obstacle along the base HEADING, which pointed at clear floor. `walk_to` now records the
   first phase that timed out and the body-frame command it was asking for (`stall` in its
   result), and the blockage is projected along THAT direction. Measured: the estimate lands
   at (2.99, -0.14) against a true box centre of (3.15, -0.28).
2. **The detour waypoint was unchecked geometry.** A fixed +-0.9 m off the midpoint of the
   original line puts it at (4.06, -1.37) -- hard against the walkway's right edge in the bench
   corner -- and the walk there fell. New `bw/locomotion/local_plan.py` plans over the
   STANDABLE surface only (walkway + table B's landing strip, shrunk by a base margin; off it
   is the 12 cm drop) and treats clearance as a soft cost, because where the box is, is a guess.
3. **A detour point was walked as a STATION**, so it got the full backup -> `far` -> `pre` ->
   creep -> trim ritual and was approached down a line running from BEHIND it; one run ended at
   x = 5.2, off the walkway entirely. `walk_to` now takes `via=True`: turn, walk, face the
   goal, nothing else. The last leg of a detour is the station's own approach point, so the
   walk that follows is the short hop and never re-runs the blocked `far`/`pre` line.

Also: the detour's legs turn CLOCKWISE only (`turn_sign=-1`). With a tool held the mirrored
left turn fell 3/3 in session 5, and a detour's first move is a big in-place turn by
construction.

**A fourth defect, found after the rng fix below made the suite reproducible:** the detour then
worked (both legs landing < 70 mm from their waypoints) but the final 0.55 m hop onto station B
**fell off the walkway**, ending at x 3.19, y -1.60 -- and at that y the standable surface is only
the spur under table B, x 2.25-3.15. A single `fine*_along` burst ran until the whole along error
was consumed, up to its 4 s cap (over a metre at V_CREEP), with no lateral re-check inside it, so
sideways drift accumulated unmeasured: it drifted 0.43 m sideways while closing 0.59 m forward.
`STEP_ALONG = 0.25` now ends a burst after a quarter metre of travel so the loop re-measures
lateral error and corrects that first (`FINE_ROUNDS` 6 -> 8 to leave room for the extra segments).
That also removed the obstacle suite's place-outside-zone loss, since the place skill plans from
the live base pose.

| obstacle suite | |
|---|---|
| session 5 | 0/2 |
| detour + local planner | 7/10 (2 falls on the final hop, 1 outside the zone) |
| **+ the along-burst cap** | **10/10** |
| transfer, as a regression check on the same change | 9/10 before, 9/10 after |

### 2. The end-to-end suite, and a bug in how it was being measured
**A seed did not identify a trial.** `trial()` reseeded the SCENE per trial but not the
ORCHESTRATOR: `Orchestrator.rng` drives every scripted skill's randomised move durations and it
carried on from wherever the previous trial left it. So an outcome depended on which trials had
run before it in the same process. Measured, same code and same seeds:
- `--suite drop` alone scores 9/10; `--suite transfer,drop` scores drop **6/10**.
- retarget seed 5 fails inside a 10-seed batch and **passes on its own** (delivered, handoff ok).

Session 5's 130-trial table was one process, so its per-suite numbers carry this noise. Fixed in
`scripts/eval_suite.py`: `orch.rng` is now reseeded per trial from the trial seed, so every
trial is reproducible in isolation. `ops/run_eval_suites.sh` additionally runs ONE PROCESS PER
SUITE, and `scripts/eval_summary.py` tabulates the dumps.

The table below is measured AFTER that fix, on the code with the obstacle recovery and the
along-burst cap in place -- one process per suite, `orch.rng` reseeded per trial.

| suite (10 seeds each, fresh process, per-trial rng) | success | failure causes |
|---|---|---|
| nominal (bring-me) | 8/10 | 2 falls on `via_step` |
| transfer | 9/10 | 1 placed outside the zone |
| missing tool | 10/10 | -- |
| mid-carry drop | 9/10 | 1 fall on `via_step` |
| ambiguous "wrench" | 8/10 | 2 falls on `via_step` |
| retarget | 6/10 | 3 falls on `via_step`, 1 grasp |
| obstacle | **10/10** | -- |
| **total** | **60/70 (86%)** | |

**8 of the 10 failures are the `via_step` leg** -- the 12 cm step DOWN off the walkway with a
tool held. The other two are one placement outside the zone and one grasp. This is T3.5 and
nothing else; the manipulation and grounding layers did not lose a trial to their own skills.

### 3. T7 full fine-tune re-run
`ops/train_vla.sh vla_full 6000 16 bw_demos 3000` on the 1,128-episode dataset: loss 0.67 ->
**0.125 by step 3,900** of 6,000 at 1.07 s/step, 4.53 GB VRAM, GPU 70 C. The step-3,000
checkpoint is saved, so T7.4 (checkpoint selection on the eval metric) has two candidates.
NOTE: `ops/*.sh` live on /mnt/d, NOT in ~/bringwrench -- `bash ops/train_vla.sh` from the WSL
home fails with "No such file or directory"; use `bash /mnt/d/bringwrench/ops/train_vla.sh`.
A `nohup`ed job does NOT survive the `wsl.exe -e` session that started it; keep the session open.

### 4. T12 montage cut
`scripts/make_montage.py` cuts `media/montage.mp4` -- 90.8 s, 4.7 MB: title, grasp (3 tools),
payload gait, table-to-table transfer, bring-me end to end, drop recovery, and a closing card
of the measured numbers. Re-run it after new clips are rendered; it is cheap and declarative.

### 5. Written and ready, not yet run (the GPU was busy training all session)
- `ops/resume_payload_l5.sh` -- T3.5. `payload_l5` stopped at 4,587,520 of 10M (success 0.25 at
  level 5, reward 2646). train.py has no resume flag, so a resume is a warm start from
  `.../payload_l5/checkpoints/step_4587520` with the remaining budget; `level_init: 5` in the
  config puts the curriculum back where it was.
- `scripts/_pliers_probe.py` -- T8. Scores candidate phrasings for the pliers ("red pliers",
  "pliers with red handles", "tool with two red handles", ...) by refusal rate and on-tool rate
  over the views where the pliers are genuinely visible, bypassing `vlm.ALIASES` so the literal
  phrase reaches the model. The control phrase "pliers" is refused outright ("There are none.")
  in 17 of 24 stored replies over seeds 0-7.


### 6. T7, why it does not grasp -- the diagnosis, in the order it was established
Every variant scores **grasp 1/20** on the fixed 20-seed protocol. What changed across the
session is not the score but the understanding, and each step was a measurement, not a guess.

| model | epochs | one-step prediction vs its trivial baseline | grasp |
|---|---|---|---|
| absolute, 6k steps | 0.92 | **2.10x WORSE** | 1/20 |
| delta, 3k steps | 0.46 | 1.20x worse | 0/20 |
| delta, 20k steps (6 GPU h) | 3.0 | **1.68x BETTER** | 1/20 |
| overfit: 40 episodes, 3k steps | 11.7 | **3.8x BETTER** | -- |

1. **Training loss cannot see this failure.** The absolute run reached loss 0.115 and could not
   grasp. `scripts/vla_replay_check.py` was written to measure what loss cannot: the policy's
   error on its OWN training frames, through the SERVING path, in radians, against two
   references -- the spread of the recorded actions, and the trivial predictor (hold the current
   joint position / command no motion).
2. **Conditioning was wrong.** Absolute joint targets mean the per-step motion (~0.026 rad) is
   only 4-8% of the action spread the normaliser divides by (0.28-0.68 rad). The absolute policy
   predicted training actions to +-0.054 rad -- TWICE the motion it had to produce, and worse
   than doing nothing. `to_lerobot.py --delta` records `q_cmd - q_state` instead (gripper stays
   absolute); measured, that rescales the target by **7-17x per joint** and the policy went to
   1.68x BETTER than its baseline. The serving convention is reported on `/health` so client and
   dataset cannot silently disagree.
3. **Capacity and pipeline are NOT the problem.** The overfit test -- 40 pick episodes seen 11.7
   times -- predicts 3.8x better than baseline on every joint. The architecture, the data
   pipeline, the delta representation and the serving path all work.
4. **What is left is COVARIATE SHIFT.** One-step prediction improved 3.5x while closed-loop
   success did not move at all. The two measure different things: prediction is scored on
   training-distribution frames, the eval is scored on 14 s of closed loop in held-out scenes.
   Every demonstration came from a SCRIPTED controller, so the data is nearly noise-free and
   covers a narrow tube of state space with **no recovery states in it** -- the demonstrator
   never made a mistake. The policy can reproduce that tube (step 3) and cannot return to it
   once outside.
   Testing this means re-collecting with noise injected into the demonstrator, which is a data
   change, not a training change. That is the next experiment, and it is NOT more GPU hours.

Also open, from the same numbers: j1 and j5 are the only joints that never beat the baseline,
in BOTH delta runs independently -- worth understanding before the next collection.

---
