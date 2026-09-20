# STATUS — "Bring me the 10mm wrench"

Last updated **2026-09-20, session 5 (paused: the D: drive filled up -- see "NEXT SESSION (6)").** Start with "NEXT SESSION (6)" below the task board. Read this, then `REMAINING.md` for what to do
next (tasks are ordered by dependency there, not by phase).

---

## TASK BOARD — read this first

Task IDs are those in `REMAINING.md`, which holds the full sub-task lists. Status as of the end
of session 3 (2026-09-19).

### DONE
| task | what it delivered | evidence |
|---|---|---|
| **T0.1/T0.2** grounding candidates | Molmo2 mirrors dead (points are special tokens, no loader); Qwen3-VL-2B chosen | Known bug #6 |
| **T0.3** grounding bake-off | 0-1000 normalised coords; 1.12 s/call; **chance on 10 vs 13 mm by size, 92.9% by colour band** | "T0.3 ANSWERED" section |
| **T1** grasp demonstrator | 5 tools, **119/125 (95%) held for 4 s**; the creep, tape grasp point, staging waypoint, settle-to-static, pliers grip height | "Scripted grasp demonstrator", "THE CREEP" |
| **T2.1** second table | table B + painted zones on both tables; A = left/far, B = right/near | `bw/sim/workshop.py` (TABLE_B_*, PLACE_ZONE) |
| **T2.2** reach audit | roll about the approach axis (joint 6); stations 0.48 m behind each zone | `scripts/reach_audit.py` |
| **T2.4** success spec | one `evaluate()` for collector + evaluator | `bw/task/spec.py` |
| **T2.5** language | paraphrased pick/transfer commands + diversity guard | `bw/task/language.py` |
| **T3** payload locomotion fine-tune | 33M steps, attitude terminations 0.20 → 0.05 | "Phase 1 gate + ablation" |
| **T4** Phase 1 gate + ablation | gate 2 FAILS at 12 cm (19/20); height sweep is the publishable result | "Step-height sweep" |
| sim fixes | 10 Hz servo stutter ("vibration") removed; screwdriver stands handle-down and is graspable | session-3 decisions 3-4 below |

### IN PROGRESS / BELOW THE BAR
| task | state | next action |
|---|---|---|
| **T3.5** curriculum L5 run | `payload_l5` (level_init 5) **stopped at 4.59M of 10M steps, never evaluated** | resume, then re-run the height sweep -- this is now the TOP locomotion item: the step DOWN off the walkway with a tool held is the dominant end-to-end failure (session 5) |
| **T7** SmolVLA fine-tune | data done (1,128 episodes); T7.1 baseline trained + scored (grasp 1/10); the FULL run was killed twice by the full D: drive | free D:, then `ops/train_vla.sh vla_full 6000 16 bw_demos 3000`, then `scripts/eval_vla.py` |
| **T0.4 / T8** grounding | implemented and scored: locate xy median 13.4 mm, miss 11%, **pliers 0/7** | fix the pliers description; then re-score 32 rendered seeds |

### STALE SECTIONS BELOW
"START HERE -- next session" and everything under "SESSION 4" and older describe earlier
sessions; the board here and "SESSION 5" are authoritative.

### DONE IN SESSION 5
| task | what it delivered | evidence |
|---|---|---|
| **T1/T2.3 on legs** | grasp **125/125**; transfer **120/125 (A) / 124/125 (B)**, every tool >= 92% | "SESSION 5" below |
| **T5** | controller == MJX element-wise (action diff 1.8e-6) | "## T5 (session 5)" |
| **T0.4 / T8** | `vlm.point()` + `locate()` behind one contract, scored | "## T0.4 + T8 (session 5)" |
| **T6** | **1,128 LeRobot v3.0 episodes** (pick + place), instructions paraphrased per episode | `scripts/collect_demos.py`, `scripts/to_lerobot.py` |
| **T9** | orchestrator state machine, scripted + VLA skill backends | `bw/orchestrator.py` |
| **T10** | recovery: missing tool, mid-carry drop, ambiguous "wrench" | `scripts/eval_suite.py` |
| **T11** | end-to-end suite + failure-cause table (scripted skills) | "SESSION 5" below |

### NOT STARTED
| task | blocked by |
|---|---|
| **T12** video, README, blog, outreach | the VLA numbers (T7) |
| T10 scenarios 3 (obstacle replan) and 5 (retarget mid-walk) | nothing -- cut for time, 1/2/4 are done |

**Critical path left:** T7 (full fine-tune + eval) -> T12. Everything else is measured.

---

## NEXT SESSION (6) -- start here

**STOP FIRST: the D: drive is FULL (68 MB free of 318 GB).** That is what killed the first full
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

## T0.3 ANSWERED (2026-09-18, session 3) -- Qwen3-VL-2B scored against ground truth

200 wrist views at 512x512, dumped on Windows with the ground-truth pixel for every tool
(`scripts/dump_wrist_views.py`, projection visually verified in `runs/t03_views/gt_check.png`),
scored in the WSL torch venv (`scripts/qwen_bakeoff.py`). Visibility comes from the rendered
DEPTH buffer, so a refusal on an occluded target is not charged to the model.

- **Coordinate scale: 0-1000 NORMALISED, confirmed.** Median error read as raw pixels
  **278.4 px**; read as `x/1000*W` **44.5 px**. Session 2's inference was right, and it is now
  measured rather than guessed. `x_px = x/1000*W`.
- **The target is occluded in 55 of 200 views (27.5%)** by the gripper, from the scan pose
  alone. That is not a model failure -- it is the measurement that justifies T8's 2-3 view
  scan strategy, and it explains session 2's "There are none." for the pliers.
- **Latency 1.12 s median**, 4.26 GB VRAM in bf16. Comfortably out of the control loop.
- Parsed a coordinate in 117/145 visible views; refused on 29 of 55 occluded ones.
- Median error **44.5 px on a 512 px image (8.7% of frame)**; only 15% land within 25 px.

### The wrench distinction: the model is at CHANCE on size, and 93% on colour

Scored only on views where BOTH wrenches are visible, asking for each in turn so a model that
always names the same one scores 50% (`scripts/qwen_prompt_ablation.py`, 254 queries each):

| prompt | parsed | median err | correct wrench |
|---|---|---|---|
| "Point to the 10mm wrench... pixel coordinates" | 242/254 | 82.9 px | **50.0%** |
| same, asking explicitly for 0-1000 normalised | 254/254 | 80.9 px | **52.4%** |
| Qwen's own grounding format (JSON `bbox_2d`) | 254/254 | 224.6 px | **49.6%** |
| **"the wrench with the blue/red grip band"** | 253/254 | **38.5 px** | **92.9%** |

**Qwen3-VL-2B cannot separate a 10 mm from a 13 mm wrench by size** -- exactly chance, across
three phrasings including the model's own native grounding format, so this is not a prompting
artifact. Naming the coloured grip band takes it to **92.9%**, past the plan's 80%
correct-wrench gate, and halves the pointing error as a side effect. The model is reading
COLOUR, not size.

**Decision (T0.4 option (a)): keep Qwen3-VL-2B, and give the size distinction a visible
feature.** `vlm.point()` maps the size in the instruction to the band colour ("10mm" -> blue
band, "13mm" -> red band). This is a real design choice with a real cost and must be stated as
a limitation in the write-up: the size discrimination is carried by the scene's colour coding
and a lookup in our code, NOT by the vision model. Colour-coded tools are ordinary in a
workshop, and the alternative (T8.6, LoRA on sim point labels) stays available as a stretch --
it is now a genuine option rather than a necessity, because the gate is met.

Note the residual 38.5 px median error is still coarse -- roughly 7.5% of the frame -- so
T8's depth lookup and multi-view merge have to absorb it. The demonstrator does not: it uses
ground truth on purpose.

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
