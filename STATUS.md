# STATUS — "Bring me the 10mm wrench"

Last updated **2026-09-21, session 8 (complete -- the queue drained; see "NEXT SESSION (9)").** This file is the CURRENT state only. Superseded
session narratives live in `docs/STATUS_archive.md`; nothing was deleted, only moved.

Read this section and "NEXT SESSION (9)". `REMAINING.md` holds the full sub-task lists.

---

## WHERE THE PROJECT IS

A quadruped with a 6-DoF arm, in MuJoCo, takes a spoken-style command, walks to a tool rack,
grounds the named tool with a VLM, grasps it, walks to a destination and places it or hands it
over — recovering from a missing tool, a dropped tool, an ambiguous name and a blocked route.

| layer | state |
|---|---|
| Grasp, on legs, 5 tools | **125/125** |
| Place, on legs, both tables | 120/125 (A), 124/125 (B) |
| Grounding `locate()`, oracle-scored | miss **2.8%**, median xy **10.7 mm**, 135/180 (75%) |
| Grounding, MISSING tool on real VLM | **9/10** -- but nominal drops 10/10 -> **3/10** (session 8) |
| End-to-end, 7 suites x 10 seeds | **60/70 (86%)** -- a swap was tried and REVERTED, see below |
| Gate 2 (0.12 m step, arm extended), `payload_l5b` | **0 falls / 20** -- but this lineage cannot turn |
| Gate 2, the DEPLOYABLE lineage `payload_nav_l5b` | **12 falls / 20 -- FAIL** (session 8) |
| Push recovery, arm extended, `payload_nav_l5b` | **320 N** (original 200 N, `payload_l5b` 240 N) |
| Learned policy (T7 SmolVLA), held out | **grasp 18/20, correct object 20/20** (was 0/20) |
| Media | 146 s montage: 5 tools carried, 5 recovery modes (`media/montage.mp4`) |

**The single thing capping the system, now with direct evidence:** the base FALLS on the 12 cm
step DOWN off the walkway while carrying a tool. This was asserted from session 3 on the strength
of the `nav_dest_failed` cause label, which is written by two different branches of the state
machine -- one for a fall, one for a stable walk that does not arrive -- so the label alone could
never prove it. `eval_suite.py` now records `loco.is_stable()` and the per-leg `fell` flag that
`navigate_to` was already computing. Re-run 2026-09-21: both nominal failures come back
`stable=False`, `fell=True` on the `via_step` leg, 749 mm short, tool still in the jaws. The
claim was right; it is now measured rather than inferred.

## TASK BOARD

### DONE
| task | what it delivered | evidence |
|---|---|---|
| **T0.1-T0.3** grounding choice + bake-off | Qwen3-VL-2B; chance on 10 vs 13 mm by size, 93% by colour band | "T0.3 ANSWERED" |
| **T0.4 / T8** grounding contract + score | `point()` / `locate()`; **pliers 0/7 -> 32/36** after `"red pliers"`; miss 11% -> **2.8%** | session 6 §3 |
| **T1** grasp demonstrator | 5 tools, 125/125 on legs; the creep, tape radial grasp, stand-lock | "Measured numbers" |
| **T2.1-T2.5** two tables, reach, spec, language | table B + zones, stations, one `evaluate()`, paraphrases | `bw/sim/`, `bw/task/` |
| **T3** payload locomotion fine-tune | 33M steps, attitude terminations 0.20 -> 0.05 | "Phase 1 gate" |
| **T4** Phase 1 gate + ablation | gate 2 FAILS at 12 cm; the height sweep is the publishable result, now across three lineages | session 8 §2 |
| **T5** controller parity | controller == MJX element-wise (action diff 1.8e-6) | archive |
| **T6** demonstrations | **1,128 episodes / 104k frames**, LeRobot v3.0, paraphrased per episode | `scripts/collect_demos.py` |
| **T9** orchestrator | state machine on observable gates, pluggable skills | `bw/orchestrator.py` |
| **T10** recovery 1/2/4/5 | missing tool, mid-carry drop, ambiguous name, retarget | `scripts/eval_suite.py` |
| **T10 #3** obstacle | **0/2 -> 10/10**: stall-direction estimate, local planner, `via` mode, burst cap | session 6 §1 |
| **T11** end-to-end suite | **60/70 (86%)**, one process + one rng per trial | `runs/eval/s7b/` |
| **T7** SmolVLA | **POSITIVE**: grasp **18/20** held out, correct object 20/20, from 0/20 -- fine phase + pick-only + quantile norm | session 8 §1 |
| ops | job queue runner, verified server shutdown, delta pipeline with `/health` handshake | `ops/queue_runner.ps1` |

### REMAINING, IN THE ORDER WORTH DOING

**1. T3.5 — the step-down policy. Still the only item with real headroom, and session 8 narrowed
it to one question.**
The step is confirmed as the binding failure (see above). The awkward part is that it IS solved,
in a lineage that cannot be deployed:

| | gate 2 @ 0.12 m | turn / back / sidestep | end-to-end |
|---|---|---|---|
| `stairs_run7` (original) | 19 falls / 20 | yes | — |
| `payload_l5b` (forward-only, session 7) | **0 falls / 20** | **no** | cannot ship |
| `payload_nav_l5b` (nav + level 5, session 8) | 12 falls / 20 | **yes**, within 8% | **28/70, reverted** |

So level-5 curriculum training under the FULL command distribution bought the best disturbance
rejection on the project (320 N extended, vs 200 N original) and did **not** buy the step. Those
are separable capabilities. The open question is how to get `payload_l5b`'s step robustness into
a policy that can also turn — candidates, none tried: train nav from the `payload_l5b` weights
rather than from nav3; keep two policies and switch per phase (the orchestrator already switches
per phase, and reverting is one file copy); or weight the forward-command mass higher at level 5
instead of sampling the full distribution uniformly.
Already tried and measured, do NOT repeat: `stow_arm()` (needed, kept), a via-waypoint past the
step (17/20, vs 16/24 crossing at full speed), tucking the tool further in (7/12), clockwise-only
turns (needed), and now **training the step curriculum directly in the nav lineage (28/70)**.

**2. T7 is no longer a negative. Write it up as the diagnostic chain it is.**
0/20 -> **18/20 on held-out scenes**, 100% correct object, from three stacked corrections with a
measurement behind each (fine phase, pick-only, quantile normalisation). See session 8 §1.
What it does NOT do: place (0/20, excluded from training by design) and the second grasp after a
tool swap (0/20 held out, 1/20 on training scenes) — that last one is unexplained and is the
first thing to look at if T7 is pushed further.

**3. T12 — write it up. The project is at a reportable state and the story got better.**
`README.md` and `docs/blog.md` need session-8 numbers. The montage is re-cut with real end-to-end
footage (`scripts/make_montage.py`). The honest headline is 60/70 with a confirmed, located
bottleneck — not the mid-60s that was hoped for and did not happen.

**4. Grounding: the absent-tool tier works, and costs more than it saves.**
On real VLM the missing-tool scenario is 9/10 (it had been passing only because the suite defaults
to `--grounding oracle`). But nominal falls from 10/10 to **3/10**, all seven failures being
`locate()` calling a present tool absent. A ~70% false-positive rate on absent tools became a
~70% false-negative rate on present ones. The operating point needs moving, not the mechanism.

**5. The tail, in one sitting.**
- 1 transfer in 10 puts the tool outside the zone (the place distribution's tail).
- j1 and j5 now BEAT the no-motion baseline (session 8) — that item is closed.
- `runs/vla_delta`, `vla_overfit`, `vla_delta_long` hold ~20 GB of checkpoints; prune to the
  evaluated ones. Ahmed's call: an agent deliberately did not delete these.
- T10 scenario 5 retarget is 6/10; 3 of those are the step again.
- `wrench_13mm` is the ONLY tool with no nominal success: seeds 1 and 6 are the two nominal
  failures and both are that tool, reproduced independently twice. Worth one look.

## NEXT SESSION (9) -- the queue is DRAINED and the runner is idle

Session 8 ran 26 jobs and emptied the queue. Nothing is in flight. First three commands:

```
type runs\queue\history.log          what ran, with exit codes and durations
dir ops\queue\hold                   three jobs parked, and why (below)
git log --oneline session-8          five branches, merged
```

**`ops/queue/hold/` holds 460/470/480, the clean+noise union jobs. Do not run them.** They exist
to test whether noise-free scripted demonstrations were what stopped the VLA learning. Session 8
answered that WITHOUT touching the data: the same dataset went 0/20 -> 18/20 once the action
scale was fixed. They are answered, not deferred. Delete them.

### The one thing worth doing next

Get `payload_l5b`'s step robustness into a policy that can also turn. That is the whole of T3.5
now, and "REMAINING 1" lists the three untried candidates. Everything else is a tail item or
writing.

### Two traps this session walked into, written down so the next one does not

**`exit=0` means nothing on this machine.** Five separate things reported success while doing
nothing or the wrong thing: `eval_suite.py` had not parsed since commit fe57f47 (job 448 ran 0 of
20 trials, filed as `done exit=0`); jobs 330 and 340 were no-ops filed the same way; the montage
had been silently dropping its closing subtitle to a `%` in the text; `run_eval_vla.bat`'s server
wait never waited. READ THE LOG, never the exit code. The runner's capture is broken in both
directions -- it filed job 457's deliberate `exit /b 1` as `done exit=0` too. Fixing that
(`|| exit /b 1` per line, and having the runner trust `%ERRORLEVEL%`) is the highest-value ops fix
available.

**A cause label can hide two causes.** `nav_dest_failed` is written by two branches of
`orchestrator.py` -- one guarded by `not loco.is_stable()` (a fall), one for a stable walk that
does not arrive. For five sessions the project's headline claim rested on that label. It turned
out to be RIGHT, but it could not have been known from the label; an interim read this session
argued the opposite from base positions and was wrong. The row now carries `stable`, `fell_any`
and the per-leg `nav` record. When a claim matters, instrument it.

### Do not re-derive these; each cost GPU hours to establish

* The VLA's failure WAS action scale, and the fix is three stacked corrections, all needed: fine
  phase, pick-only, quantile normalisation. 0/20 -> 18/20 held out.
* Eliminated by measurement: capacity, the pipeline and serving path, the delta representation,
  covariate shift and generalisation, the stance mismatch, the image domain gap, the chunk
  horizon. `n_action_steps` stays at 10.
* The training success EMA of a stairs run says nothing about gate 2: it scores the level-5 task
  (0.130 m), the gate is 0.120 m. Reading it cost a wrong call in session 7.
* The VLA's training loss is blind. It fell 0.438 -> 0.065 in the run that worked, and looked the
  same in every run that did not.
* One process per suite. Trials are not independent inside a process: `transfer,drop` scored drop
  6/10 where `drop` alone scored 9/10, same code, same seeds.
* Windows `timeout /t` cannot be used in a queued job: GNU coreutils shadows it on PATH, AND
  Windows' own refuses to run under redirected stdin. Use `ping -n N+1 127.0.0.1`.

### Still open

1. The `scripts/` reorganisation in `docs/REPO_LAYOUT_PLAN.md`. The queue is empty now, so the
   path references it was waiting on are no longer a blocker.
2. The place tail: 119/125 on table A; the residual is the wrench's post-release travel.
3. The screwdriver is no longer the worst tool -- it is 4/4 in the VLA eval. `wrench_13mm` is now
   the outlier (no nominal success, reproduced twice).
4. Five scan views instead of three for grounding, and the absent-tool operating point.
5. Outreach, which is Ahmed's call, not an agent's.

---

## SESSION 8 (2026-09-21) -- T7 reversed; T3.5's deployable lineage failed and was reverted

26 queued jobs, no GPU left idle. Every number below is measured.

### 1. T7 IS NO LONGER A NEGATIVE: 0/20 -> 18/20 on held-out scenes

The action-scale diagnosis from session 7 was correct and all three corrections were needed.
`vla_fine_pick`, 20,000 steps, final loss 0.065.

Replay check through the serving path, 91 frames -- every joint beats the no-motion baseline,
which j1 and j5 had failed to do in BOTH previous delta runs:

| joint | policy err | no-motion | spread | policy/spread |
|---|---|---|---|---|
| j1 | **0.0067** | 0.0101 | 0.0213 | 0.32 |
| j5 | **0.0053** | 0.0069 | 0.0157 | 0.34 |
| grip | **0.0011** | 0.0095 | 0.0121 | 0.09 |
| ALL | 0.0083 | 0.0160 | 0.0304 | 0.27 |

In millimetres at the gripper (the project's own anchor: 0.0289 rad ~ 15 mm), j1 is ~3.5 mm
against a no-motion 5.2 mm, where the failing run sat at ~15 mm. Read millimetres, not ratios.

End-to-end scoring, `--from-pregrasp --pick-s 7` (the scripted stack transports to the pre-grasp
pose; the policy does approach, close and lift with 7 s of control):

| | held out | training scenes |
|---|---|---|
| grasp | **18/20** | 17/20 |
| correct object | **20/20** | 20/20 |
| swap grasp | 0/20 | 1/20 |
| transfer | 0/20 | 0/20 |

Per tool, held out: wrench_10mm 4/4, wrench_13mm 4/4, screwdriver **4/4**, pliers 3/4,
tape_roll 3/4. The screwdriver had been the tool that failed across every layer (0/3 in the
overfit control).

Four caveats, so nobody overclaims this:
* It does the GRASP, not the task. Transport is scripted; the policy is handed the pre-grasp pose.
* `transfer` 0/20 is by construction -- place was excluded from training because its 2.4 rad IK
  jumps set the normaliser. This checkpoint was never asked to place.
* `swap_grasp` 0/20 is NOT explained by that. First grasp 18/20 and second grasp 0/20 in the same
  trials, same tools. Unexplained; the first thing to investigate if T7 continues.
* Prediction 3 ("training scenes are the upper bound") was NOT met -- held out beat training
  scenes, 18 vs 17. One trial at n=20 with an unmatched tool mix, so: noise. But it means the run
  gives no usable ceiling. Say "90% held out", not "90% of a known ceiling".

### 2. T3.5: the deployable lineage FAILED the gate, was swapped in anyway, and was reverted

`payload_nav_l5b` (nav command distribution, level_init 5), best checkpoint chosen from the run's
own eval rows at step 1,638,400 (reward 3732.62 -- the peak is mid-run again; `final` was worse).

Flat-ground tracking is fine and all three pipeline-critical skills survived: turn +0.63/-0.57
against +/-0.60, back -0.26 against -0.25, sidestep +0.17/-0.18 against +/-0.20, 0 falls in 40
trials. One anomaly: `slow` tracks **+0.06 against a commanded +0.15**.

The gate, which had never been run on this lineage (job 440 ran flat-ground tracking only):

| height | original | `payload_l5b` | `payload_nav_l5b` |
|---|---|---|---|
| 0.10 | 14 | 0 | 1 |
| 0.11 | 18 | 0 | 8 |
| **0.12** | **19** | **0** | **12 FAIL** |
| 0.13 | 20 | 5 | 12 |

Push recovery went the other way and is the best on the project: **280 N stowed / 320 N extended**
against the original's 160/200 and `payload_l5b`'s 240/240. Disturbance rejection and step
descent are separable capabilities.

It was swapped in regardless -- the reasoning being that the gate tests the arm EXTENDED while the
orchestrator stows for transit, so the gate might be harsher than deployment. It was not:

```
nominal 0/10   drop 0/10   ambiguous 0/10   retarget 0/10     <- every human destination
transfer 9/10  obstacle 9/10  missing 10/10                   <- every table destination
TOTAL 28/70 (40%), against a 60/70 baseline
```

Reverted by hash (`payload_nav_policy.npz` 1dce46bd -> 50c16e82) and verified: nominal back to
8/10, every success ending at the handoff tray. **The gate said FAIL and the gate was right.** A
plausible story about why a gate might not transfer is not evidence.

### 3. The step-down claim, finally measured rather than inferred

`nav_dest_failed` conflated a fall with a stable non-arrival, and the project's headline claim had
rested on that label since session 3. `eval_suite.py` now records `stable`, `fell_any` and the
per-leg `nav` log -- data `navigate_to` was already computing and throwing away. Re-run: both
nominal failures come back `fell=True`, `stable=False` on the `via_step` leg, 749 mm short, tool
still in the jaws. The step IS the bottleneck, and `payload_nav_l5b` took that same fall from
2/10 to 10/10.

### 4. Grounding on a real VLM: the absent-tool tier works, and costs more than it saves

| | oracle | real VLM |
|---|---|---|
| missing | 10/10 | **9/10** |
| nominal | 10/10 | **3/10** |

All seven nominal failures are `locate()` reporting a present tool absent. A ~70% false-positive
rate on absent tools became a ~70% false-negative rate on present ones. Median `locate` latency
9.3 s on nominal against 5.4 s on missing -- the model exhausts its scan views before giving up.
The operating point needs moving, not the mechanism. (Job 458 straddled the revert: `missing` ran
under the bad policy, but is policy-insensitive since nothing is delivered.)

### 5. Five things that reported success while doing nothing

`eval_suite.py` had not parsed since fe57f47 -- an `if args.out:` whose body was dedented out from
under it -- so job 448 ran 0 of 20 trials and was filed `done exit=0`; it would have taken the
seven end-to-end suites with it. Jobs 330 and 340 were the same shape. The montage's closing card
had been silently dropping its subtitle, because `drawtext` expands `%` and discards the whole
string on failure ("2.8% miss"), while its title overflowed the 960 px frame and was clipped at
both edges. `run_eval_vla.bat`'s 40 s server wait never waited, for two independent reasons, and
only passed because PowerShell's startup cost accidentally supplied the delay.
`make_orch_video.py`'s usage line documented an argument it does not take. All fixed.

### 6. Media

`media/montage.mp4` re-cut, 146 s: all five tools carried end to end, and all five recovery modes
(drop 9/10, obstacle 10/10, ambiguous 8/10, missing 10/10, retarget 6/10), each clip rendered from
a seed the suite scored as a SUCCESS, with its suite score burned into the caption. Nine source
clips committed at 9.0 MB (203 MB before compression, matching the repo's existing convention).

---

## SESSION 7 (2026-09-21) -- T3.5 in both lineages, the place tail, T7's data test

Complete. Everything below is measured. What session 7 left running was finished in session 8.

### 1. T3.5, and a bug that had kept it from ever starting
`ops/resume_payload_l5.sh` was "written and ready" since session 6 and had never been run. Its own
guard is `[ -d "$L5" ]` -- but brax writes a checkpoint as a FILE here, not a directory, so the
first launch exited instantly with "no checkpoint at .../step_4587520" against a checkpoint that
was plainly there. Fixed to `-e`. (The lesson is the one session 6 already wrote down about
scripts that have never been executed.)

**GATE 2 NOW PASSES -- and the training metric said the opposite.** The Phase 1 gate on the chosen
checkpoint (step_1310720), against stairs run 7 as the control, 20 trials per cell:

| | gate 1 (5 m flat, stowed) | gate 2 (0.12 m step, arm extended) | height sweep, falls/20 |
|---|---|---|---|
| original (stairs run 7) | 20/20 PASS | **19 falls, FAIL** | 0.06:0  0.08:3  0.10:14  0.11:18  0.12:19  0.13:20 |
| **payload_l5b @ 1,310,720** | 20/20 PASS | **0 falls, PASS** | 0.06:0  0.08:0  0.10:0  0.11:0  0.12:0  **0.13:5** |

Push recovery, same run: payload 240 N stowed and extended, against the original's 160 N / 200 N.

So the 12 cm step DOWN -- the thing that lost 8 of 10 end-to-end trials and had failed this gate
since session 3 -- is now clean in MJX, and the policy's own failure edge has moved to 0.13 m.
**The training success EMA was the wrong thing to read.** It stayed in the 0.22-0.30 band all run
because it scores the level-5 task, i.e. the 0.130 m step, where the policy still falls 5 in 20; the
gate's geometry is 0.120 m, where it now falls none. An interim read of this session called the run
a negative on the strength of that EMA. It was not: the gate is the metric that was asked for, and
it moved from FAIL to PASS.

What is still open is DEPLOYMENT, not locomotion: this lineage trains on the forward-only command
distribution and cannot turn in place, back up or sidestep, which the pipeline cannot do without.
That is what `payload_nav_l5` is for, and it is now the run that matters.

**The L5 continuation's own reward curve, for the record:** 5,898,240 steps in 130.7 min
(753 sps): level-5 success stayed inside the 0.22-0.30 band it started in (final 0.27 against 0.25
at session 5's stop), reward peaked at **2,935.97 at step 1,310,720** and the remaining 4.6M steps
produced nothing better -- 2,374 at its worst, with contact terminations (the trunk striking the
riser, which is gate 2's own failure mode) rising 0.38 -> 0.53 as it went. So the step is not
step-count-limited at this level, and `checkpoints/final` is the WORST policy the run produced.
`scripts/pick_best_ckpt.py` now chooses from a run's own eval rows for both locomotion exports --
the same rule T7.4 applies to the VLA, and it would have been needed here: every ops script until
now exported `final`.

| payload_l5b (5.5M steps from step 4,587,520) | reward | succ | contact term |
|---|---|---|---|
| warm start (step 0 eval) | 2,742 | 0.30 | 0.45 |
| **best, step 1,310,720** | **2,936** | 0.25 | 0.38 |
| step 3,932,160 (worst) | 2,375 | 0.20 | 0.53 |
| final, step 5,898,240 | 2,867 | 0.27 | 0.44 |

**Two lineages, not one, because they are not interchangeable.** `payload_l5` trains the STEP
(level_init 5 = 0.130 m) on the ORIGINAL forward-only command distribution, so it can serve the
Phase 1 gate and the height sweep -- neither needs turning. But the DEPLOYED policy is
`payload_nav_policy.npz` (= payload_nav3 final), which is what gives the pipeline turn-in-place,
back-up and sidestep, and nav3 trained at level_init 4 / level_min 3. So the 12 cm step DOWN that
loses 8 of 10 end-to-end trials is being taken by a policy whose curriculum never contained it,
and swapping payload_l5b in would lose navigation. `configs/payload_nav_l5.yaml` is nav3's config
with ONE change -- level_init 4 -> 5, level_min 3 -> 4, warm-started from nav3 final -- so the
comparison at a fixed step count is a comparison of the curriculum alone.

### 2. The place distribution's tail: a silent give-up in the align, and an aim shift that still earns its keep
`scripts/topple_diagnose.py`, 14 pliers transfers onto table A: **displacement after release is a
median 0 mm and cos(displacement, lean) is -0.00.** The topple that `PLACE_AIM_SHIFT` (45 mm) was
introduced for -- session 5 measured cos = +1.00 over 25 transfers -- does not happen on the
current release poses. Meanwhile the tool's position AT RELEASE scattered +-70 mm in the zone frame
against a 100 mm half-width. So the 1-transfer-in-10 that lands outside the zone is a
RELEASE-POSITION failure, not a topple, and two mechanisms feed it:
  * `_servo_xy` gives up SILENTLY -- it returns as soon as one pass's IK fails to converge (`ok`
    false or residual > 1 cm), and nothing checked the achieved offset before the jaws opened.
    Reproduced: transfer seed 8 releases the pliers 55.6 mm from its own aim point.
  * the aim point itself is centre + 45 mm (the shift) + up to 20 mm (jitter), so a converged
    align can still sit 65 mm out before the tool has moved at all.
Two changes: a pre-release guard (`PLACE_MAX_OFFSET` 50 mm) re-servos at the jitter-free aim with
more passes and a tighter tolerance, and the shift is now conditional on the tool's MEASURED lean
(vertical hang, tape roll, or standing upright -> aim at the centre).

**The obvious next step -- drop the 45 mm shift entirely -- was A/B'd and REJECTED by its own
numbers.** 10 seeds x 4 tools on table A, teleported base, one arm per value of the shift
(`BW_PLACE_AIM_SHIFT`, added so the sweep needs no edit to the skill):

| table A, 10 seeds per tool | shift 45 mm (kept) | shift removed |
|---|---|---|
| wrench_10mm | 10/10 | 10/10 |
| wrench_13mm | **10/10** | **7/10** |
| pliers | 10/10 | 9/10 |
| tape_roll | 6/6 | 6/6 |
| total | **36/36** | 32/33 |

The "it pays for a topple that stopped happening" reading held for the PLIERS, which is what was
measured first, and not for the wrenches, which still travel when released leaning. So the shift
stays wherever the tool hangs leaning and is dropped only for a near-vertical hang -- the one case
the measurement supports. `eval_suite.py` also records
where the tool actually ended up (`gt`: position, zone delta, dz, held, base) on every trial, so a
failed trial no longer needs re-running by hand to find out what it was.

### 3. T7 TIER 1: the VLA was evaluated with the legs FREE, which no demonstration ever was

Before spending 6 more GPU hours on the covariate-shift dataset, two cheap diagnostics. The first
one changed the picture.

`sim.lock_stance()` -- legs held on joint PD while the arm works -- is called by `run_grasp`,
`run_place` and the orchestrator. It was called NOWHERE in the VLA path: not in
`bw/policy/vla.py:run_skill`, not in `scripts/eval_vla.py`. `Locomotion.lock_stance` exists because,
measured in session 5, "under the policy the arm's reach and pull push the standing base back
25-260 mm (it steps away from the load), so the jaws arrive short or the tool is dragged against the
rack". So every demonstration was recorded on a LOCKED base and every rollout behind the 1/20 ran on
a base walking away from the rack -- a regime that appears in no training frame, and one the
ORCHESTRATOR never uses, because it locks the stance itself before calling a skill.

`scripts/vla_exec_check.py` replays each demonstration's OWN recorded actions through the serving
loop (10-action chunks at 10 Hz, deltas re-applied to the live joint position, exactly as the client
does) into the scene it was collected in, and scores it with the same `evaluate(Task("pick"))` the
VLA eval uses. No policy, no GPU. 8 demonstrations, four cells:

| cell | grasp with PERFECT actions | mean base movement |
|---|---|---|
| absolute, legs locked | **8/8** | 4.3 mm |
| absolute, legs free | 5/8 | 35.1 mm |
| delta, legs locked | **8/8** | 4.3 mm |
| delta, legs free (**what the 1/20 was measured in**) | **3/8** | 39.4 mm |

Two things follow, and they point in opposite directions:
  * **The delta serving convention is NOT lossy.** Locked, delta replay equals absolute replay at
    8/8. The representation is fine and does not need changing.
  * **The harness was capping the achievable score at ~3/8.** A perfect policy grasps 3 of 8 in the
    regime the evaluation ran in. The policy scored 1/20, so it is genuinely bad as well -- but the
    session-6 inference from "prediction improved 3.5x while closed-loop success did not move" was
    drawn against a ceiling of ~37%, not 100%, so covariate shift is no longer established. It is
    back to being one candidate among others.

`run_skill` now locks the stance (with the measurement in its docstring), and `eval_vla.py
--train-scenes` can score the policy on the scenes the DEMONSTRATIONS were collected in, using
`collect_demos`'s own rng stream -- so "cannot grasp anywhere" and "cannot grasp in a NEW scene" can
finally be told apart. Queue job 327 runs both, on the existing delta-20k checkpoint, BEFORE the
union training.

### 4. T7's covariate-shift dataset (built, and now demoted to one candidate among others)
`bw/manip/disturb.py` + `collect_demos.py --noise`: DART-style kicks on the ARM (a servo offset)
with the LABEL left nominal, so every frame from a kick until the arm is back on the path pairs an
off-tube state with the command that corrects it. The gripper is never kicked (that just drops the
tool) and neither are the close / lift / descend / release phases (that knocks the scene). Yield
sweep, seeds 90000-90013 at sigma 0.04 / 0.08 / 0.12: 8 of 8 runs still produced a clean pick AND
place, while |q_cmd - q_state| widened where it matters -- j1 3.0 -> 5.0 mrad median and j5
2.5 -> 5.0, which are precisely the two joints that never beat the trivial baseline in EITHER delta
run. Collection runs at sigma 0.10 / p 0.15 (`ops/collect_noise.sh`, 600 runs, 3 lanes).

### 5. Bookkeeping that was wrong rather than missing
Ten REMAINING items were finished in earlier sessions and never ticked (T0.4, T1.3, T1.5, T1.6,
T3.1-T3.4, T7.2, gate 1/2 + the ablation). T0.2 is closed as obsolete -- it asks for Molmo2's point
encoding and Molmo2 was dropped in T0.1. T7.5 (language sensitivity) has been RUN and cannot say
anything: `swap_grasp` 0.05 against `grasp` 0.05 is 1/20 either way, so the test has no resolution
until a variant grasps. T7.6 (data-scaling curve) would read 1/20 at every point for the same
reason.

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
- **JOB QUEUE (session 6) -- how long jobs should be started now.** `ops/queue_runner.ps1` is a
  detached runner: it takes jobs from `ops/queue/pending/*.cmd` in filename order, runs them ONE
  AT A TIME, waits while the GPU is hot (>= 88 C) or still busy with someone else's process
  (> 1500 MiB), and writes `runs/queue/<job>.log`, `runs/queue/status.json` and
  `runs/queue/history.log`. Finished jobs move to `done/` or `failed/` by exit code.
  ```
  powershell -NoProfile -ExecutionPolicy Bypass -File ops\queue_runner.ps1   # in a terminal
  ops\install_queue_task.cmd                                                 # or detached, at logon
  ```
  WHY: an agent must never OWN a long job. Claude Code reaps its own background shells under
  memory pressure and killed the same SmolVLA fine-tune twice, at steps 3,900 and 4,700 of
  6,000; the same run launched from a plain terminal finished untouched. The agent now ENQUEUES
  by writing a file -- which nothing can reap -- and the runner, started once by a human, does
  the work. Note PowerShell 5.1 writes UTF-16 from `Tee-Object` and a BOM from
  `Set-Content -Encoding utf8`; the runner writes BOM-less UTF-8 through .NET so `grep` and
  `json.load` can read its output.
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

