# Bring me the 10mm wrench

Language-commanded mobile manipulation in simulation, trained and evaluated end to end on **one
6 GB consumer GPU (RTX 3060 laptop), no API keys, no cloud**.

You type *"bring me the 10mm wrench"*. A Unitree Go2 with a Z1 arm, in MuJoCo, walks to the
workbench, finds the wrench, picks it out of the rack, carries it across a 12 cm step and hands
it to the person — and says something useful when it cannot.

```
"bring me the 10mm wrench"
        |
        v
  orchestrator (state machine, bw/orchestrator.py)
        |            gates: base at station? point returned? gripper holding? gripper empty?
        +--> navigate_to   Layer 3: RL walking policy (numpy, 50 Hz)      bw/locomotion/
        +--> locate        Layer 1: Qwen3-VL-2B point -> wrist depth -> 3D bw/perception/
        +--> grasp/place   Layer 2: SmolVLA, or the scripted demonstrator  bw/manip/, bw/policy/
        +--> ask_human     when the command is ambiguous or the tool is gone
```

## Results (all measured in this repo, seeds and scripts given)

| what | number | how |
|---|---|---|
| Grasp, legs standing on the walking policy | **125/125** (5 tools x 25 seeds) | `scripts/try_grasp.py 25 --walk` |
| Two-table transfer, walking between stations | **120/125** table A, **124/125** table B | `scripts/try_place.py 25 --walk --table X` |
| Walks reaching the station | 214/214, no falls | session 4 |
| Grounding `locate()` | **10.7 mm** median xy error, **2.8%** miss, 135/180 right tool within 30 mm | `scripts/bench_locate.py`, `scripts/score_locate.py` |
| Layer 3 numpy controller vs MJX | action difference 1.8e-6 | `scripts/t5_obs_check.py` |
| Demonstration data | **1,128 episodes** / 104k frames, LeRobot v3.0, paraphrased instructions | `scripts/collect_demos.py`, `scripts/to_lerobot.py` |
| End to end, scripted skills + oracle grounding, 7 suites x 10 seeds | **60/70 (86%)** | `ops/run_eval_suites.sh`, `scripts/eval_summary.py` |
| Fine-tuned SmolVLA, closed loop | **1/20 -- a measured negative, diagnosed** (below) | `scripts/eval_vla.py`, `scripts/vla_replay_check.py` |

Per suite, ten seeds each, one process and one reseeded rng per trial:

| suite | success | what lost the rest |
|---|---|---|
| nominal ("bring me the ...") | 8/10 | 2 falls stepping down off the walkway |
| two-table transfer | 9/10 | 1 tool placed outside the zone |
| missing tool -> report it | 10/10 | -- |
| mid-carry drop -> re-grasp | 9/10 | 1 fall on the step |
| ambiguous "wrench" -> ask | 8/10 | 2 falls on the step |
| retarget mid-walk | 6/10 | 3 falls on the step, 1 grasp |
| obstacle on the route -> detour | 10/10 | -- |

The dominant end-to-end failure is **locomotion, not manipulation**: **8 of the 10 failures** are
the base falling on the 12 cm step DOWN off the walkway with a tool in the jaws. No trial was lost
to the grasp, the place or the grounding layer's own skill. See `STATUS.md`.

## The learned policy is a negative result, and it is reported as one

SmolVLA was fine-tuned on the 1,128 demonstrations and **grasps 1 of 20 held-out seeds**. That is
the honest headline for Layer 2, and the diagnosis is the part worth reading, because each step is
a measurement rather than a guess:

| model | epochs | one-step prediction vs the trivial "command no motion" predictor | grasp |
|---|---|---|---|
| absolute joint targets, 6k steps | 0.92 | **2.10x WORSE** | 1/20 |
| delta targets, 3k steps | 0.46 | 1.20x worse | 0/20 |
| delta targets, 20k steps | 3.0 | **1.68x BETTER** | 1/20 |
| overfit control: 40 episodes seen 11.7x | 11.7 | **3.8x BETTER** | -- |

1. **Training loss cannot see the failure.** The absolute run reached loss 0.115 and could not
   grasp. `scripts/vla_replay_check.py` scores the policy on its own training frames, through the
   serving path, against the trivial predictor -- which is what loss does not tell you.
2. **The action representation was wrong.** With absolute targets, per-step motion (~0.026 rad) is
   4-8% of the spread the normaliser divides by, so the policy was worse than doing nothing.
   `to_lerobot.py --delta` records `q_cmd - q_state`, rescaling the target 7-17x per joint.
3. **Capacity, pipeline and serving are not the problem** -- the overfit control predicts 3.8x
   better than baseline on every joint.
4. **What remains is covariate shift.** One-step prediction improved 3.5x while closed-loop
   success did not move. Every demonstration came from a scripted controller that never made a
   mistake, so the data contains no recovery states. `--noise` in `scripts/collect_demos.py`
   (`bw/manip/disturb.py`) injects kicks into the demonstrator to test exactly that, and it is a
   DATA change, not more GPU hours.

## What is honest about this

- Simulation only. Nothing here has run on hardware.
- The demonstrator uses ground-truth object poses **on purpose** — it is the teacher. The policy
  trained on it sees only two 256x256 RGB cameras, the 7-D arm state and the instruction.
- The 10 mm vs 13 mm distinction is carried by a **coloured grip band plus a size->colour lookup**,
  because the VLM is at chance on the size itself (92.9% on the band). That is a stated
  limitation, measured, not hidden.
- Every number above comes from a script in this repo, on fixed seeds, with the failures counted
  and tagged. `STATUS.md` records what was tried and rejected, with the measurement that killed it.

## Layout

```
bw/sim/          the workshop scene and the MuJoCo wrapper (WorkshopSim)
bw/locomotion/   Layer 3: MJX/brax training, the numpy controller, the navigator
bw/manip/        scripted grasp / place / handoff demonstrator + IK
bw/perception/   Layer 1: vlm.point() (model in its own process) and locate()
bw/policy/       SmolVLA inference server + the skill client
bw/task/         the one success spec, and the instruction paraphrases
bw/orchestrator.py  the state machine
scripts/         benchmarks, collectors, evaluation suites, video renderers
ops/             run scripts (training, collection, conversion)
```

## Running it

Three environments (see STATUS.md "Environment and operations"):
`~/go2-stairs/.venv` (JAX/MJX, WSL), `~/bringwrench/.venv-vla` (torch + lerobot, WSL),
`D:\hexapod\render_venv` (Windows, all rendering — EGL fails inside WSL).

```bash
# rebuild the models
python -m bw.sim.build_models

# benchmarks (Windows render venv)
python scripts/try_grasp.py 25 --walk
python scripts/try_place.py 25 --walk --table table_b

# demonstration data -> LeRobot dataset
bash ops/collect_queue.sh 0 600 3          # Windows sims, 3 workers
bash ops/convert_all.sh                    # WSL, 8 shards + merge

# fine-tune SmolVLA and evaluate it
bash ops/train_vla.sh vla_full 8000 16 bw_demos
bash ops/vla_server.sh ~/bringwrench/runs/vla_full/checkpoints/last/pretrained_model
python scripts/eval_vla.py 20 --transfer --swap

# end to end
python scripts/eval_suite.py --suite all --n 10
python scripts/make_orch_video.py 5 media/bring.mp4
```

## Credits

Built on MuJoCo/MJX, brax, LeRobot (SmolVLA), Qwen3-VL, and Unitree's Go2 + Z1 models.
