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
| Grounding `locate()` | 13.4 mm median xy error, 11% miss (pliers: 0/7) | `scripts/bench_locate.py`, `scripts/score_locate.py` |
| Layer 3 numpy controller vs MJX | action difference 1.8e-6 | `scripts/t5_obs_check.py` |
| Demonstration data | **1,128 episodes**, LeRobot v3.0, paraphrased instructions | `scripts/collect_demos.py`, `scripts/to_lerobot.py` |
| End to end, scripted skills + oracle grounding | transfer 9/10, bring-me 7/10, missing tool 6/6, drop recovery 9/10 | `scripts/eval_suite.py` |

The dominant end-to-end failure is **locomotion, not manipulation**: stepping DOWN off the 12 cm
walkway while carrying a tool (3 of 10 bring-me trials). See `STATUS.md`.

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
