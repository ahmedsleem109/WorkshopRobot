# Architecture

Four layers, each with one contract. The point of the design is that **no learned model sits inside a
control loop**: the VLM is called 4-6 times per episode, the VLA emits 50 actions per forward pass,
and the only thing running at 50 Hz is a numpy MLP.

```
  "bring me the 10mm wrench"
            |
            v
  +---------------------------+   bw/orchestrator.py
  |  Layer 4: orchestrator    |   a state machine on OBSERVABLE gates:
  |                           |   base at station? point returned? gripper holding? empty?
  +---------------------------+   every state has a timeout and an on_failure edge
     |          |          |
     v          v          v
  +--------------------+  +---------------------------+  +---------------------+
  | Layer 1: grounding |  | Layer 2: manipulation     |  | Layer 3: locomotion |
  | bw/perception/     |  | bw/manip/ (scripted)      |  | bw/locomotion/      |
  |                    |  | bw/policy/ (learned)      |  |                     |
  | point(img, text)   |  | run_grasp / run_place     |  | set_velocity()      |
  |   -> (u,v) or None |  |   -> {success, reason}    |  | get_base_pose()     |
  | locate(desc)       |  | vla.run_skill(task)       |  | is_stable()         |
  |   -> Located/None  |  |                           |  | walk_to(station)    |
  +--------------------+  +---------------------------+  +---------------------+
   Qwen3-VL-2B, 4-bit,     scripted demonstrator 125/125   RL policy trained in MJX,
   in its OWN process,     learned policy: see the T7       exported to .npz and run by
   2-5 s per call          section of STATUS.md             numpy at 50 Hz
```

## Where the contracts live

| contract | file | note |
|---|---|---|
| `point(image, description)` | `bw/perception/vlm.py` | model in a separate process; swapping it is a one-file change |
| `locate(description)` | `bw/perception/locate.py` | point -> wrist depth -> camera -> base frame, merged over 3 scan views |
| `run_grasp` / `run_place` | `bw/manip/scripted_*.py` | the demonstrator; allowed ground truth **on purpose**, it is the teacher |
| `vla.run_skill(sim, task, max_s)` | `bw/policy/vla.py` | the learned skill, same call shape as the scripted one |
| `set_velocity` / `get_base_pose` / `is_stable` | `bw/locomotion/controller.py` | element-wise equal to MJX (action diff 1.8e-6) |
| success, for **both** training and scoring | `bw/task/spec.py` | one `evaluate()`, so collector and evaluator cannot drift apart |

## The three environments, and why they cannot be one

| environment | holds | why separate |
|---|---|---|
| `D:\hexapod\render_venv` (Windows) | CPU MuJoCo + **all** rendering | MuJoCo's EGL/OpenGL path does not work inside WSL2 |
| `~/go2-stairs/.venv` (WSL) | JAX/MJX, brax | CUDA training; JAX and the Windows renderer do not share a 6 GB card |
| `~/bringwrench/.venv-vla` (WSL) | torch, lerobot | SmolVLA fine-tune + policy server, ~4.5 GB VRAM |

This is also why there is no `docker compose up`. A compose file covering the two WSL halves would
advertise a reproduction it does not deliver.

## Data flow, collection to policy

```
collect_demos.py   ->  D:/bw_data/raw/<seed>_<kind>/{wrist.mp4, mast.mp4, data.npz, meta.json}
      |                successes only, scored by bw/task/spec.evaluate
      v
to_lerobot.py      ->  LeRobotDataset v3.0
      |                --delta       action = q_cmd - q_state, gripper stays absolute
      |                --fine-phase  drop the leading transport swing
      v
ops/train_vla.sh   ->  runs/<name>/checkpoints/<step>/pretrained_model
      |
      v
ops/vla_server.sh  ->  HTTP :8766 /act  ->  bw/policy/vla.py  ->  WorkshopSim
```

## Conventions that are easy to violate by accident

* **Score after a hold, not at the end of the motion.** Grasp success is checked after a static hold;
  the first benchmark read 69% and became 31% under that rule, because tools were creeping through
  the jaws. Numbers from before 2026-09-18 are not comparable.
* **One process per eval suite, one rng per trial.** Trials are not independent inside a process:
  `--suite drop` alone scored 9/10 while `--suite transfer,drop` scored the same seeds 6/10.
* **The delta convention travels with the checkpoint**, reported on the policy server's `/health`, so
  a client and a dataset cannot silently disagree.
* **The legs are stand-locked during manipulation.** Every demonstration was recorded that way, and a
  rollout with the legs free walks the base 25-260 mm away from the rack while the arm reaches.
* **No agent owns a long job.** `ops/queue_runner.ps1` runs queued `.cmd` files one at a time, waits
  for a cool and free GPU, and survives the shell that enqueued them.
* **Ground truth is for the demonstrator and the scorer, never for a trained policy's input.**
