# "Bring me the 10mm wrench"

**Language-commanded mobile manipulation in simulation — open weights, $0 budget, three phases**

Owner: Ahmed Selim
Target: legged/humanoid robotics roles in Europe (Neura, Agile Robots, ANYbotics, RIVR, 1X)
Duration: 12 weeks part-time · Cost: $0 · Hardware: 6GB GPU + free Kaggle tier

---

## The one-paragraph version

You type "bring me the 10mm wrench." A Unitree Go2 with a Z1 arm walks across a cluttered workshop in MuJoCo, finds the wrench, picks it up, brings it back, and recovers when something goes wrong. Three components: your existing locomotion policy retrained for the arm's weight, a fine-tuned SmolVLA for grasping, and a state machine that calls a VLM to answer "where is it?".

---

## How to read this document

Every item is tagged:

- **[CORE]** — the project fails without it. Do these first, always.
- **[STRETCH]** — real added value, do it only if the phase gate passed early.
- **[LATER]** — a follow-up project, not this one. Listed so you stop thinking about it.

**Rule:** no STRETCH item starts until every CORE item in that phase is done and its gate has passed. This rule is the whole reason the document has tags.

---

## What each piece is

| Component | Type | Input | Output | Status |
|---|---|---|---|---|
| Locomotion policy | RL — no vision, no language | body state | leg torques | **You already have this** |
| SmolVLA | **VLA** — vision-language-**action** | camera + instruction + arm state | arm joint commands | Fine-tune — *the centerpiece* |
| Molmo2-ER | **VLM** — vision-language | camera + "the 10mm wrench" | pixel coordinates | Off the shelf, one function |
| Orchestrator | Hand-written state machine | everything | tool calls | You write ~300 lines |

**The VLA is what matters for your CV.** It is the keyword every robotics ad uses, and you will have fine-tuned one on data you generated. The VLM is supporting cast.

### Honest note on the reasoning layer

An earlier draft used Gemini Robotics ER 2 and claimed open models were near-equivalent. They are not. ER 2 does streaming video progress tracking, trained tool orchestration, and mid-task self-correction. Molmo is a **stateless pointing function** and does none of that.

So the design is not "Molmo replaces ER 2." It is: Molmo does grounding, you write the orchestration explicitly. More work, less general — but every decision is a line of code you can point at in an interview, and every failure has a traceable cause.

---

## Spec — lock this, do not change it

| Item | Decision |
|---|---|
| Robot | Unitree Go2 + Z1 arm (both in MuJoCo Menagerie) |
| Simulator | MuJoCo / MJX |
| Scene | One workshop: bench (75cm), tool tray, floor clutter, one 12cm step |
| Objects | Exactly five: 10mm wrench, 13mm wrench, screwdriver, pliers, tape roll |
| Commands | Typed natural language: "bring me the X" |
| Done means | 50 logged trials (5 objects × 10) + 5 filmed recovery scenarios |

**Out of scope, permanently:** real hardware · more than one scene · bimanual · more than five objects · training any foundation model from scratch · any paid service.

**Why Go2 + Z1:** base and arm decouple, so you debug locomotion and manipulation separately. G1 couples them — that is the sequel, not this.

---

## Architecture

| Layer | Component | Rate | Where |
|---|---|---|---|
| 3. Locomotion | Your terrain policy, payload-adapted | 50 Hz | Local GPU |
| 2. Manipulation | Fine-tuned SmolVLA (~450M) | ~10 Hz | Local GPU |
| 1. Grounding | Molmo2-ER-4B, 4-bit (~3 GB VRAM) | 4–6 calls/episode | Local, separate process |
| 0. Orchestration | Your state machine | event-driven | CPU |

### Interface contracts — write these in Week 1, before anything else

```python
# Layer 3 — locomotion (wrap your existing policy)
locomotion.set_velocity(vx, vy, wz)          # 50 Hz closed loop
locomotion.get_base_pose() -> SE3
locomotion.is_stable() -> bool

# Layer 2 — manipulation (SmolVLA)
manip.execute(instruction: str, obs: Obs) -> ActionChunk
manip.gripper_state() -> {open, closed, holding}

# Layer 1 — grounding (Molmo, stateless)
vlm.point(image, description: str) -> (u, v) | None

# Layer 0 — orchestration (yours)
tools = [navigate_to, locate, grasp, release, stow_arm, ask_human]
```

Keeping Layer 1 behind `point()` makes swapping Molmo → Embodied-R1.5 → a hosted API a one-file change.

### The latency rule

**The VLM never enters a control loop.** A quantized 4B model takes 2–5 seconds per call and competes with MuJoCo for the same GPU. Separate process, called only at decision points. If VRAM contention hurts rollout speed, move it to CPU. Local policies close their loops at full rate regardless.

Put this in the README with a diagram — it is the same separation real deployments use, and it is the engineering contribution.

---

# PHASE 1 — Body · Weeks 1–4

**Goal:** a quadruped carrying a 4.3kg arm that walks a cluttered workshop without falling.

**This is the phase that uses what you already have.** It should be the easiest for you and it produces your best quantitative result.

### [CORE] Scene

- MJCF workshop: floor, bench at 75cm, tool tray, five objects with realistic mass and friction, one 12cm step
- Two cameras: head RGB-D (grounding, navigation), wrist RGB (SmolVLA)
- Domain randomization hooks for lighting and object pose, wired in from day one — retrofitting them later is painful

### [CORE] Embodiment

- Mount Z1 on Go2 (Menagerie has both — you write the attachment joint and fix mass/inertia)
- Wrap your existing policy behind the `locomotion` interface above

**Expect this:** the Z1 weighs ~4.3kg. Your Go2 policy never saw it. It will walk drunk or fall when the arm extends. Not a bug — that is the next item.

### [CORE] Payload-aware locomotion

Fine-tune your existing policy. Do **not** retrain from scratch.

- Add arm mass and inertia to the model
- Randomize arm joint configuration during training → the policy learns a shifting centre of mass
- Small reward term for base stability while the arm is extended
- Port the loop to **MJX** (MuJoCo's JAX backend) — doubles as the JAX credential you need for DeepMind

### [CORE] The ablation — two days, most rigorous part of the whole project

Max recoverable push force:

| | Arm stowed | Arm extended |
|---|---|---|
| Original policy | | |
| Payload-aware policy | | |

This is your existing push-recovery study with one new variable. You already know how to run it, including how to catch the silent evaluation bugs.

### [STRETCH] Phase 1 extras

- **Terrain variation** — reuse your stair-climbing policy; add a second step height or a ramp
- **Arm-swing compensation** — an explicit counter-motion term, ablated against the learned solution
- **Carry-load robustness** — randomize held-object mass 0–1kg during training so grasping a heavy object does not destabilize the walk
- **Failure taxonomy** — classify every fall (edge, trip, overextension), the way you classified edge falls at 49% before

### [LATER]
- Port to G1 humanoid (whole-body reach-while-balance)
- Real hardware deployment

### GATE — do not start Phase 2 until all three pass
1. Walks 5m on flat, arm stowed, 20/20 trials
2. Crosses the step with arm extended, 0 falls in 20 trials
3. Ablation table filled in with real numbers

---

# PHASE 2 — Senses · Weeks 5–8

**Goal:** the robot can find any of the five objects and pick it up on command.

**This is the phase that is new for you, and the one that matters most on your CV.**

### [CORE] Grounding

```
vlm.point(head_image, "the 10mm wrench")
  → (u,v) → depth lookup → camera frame → TF → base frame → goal pose
```

Build `locate(description) -> Pose3D | None`. **The `None` case matters** — it triggers the floor-sweep recovery in Phase 3. Do not skip it.

**Free measurement:** sim gives you ground-truth object poses. Use them *only* to score grounding error, never to feed the pipeline. "Median grounding error 3.1cm, 8% miss rate" is a clean result that costs nothing and that closed-API projects usually cannot produce.

### [CORE] Demonstration data — script it, no teleop hardware needed

1. Scripted grasp controller using MuJoCo IK: approach from above, close, lift
2. Randomize object pose, lighting, clutter, initial arm configuration
3. ~400 episodes, keep successes → ~250–300 demos
4. Log in **LeRobot dataset format v2.0**, language instruction attached per episode

**Critical: vary the instruction phrasing.** "grab the wrench", "pick up the 10mm", "get me the small wrench", "hand me the 10 millimetre spanner". If every episode for an object carries one fixed string, the policy learns the object from vision and **ignores language entirely** — a known, widespread VLA failure mode. Generate paraphrases with a local LLM (Ollama) or Molmo itself.

### [CORE] Fine-tune the VLA

SmolVLA from `lerobot/smolvla_base`. 1–2 hours on one GPU. Documented range is 50–200 demos, so 250–300 gives headroom.

**Free compute, in order:**
1. **Kaggle Notebooks** — ~30 GPU hours/week, 16GB VRAM, no credit card. This is the answer.
2. Colab free T4 — checkpoint to Drive often
3. Local LoRA + bf16 + batch 1–2 + gradient accumulation — works on 6GB, slower

### [CORE] Evaluate the VLA properly — report two numbers, separately

- Grasp success across the five objects: **≥70%**
- Correct-wrench selection when both 10mm and 13mm are present: **≥80%**

Almost nobody separates these. That is exactly why it is a sharp detail in an interview.

### [STRETCH] Phase 2 extras

- **Language sensitivity test** — swap the instruction while holding the scene fixed, measure how much success drops. If it barely drops, the policy is ignoring language and you have found something publishable
- **Data-scaling curve** — retrain at 50 / 100 / 200 / 300 demos, plot success. Cheap, and directly answers "how much data does this need?"
- **Held-out object** — never train on the pliers, test zero-shot. Real generalization evidence
- **Compare a second VLA** — π0 or ACT on the same dataset
- **Grasp-quality metric** — not just success, but slip rate during the subsequent walk

### [LATER]
- Fine-tune the VLM itself for your scene
- Multi-camera fusion

### GATE — do not start Phase 3 until all three pass
1. `locate()` correct on all five objects from three viewpoints
2. `locate()` returns `None` correctly when the object is absent
3. Grasp ≥70% and correct-wrench ≥80%

---

# PHASE 3 — Mind · Weeks 9–12

**Goal:** the whole thing runs end to end, recovers from failure, and is published.

### [CORE] Orchestration

Write the state machine. Molmo gets called 4–6 times per episode:

1. Arriving at the bench — where is it?
2. Before the grasp — re-point from close range (the approach shifts the view)
3. After a failed grasp — still there? moved?
4. Floor sweep during recovery scenario 1

**What you build in place of ER 2's orchestration:**

| ER 2 would give you | You write |
|---|---|
| Task sequencing | State machine over the 6 tools |
| Video progress tracking | Explicit gate checks — base reached pose? gripper closed? |
| Failure detection | Timeouts · `holding == False` after grasp · `point()` returns `None` |
| Self-correction | An `on_failure` transition per state |
| Disambiguation | `point()` twice (10mm, 13mm), compare distance, `ask_human` if too close |

~300 lines. Tedious, only handles failures you anticipated — but fully inspectable.

Nominal decomposition:
```
navigate_to(bench) → locate("10mm wrench") → grasp() → navigate_to(human) → release()
```

### [CORE] Recovery behaviors — *the part that makes the project worth watching*

| # | Scenario | Demonstrates |
|---|---|---|
| 1 | Wrench on the floor, not the bench → `locate` returns `None` → floor sweep | Perception-driven replanning |
| 2 | Mid-carry drop → gripper state transition detected → return, re-grasp | Failure detection and retry |
| 3 | Box blocks the path after planning → replan route | Dynamic replanning |
| 4 | Both wrenches present, command says only "wrench" → `ask_human` instead of guessing | Language grounding + interaction |
| 5 | "actually the 13mm" typed mid-walk → abandon, re-target | Interruptibility |

**Scenario 4 stops the scroll.** A robot that asks a clarifying question reads as intelligent in a way a successful grasp never does. If time is short, keep 1, 2, 4 and cut 3, 5.

### [CORE] Evaluation

50 nominal trials + 5 scenarios × 5 trials. Tag every failure with a cause. **The failure breakdown table is a better result than the success rate.**

**Watch for the silent eval bug.** You caught one before. Assert on the actual instruction string reaching the policy each episode — a stale cache will hand you a fake result.

### [CORE] Publish

- **90-second video** — nominal (25s) · recovery montage (45s) · push-recovery ablation (20s). On-screen captions; most people watch muted.
- **GitHub** — architecture diagram, `docker compose up` repro, eval logs committed. Lead the README with: *runs entirely on one 6GB consumer GPU, no API keys, no paid services.*
- **Blog post** — the bugs you root-caused, the payload ablation, grounding error stats, state-machine design.
- **Send it** — LinkedIn and X (tag LeRobot, Menagerie, Ai2 maintainers), then directly to Neura, Agile Robots, ANYbotics, RIVR, 1X with a two-line note.

Week 12, not "when it's polished."

### [STRETCH] Phase 3 extras

- **ROS 2 bridge** — expose the stack as ROS 2 nodes with proper topics and TF. **This is the highest-value stretch item on the whole list for your CV**, because ROS 2 is already on your resume and every one of your target employers runs it. Half a week of work.
- **ER 2 comparison arm** — ER 2 has a free AI Studio tier, no credit card (reporting conflicts on whether the ER models are included — check it yourself in five minutes). If available, run it against your state machine on the same 50 trials. "Open state machine vs. frontier orchestrator" is a better result than either alone.
- **Multi-object commands** — "bring me the wrench and the screwdriver". Tests sequential planning, cheap to add once the state machine exists
- **Spatial language** — "the wrench on the left", "the tool next to the pliers". Plays directly to Molmo's pointing strength
- **Object memory** — remember where things were seen, skip the search on repeat requests
- **Voice input** — Whisper locally, $0. Adds nothing technically but makes the video far more compelling
- **Web dashboard** — live view of state machine, VLM calls, confidence. Excellent for the video
- **Latency budget table** — per-layer timing. Systems engineers love this and it costs an afternoon

### [LATER]
- Multi-robot coordination
- Sim-to-real on borrowed hardware
- G1 port as project #2

### GATE — the project is done when
1. End-to-end nominal success ≥60% over 50 trials
2. 5 recovery scenarios implemented and filmed (or 3, if you cut)
3. Video, repo, and blog post published
4. Five companies contacted

---

## Budget

| Item | Cost |
|---|---|
| MuJoCo / MJX / Menagerie | $0 — Apache 2.0 |
| Molmo2-ER-4B weights | $0 — open weights |
| SmolVLA + LeRobot | $0 — Apache 2.0 |
| All training (local 6GB or Kaggle free tier) | $0 |
| All inference and rollouts | $0 |
| **Total** | **$0** — only real cost is ~10–15GB of one-time downloads |

---

## Risks and cut lines

| Risk | Cut to |
|---|---|
| Go2+Z1 will not stabilize | Fix the base (wheeled or static). Lose the locomotion story, keep the rest. |
| Grasp success under 50% | Drop to 3 objects; or scripted grasp, VLA for object selection only. |
| Molmo pointing too weak in clutter | Classical CV localization on the sim render. Document the swap as a finding. |
| VLM starves the sim for VRAM | Move Layer 1 to CPU, or pause the sim during planning calls. |
| Tool-call parse failures | Grammar-constrained decoding (llama.cpp GBNF), or reduce to 4 tools. |
| Behind at Week 10 | Cut recovery scenarios 3 and 5. Cut every STRETCH item. |

**Hard rule:** at Week 11, whatever works becomes the deliverable. Film and ship on schedule regardless of state. An honest 70%-working demo posted on time beats a perfect one that never leaves your laptop.

---

## Tracker

| Phase | Weeks | Gate | ✓ |
|---|---|---|---|
| **1 — Body** | 1–4 | Walks 5m + step, arm extended, 0 falls in 20 · ablation table filled | ☐ |
| **2 — Senses** | 5–8 | `locate()` works + `None` case · grasp ≥70% · correct-wrench ≥80% | ☐ |
| **3 — Mind** | 9–12 | ≥60% end-to-end · recoveries filmed · published · 5 companies contacted | ☐ |

---

## Parallel track — start Week 1, do not wait

- [ ] **Push-recovery study → arXiv.** Two weeks of writing, work already done. Free. Then a CoRL or ICRA workshop.
- [ ] **Apply to European companies now**, not when the portfolio is finished.
- [ ] **German Chancenkarte** (points-based job-seeker visa) and EU Blue Card shortage-occupation thresholds. Your German-curriculum GUC degree, intermediate German, and mechatronics background fit well.
- [ ] **Post clips monthly.** Hiring managers find people this way constantly.

**Timeline:** 6–12 months to land in Europe → 2–3 years of real-hardware experience → DeepMind becomes a live application rather than a lottery ticket.

---

## This week

Open MuJoCo. Load Go2 and Z1 from Menagerie. Bolt them together. Make it walk 5 meters without falling.

Nothing else on this page matters until that works.
