# Repository layout: what to move, and why it has not been moved yet

The code is organised by layer (`bw/sim`, `bw/locomotion`, `bw/manip`, `bw/perception`, `bw/policy`,
`bw/task`) and that part is sound. What a newcomer trips over is `scripts/`: 40+ files with no
distinction between a benchmark you are meant to run and a one-off probe that answered one question in
one session.

**Why this is a plan and not a commit:** background jobs reference these paths right now — the files
in `ops/queue/pending/` call `scripts/to_lerobot.py`, `scripts/eval_vla.py` and others, and the GPU
queue is mid-pipeline. Moving files would break running work for a cosmetic gain. Execute this once
the queue drains, in one commit, updating the `ops/` and docs references in the same change.

## The move

```
scripts/                   ->  becomes                     rationale
  try_grasp.py                 benchmarks/                 things you are meant to run:
  try_place.py                 benchmarks/                 each prints a table and takes --seeds
  eval_suite.py                benchmarks/
  eval_summary.py              benchmarks/
  eval_vla.py                  benchmarks/
  bench_locate.py              benchmarks/
  score_locate.py              benchmarks/

  collect_demos.py             pipeline/                   data -> dataset -> policy -> media
  to_lerobot.py                pipeline/
  make_montage.py              pipeline/
  make_orch_video.py           pipeline/
  render_traj.py               pipeline/

  vla_exec_check.py            diagnostics/                each of these located a real defect
  vla_replay_check.py          diagnostics/                and is worth re-running
  vla_frame_gap.py             diagnostics/
  grasp_diagnose.py            diagnostics/
  topple_diagnose.py           diagnostics/
  score_absent.py              diagnostics/
  pick_best_ckpt.py            diagnostics/
  t5_obs_check.py              diagnostics/
  solver_sweep.py              diagnostics/
  stability_baseline.py        diagnostics/
  nav_tracking.py              diagnostics/
  reach_audit.py               diagnostics/

  _knock_probe.py  _tape_sweep.py  _creep_probe.py  _creep_sweep.py  _pliers_probe.py
  _absent_probe.py  _grip_dbg.py  _grip_force_probe.py  _grip_variants.py  _gripforce.py
  _pad_fix_sweep.py  _diag_gate2.py  _diag_roll_at_close.py  _liftdbg.py  _retreatdbg.py
  _closeup.py  debug_nan.py  debug_nan2.py  debug_nan3.py
                               attic/                      kept for the record, not for use
```

Rules that keep it clean afterwards:

1. A file in `benchmarks/` prints a table and takes a seed count. If it prints a one-off number, it is
   a diagnostic.
2. A file whose name starts with `_` belongs in `attic/` from birth. It answered one question; the
   answer lives in `STATUS.md`, and the file is kept only so the measurement can be re-run.
3. `ops/` holds runnable job scripts. `ops/queue/` is runtime state and stays gitignored.
4. Every benchmark states its acceptance bar in its own docstring, next to how to run it.

## Worth doing in the same commit

* `MUJOCO_LOG.TXT` and `scripts/MUJOCO_LOG.TXT` are MuJoCo's own crash logs — now gitignored, and the
  tracked copies should be deleted.
* `bring-me-the-10mm-wrench-plan (1).md` is the original plan carrying a download-mangled filename.
  Move it to `docs/original-plan.md`; `STATUS.md` refers to its phases and gates, so it stays.
* Add a `benchmarks/README.md` listing each benchmark, its acceptance bar and its current number, so
  the bars live next to the code rather than only in `STATUS.md`.
