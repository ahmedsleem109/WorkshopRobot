"""T6.1 -- demonstration collector: scripted pick + walk + place on legs, recorded for SmolVLA.

    render_venv\\Scripts\\python.exe scripts\\collect_demos.py START END [--out DIR]

One RUN = the full transfer the orchestrator will do: pick from the rack (the VLA's `grasp`
skill), walk to the destination table (Layer 3, NOT recorded -- the VLA does not walk), place in
the zone (the VLA's `place` skill). Each run yields up to two episodes, each kept only if the
SHARED success spec (bw/task/spec.py) says so:
    pick   -- instruction from PICK_TEMPLATES, scored as Task("pick") after the hold check
    place  -- instruction from TRANSFER_TEMPLATES, scored as Task("transfer") at the end
A failed pick ends the run; a failed walk or place drops only the place episode.

Per 10 Hz tick (the arm's command rate): wrist + mast RGB (256x256), arm state q (7: six joints +
finger travel) and the commanded target (7) -- the action is exactly what the servos were sent.
Written per episode as <out>/<run>_<kind>/{wrist.mp4, mast.mp4, data.npz, meta.json};
scripts/to_lerobot.py (WSL, torch venv) converts the lot into a LeRobotDataset.

Randomised per run: tool subset + slot order + lean + flip, lighting, rack colour (sim.reset);
the base pose around the rack station (+-3 cm, +-6 deg: where walk_to actually stops); the
scan pose (+-0.05 rad per joint); the destination table; the instruction phrasing.
Seeds START..END-1 -- keep them disjoint from the evaluation seeds (0..99, try_grasp/try_place).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np

from bw.locomotion.navigate import walk_to
from bw.manip.disturb import Disturber
from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp
from bw.manip.scripted_place import run_place
from bw.sim.workshop import GRASP_TOOLS, PLACE_STATION, RACK_STATION
from bw.sim.workshop_sim import WorkshopSim
from bw.task.language import instruction
from bw.task.spec import Snapshot, Task, evaluate

CAMERAS = ("wrist", "mast")
FPS = 10


class Recorder:
    def __init__(self, sim: WorkshopSim, disturber=None):
        self.sim = sim
        self.dz = disturber
        self.frames = {c: [] for c in CAMERAS}
        self.state, self.action = [], []

    def __call__(self, q_cmd):
        if self.dz is not None:          # decide THIS tick's kick before it is executed
            self.dz.tick()
        for c in CAMERAS:
            self.frames[c].append(self.sim.render(c))
        self.state.append(self.sim.arm_q().astype(np.float32))
        self.action.append(np.asarray(q_cmd, np.float32))

    def __len__(self):
        return len(self.action)

    def save(self, path: Path, meta: dict):
        path.mkdir(parents=True, exist_ok=True)
        for c in CAMERAS:
            imageio.mimwrite(path / f"{c}.mp4", self.frames[c], fps=FPS, codec="libx264",
                             quality=9, macro_block_size=1, pixelformat="yuv420p")
        np.savez(path / "data.npz", state=np.stack(self.state), action=np.stack(self.action))
        (path / "meta.json").write_text(json.dumps({**meta, "length": len(self)}))


def run(sim, ik, seed: int, out: Path, noise: float = 0.0, sigma: float = None) -> dict:
    rng = np.random.default_rng(10_000_019 + seed)
    tool = GRASP_TOOLS[int(rng.integers(len(GRASP_TOOLS)))]
    table = ("table_a", "table_b")[int(rng.integers(2))]
    base = (RACK_STATION[0] + rng.uniform(-0.03, 0.03), RACK_STATION[1] + rng.uniform(-0.03, 0.03),
            RACK_STATION[2] + np.radians(rng.uniform(-6, 6)))
    scan = SCAN_Q.copy()
    scan[:6] += rng.uniform(-0.05, 0.05, 6)
    present = sim.reset(rng, target=tool, arm_q=scan, base_pose=base)
    start = Snapshot.take(sim, present)
    res = {"seed": seed, "tool": tool, "table": table, "pick": False, "place": False}

    pick_task = Task("pick", tool)
    dz = Disturber(sim, rng, p=noise, **({} if sigma is None else {"sigma": sigma})) if noise > 0 else None
    rec = Recorder(sim, dz)
    if dz is None:
        g = run_grasp(sim, ik, tool, rng, record=rec)
    else:
        with dz.attached():
            g = run_grasp(sim, ik, tool, rng, record=rec, on_phase=dz.on_phase)
        res["pick_kicks"] = len(dz.kicks)
    ev = evaluate(sim, pick_task, start) if g["success"] else {"success": False,
                                                               "reason": g["reason"]}
    res["pick_reason"] = ev["reason"]
    if not ev["success"]:
        return res
    rec.save(out / f"{seed:06d}_pick", {"kind": "pick", "tool": tool, "table": None,
                                        "seed": seed, "instruction": instruction(pick_task, rng),
                                        "present": sorted(present),
                                        **({} if dz is None else dz.summary())})
    res["pick"] = True

    w = walk_to(sim, PLACE_STATION[table])
    if not w["success"]:
        res["place_reason"] = "walk_fell" if w["fell"] else "walk_timeout"
        return res
    sim.settle(0.3)
    task = Task("transfer", tool, table)
    dzp = Disturber(sim, rng, p=noise, **({} if sigma is None else {"sigma": sigma})) if noise > 0 else None
    rec = Recorder(sim, dzp)
    if dzp is None:
        p = run_place(sim, ik, tool, table, rng, record=rec)
    else:
        with dzp.attached():
            p = run_place(sim, ik, tool, table, rng, record=rec, on_phase=dzp.on_phase)
        res["place_kicks"] = len(dzp.kicks)
    if p.get("reason") not in ("place_ik_fail", "not_holding"):
        p = evaluate(sim, task, start)
    res["place_reason"] = p["reason"]
    if p["success"]:
        rec.save(out / f"{seed:06d}_place", {"kind": "place", "tool": tool, "table": table,
                                             "seed": seed, "instruction": instruction(task, rng),
                                             "present": sorted(present),
                                             **({} if dzp is None else dzp.summary())})
        res["place"] = True
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", type=int)
    ap.add_argument("end", type=int)
    ap.add_argument("--out", default="D:/bw_data/raw")
    ap.add_argument("--noise-sigma", type=float, default=None,
                    help="rad, per joint, of a kick (default bw/manip/disturb.SIGMA)")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="T7 covariate shift: probability per 10 Hz tick that a joint-space "
                         "kick starts during a free-space phase (bw/manip/disturb.py). "
                         "0 = the session-5/6 noise-free demonstrator.")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = WorkshopSim()
    sim.attach_locomotion()
    ik = ArmIK(sim.m)
    log = out / f"log_{args.start:06d}_{args.end:06d}.jsonl"
    t0 = time.time()
    with log.open("a", encoding="utf-8") as fh:
        for seed in range(args.start, args.end):
            if (out / f"{seed:06d}_pick").exists():     # resumable
                continue
            try:
                r = run(sim, ik, seed, out, noise=args.noise, sigma=args.noise_sigma)
            except Exception as e:                    # one bad episode must not kill a lane
                r = {"seed": seed, "error": repr(e)}
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"{time.time() - t0:6.0f}s {r}", flush=True)


if __name__ == "__main__":
    main()
