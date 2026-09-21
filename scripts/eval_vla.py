"""T7.3 -- the fixed SmolVLA evaluation protocol (decided BEFORE looking at any checkpoint).

    # WSL: ~/bringwrench/.venv-vla/bin/python bw/policy/vla_server.py --ckpt <dir>
    render_venv\\Scripts\\python.exe scripts\\eval_vla.py N [--transfer] [--swap] [--out F.json]

Held-out scenes: seeds 0..N-1 with the benchmark seeding (rng 1000*seed + tool index) -- the
collector used rng(10_000_019 + seed), so no evaluation scene was trained on. ALL five tools are
present in every scene (both wrenches always), base +-3 cm / +-6 deg around the rack station.
The instruction is a paraphrase drawn per episode and is printed with the result, so what
actually reached the policy is on record (a stale instruction faked a result last project).

Three numbers, reported separately (bw/task/spec.py decides success, same rule as the data):
    grasp          the NAMED tool is held and lifted clear            (Task "pick")
    correct_object among grasp attempts that lifted SOMETHING, the named one
    transfer       --transfer: after a successful pick, walk (Layer 3) and place with the VLA;
                   the named tool rests in the named table's zone  (Task "transfer")
--swap (T7.5): the same scene is run twice with instructions naming two DIFFERENT present tools;
    if the policy ignores language it picks the same tool both times.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.locomotion.navigate import walk_to
from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q
from bw.policy import vla
from bw.sim.workshop import GRASP_TOOLS, PLACE_STATION, RACK_STATION, TOOL_NAMES
from bw.sim.workshop_sim import WorkshopSim
from bw.task.language import instruction
from bw.task.spec import Snapshot, Task, evaluate

PICK_S, PLACE_S = 14.0, 12.0     # demos are ~9-12 s (pick) and ~8-10 s (place) at 10 Hz


def train_scene(sim, seed):
    """Rebuild a scene a DEMONSTRATION was collected in (scripts/collect_demos.py's rng stream), so
    the policy can be scored on states it was actually trained on. Distinguishing "cannot grasp
    anywhere" from "cannot grasp in a NEW scene" is what separates an execution problem from
    covariate shift, and the held-out protocol cannot tell them apart on its own.
    Returns (tool, rng, start) -- the tool is the one the demonstration used, not seed % 5."""
    from scripts.vla_exec_check import collect_scene
    tool, start = collect_scene(sim, seed)
    return tool, np.random.default_rng(31_000 + seed), start


def scene(sim, seed, tool):
    ti = TOOL_NAMES.index(tool)
    rng = np.random.default_rng(1000 * seed + ti)
    j = np.random.default_rng(777 + 1000 * seed + ti)
    base = (RACK_STATION[0] + j.uniform(-0.03, 0.03), RACK_STATION[1] + j.uniform(-0.03, 0.03),
            RACK_STATION[2] + np.radians(j.uniform(-6, 6)))
    present = sim.reset(rng, target=tool, arm_q=SCAN_Q, base_pose=base,
                        tools_present=set(TOOL_NAMES))
    return rng, Snapshot.take(sim, present)


def lifted_tool(sim, start):
    held = sim.held_tool()
    if held and sim.gt_tool_pos(held)[2] - start.pos[held][2] > 0.08:
        return held
    return None


def pick(sim, tool, rng, start, from_pregrasp: bool = False, ik=None, max_s: float = None):
    text = instruction(Task("pick", tool), rng)
    if from_pregrasp:
        # Score the policy on the stretch it is trained for: the scripted stack transports to the
        # pre-grasp pose (in the full pipeline that target comes from locate(), median 10.7 mm) and
        # the policy does the approach, the close and the lift. See scripted_grasp.to_pregrasp.
        from bw.manip.scripted_grasp import to_pregrasp
        if to_pregrasp(sim, ik, tool, rng) is None:
            return text, {"success": False, "reason": "pregrasp_ik_fail"}, None
    vla.run_skill(sim, text, max_s or PICK_S)
    sim.settle(1.0)
    ev = evaluate(sim, Task("pick", tool), start)
    return text, ev, lifted_tool(sim, start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-pregrasp", action="store_true",
                    help="scripted transport to the pre-grasp pose, then hand over to the policy "
                         "(matches to_lerobot.py --fine-phase)")
    ap.add_argument("--pick-s", type=float, default=None,
                    help="seconds of policy control for the pick (default 14; use ~7 with "
                         "--from-pregrasp, since transport is no longer the policy's job)")
    ap.add_argument("--train-scenes", action="store_true",
                    help="score on the scenes the demonstrations were collected in (T7 Tier 1), "
                         "instead of held-out seeds")
    ap.add_argument("n", type=int)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert vla.health(), "start bw/policy/vla_server.py first"
    sim = WorkshopSim()
    sim.attach_locomotion()
    ik = ArmIK(sim.m) if args.from_pregrasp else None
    rows = []
    t0 = time.time()
    for seed in range(args.start, args.start + args.n):
        if args.train_scenes:
            tool, rng, start = train_scene(sim, seed)
        else:
            tool = GRASP_TOOLS[seed % len(GRASP_TOOLS)]
            rng, start = scene(sim, seed, tool)
        text, ev, got = pick(sim, tool, rng, start, args.from_pregrasp, ik, args.pick_s)
        row = {"seed": seed, "tool": tool, "instruction": text, "grasp": ev["success"],
               "grasp_reason": ev["reason"], "lifted": got}
        if args.transfer and ev["success"]:
            table = ("table_a", "table_b")[seed % 2]
            w = walk_to(sim, PLACE_STATION[table])
            row["table"] = table
            if w["success"]:
                sim.settle(0.3)
                t2 = instruction(Task("transfer", tool, table), rng)
                vla.run_skill(sim, t2, PLACE_S)
                sim.settle(2.0)
                e2 = evaluate(sim, Task("transfer", tool, table), start)
                row.update(place_instruction=t2, transfer=e2["success"], transfer_reason=e2["reason"])
            else:
                row.update(transfer=False, transfer_reason="walk")
        if args.swap:
            other = GRASP_TOOLS[(seed + 2) % len(GRASP_TOOLS)]
            rng2, start2 = scene(sim, seed, tool)          # the SAME scene again
            t3, e3, got3 = pick(sim, other, rng2, start2)
            row.update(swap_tool=other, swap_instruction=t3, swap_grasp=e3["success"],
                       swap_lifted=got3)
        rows.append(row)
        print(f"{time.time() - t0:6.0f}s {row}", flush=True)

    n = len(rows)
    per = defaultdict(Counter)
    for r in rows:
        per[r["tool"]]["n"] += 1
        per[r["tool"]]["grasp"] += r["grasp"]
    lifted = [r for r in rows if r["lifted"]]
    out = {"n": n, "grasp": sum(r["grasp"] for r in rows) / n,
           "correct_object": (sum(r["lifted"] == r["tool"] for r in lifted) / len(lifted)) if lifted else None,
           "per_tool": {t: f"{c['grasp']}/{c['n']}" for t, c in per.items()}}
    if args.transfer:
        out["transfer"] = sum(bool(r.get("transfer")) for r in rows) / n
    if args.swap:
        out["swap_grasp"] = sum(r["swap_grasp"] for r in rows) / n
        both = [r for r in rows if r["lifted"] and r["swap_lifted"]]
        out["swap_same_tool_both_times"] = (sum(r["lifted"] == r["swap_lifted"] for r in both) / len(both)) if both else None
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": out, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
