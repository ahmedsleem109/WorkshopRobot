"""T1.1 -- instrument the slip: where, when and under what force does the tool leave the jaws?

    render_venv\\Scripts\\python.exe scripts\\grasp_diagnose.py [seeds] [--tool NAME] [--csv PATH]

Samples the live sim every ~20 ms through a whole scripted grasp and logs, per sample:
the tool's pose IN THE GRIPPER FRAME, the normal force on each pad, the normal force from
everything that is NOT the gripper (the rack), and the finger travel. From that it derives
the one number T1 needs -- the instant the tool first moves >2 mm in the gripper frame,
and the phase it happened in:

    never          the tool never left the pose it was gripped in       -> success
    close/regrasp  it moved while the jaws were closing                 -> extruded
    lift           it slipped on the way up                             -> grip too weak,
                                                                           or the rack is
                                                                           holding it
    retreat        it slipped only once moving horizontally             -> inertia / jerk

Writes one row per sample to CSV and prints a per-episode and per-tool summary.
"""
import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from bw.manip.ik import ArmIK
from bw.manip.scripted_grasp import SCAN_Q, run_grasp
from bw.sim.workshop import GRASP_TOOLS, RACK_STATION, TOOL_NAMES
from bw.sim.workshop_sim import FING_A, WorkshopSim

SLIP_MM = 2.0          # the threshold the task list asks for
SAMPLE_STEPS = 10      # physics steps between samples (~20 ms at dt=0.002)


class Probe:
    """Wraps WorkshopSim.physics_step so every grasp is sampled at a fixed rate."""

    def __init__(self, sim: WorkshopSim, name: str):
        self.sim, self.name = sim, name
        self.tb = sim.tool_body[name]
        self.fingers = {sim.mover_body: "pad_a", sim.stator_body: "pad_b"}
        self.phase = "init"
        self.rows: list[dict] = []
        self._orig = sim.physics_step
        sim.physics_step = self._stepped
        self._f6 = np.zeros(6)

    def close(self):
        self.sim.physics_step = self._orig

    def set_phase(self, label):
        self.phase = label

    def _stepped(self, n: int = 1):
        left = n
        while left > 0:
            k = min(SAMPLE_STEPS, left)
            self._orig(k)
            left -= k
            self.rows.append(self._sample())

    def _sample(self) -> dict:
        sim, d, m = self.sim, self.sim.d, self.sim.m
        p_ee = d.site_xpos[sim.ee_site]
        R_ee = d.site_xmat[sim.ee_site].reshape(3, 3)
        p_t = d.xpos[self.tb]
        R_t = d.xmat[self.tb].reshape(3, 3)
        rel_p = R_ee.T @ (p_t - p_ee)                 # tool origin in the gripper frame
        rel_R = R_ee.T @ R_t
        f = {"pad_a": 0.0, "pad_b": 0.0, "rack": 0.0}
        ncon = {"pad_a": 0, "pad_b": 0, "rack": 0}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self.tb not in (b1, b2):
                continue
            other = b2 if b1 == self.tb else b1
            key = self.fingers.get(other, "rack")
            mujoco.mj_contactForce(m, d, i, self._f6)
            f[key] += abs(float(self._f6[0]))
            ncon[key] += 1
        return dict(t=round(sim.time, 4), phase=self.phase,
                    rx=rel_p[0], ry=rel_p[1], rz=rel_p[2], rel_R=rel_R.copy(),
                    fa=f["pad_a"], fb=f["pad_b"], frack=f["rack"],
                    na=ncon["pad_a"], nb=ncon["pad_b"], nrack=ncon["rack"],
                    grip_q=float(d.qpos[FING_A]), tool_z=float(p_t[2]), ee_z=float(p_ee[2]),
                    bx=float(d.qpos[0]), by=float(d.qpos[1]), bz=float(d.qpos[2]),
                    ex=float(p_ee[0]), ey=float(p_ee[1]))


def analyse(rows):
    """Reference pose = the last sample of the close (or regrasp) phase, i.e. the pose the
    tool was actually gripped in. Slip is measured from there."""
    grip_phases = ("close", "regrasp")
    idx = [i for i, r in enumerate(rows) if r["phase"] in grip_phases]
    if not idx:
        return {"slip_phase": "no_close", "slip_t": None, "slip_mm_final": None}
    ref_i = idx[-1]
    ref = np.array([rows[ref_i][k] for k in ("rx", "ry", "rz")])
    ref_R = rows[ref_i]["rel_R"]
    out = {"slip_phase": "never", "slip_t": None, "slip_mm_at_onset": None}
    dists, rolls = [], []
    for r in rows[ref_i:]:
        d_mm = 1000.0 * float(np.linalg.norm(np.array([r["rx"], r["ry"], r["rz"]]) - ref))
        dists.append(d_mm)
        dR = ref_R.T @ r["rel_R"]
        rolls.append(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))))
        if out["slip_t"] is None and d_mm > SLIP_MM:
            out.update(slip_phase=r["phase"], slip_t=r["t"], slip_mm_at_onset=round(d_mm, 1))
    hold = rows[ref_i:]
    out["slip_mm_final"] = round(dists[-1], 1)
    out["rot_deg_final"] = round(rolls[-1], 1)
    out["fa_close"] = round(rows[ref_i]["fa"], 1)
    out["fb_close"] = round(rows[ref_i]["fb"], 1)
    out["fmin_hold"] = round(min(min(r["fa"], r["fb"]) for r in hold), 1)
    out["frack_close"] = round(rows[ref_i]["frack"], 1)
    out["frack_max_lift"] = round(max([r["frack"] for r in hold if r["phase"] == "lift"] or [0.0]), 1)
    # first sample after the grip where BOTH pads lose contact with the tool
    lost = [r for r in hold if r["na"] == 0 and r["nb"] == 0]
    out["pads_lost_t"] = lost[0]["t"] if lost else None
    out["pads_lost_phase"] = lost[0]["phase"] if lost else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="?", type=int, default=4)
    ap.add_argument("--tool", default=None, help="only this tool")
    ap.add_argument("--csv", default=str(ROOT / "runs/grasp_diag.csv"))
    ap.add_argument("--walk", nargs="?", const=str(ROOT / "models/payload_nav_policy.npz"),
                    default=None, help="legs on this walking policy (base at RACK_STATION)")
    args = ap.parse_args()

    tools = [args.tool] if args.tool else list(GRASP_TOOLS)
    sim = WorkshopSim()
    if args.walk:
        sim.attach_locomotion(args.walk)
    ik = ArmIK(sim.m)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "tool", "t", "phase", "rx", "ry", "rz", "fa", "fb", "frack",
              "na", "nb", "nrack", "grip_q", "tool_z", "ee_z", "bx", "by", "bz", "ex", "ey"]
    summaries = []
    t0 = time.time()
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for seed in range(args.seeds):
            for tool in tools:
                ti = TOOL_NAMES.index(tool)
                rng = np.random.default_rng(1000 * seed + ti)
                sim.reset(rng, target=tool, arm_q=SCAN_Q,
                          base_pose=RACK_STATION if args.walk else None)
                probe = Probe(sim, tool)
                try:
                    r = run_grasp(sim, ik, tool, rng, on_phase=probe.set_phase)
                finally:
                    probe.close()
                s = analyse(probe.rows)
                s.update(seed=seed, tool=tool, success=r["success"],
                         lifted=r.get("lifted"), reason=r.get("reason"))
                summaries.append(s)
                for row in probe.rows:
                    out = {k: (round(row[k], 5) if isinstance(row[k], float) else row[k])
                           for k in fields if k in row}
                    out["seed"], out["tool"] = seed, tool
                    w.writerow(out)
                print("  seed {} {:12s} success={!s:5s} slip={:9s} @{} final={}mm rot={}deg "
                      "F=({},{})N min={}N rack={}/{}N".format(
                          seed, tool, r["success"], s["slip_phase"], s["slip_t"],
                          s["slip_mm_final"], s.get("rot_deg_final"), s.get("fa_close"),
                          s.get("fb_close"), s.get("fmin_hold"), s.get("frack_close"),
                          s.get("frack_max_lift")), flush=True)

    print("\n{:.0f}s, {} episodes -> {}".format(time.time() - t0, len(summaries), csv_path))
    print("\nslip phase x outcome")
    tab = Counter((s["slip_phase"], "ok" if s["success"] else "FAIL") for s in summaries)
    for (ph, oc), n in sorted(tab.items()):
        print("  {:10s} {:5s} {}".format(ph, oc, n))
    print("\nper tool")
    by = defaultdict(list)
    for s in summaries:
        by[s["tool"]].append(s)
    for tool, ss in by.items():
        ok = sum(s["success"] for s in ss)
        ph = Counter(s["slip_phase"] for s in ss)
        med_f = np.median([s.get("fmin_hold") or 0.0 for s in ss])
        print("  {:12s} {}/{}  slip={}  median min pad force {:.1f} N".format(
            tool, ok, len(ss), dict(ph), med_f))
    return summaries


if __name__ == "__main__":
    main()
