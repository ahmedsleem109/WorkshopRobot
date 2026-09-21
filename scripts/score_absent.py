"""T8 -- choose the ABSENT-tool rejection rule, offline, from one pass of model calls.

Input: runs/t8_views/absent_probe.json (scripts/_absent_probe.py) + the view dumps beside it.
Nothing here calls the model, so rules are free to compare and the comparison is reproducible.

Every rule is scored on the three numbers that trade against each other:

    FP      a point returned for a tool that is NOT IN THE SCENE   (today 14/20 = 70%)
    MISS    None although the tool is present                      (today 2.8%)
    SUCCESS found, nearest present tool is the target, xy < 30 mm   (today 135/180 = 75%)

A rule is only worth taking if it cuts FP without eating SUCCESS -- the missing-tool recovery
needs the refusal, but the other six suites need the tool found.

    D:\\hexapod\\render_venv\\Scripts\\python.exe scripts\\score_absent.py [runs/t8_views]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.perception.locate import View, locate_in_views
from bw.sim.workshop import TOOL_NAMES

# What an open "name the tool" answer must contain to count as the expected tool.
NAME_WORDS = {"wrench_10mm": ("wrench", "spanner"), "wrench_13mm": ("wrench", "spanner"),
              "screwdriver": ("screwdriver", "driver"), "pliers": ("plier", "pincer"),
              "tape_roll": ("tape", "roll")}


def yes(raw):
    """A yes/no answer -> True/False/None. The model is agreeable: anything that does not
    start with a refusal counts as yes, which is the charitable reading for a rejection rule."""
    if raw is None:
        return None
    t = raw.strip().lower()
    if t.startswith(("no", "there is no", "there are no", "nope")):
        return False
    return True


def name_ok(tool, raw):
    if raw is None:
        return None
    t = raw.strip().lower()
    return any(w in t for w in NAME_WORDS[tool])


RULES = {
    # name: (per-view filter(probe_row) -> keep?, min views in the winning cluster)
    "current (accept any point)":            (lambda r: True, 1),
    "cross-view: 2 of 3 agree":              (lambda r: True, 2),
    "crop verify":                           (lambda r: yes(r["crop"]) is not False, 1),
    "full-image verify":                     (lambda r: yes(r["full"]) is not False, 1),
    "name matches on the crop":              (lambda r: name_ok(r["tool"], r["name"]) is not False, 1),
    "full-image verify + 2 of 3":            (lambda r: yes(r["full"]) is not False, 2),
    "name matches + 2 of 3":                 (lambda r: name_ok(r["tool"], r["name"]) is not False, 2),
    "full + name":                           (lambda r: yes(r["full"]) is not False
                                              and name_ok(r["tool"], r["name"]) is not False, 1),
}


def load(D: Path):
    probe = {(r["seed"], r["view"], r["tool"]): r for r in
             json.loads((D / "absent_probe.json").read_text())}
    seeds = sorted({k[0] for k in probe})
    return probe, seeds


def score_tiered(D: Path, probe, seeds, keep, views=(0, 1, 2), agree=2):
    """TIERED, which is the point of the exercise: a point CORROBORATED BY ANOTHER VIEW is accepted
    as it stands (its false-positive rate is 2/20, not 14/20), and only an uncorroborated one is put
    to `keep` -- the extra model call. A blanket rule spends its recall on the answers that were
    already fine; this spends it only where the evidence is thin."""
    res = {"fp": 0, "absent": 0, "miss": 0, "present": 0, "success": 0, "asked": 0}
    for sd in seeds:
        meta = json.loads((D / f"{sd}.json").read_text())
        if not meta.get("replies"):
            continue
        d = dict(np.load(D / f"{sd}.npz"))
        rep = {(r["view"], r["tool"]): r for r in meta["replies"]}
        V = [View(rgb=None, depth=d["depth"][k], K=d["K"][k], cam_pos=d["cam_pos"][k],
                  cam_mat=d["cam_mat"][k], pan=float(d["pans"][k])) for k in views]
        for ti, tool in enumerate(TOOL_NAMES):
            raw = [rep[(k, tool)].get("uv_fine") or rep[(k, tool)].get("uv_coarse") for k in views]
            it = iter(raw)
            L = locate_in_views(V, tool, lambda rgb, q: next(it), d["base_pos"], d["base_quat"])
            if L is not None and L.n_views < agree:          # thin evidence: ask, then re-merge
                res["asked"] += 1
                flt = [uv if (uv is None or keep(probe[(sd, k, tool)])) else None
                       for uv, k in zip(raw, views) if True]
                it2 = iter(flt)
                L = locate_in_views(V, tool, lambda rgb, q: next(it2), d["base_pos"], d["base_quat"])
            present = bool(d["present"][ti])
            if not present:
                res["absent"] += 1
                res["fp"] += L is not None
                continue
            res["present"] += 1
            if L is None:
                res["miss"] += 1
                continue
            gt = d["gt"][ti]
            errxy = float(np.linalg.norm(L.world[:2] - gt[:2]))
            dist = [np.linalg.norm(L.world[:2] - d["gt"][tj][:2]) if d["present"][tj] else np.inf
                    for tj in range(len(TOOL_NAMES))]
            res["success"] += int(np.argmin(dist)) == ti and errxy < 0.03
    return res


def score(D: Path, probe, seeds, keep, min_views, views=(0, 1, 2)):
    res = {"fp": 0, "absent": 0, "miss": 0, "present": 0, "success": 0}
    for sd in seeds:
        meta = json.loads((D / f"{sd}.json").read_text())
        if not meta.get("replies"):
            continue
        d = dict(np.load(D / f"{sd}.npz"))
        rep = {(r["view"], r["tool"]): r for r in meta["replies"]}
        V = [View(rgb=None, depth=d["depth"][k], K=d["K"][k], cam_pos=d["cam_pos"][k],
                  cam_mat=d["cam_mat"][k], pan=float(d["pans"][k])) for k in views]
        for ti, tool in enumerate(TOOL_NAMES):
            uvs = []
            for k in views:
                r = rep[(k, tool)]
                uv = r.get("uv_fine") or r.get("uv_coarse")
                pr = probe.get((sd, k, tool))
                if uv is not None and pr is not None and not keep(pr):
                    uv = None                      # this view's answer is rejected
                uvs.append(uv)
            it = iter(uvs)
            L = locate_in_views(V, tool, lambda rgb, q: next(it), d["base_pos"], d["base_quat"])
            if L is not None and L.n_views < min_views:
                L = None                           # not enough views agreed: refuse
            present = bool(d["present"][ti])
            if not present:
                res["absent"] += 1
                res["fp"] += L is not None
                continue
            res["present"] += 1
            if L is None:
                res["miss"] += 1
                continue
            gt = d["gt"][ti]
            errxy = float(np.linalg.norm(L.world[:2] - gt[:2]))
            dist = [np.linalg.norm(L.world[:2] - d["gt"][tj][:2]) if d["present"][tj] else np.inf
                    for tj in range(len(TOOL_NAMES))]
            res["success"] += int(np.argmin(dist)) == ti and errxy < 0.03
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(ROOT / "runs/t8_views"))
    args = ap.parse_args()
    D = Path(args.dir)
    probe, seeds = load(D)
    print(f"{len(seeds)} seeds, {len(probe)} probe rows\n")
    print(f"{'rule':30s} {'FP (absent->point)':>20s} {'MISS':>12s} {'SUCCESS':>14s}")
    out = {}
    TIERED = {
        "2 views agree, else full-image verify": lambda r: yes(r["full"]) is not False,
        "2 views agree, else crop verify": lambda r: yes(r["crop"]) is not False,
        "2 views agree, else name matches": lambda r: name_ok(r["tool"], r["name"]) is not False,
        "2 views agree, else full AND name": lambda r: (yes(r["full"]) is not False
                                                       and name_ok(r["tool"], r["name"]) is not False),
    }
    for name, (keep, mv) in RULES.items():
        r = score(D, probe, seeds, keep, mv)
        out[name] = r
        f = f"{r['fp']}/{r['absent']} ({100 * r['fp'] / max(1, r['absent']):.0f}%)"
        m = f"{r['miss']}/{r['present']}"
        s = f"{r['success']}/{r['present']} ({100 * r['success'] / max(1, r['present']):.0f}%)"
        print(f"{name:30s} {f:>20s} {m:>12s} {s:>14s}")
    print()
    for name, keep in TIERED.items():
        r = score_tiered(D, probe, seeds, keep)
        out[name] = r
        f = f"{r['fp']}/{r['absent']} ({100 * r['fp'] / max(1, r['absent']):.0f}%)"
        m = f"{r['miss']}/{r['present']}"
        sc = f"{r['success']}/{r['present']} ({100 * r['success'] / max(1, r['present']):.0f}%)"
        print(f"{name:30s} {f:>20s} {m:>12s} {sc:>14s}   extra calls {r['asked']}")
    (D / "absent_rules.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
