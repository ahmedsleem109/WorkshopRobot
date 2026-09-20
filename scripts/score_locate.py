"""T0.4 + T8 benchmark, stage 2 (numpy only, any venv): score the dumps of bench_locate.py.

    python scripts/score_locate.py [runs/t8_bench] [--win 10 --pct 20 --tol 0.05 --views 0,1,2]

T0.4 (per view, per query):
  visible    the tool covers >= MIN_PIX pixels of the GT segmentation map
  HIT        the returned pixel lands on the queried tool's pixels (within HIT_R px)
  wrong obj  lands on a DIFFERENT tool -- for the wrenches this is the 10 vs 13 mm question
  None rate  on visible targets (false refusal) and on absent/invisible ones (correct refusal)
T8 (per seed, per tool): locate_in_views() on the stored replies, error against the tool
body's ground-truth position (3D and horizontal), the distance to the tool's nearest visible
surface point, the miss rate (None although the tool is present) and the false-positive
rate (a point although the tool is absent).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bw.perception.locate import MERGE_TOL, View, locate_in_views
from bw.sim.workshop import TOOL_NAMES

MIN_PIX = 40
HIT_R = 3


def on_tool(seg, uv, r=HIT_R):
    h, w = seg.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    patch = seg[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)]
    return set(int(x) for x in np.unique(patch)) - {-1}


def surface_points(d, k, ti, step=2):
    """World points of tool ti's visible pixels in view k (GT; scoring only)."""
    vv, uu = np.nonzero(d["seg"][k][::step, ::step] == ti)
    vv, uu = vv * step, uu * step
    z = d["depth"][k][vv, uu]
    K, cp, cm = d["K"][k], d["cam_pos"][k], d["cam_mat"][k]
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    pc = np.stack([(uu - cx) * z / f, -(vv - cy) * z / f, -z], 1)
    return cp + pc @ cm.T


def pick(r, stage, verify):
    """The answer a given pipeline variant would have returned (see vlm.point_ex)."""
    if "uv_coarse" not in r:          # old dumps: a single-stage answer
        return r["uv"]
    if stage == "coarse":
        uv = r.get("uv_coarse")
    else:                                # fine, falling back to coarse (what point_ex does)
        uv = r.get("uv_fine") or r.get("uv_coarse")
    if verify and r.get("verify") is False:
        return None
    return uv


def score(D, use=(0, 1, 2), tol=MERGE_TOL, stage="fine", verify=True, **kw):
    seeds = sorted(p.stem for p in D.glob("*.json") if p.stem != "scores")
    rows, loc = [], []
    nt = len(TOOL_NAMES)
    for sd in seeds:
        meta = json.loads((D / f"{sd}.json").read_text())
        d = dict(np.load(D / f"{sd}.npz"))
        rep = {(r["view"], r["tool"]): dict(r, uv=pick(r, stage, verify))
               for r in meta["replies"]}
        if not rep:
            continue
        counts = [[int((d["seg"][k] == ti).sum()) for ti in range(nt)]
                  for k in range(len(d["pans"]))]
        for k in range(len(d["pans"])):
            for ti, tool in enumerate(TOOL_NAMES):
                r = rep[(k, tool)]
                vis = bool(d["present"][ti]) and counts[k][ti] >= MIN_PIX
                ids = on_tool(d["seg"][k], r["uv"]) if r["uv"] else set()
                row = {"seed": sd, "view": k, "tool": tool, "visible": vis,
                       "present": bool(d["present"][ti]), "uv": r["uv"], "raw": r["raw"],
                       "hit": ti in ids, "wrong": bool(ids) and ti not in ids,
                       "latency": r["latency"], "roundtrip": r["roundtrip"]}
                if r["uv"] and vis:
                    vv, uu = np.nonzero(d["seg"][k] == ti)
                    row["px_err"] = float(np.min(np.hypot(uu - r["uv"][0], vv - r["uv"][1])))
                if tool.startswith("wrench"):
                    o = TOOL_NAMES.index("wrench_13mm" if tool == "wrench_10mm" else "wrench_10mm")
                    row["other_wrench_vis"] = bool(d["present"][o]) and counts[k][o] >= MIN_PIX
                    row["other_wrench_hit"] = o in ids
                rows.append(row)
        # ---- T8: locate every tool from the stored replies (the SAME function the robot runs)
        views = [View(rgb=None, depth=d["depth"][k], K=d["K"][k], cam_pos=d["cam_pos"][k],
                      cam_mat=d["cam_mat"][k], pan=float(d["pans"][k])) for k in use]
        for ti, tool in enumerate(TOOL_NAMES):
            uvs = iter([rep[(k, tool)]["uv"] for k in use])
            L = locate_in_views(views, tool, lambda rgb, q: next(uvs), d["base_pos"],
                                d["base_quat"], tol=tol, **kw)
            present = bool(d["present"][ti])
            e = {"seed": sd, "tool": tool, "present": present,
                 "visible_any": present and any(counts[k][ti] >= MIN_PIX for k in use),
                 "found": L is not None}
            if L is not None:
                gt = d["gt"][ti]
                e.update(err3d=float(np.linalg.norm(L.world - gt)),
                         errxy=float(np.linalg.norm(L.world[:2] - gt[:2])),
                         n_views=L.n_views, n_pointed=L.n_pointed, base=L.base.tolist())
                if present:
                    S = np.concatenate([surface_points(d, k, ti) for k in use])
                    if len(S):
                        e["err_surf"] = float(np.min(np.linalg.norm(S - L.world, axis=1)))
                    dist = [np.linalg.norm(L.world[:2] - d["gt"][tj][:2]) if d["present"][tj]
                            else np.inf for tj in range(nt)]
                    e["nearest_is_target"] = int(np.argmin(dist)) == ti
            loc.append(e)
    return rows, loc


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else f"{a}/0"


def report(rows, loc):
    print("=" * 76)
    print("T0.4  vlm.point()  -- per view, per query")
    vis = [r for r in rows if r["visible"]]
    inv = [r for r in rows if not r["visible"]]
    print(f"  queries {len(rows)}   target visible {len(vis)}   not visible/absent {len(inv)}")
    print(f"  HIT on visible target     {pct(sum(r['hit'] for r in vis), len(vis))}")
    print(f"  wrong object              {pct(sum(r['wrong'] for r in vis), len(vis))}")
    print(f"  false None (visible)      {pct(sum(r['uv'] is None for r in vis), len(vis))}")
    print(f"  correct None (not vis)    {pct(sum(r['uv'] is None for r in inv), len(inv))}")
    ab = [r for r in rows if not r["present"]]
    print(f"    of which truly ABSENT   {pct(sum(r['uv'] is None for r in ab), len(ab))}")
    e = [r["px_err"] for r in vis if "px_err" in r]
    if e:
        print(f"  px distance to target mask: median {np.median(e):.1f}  p75 "
              f"{np.percentile(e, 75):.1f}  p90 {np.percentile(e, 90):.1f}")
    w = [r for r in vis if r["tool"].startswith("wrench") and r.get("other_wrench_vis")
         and (r["hit"] or r["other_wrench_hit"])]
    print(f"  10 vs 13 mm (both visible, landed on a wrench): correct "
          f"{pct(sum(r['hit'] for r in w), len(w))}")
    for t in TOOL_NAMES:
        tv = [r for r in vis if r["tool"] == t]
        print(f"    {t:12s} hit {pct(sum(r['hit'] for r in tv), len(tv))}   None "
              f"{sum(r['uv'] is None for r in tv)}   wrong {sum(r['wrong'] for r in tv)}")
    lat = [r["latency"] for r in rows]
    rt = [r["roundtrip"] for r in rows]
    print(f"  latency: model median {np.median(lat):.2f}s p90 {np.percentile(lat, 90):.2f}s;"
          f" round trip median {np.median(rt):.2f}s p90 {np.percentile(rt, 90):.2f}s")

    print("\nT8  locate()  -- per seed, per tool")
    pres = [e for e in loc if e["present"]]
    pv = [e for e in pres if e["visible_any"]]
    ab = [e for e in loc if not e["present"]]
    f = [e for e in pres if e["found"]]
    print(f"  present {len(pres)} (visible in >=1 view {len(pv)})   absent {len(ab)}")
    print(f"  MISS rate (present -> None)          {pct(len(pres) - len(f), len(pres))}")
    print(f"  false positive (absent -> a point)   {pct(sum(e['found'] for e in ab), len(ab))}")
    if f:
        for k, name in (("err3d", "3D error vs body origin"), ("errxy", "horizontal (xy) error"),
                        ("err_surf", "dist to visible surface")):
            a = [e[k] for e in f if k in e]
            print(f"  {name:28s} median {1000 * np.median(a):6.1f} mm  p75 "
                  f"{1000 * np.percentile(a, 75):6.1f}  p90 {1000 * np.percentile(a, 90):6.1f}")
        print(f"  nearest present tool is the target   "
              f"{pct(sum(e.get('nearest_is_target', False) for e in f), len(f))}")
        good = sum(e.get("nearest_is_target", False) and e["errxy"] < 0.03 for e in f)
        print(f"  SUCCESS (found, right tool, xy<30mm) {pct(good, len(pres))}")
        for t in TOOL_NAMES:
            tf = [e for e in pres if e["tool"] == t]
            ok = [e for e in tf if e["found"]]
            me = np.median([e["errxy"] for e in ok]) * 1000 if ok else float("nan")
            print(f"    {t:12s} found {pct(len(ok), len(tf))}  median xy {me:5.1f} mm  "
                  f"right tool {sum(e.get('nearest_is_target', False) for e in ok)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(ROOT / "runs/t8_bench"))
    ap.add_argument("--mode", choices=("snap", "pct"), default=None)
    ap.add_argument("--r", type=int, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--win", type=int, default=None)
    ap.add_argument("--pct", type=float, default=None)
    ap.add_argument("--tol", type=float, default=MERGE_TOL)
    ap.add_argument("--views", default="0,1,2", help="which scan views T8 uses")
    ap.add_argument("--stage", choices=("coarse", "fine"), default="fine")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    kw = {k: v for k, v in (("win", args.win), ("pct", args.pct), ("mode", args.mode),
                            ("r", args.r), ("margin", args.margin)) if v is not None}
    D = Path(args.dir)
    rows, loc = score(D, [int(x) for x in args.views.split(",")], args.tol, args.stage,
                      not args.no_verify, **kw)
    print(f"variant: stage={args.stage} verify={not args.no_verify} tol={args.tol} {kw}")
    report(rows, loc)
    (D / "scores.json").write_text(json.dumps({"t04": rows, "t8": loc}), encoding="utf-8")


if __name__ == "__main__":
    main()
