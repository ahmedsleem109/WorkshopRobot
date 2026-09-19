"""T2.4 -- the ONE definition of task success, called by both the data collector and the
evaluator, so training data and scores can never be judged by different rules.

Two task kinds:
  pick      -- lift the named tool out of the rack and keep holding it (HOLD_VERIFY in
               scripted_grasp.py). Success: the named tool is in the jaws and clear of the rack.
  transfer  -- pick the named tool and place it in a table's marked zone. Success: the NAMED
               tool is resting inside the NAMED table's zone and has stopped moving, the
               gripper is empty, and nothing else was knocked out of place.

Every check reads ground truth (`gt_*`), because this is scoring, not the policy's input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bw.sim.workshop import TOOL_NAMES, in_place_zone
from bw.sim.workshop_sim import WorkshopSim

KNOCK_TOL = 0.03       # m: a bystander tool moving further than this was knocked
REST_SPEED = 0.002     # m/s: "stopped moving", same threshold as settle_static
LIFT_CLEAR = 0.08      # m: a picked tool must be at least this far above where it stood


@dataclass(frozen=True)
class Task:
    kind: str                      # "pick" | "transfer"
    tool: str                      # body name, e.g. "wrench_10mm"
    table: str | None = None       # destination for "transfer": "table_a" | "table_b"

    def __post_init__(self):
        assert self.kind in ("pick", "transfer"), self.kind
        assert self.tool in TOOL_NAMES, self.tool
        assert (self.table is not None) == (self.kind == "transfer"), self


@dataclass
class Snapshot:
    """Where every present tool was when the episode started."""
    pos: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def take(cls, sim: WorkshopSim, present) -> "Snapshot":
        return cls({n: sim.gt_tool_pos(n).copy() for n in present})


def _speed(sim: WorkshopSim, name: str) -> float:
    a = sim.tool_dadr[name]
    return float(np.linalg.norm(sim.d.qvel[a:a + 3]))


def evaluate(sim: WorkshopSim, task: Task, start: Snapshot) -> dict:
    """Score the CURRENT sim state against `task`. Returns {"success": bool, "reason": str,
    ...checks}. Call it after the episode has finished and the scene has had time to settle."""
    held = sim.held_tool()
    knocked = sorted(n for n, p in start.pos.items()
                     if n != task.tool
                     and float(np.linalg.norm(sim.gt_tool_pos(n) - p)) > KNOCK_TOL)
    out = {"held": held, "knocked": knocked}
    if task.kind == "pick":
        lifted = float(sim.gt_tool_pos(task.tool)[2] - start.pos[task.tool][2])
        out["lifted"] = round(lifted, 3)
        checks = [(held == task.tool, "wrong_object" if held else "not_held"),
                  (lifted > LIFT_CLEAR, "not_lifted"),
                  (not knocked, "knocked")]
    else:
        pos = sim.gt_tool_pos(task.tool)
        wrong = [n for n in start.pos if n != task.tool and in_place_zone(sim.gt_tool_pos(n), task.table)]
        out.update(in_zone=in_place_zone(pos, task.table), speed=round(_speed(sim, task.tool), 4),
                   wrong_in_zone=wrong)
        checks = [(not wrong, "wrong_object"),
                  (out["in_zone"], "outside_zone"),
                  (out["speed"] < REST_SPEED, "not_settled"),
                  (held is None, "still_held"),
                  (not knocked, "knocked")]
    for ok, why in checks:
        if not ok:
            return {"success": False, "reason": why, **out}
    return {"success": True, "reason": "ok", **out}
