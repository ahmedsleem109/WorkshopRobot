"""T2.5 / T6.2 -- instruction generation for pick and transfer, with paraphrases.

The known VLA failure mode: if an object is always named by one fixed string, the policy
learns the object from vision and ignores the language entirely. So every episode draws a
template x tool synonym x table alias, and `check_diversity` asserts at collection time that
no tool (and no destination) is ever tied to a single phrasing.

Destinations can be named in two independent schemes that agree (see TABLES in
bw/sim/workshop.py): left/right from the robot's start pose, and near/far by distance from
it -- plus plain names, and "the other table" when the destination is not the table the tool
starts on. Picks always start in the rack on table A, so "the other table" means table B.

The size in a wrench name stays in the INSTRUCTION ("the 10mm wrench"). Mapping size to the
grip-band colour is Layer 1's job (vlm.point, see STATUS.md T0.3), not the language's.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from bw.sim.workshop import TABLES
from bw.task.spec import Task

SOURCE_TABLE = "table_a"          # every tool starts in the rack, which is on table A

TOOL_SYNONYMS = {
    "wrench_10mm": ("10mm wrench", "10 mm wrench", "ten-millimetre wrench", "10mm spanner",
                    "small wrench", "10mm"),
    "wrench_13mm": ("13mm wrench", "13 mm wrench", "thirteen-millimetre wrench",
                    "13mm spanner", "big wrench", "13mm"),
    "pliers": ("pliers", "pair of pliers", "red pliers", "pliers with the red handles"),
    "tape_roll": ("tape roll", "roll of tape", "tape", "roll of duct tape"),
    "screwdriver": ("screwdriver", "yellow screwdriver", "flathead screwdriver"),
}

PICK_TEMPLATES = (
    "pick up the {tool}", "grab the {tool}", "bring me the {tool}", "hand me the {tool}",
    "get the {tool} from the rack", "fetch the {tool}", "take the {tool} out of the rack",
    "I need the {tool}", "can you get me the {tool}?", "lift the {tool}",
)

TRANSFER_TEMPLATES = (
    "put the {tool} on {table}", "move the {tool} to {table}", "place the {tool} on {table}",
    "take the {tool} over to {table}", "carry the {tool} to {table}",
    "set the {tool} down on {table}", "the {tool} goes on {table}",
    "could you move the {tool} onto {table}?", "put the {tool} in the green zone on {table}",
    "transfer the {tool} to {table}",
)


def table_aliases(table: str) -> tuple[str, ...]:
    aliases = [a for a in TABLES[table]["aliases"] if a != "the other table"]
    if table != SOURCE_TABLE:
        aliases.append("the other table")
    return tuple(aliases)


def instruction(task: Task, rng: np.random.Generator) -> str:
    tool = TOOL_SYNONYMS[task.tool][rng.integers(len(TOOL_SYNONYMS[task.tool]))]
    if task.kind == "pick":
        t = PICK_TEMPLATES[rng.integers(len(PICK_TEMPLATES))]
        return t.format(tool=tool)
    aliases = table_aliases(task.table)
    table = aliases[rng.integers(len(aliases))]
    t = TRANSFER_TEMPLATES[rng.integers(len(TRANSFER_TEMPLATES))]
    return t.format(tool=tool, table=table)


def check_diversity(pairs, min_phrasings: int = 3) -> None:
    """Collection-time guard: `pairs` is an iterable of (Task, instruction). Raises if any
    tool, or any destination, appears with fewer than `min_phrasings` distinct strings."""
    by_tool, by_table = defaultdict(set), defaultdict(set)
    for task, text in pairs:
        by_tool[task.tool].add(text)
        if task.kind == "transfer":
            by_table[task.table].add(text)
    thin = {k: len(v) for k, v in {**by_tool, **by_table}.items() if len(v) < min_phrasings}
    if thin:
        raise AssertionError(f"instruction diversity too low (distinct phrasings): {thin} -- "
                             "a VLA trained on this learns the object from vision and "
                             "ignores the language")
