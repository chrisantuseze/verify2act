"""Relabel twin frames with language goals: any goal text + a saved block state (``states.json``) -> satisfied or not.

The predicates mirror ``DofbotTwin.on`` / ``side_of`` / bin presence (``verify2act/twin/scene.py``), but work on the
saved per-frame states, so every frame can be labelled against every goal, including counterfactual ones (swapped
arguments, the other side, a different bin subset). This gives the goal critic hard negatives the episodes alone lack.
"""

import math
import random
from typing import Dict, List, Sequence, Tuple

COLORS = ("red", "green", "blue", "yellow")
WARM, COOL = ("red", "yellow"), ("green", "blue")
SIZE = (0.06, 0.03, 0.03)                      # block L, W, H (twin config)

State = Dict[str, Dict]


def present(s: State, c: str) -> bool:
    return bool(s[c]["present"])


def on(s: State, c: str, b: str) -> bool:
    if c == b or not (present(s, c) and present(s, b)):
        return False
    pc, pb, yb = s[c]["pos"], s[b]["pos"], s[b]["yaw"]
    dz = pc[2] - pb[2]
    if not (0.6 * SIZE[2] < dz < 1.4 * SIZE[2]):
        return False
    cs, sn = math.cos(math.radians(yb)), math.sin(math.radians(yb))
    dx, dy = pc[0] - pb[0], pc[1] - pb[1]
    return abs(cs * dx + sn * dy) <= SIZE[0] / 2 and abs(-sn * dx + cs * dy) <= SIZE[1] / 2


def side_of(s: State, c: str, b: str, side: str) -> bool:
    """left = +y = image left, 0.6-3.5 block widths away, level, ahead/behind by less than a block length."""
    if c == b or not (present(s, c) and present(s, b)):
        return False
    pc, pb = s[c]["pos"], s[b]["pos"]
    dy = (pc[1] - pb[1]) * (1 if side == "left" else -1)
    return 0.6 * SIZE[1] <= dy <= 3.5 * SIZE[1] and abs(pc[0] - pb[0]) < SIZE[0] and abs(pc[2] - pb[2]) < SIZE[2]


def binned(s: State, targets: Sequence[str], keep: Sequence[str] = ()) -> bool:
    return all(not present(s, c) for c in targets) and all(present(s, c) for c in keep)


def _blocks(cs: Sequence[str]) -> str:
    n = [f"the {c} block" for c in cs]
    return n[0] if len(n) == 1 else f"{n[0]} and {n[1]}" if len(n) == 2 else ", ".join(n[:-1]) + f" and {n[-1]}"


# The real-robot eval tasks verbatim (verify2act/twin/tasks.py EVAL_TASKS) as (text, predicate).
EVAL_GOALS: List[Tuple[str, callable]] = [
    ("Put the blue block and the yellow block into the bin", lambda s: binned(s, ["blue", "yellow"], ["red", "green"])),
    ("Clear all cool-colored blocks into the bin and leave the yellow block",
     lambda s: binned(s, ["green", "blue"], ["red", "yellow"])),
    ("Clear all warm-colored blocks into the bin and leave the green block",
     lambda s: binned(s, ["red", "yellow"], ["green", "blue"])),
    ("Put the red block, the green block and the blue block into the bin except the yellow block",
     lambda s: binned(s, ["red", "green", "blue"], ["yellow"])),
    ("Put the red block to the left of the blue block", lambda s: side_of(s, "red", "blue", "left")),
    ("Put the green block to the right of the yellow block", lambda s: side_of(s, "green", "yellow", "right")),
    ("Stack the blue block on top of the yellow block", lambda s: on(s, "blue", "yellow")),
]


def side_text(rng: random.Random, c: str, b: str, side: str) -> str:
    return f"{rng.choice(['Put', 'Place', 'Move'])} the {c} block to the {side} of the {b} block"


def stack_text(rng: random.Random, c: str, b: str) -> str:
    return rng.choice([f"Stack the {c} block on top of the {b} block", f"Put the {c} block on the {b} block",
                       f"Place the {c} block on top of the {b} block"])


def bin_goal(rng: random.Random, s: State) -> Tuple[str, bool]:
    """A random bin goal (same styles as tasks.bin_goal); ~half are built from the frame's actual bin state."""
    style = rng.choice(["explicit", "explicit", "group", "except", "all"])
    gone = [c for c in COLORS if not present(s, c)]
    if style == "group":
        name, group = rng.choice([("warm", WARM), ("cool", COOL)])
        others = [c for c in COLORS if c not in group]
        keep = [rng.choice(others)] if rng.random() < 0.7 else []
        text = f"{rng.choice(['Clear all', 'Put all', 'Move all', 'Remove all'])} {name}-colored blocks into the bin"
        if keep:
            text += f" and {rng.choice(['leave', 'keep'])} {_blocks(keep)}"
        return text, binned(s, group, keep)
    if style == "except":
        spared = rng.choice(COLORS)
        targets = [c for c in COLORS if c != spared]
        text = rng.choice([f"Put {_blocks(targets)} into the bin except the {spared} block",
                           f"Put every block into the bin except the {spared} block",
                           f"Clear the table except the {spared} block",
                           f"Clear all blocks into the bin but leave the {spared} block"])
        return text, binned(s, targets, [spared])
    if style == "all":
        return rng.choice(["Put all the blocks into the bin", "Clear the table", "Clear every block into the bin"]), \
            binned(s, COLORS)
    if gone and rng.random() < 0.5:                  # targets drawn from what is actually gone -> often positive
        k = rng.randint(1, len(gone))
        targets = rng.sample(gone, k)
    else:
        targets = rng.sample(COLORS, rng.randint(1, 3))
    others = [c for c in COLORS if c not in targets]
    keep = [rng.choice(others)] if others and rng.random() < 0.25 else []
    text = f"{rng.choice(['Put', 'Move', 'Place', 'Drop'])} {_blocks(targets)} into the bin"
    if keep:
        text += f" and leave {_blocks(keep)}"
    return text, binned(s, targets, keep)


def sample_goals(rng: random.Random, s: State, k: int = 8) -> List[Tuple[str, int]]:
    """k (text, label) goals for one frame: eval tasks, true relations and their hard negatives, bin goals."""
    out: List[Tuple[str, int]] = []
    tab = [c for c in COLORS if present(s, c)]
    # true side / stack relations in this frame and their minimal counterfactuals
    sides = [(c, b, d) for c in tab for b in tab for d in ("left", "right") if side_of(s, c, b, d)]
    stacks = [(c, b) for c in tab for b in tab if on(s, c, b)]
    if sides:
        c, b, d = rng.choice(sides)
        other = "right" if d == "left" else "left"
        out.append((side_text(rng, c, b, d), 1))
        out.append((side_text(rng, c, b, other), 0))                            # flipped side
        out.append((side_text(rng, b, c, d), int(side_of(s, b, c, d))))         # swapped arguments
    if stacks:
        c, b = rng.choice(stacks)
        out.append((stack_text(rng, c, b), 1))
        out.append((stack_text(rng, b, c), int(on(s, b, c))))
    # eval tasks verbatim (these are what the robot is asked)
    for text, pred in rng.sample(EVAL_GOALS, 2):
        out.append((text, int(pred(s))))
    while len(out) < k:
        r = rng.random()
        if r < 0.35 and len(tab) >= 2:
            c, b = rng.sample(tab, 2); d = rng.choice(["left", "right"])
            out.append((side_text(rng, c, b, d), int(side_of(s, c, b, d))))
        elif r < 0.55 and len(tab) >= 2:
            c, b = rng.sample(tab, 2)
            out.append((stack_text(rng, c, b), int(on(s, c, b))))
        else:
            t, y = bin_goal(rng, s)
            out.append((t, int(y)))
    rng.shuffle(out)
    return out[:k] if len(out) > k else out
