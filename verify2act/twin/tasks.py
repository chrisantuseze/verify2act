"""Language goals for twin episodes: the eval families (EVAL_TASKS.md) plus paraphrases, each with a predicate on the
twin and an oracle plan in the robot subtask vocabulary.

  1 bin clearing   explicit colours, warm/cool groups, "leave"/"except" exclusions, all blocks
  2 rearrangement  <c> to the left/right of <b>
  3 stacking       <c> on <b>
  4 compound       two of the above, in order
  eval             the real-robot eval tasks verbatim (task1a–1d, 2a/2b, 3a; dofbot-controller EVAL_TASKS.md)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from verify2act.twin.scene import DofbotTwin

WARM = ("red", "yellow")
COOL = ("green", "blue")


@dataclass
class Goal:
    text: str
    family: str
    check: Callable[[DofbotTwin], bool]
    plan: Callable[[DofbotTwin, np.random.Generator], Optional[List[str]]]
    parts: List["Goal"] = field(default_factory=list)
    moves: frozenset = frozenset()              # blocks the goal moves
    task: str = ""                              # eval task id (task1a ...) for the eval family


def _bin(c: str) -> str:
    return f"pick and place {c} block into the bin"


def _list_blocks(cs: List[str]) -> str:
    names = [f"the {c} block" for c in cs]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _bin_plan(targets: List[str]) -> Callable[[DofbotTwin, np.random.Generator], Optional[List[str]]]:
    """Targets still on the table, in a random order that removes stacked blocks top-down. None when a block that must
    stay sits on a target (the plan would have to move it)."""
    def plan(t: DofbotTwin, rng: np.random.Generator) -> Optional[List[str]]:
        left = [c for c in targets if t.present[c]]
        above = {c: [o for o in t.colors if t.on(o, c)] for c in left}
        if any(o not in targets for c in left for o in above[c]):
            return None
        order: List[str] = []
        remaining = set(left)
        while remaining:
            free = [c for c in sorted(remaining) if not any(o in remaining for o in above[c])]
            if not free:
                return None
            c = free[int(rng.integers(len(free)))]
            order.append(c)
            remaining.remove(c)
        return [_bin(c) for c in order]
    return plan


def bin_goal(t: DofbotTwin, rng: np.random.Generator) -> Optional[Goal]:
    present = [c for c in t.colors if t.present[c]]
    if not present:
        return None
    style = rng.choice(["explicit", "explicit", "group", "except", "all"])
    keep_phrase = ""
    if style == "group":
        group_name, group = (("warm", WARM), ("cool", COOL))[int(rng.integers(2))]
        targets = [c for c in present if c in group]
        keep = [c for c in present if c not in group]
        if not targets:
            return None
        if keep and rng.random() < 0.7:
            keep_phrase = f" and {rng.choice(['leave', 'keep'])} {_list_blocks(list(rng.permutation(keep)[:1]))}"
        verb = rng.choice(["Clear all", "Put all", "Move all", "Remove all"])
        text = f"{verb} {group_name}-colored blocks into the bin{keep_phrase}"
    elif style == "except":
        if len(present) < 2:
            return None
        spared = str(rng.choice(present))
        targets = [c for c in present if c != spared]
        text = str(rng.choice([
            f"Put {_list_blocks(targets)} into the bin except the {spared} block",
            f"Put every block into the bin except the {spared} block",
            f"Clear the table except the {spared} block",
            f"Clear all blocks into the bin but leave the {spared} block",
        ]))
    elif style == "all":
        targets = present
        text = str(rng.choice(["Put all the blocks into the bin", "Clear the table", "Clear every block into the bin"]))
    else:
        k = int(rng.integers(1, min(3, len(present)) + 1))
        targets = [str(c) for c in rng.permutation(present)[:k]]
        verb = rng.choice(["Put", "Move", "Place", "Drop"])
        text = f"{verb} {_list_blocks(targets)} into the bin"
        others = [c for c in present if c not in targets]
        if others and rng.random() < 0.25:
            text += f" and leave {_list_blocks(others[:1])}"
    targets = [str(c) for c in targets]
    return _make_bin(text, targets, [c for c in present if c not in targets])


def _make_bin(text: str, targets: List[str], keep: List[str], family: str = "bin", task: str = "") -> Goal:
    def check(tw: DofbotTwin) -> bool:
        return all(not tw.present[c] for c in targets) and all(tw.present[c] for c in keep)

    return Goal(text, family, check, _bin_plan(targets), moves=frozenset(targets), task=task)


def side_goal(t: DofbotTwin, rng: np.random.Generator) -> Optional[Goal]:
    present = [c for c in t.colors if t.present[c]]
    if len(present) < 2:
        return None
    # Like the eval setup (>= 10 cm free on the placement side), pick a combination that has room.
    combos = [(str(c), str(b), s) for c in rng.permutation(present) for b in rng.permutation(present)
              for s in rng.permutation(["left", "right"]) if c != b and t.clear(str(c))]
    for c, b, side in combos:
        if not t.side_of(c, b, side) and t.find_side_spot(c, b, side, rng, max_tries=25) is not None:
            break
    else:
        return None
    verb = rng.choice(["Put", "Place", "Move"])
    return _make_side(f"{verb} the {c} block to the {side} of the {b} block", c, b, side)


def _make_side(text: str, c: str, b: str, side: str, family: str = "side", task: str = "") -> Goal:
    def check(tw: DofbotTwin) -> bool:
        return tw.side_of(c, b, side)

    def plan(tw: DofbotTwin, _rng: np.random.Generator) -> Optional[List[str]]:
        if check(tw) or not tw.clear(c) or not tw.present[b]:
            return None
        return [f"pick and place {c} block to the {side} of {b} block"]

    return Goal(text, family, check, plan, moves=frozenset([c]), task=task)


def stack_goal(t: DofbotTwin, rng: np.random.Generator) -> Optional[Goal]:
    present = [c for c in t.colors if t.present[c]]
    if len(present) < 2:
        return None
    c, b = (str(x) for x in rng.permutation(present)[:2])
    text = str(rng.choice([f"Stack the {c} block on top of the {b} block", f"Put the {c} block on the {b} block",
                           f"Place the {c} block on top of the {b} block"]))
    return _make_stack(text, c, b)


def _make_stack(text: str, c: str, b: str, family: str = "stack", task: str = "") -> Goal:
    def check(tw: DofbotTwin) -> bool:
        return tw.on(c, b)

    def plan(tw: DofbotTwin, _rng: np.random.Generator) -> Optional[List[str]]:
        if check(tw) or not tw.clear(c) or not tw.clear(b):
            return None
        return [f"pick and place {c} block on {b} block"]

    return Goal(text, family, check, plan, moves=frozenset([c]), task=task)


def compound_goal(t: DofbotTwin, rng: np.random.Generator) -> Optional[Goal]:
    """Two goals in order. The second one's plan is computed after the first has been executed (see ``run_goal``)."""
    makers = [bin_goal, side_goal, stack_goal]
    g1 = makers[int(rng.integers(3))](t, rng)
    g2 = makers[int(rng.integers(3))](t, rng)
    if g1 is None or g2 is None or g1.moves & g2.moves:        # two different instructions, not a paraphrase
        return None
    text = f"{g1.text}, then {g2.text[0].lower()}{g2.text[1:]}"

    def check(tw: DofbotTwin) -> bool:
        return g1.check(tw) and g2.check(tw)

    return Goal(text, "compound", check, g1.plan, parts=[g1, g2], moves=g1.moves | g2.moves)


# The real-robot evaluation tasks, verbatim (dofbot-controller verify2act/EVAL_TASKS.md). The eval scenes have all four
# blocks on the sheet; family 2 leaves room on the placement side.
EVAL_TASKS = {
    "task1a": lambda: _make_bin("Put the blue block and the yellow block into the bin",
                                ["blue", "yellow"], ["red", "green"], "eval", "task1a"),
    "task1b": lambda: _make_bin("Clear all cool-colored blocks into the bin and leave the yellow block",
                                ["green", "blue"], ["red", "yellow"], "eval", "task1b"),
    "task1c": lambda: _make_bin("Clear all warm-colored blocks into the bin and leave the green block",
                                ["red", "yellow"], ["green", "blue"], "eval", "task1c"),
    "task1d": lambda: _make_bin("Put the red block, the green block and the blue block into the bin except the "
                                "yellow block", ["red", "green", "blue"], ["yellow"], "eval", "task1d"),
    "task2a": lambda: _make_side("Put the red block to the left of the blue block", "red", "blue", "left",
                                 "eval", "task2a"),
    "task2b": lambda: _make_side("Put the green block to the right of the yellow block", "green", "yellow", "right",
                                 "eval", "task2b"),
    "task3a": lambda: _make_stack("Stack the blue block on top of the yellow block", "blue", "yellow",
                                  "eval", "task3a"),
}


def eval_goal(t: DofbotTwin, rng: np.random.Generator, task: Optional[str] = None) -> Optional[Goal]:
    """One of the eval tasks (random unless ``task``); None when the scene cannot host it (e.g. no room)."""
    task = task or str(rng.choice(sorted(EVAL_TASKS)))
    g = EVAL_TASKS[task]()
    if not all(t.present[c] for c in g.moves) or g.check(t):     # eval scenes start with the goal not yet met
        return None
    side_args = {"task2a": ("red", "blue", "left"), "task2b": ("green", "yellow", "right")}.get(task)
    if side_args:
        c, b, side = side_args
        if not t.clear(c) or t.find_side_spot(c, b, side, rng, max_tries=25) is None:
            return None
    return g


FAMILY_WEIGHTS = {"eval": 0.3, "bin": 0.3, "side": 0.17, "stack": 0.15, "compound": 0.08}
_MAKERS = {"eval": eval_goal, "bin": bin_goal, "side": side_goal, "stack": stack_goal, "compound": compound_goal}


def sample_goal(t: DofbotTwin, rng: np.random.Generator, family: Optional[str] = None,
                max_tries: int = 30, weights: Optional[dict] = None) -> Optional[Goal]:
    """A goal whose oracle plan exists and is non-empty in the current scene."""
    weights = weights or FAMILY_WEIGHTS
    names = list(weights)
    p = np.array([weights[n] for n in names], float)
    for _ in range(max_tries):
        fam = family or str(rng.choice(names, p=p / p.sum()))
        g = _MAKERS[fam](t, rng)
        if g is None or g.check(t):
            continue
        steps = g.plan(t, rng)
        if steps:
            return g
    return None


def run_goal(t: DofbotTwin, g: Goal, rng: np.random.Generator,
             on_step: Optional[Callable[[str], None]] = None) -> List[str]:
    """Execute the oracle plan (each part re-planned from the current state); returns the executed subtasks.
    ``on_step(subtask)`` is called after every subtask. Raises ``InvalidSubtask`` / ``RuntimeError`` on failure."""
    executed: List[str] = []
    for part in (g.parts or [g]):
        if part.check(t):
            continue
        steps = part.plan(t, rng)
        if not steps:
            raise RuntimeError(f"no plan for {part.text!r}")
        for s in steps:
            t.apply(s, rng)
            executed.append(s)
            if on_step:
                on_step(s)
    if not g.check(t):
        raise RuntimeError(f"goal not reached: {g.text!r}")
    return executed
