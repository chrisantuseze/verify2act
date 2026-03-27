"""Sub-skill plan decomposition for inference.

Option A (primary path): VLM outputs NUT NAMES in assembly order.
``expand_nut_plan`` expands each nut name into its 5 ordered sub-skill prompts.

    VLM plan:  ["left round nut", "right round nut"]
                             ↓  expand_nut_plan()
    Imagination steps:
        ("left round nut",  "approach left round nut from above")
        ("left round nut",  "grasp left round nut and lift")
        ("left round nut",  "carry left round nut toward peg")
        ("left round nut",  "align left round nut over peg")
        ("left round nut",  "lower left round nut onto peg")
        ("right round nut", "approach right round nut from above")
        ... (5 more)

The tuple ``(nut_name, sub_skill_prompt)`` keeps the originating nut name
available so that:
  - Real execution calls ``env_wrapper.execute_nut_assembly(nut_name)``.
  - Reflection is framed in terms of nuts (VLM level), not sub-skills.

Legacy helper ``expand_plan`` (for "pick/insert" style HL plans) is
retained for ablations.

Template strings are kept identical to those in
``robosuite/data_capture_wm/prompt_utils.py::build_subskill_action_prompt``
so inference-time prompts stay in-distribution.
"""

from __future__ import annotations

from typing import List, Tuple

# ── Sub-skill templates — must match prompt_utils.build_subskill_action_prompt ─

_PICK_SUBSKILLS: List[str] = [
    "approach {obj} from above",
    "grasp {obj} and lift",
    "carry {obj} toward peg",
]

_INSERT_SUBSKILLS: List[str] = [
    "align {obj} over peg",
    "lower {obj} onto peg",
]

# All 5 ordered sub-skills for a complete nut assembly (pick + insert).
_ALL_SUBSKILLS: List[str] = _PICK_SUBSKILLS + _INSERT_SUBSKILLS

# Map the first token of a legacy VLM action to its ordered sub-skill templates.
_SKILL_TO_SUBSKILLS = {
    "pick":     _PICK_SUBSKILLS,
    "insert":   _INSERT_SUBSKILLS,
    "place":    _INSERT_SUBSKILLS,
    "put_down": _INSERT_SUBSKILLS,
}


def expand_nut_plan(nut_names: List[str]) -> List[Tuple[str, str]]:
    """Expand a nut-ordering plan into ``(nut_name, sub_skill_prompt)`` pairs.

    This is the **primary** decomposition for Option A inference, where the
    VLM outputs only nut names in assembly order.

    Parameters
    ----------
    nut_names : List[str]
        Nut labels as returned by the VLM, e.g.
        ``["left round nut", "right round nut"]``.

    Returns
    -------
    List[Tuple[str, str]]
        Five ``(nut_name, sub_skill_prompt)`` pairs per nut.
        Sub-skill prompts go to the world model; nut names are retained for
        real execution (``env_wrapper.execute_nut_assembly``) and reflection.
    """
    result: List[Tuple[str, str]] = []
    for nut in nut_names:
        if nut.strip().lower() == "done":
            continue
        for template in _ALL_SUBSKILLS:
            result.append((nut, template.format(obj=nut)))
    return result


def decompose_action(action_text: str) -> List[str]:
    """Expand one high-level VLM action into ordered sub-skill strings.

    Parameters
    ----------
    action_text : str
        e.g. ``"pick left round nut"``

    Returns
    -------
    List[str]
        e.g. ``["approach left round nut from above",
                "grasp left round nut and lift",
                "carry left round nut toward peg"]``

    Notes
    -----
    Unknown skills (including ``"done"``) are returned as a single-element
    list containing the original string unchanged.
    """
    parts = action_text.strip().lower().split(None, 1)
    if not parts:
        return [action_text]

    skill = parts[0]
    obj = parts[1] if len(parts) > 1 else ""

    templates = _SKILL_TO_SUBSKILLS.get(skill)
    if templates is None:
        return [action_text]

    return [t.format(obj=obj) for t in templates]


def expand_plan(plan: List[str]) -> List[Tuple[str, str]]:
    """Expand a VLM-level plan into ``(orig_action, subskill_prompt)`` pairs.

    Parameters
    ----------
    plan : List[str]
        VLM plan, e.g. ``["pick left round nut", "insert left round nut"]``

    Returns
    -------
    List[Tuple[str, str]]
        Each element is ``(original_vl_action, sub_skill_prompt)``.
        The sub-skill prompts go to the world model; the original VLM actions
        are retained for real execution and VLM reflection.
    """
    result: List[Tuple[str, str]] = []
    for action in plan:
        for sub in decompose_action(action):
            result.append((action, sub))
    return result
