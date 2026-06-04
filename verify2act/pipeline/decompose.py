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


def expand_nut_plan(nut_names: List[str | dict]) -> List[Tuple[str | dict, str]]:
    """Expand a nut-ordering plan into ``(nut_name, action_prompt)`` pairs.

    This is the **primary** decomposition for Option A inference, where the
    VLM outputs only nut names in assembly order.

    Parameters
    ----------
    nut_names : List[str | dict]
        Nut labels as returned by the VLM, e.g.
        ``["left round nut", "right round nut"]`` or paired dicts.

    Returns
    -------
    List[Tuple[str | dict, str]]
        Two ``(nut_name, action_prompt)`` pairs (pick and insert) per nut.
        Action prompts go to the world model; nut names are retained for
        real execution (``env_wrapper.execute_nut_assembly``) and reflection.
    """
    result: List[Tuple[str | dict, str]] = []
    for nut in nut_names:
        if isinstance(nut, dict):
            label_str = nut.get("label", "")
            if label_str.strip().lower() == "done":
                continue
            orig_nut = nut
        else:
            if nut.strip().lower() == "done":
                continue
            label_str = nut
            orig_nut = nut
        result.append((orig_nut, f"pick {label_str}"))
        result.append((orig_nut, f"insert {label_str}"))
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
        e.g. ``["pick left round nut"]``
    """
    return [action_text]


def expand_plan(plan: List[str]) -> List[Tuple[str, str]]:
    """Expand a VLM-level plan into ``(orig_action, action_prompt)`` pairs.

    Parameters
    ----------
    plan : List[str]
        VLM plan, e.g. ``["pick left round nut", "insert left round nut"]``

    Returns
    -------
    List[Tuple[str, str]]
        Each element is ``(original_vl_action, action_prompt)``.
    """
    result: List[Tuple[str, str]] = []
    for action in plan:
        result.append((action, action))
    return result
