"""
Action Prompt Template Builder for World Model Data Collection

Generates fixed-format text prompts for InstructPix2Pix conditioning.
"""


def build_action_prompt(skill: str, object_name: str, cartesian_target) -> str:
    """
    Build a standardised action prompt string.

    Args:
        skill: One of 'pick', 'place', 'insert'.
        object_name: e.g. 'round nut', 'square nut'.
        cartesian_target: 3-element array/list [x, y, z] in world frame (metres).

    Returns:
        Prompt string, e.g. "pick round nut. position: (0.12, -0.05, 0.92)."
    """
    skill = skill.lower().strip()
    assert skill in ("pick", "place", "insert"), f"Unknown skill: {skill}"
    x, y, z = float(cartesian_target[0]), float(cartesian_target[1]), float(cartesian_target[2])
    return f"{skill} {object_name}. position: ({x:.4f}, {y:.4f}, {z:.4f})."


def skill_from_gripper(gripper_action: float, prev_skill: str | None) -> str:
    """
    Map low-level gripper action to high-level skill name.

    Robosuite convention: gripper_action > 0 → close (pick), < 0 → open (place/insert).
    We treat the first open after a close as 'place'; override to 'insert' externally
    when the target is a peg.

    Args:
        gripper_action: Scalar gripper command.
        prev_skill: Previous skill string or None.

    Returns:
        One of 'pick', 'place'.
    """
    if gripper_action > 0:
        return "pick"
    else:
        return "place"
