"""
Action Prompt Template Builder for World Model Data Collection

Generates fixed-format text prompts for InstructPix2Pix conditioning.

Action text is intentionally SEMANTIC ONLY — no coordinates.
InstructPix2Pix was pretrained on natural-language image editing instructions
and conditions on visual descriptions of change, not metric positions.
Coordinates belong in action_params.cartesian_target for the robot controller,
not in the diffusion prompt.
"""


def build_action_prompt(skill: str, object_name: str, cartesian_target=None) -> str:
    """
    Build a semantic action prompt string for InstructPix2Pix conditioning.

    Args:
        skill: One of 'pick', 'place', 'insert'.
        object_name: e.g. 'round nut', 'square nut'.
        cartesian_target: Accepted but ignored. Coordinates are stored separately
            in action_params and are consumed by the robot controller, not the
            diffusion world model.

    Returns:
        Semantic prompt string, e.g. "pick round nut", "insert square nut".
    """
    skill = skill.lower().strip()
    assert skill in ("pick", "place", "insert"), f"Unknown skill: {skill}"
    return f"{skill} {object_name}"


def build_subskill_action_prompt(
    sub_skill: str,
    object_name: str,
    cartesian_target=None,
) -> str:
    """
    Build an enriched sub-skill action prompt for finer-grained transitions.

    Each sub-skill describes a specific phase of manipulation with a natural-
    language template that gives the diffusion model more discriminative
    conditioning signal than the coarse skill-level prompts.

    Args:
        sub_skill: One of 'approach', 'grasp', 'carry', 'align', 'lower_insert',
                   or falls back to the coarse skill name.
        object_name: e.g. 'left round nut', 'right square nut'.
        cartesian_target: Optional (x, y, z). If provided, a spatial description
                          is appended (e.g. "on the left near the back").

    Returns:
        Enriched prompt string,
        e.g. "approach left round nut from above on the left".
    """
    sub_skill = sub_skill.lower().strip()

    _TEMPLATES = {
        "approach":     "approach {obj} from above",
        "grasp":        "grasp {obj} and lift",
        "carry":        "carry {obj} toward peg",
        "align":        "align {obj} over peg",
        "lower_insert": "lower {obj} onto peg",
        # Coarse skill fallbacks
        "pick":         "pick {obj}",
        "insert":       "insert {obj}",
        "place":        "place {obj}",
    }

    template = _TEMPLATES.get(sub_skill, "{skill} {obj}")
    prompt = template.format(obj=object_name, skill=sub_skill)

    # Optionally append a spatial description from coordinates
    if cartesian_target is not None:
        try:
            x, y = float(cartesian_target[0]), float(cartesian_target[1])
            parts = []
            if x < -0.05:
                parts.append("on the left")
            elif x > 0.05:
                parts.append("on the right")
            if y < -0.05:
                parts.append("near the front")
            elif y > 0.05:
                parts.append("near the back")
            if parts:
                prompt += " " + " ".join(parts)
        except (IndexError, TypeError, ValueError):
            pass

    return prompt


def spatial_qualifier(target_x: float, target_y: float, all_positions: list) -> str:
    """
    Returns a spatial label for an object at (target_x, target_y) relative to all
    same-type objects in the scene.

    Axis convention (robosuite world frame, agentview camera):
      x-axis: robot's left (more negative) → robot's right (more positive)
      y-axis: front/near robot (more negative) → back/far from robot (more positive)

    Args:
        target_x: x world-frame coordinate of the target object.
        target_y: y world-frame coordinate of the target object.
        all_positions: list of (x, y) tuples for ALL same-type objects
            (including the target itself).

    Returns:
        "" if there is only one object.
        A label like "left", "right", "front-left", "back-right", "front-center", etc.

    Strategy:
        1. Assign an x-label (left / center / right) by bucketing each object's
           x rank into 3 equal groups (or left/right for exactly 2 distinct x values).
        2. If the x-label alone uniquely identifies the target among its siblings,
           return it.
        3. Otherwise, also assign a y-label (front / middle / back) by the same
           bucketing rule and return the compound label, e.g. "front-left".
        4. If all objects share the same x (vertical column), return the y-label only.
    """
    n = len(all_positions)
    if n <= 1:
        return ""

    def _bin(val: float, all_vals: list, labels_2: tuple, labels_3: tuple) -> str:
        """
        Bucket val into a label using distinct sorted values.
          2 distinct values → labels_2  (e.g. ("left", "right"))
          3+ distinct values → labels_3 tertiles  (e.g. ("left", "center", "right"))
        """
        distinct = sorted(set(round(v, 4) for v in all_vals))
        nd = len(distinct)
        if nd <= 1:
            return ""
        # find nearest distinct bucket
        idx = min(range(nd), key=lambda i: abs(distinct[i] - round(val, 4)))
        if nd == 2:
            return labels_2[idx]
        # 3+ distinct values → map index linearly to 3 buckets
        bucket = min(int(idx * 3 / nd), 2)
        return labels_3[bucket]

    all_x = [p[0] for p in all_positions]
    all_y = [p[1] for p in all_positions]

    x_label = _bin(target_x, all_x, ("left", "right"), ("left", "center", "right"))
    y_label = _bin(target_y, all_y, ("front", "back"), ("front", "middle", "back"))

    # Check whether x_label alone uniquely identifies the target.
    # Count siblings that fall into the same x-bucket.
    same_x_count = sum(
        1 for p in all_positions
        if _bin(p[0], all_x, ("left", "right"), ("left", "center", "right")) == x_label
    )

    if same_x_count == 1 and x_label:
        return x_label  # x is sufficient

    # x is ambiguous (or absent) — combine with y
    parts = [p for p in [y_label, x_label] if p]
    return "-".join(parts) if parts else ""


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
