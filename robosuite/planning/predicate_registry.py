"""
Centralized Predicate Registry

This module defines all predicates used across the planning system,
ensuring consistency between:
- LLM prompts (YAML configs)
- Dynamics model (trained on these indices)
- State converter and goal checking
- Action primitives

IMPORTANT: The indices MUST match the training data format from Points2Plans
Order from get_predicates() in dataloader.py:
Index 0: Left, 1: Right, 2: Below, 3: Above, 4: Front, 5: Behind, 
      6: Contact/On, 7: Boundary, 8: Inside
"""

from typing import Dict, List, Tuple
from enum import IntEnum


class PredicateType(IntEnum):
    """Predicate types matching training data indices."""
    LEFT = 0
    RIGHT = 1
    BELOW = 2
    ABOVE = 3
    FRONT = 4
    BEHIND = 5
    ON = 6          # Contact/On - used for stacking, placing, assembly
    BOUNDARY = 7    # Boundary relationship (edge of workspace, etc.)
    INSIDE = 8      # Inside relationship (object in container)


# Predicate names in order (for display/parsing)
PREDICATE_NAMES = [
    'Left',      # 0
    'Right',     # 1
    'Below',     # 2
    'Above',     # 3
    'Front',     # 4
    'Behind',    # 5
    'On',        # 6 - Contact/On (also covers Stacked, OnPeg, etc.)
    'Boundary',  # 7
    'Inside',    # 8
]

# Alternative names that map to the same predicate
PREDICATE_ALIASES = {
    # Spatial relations
    'leftof': PredicateType.LEFT,
    'rightof': PredicateType.RIGHT,
    'under': PredicateType.BELOW,
    'over': PredicateType.ABOVE,
    'infront': PredicateType.FRONT,
    'infrontof': PredicateType.FRONT,
    
    # Contact relations (all map to ON)
    'on': PredicateType.ON,
    'contact': PredicateType.ON,
    'touching': PredicateType.ON,
    'stacked': PredicateType.ON,     # Stacked is just On with vertical alignment
    'onpeg': PredicateType.ON,       # OnPeg is just On(nut, peg)
    'assembled': PredicateType.ON,   # Assembly is contact relationship
    
    # Container relations
    'in': PredicateType.INSIDE,
    'within': PredicateType.INSIDE,
}


def get_predicate_index(predicate_name: str) -> int:
    """
    Get predicate index from name (case-insensitive).
    
    Args:
        predicate_name: Predicate name (e.g., "On", "Stacked", "OnPeg")
    
    Returns:
        Predicate index (0-8)
    
    Raises:
        ValueError: If predicate name is unknown
    """
    pred_lower = predicate_name.lower().replace('_', '').replace('-', '')
    
    # Check aliases first (includes all variations)
    if pred_lower in PREDICATE_ALIASES:
        return int(PREDICATE_ALIASES[pred_lower])
    
    # Check canonical names
    for idx, name in enumerate(PREDICATE_NAMES):
        if name.lower() == pred_lower:
            return idx
    
    raise ValueError(f"Unknown predicate: {predicate_name}")


def get_predicate_name(predicate_index: int) -> str:
    """
    Get canonical predicate name from index.
    
    Args:
        predicate_index: Predicate index (0-8)
    
    Returns:
        Canonical predicate name
    """
    if 0 <= predicate_index < len(PREDICATE_NAMES):
        return PREDICATE_NAMES[predicate_index]
    raise ValueError(f"Invalid predicate index: {predicate_index}")


def get_all_predicates() -> List[str]:
    """Get list of all canonical predicate names."""
    return PREDICATE_NAMES.copy()


def get_predicate_description(predicate_name: str) -> str:
    """
    Get human-readable description of a predicate.
    
    Args:
        predicate_name: Predicate name
    
    Returns:
        Description string
    """
    descriptions = {
        'Left': 'Object a is to the left of object b',
        'Right': 'Object a is to the right of object b',
        'Below': 'Object a is below object b',
        'Above': 'Object a is above object b',
        'Front': 'Object a is in front of object b',
        'Behind': 'Object a is behind object b',
        'On': 'Object a is on/touching object b (includes stacking, assembly, contact)',
        'Boundary': 'Object a is at the boundary relative to b',
        'Inside': 'Object a is inside object b (container relationship)',
    }
    
    canonical_name = get_predicate_name(get_predicate_index(predicate_name))
    return descriptions.get(canonical_name, 'Unknown predicate')


def get_llm_predicate_definitions() -> str:
    """
    Get predicate definitions formatted for LLM system prompts.
    
    Returns:
        Multi-line string with predicate definitions
    """
    definitions = [
        "The robot can detect the following relationships among objects:\n",
        "  - On(a, b): Object a is on object b (contact relationship)",
        "  - Inside(a, b): Object a is inside object b (container relationship)",
        "  - Above(a, b): Object a is above object b (spatial relationship)",
        "  - Below(a, b): Object a is below object b (spatial relationship)",
        "  - Left(a, b): Object a is to the left of object b",
        "  - Right(a, b): Object a is to the right of object b",
        "  - Front(a, b): Object a is in front of object b",
        "  - Behind(a, b): Object a is behind object b",
        "  - Grasped(a): Object a is currently held by the robot (special state)\n",
        "\nNote: Stacked(a, b) is equivalent to On(a, b) with vertical alignment.",
        "Assembly tasks use On(nut, peg) to represent successful assembly.",
    ]
    return '\n'.join(definitions)


# Action primitives (for reference, not strictly predicates)
ACTION_PRIMITIVES = [
    'Pick',          # Pick(object, location)
    'Place',         # Place(object, target) - covers PlaceOnPeg, PlaceOnTable, etc.
    'Open',          # Open(object)
    'Close',         # Close(object)
]


def normalize_action(action_name: str) -> str:
    """
    Normalize action names to canonical form.
    
    Args:
        action_name: Action name (e.g., "PlaceOnPeg", "PlaceOnTable")
    
    Returns:
        Canonical action name (e.g., "Place")
    """
    action_lower = action_name.lower()
    
    # Normalize variations of Place
    if 'place' in action_lower:
        return 'Place'
    
    # Keep other actions as-is (capitalized)
    return action_name.capitalize()


def get_action_description(action_name: str) -> str:
    """
    Get human-readable description of an action.
    
    Args:
        action_name: Action name
    
    Returns:
        Description string
    """
    descriptions = {
        'Pick': 'Pick object from location',
        'Place': 'Place object on target (table, peg, other object, etc.)',
        'Open': 'Open object (door, drawer, etc.)',
        'Close': 'Close object (door, drawer, etc.)',
    }
    
    canonical = normalize_action(action_name)
    return descriptions.get(canonical, f'Execute {action_name} action')


def get_task_relevant_predicates(task_type: str) -> List[int]:
    """
    Get list of relevant predicate indices for a specific task type.
    This filters out noisy/irrelevant spatial relationships.
    
    Args:
        task_type: Task type ("assembly", "stacking", "pickplace", "door", "all")
    
    Returns:
        List of relevant predicate indices
    """
    task_type_lower = task_type.lower()
    
    if task_type_lower in ['assembly', 'nut_assembly', 'nutassembly', 'clutterednutassembly']:
        # Assembly tasks: Only care about contact (On) and vertical relationships
        # On(nut, peg), On(nut, table), On(nut, nut) for stacking
        return [PredicateType.ON, PredicateType.ABOVE, PredicateType.BELOW]
    
    elif task_type_lower in ['stacking', 'stack']:
        # Stacking tasks: Contact and vertical relationships
        return [PredicateType.ON, PredicateType.ABOVE, PredicateType.BELOW]
    
    elif task_type_lower in ['pickplace', 'pick_place']:
        # Pick and place: Contact and container relationships
        return [PredicateType.ON, PredicateType.INSIDE]
    
    elif task_type_lower in ['door']:
        # Door tasks: Just contact relationship (handle touching door)
        return [PredicateType.ON]
    
    else:
        # Default: all predicates
        return list(range(len(PREDICATE_NAMES)))


def should_include_predicate(
    predicate_idx: int,
    obj1_name: str,
    obj2_name: str,
    task_type: str = "all"
) -> bool:
    """
    Determine if a predicate should be included in LLM prompt.
    
    Filters out:
    - Irrelevant predicate types for the task
    - Static object relationships (peg-to-peg, peg-to-table)
    - Redundant relationships
    
    Args:
        predicate_idx: Predicate index (0-8)
        obj1_name: First object name
        obj2_name: Second object name
        task_type: Task type for filtering
    
    Returns:
        True if predicate should be included
    """
    # Get task-relevant predicates
    relevant_predicates = get_task_relevant_predicates(task_type)
    
    if predicate_idx not in relevant_predicates:
        return False
    
    # Filter out static object relationships
    obj1_lower = obj1_name.lower()
    obj2_lower = obj2_name.lower()
    
    # Skip peg-to-peg relationships (static)
    if 'peg' in obj1_lower and 'peg' in obj2_lower:
        return False
    
    # Skip table-to-table (if multiple tables)
    if 'table' in obj1_lower and 'table' in obj2_lower:
        return False
    
    # For spatial predicates (Left, Right, etc), skip if either object is a peg
    # Pegs are static, so their spatial relationships don't change
    if predicate_idx in [PredicateType.LEFT, PredicateType.RIGHT, 
                         PredicateType.FRONT, PredicateType.BEHIND]:
        if 'peg' in obj1_lower or 'peg' in obj2_lower:
            return False
        if 'table' in obj1_lower or 'table' in obj2_lower:
            return False
    
    return True


if __name__ == "__main__":
    # Test the registry
    print("Predicate Registry")
    print("=" * 80)
    print("\nCanonical Predicates:")
    for idx, name in enumerate(PREDICATE_NAMES):
        print(f"  {idx}: {name} - {get_predicate_description(name)}")
    
    print("\nAlias Examples:")
    test_aliases = ['Stacked', 'OnPeg', 'Assembled', 'In', 'Under']
    for alias in test_aliases:
        idx = get_predicate_index(alias)
        canonical = get_predicate_name(idx)
        print(f"  {alias} -> {canonical} (index {idx})")
    
    print("\nLLM Prompt Format:")
    print(get_llm_predicate_definitions())
    
    print("\nAction Normalization:")
    test_actions = ['PlaceOnPeg', 'PlaceOnTable', 'Pick', 'place']
    for action in test_actions:
        normalized = normalize_action(action)
        print(f"  {action} -> {normalized}")
