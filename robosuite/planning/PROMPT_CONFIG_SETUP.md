# Prompt Configuration Setup for Robosuite LLM Task Planner

## Summary

Successfully created prompt configuration files for the LLM Task Planner to work with robosuite stacking tasks. The configuration follows Points2Plans' YAML-based prompt structure but adapted for robosuite manipulation tasks.

## What Was Created

### 1. Configuration Directory Structure

```
robosuite/planning/configs/prompts/
├── system/
│   ├── goal_prediction_v1.yaml
│   └── task_planning_goal_condition_v1.yaml
├── examples/
│   ├── example_1.yaml
│   ├── example_2.yaml
│   └── example_3.yaml
└── tasks/
    └── stack_task.yaml
```

### 2. System Prompts

**goal_prediction_v1.yaml**: Defines how the LLM should predict symbolic goals from task instructions
- Adapted for tabletop manipulation (vs. household environment)
- Supports predicates: On, Stacked, Grasped
- Removed household-specific predicates (Opened, Closed, Inside)

**task_planning_goal_condition_v1.yaml**: Defines how the LLM should generate task plans
- Supports actions: Pick(obj, location), Place(obj, location)
- Removed household-specific actions (Open, Close)
- Maintains action preconditions for manipulator

### 3. Example Prompts

Three few-shot examples for the LLM:
1. **example_1.yaml**: Simple single-object placement
2. **example_2.yaml**: Basic two-object stacking
3. **example_3.yaml**: Multi-object stacking (similar to Stack3 task)

### 4. Task Config

**stack_task.yaml**: Main task configuration template
- Can be dynamically updated with actual task details
- References system and example prompts using absolute paths
- Includes default objects and predicates for stack tasks

## Code Changes

### llm_task_planner.py

1. **Added `prompt_config_path` parameter** to `__init__`:
   ```python
   def __init__(
       self,
       model_config_path: Optional[str] = None,
       prompt_config_path: Optional[str] = None,  # NEW
       api_key: Optional[str] = None,
       use_examples: bool = True  # Changed default to True
   ):
   ```

2. **Updated `generate_goals_and_plans` to use prompt config**:
   - Loads BehaviorPromptManager from YAML
   - Dynamically updates task_prompt with actual task details
   - Properly parses LLM responses (handles "Goals:" and "Plans:" prefixes)
   - Returns first goal/plan set

3. **Fixed `goals_to_predicates` method**:
   - Added `object_name_to_id` parameter
   - Properly maps object names to indices
   - Added "Stacked" predicate (maps to "On" index)

### demo_phase2.py

1. **Fixed goal/plan indexing**:
   - Changed `goals[0]` to `goals` (already extracted in planner)
   - Changed `plans[0]` to `plans` (pass full plan list)

## Usage

The prompt config is automatically loaded by default:

```python
llm_planner = LLMTaskPlanner()  # Uses default stack_task.yaml

# Or specify custom config:
llm_planner = LLMTaskPlanner(
    prompt_config_path="/path/to/custom_task.yaml"
)
```

The task details are dynamically updated at runtime:

```python
goals, plans = llm_planner.generate_goals_and_plans(
    task_description="Stack all objects on top of each other",
    objects=["cubeA", "cubeB", "cubeC", "table"],
    initial_predicates=["On(cubeA, table)", "On(cubeB, table)", "On(cubeC, table)"]
)
```

## Test Results

Successfully tested with Stack3 task:
- LLM generates multiple goal hypotheses
- LLM generates corresponding plans for each goal
- Selects first goal/plan pair
- Properly converts to predicate tensors
- Integrates with dynamics model planner

Example output:
```
Predicted goals: [
    ['Stacked(cubeA, cubeB)', 'Stacked(cubeB, cubeC)'],
    ['Stacked(cubeA, cubeC)', 'Stacked(cubeC, cubeB)'],
    ...
]

Predicted plans: [
    ['Pick(cubeA, table)', 'Place(cubeA, cubeB)', 'Pick(cubeB, table)', 'Place(cubeB, cubeC)'],
    ...
]
```

## Creating New Task Configs

To create a config for a new task:

1. Copy `stack_task.yaml` to `tasks/your_task.yaml`
2. Update the task-specific fields:
   ```yaml
   task: your_task_name
   instruction: "Your task description"
   objects: ["obj1", "obj2", ...]
   predicates: ["On(obj1, table)", ...]
   ```
3. Keep the system_prompts and example_prompts unchanged (or customize if needed)
4. Pass the new config path to LLMTaskPlanner

## Key Design Decisions

1. **Absolute paths**: Used absolute paths for system/example prompts to avoid working directory issues
2. **Predicate mapping**: "Stacked" maps to "On" predicate (index 0) for compatibility with dynamics model
3. **Few-shot learning**: Enabled by default (`use_examples=True`) for better performance
4. **Dynamic updates**: Task details updated at runtime, not in YAML file
5. **First-goal selection**: Currently uses first predicted goal/plan pair (can be extended for multi-hypothesis tracking)

## Next Steps

To extend the system:

1. **Multiple goal tracking**: Modify to track all goal/plan hypotheses for re-planning
2. **Custom predicates**: Add new predicates to system prompts and `_predicate_type_to_idx`
3. **Task-specific prompts**: Create separate configs for different task types (PickPlace, NutAssembly, etc.)
4. **Prompt optimization**: Fine-tune system prompts based on LLM performance
