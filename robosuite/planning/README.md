# LLM Task Planner - Refactored

The `LLMTaskPlanner` has been refactored to use Points2Plans' `BehaviorPromptManager` for YAML-based prompt configuration.

## Key Changes

### Before
- Hardcoded prompts in Python strings
- No few-shot learning support
- Less flexible prompt tuning

### After
- YAML-based prompt configuration via `BehaviorPromptManager`
- Few-shot learning with example demonstrations
- Easy prompt tuning without code changes
- Backward compatible fallback mode

## Usage

### Recommended: YAML-based Prompts

```python
from robosuite.planning import LLMTaskPlanner

# Initialize with YAML config
planner = LLMTaskPlanner(
    prompt_config_path="planning/prompts/robosuite_pickplace.yaml",
    use_examples=True  # Enable few-shot learning
)

# Generate goals and plans
goals, plans = planner.generate_goals_and_plans(
    task_description="Put all objects in the bin",
    objects=["milk", "cereal", "bread", "bin"],
    initial_predicates=["On(milk, table)", "On(cereal, table)", ...]
)
```

### Fallback Mode (No YAML Config)

```python
# Initialize without YAML config (uses hardcoded prompts)
planner = LLMTaskPlanner()

goals, plans = planner.generate_goals_and_plans(
    task_description="Put the milk in the bin",
    objects=["milk", "bin", "table"],
    initial_predicates=["On(milk, table)"]
)
```

## YAML Prompt Configuration

Create a YAML file with task examples:

```yaml
# prompts/robosuite_pickplace.yaml
task: robosuite_pickplace
instruction: "Put all objects in the bin."
objects: 
  - "milk"
  - "cereal"
  - "bin"
predicates:
  - "On(milk, table)"
  - "On(cereal, table)"

goals: [["On(milk, bin)", "On(cereal, bin)"]]
plans: [["Pick(milk, table)", "Place(milk, bin)", "Pick(cereal, table)", "Place(cereal, bin)"]]

role: system
name_query: example_user
name_response: example_assistant
```

## Demo Script

Run the demo to see both modes in action:

```bash
# YAML mode (recommended)
cd robosuite/planning
python demo_llm_planner.py --prompt-config prompts/robosuite_pickplace.yaml

# Fallback mode
python demo_llm_planner.py
```

## Benefits

1. **Few-shot Learning**: YAML configs support example-based learning for better LLM performance
2. **Flexibility**: Easy to tune prompts without modifying code
3. **Consistency**: Matches Points2Plans' original architecture
4. **Backward Compatible**: Fallback mode works without YAML configs

## Next Steps

Proceed to Phase 2:
- Dynamics model planner interface with rejection sampling
- Action executor for primitive-level execution
