# Phase 2: Dynamics Model Planning + Primitive Execution

Phase 2 completes the core planning and execution pipeline by adding:
1. **DynamicsModelPlanner**: Closed-loop primitive planning with rejection sampling
2. **PrimitiveExecutor**: High-level primitive to low-level robosuite control

## Architecture

```
Episode Start:
  ↓
LLM Task Planner (ONCE)
  ├─→ Goals: [["On(milk, bin)", "On(cereal, bin)", ...]]
  └─→ Plans: [["Pick(milk, table)", "Place(milk, bin)", ...]]
  ↓
CLOSED-LOOP Planning + Execution:
  ↓
  [Observe State] ←─────────────────┐
    ↓                                │
  StateConverter                     │
    ↓                                │
  DynamicsModelPlanner               │
    ├─ Sample K actions              │
    ├─ Forward simulate              │
    ├─ Check feasibility             │
    └─→ Best primitive action        │
         ↓                           │
  PrimitiveExecutor                  │
    ├─ Pick: Approach→Grasp→Lift    │
    ├─ Place: Approach→Place→Retreat│
    └─→ Execute (~200-500 steps)    │
         ↓                           │
  [New Observation] ─────────────────┘
```

## Components

### 1. DynamicsModelPlanner

Wraps trained Points2Plans model for closed-loop planning.

**Key Features:**
- **Rejection Sampling**: Samples K candidate actions around goal locations
- **Forward Simulation**: Predicts next state through dynamics model
- **Binary Feasibility**: Checks if predicted state satisfies goals
- **Closed-loop**: Replans BEFORE EACH PRIMITIVE (not trajectory-level)

**Usage:**
```python
from robosuite.planning import DynamicsModelPlanner

# Initialize with trained checkpoint
planner = DynamicsModelPlanner(
    checkpoint_path="Points2Plans/ckpt/checkpoint/cp_1.pth",
    num_samples=50  # Number of action samples for rejection sampling
)

# Plan next primitive (called before each primitive execution)
primitive, action_params, feasibility = planner.plan_next_primitive(
    state_dict=state_dict,        # From StateConverter
    goal_predicates=goal_predicates,  # From LLMTaskPlanner
    primitive_plan=primitive_plan  # High-level plan from LLM
)

# Returns:
#   primitive: "Pick(milk, table)" or "Place(milk, bin)"
#   action_params: Target position [x, y, z]
#   feasibility: Score [0, 1] indicating success likelihood
```

**Implementation Details:**
- Encodes state through PointConv encoder
- Samples actions around target locations with random perturbations
- Forward simulates each action through graph dynamics
- Decodes predicted state and extracts relations
- Compares predicted relations with goal predicates
- Returns action with highest feasibility score

### 2. PrimitiveExecutor

Executes high-level primitives as low-level robosuite OSC commands.

**Key Features:**
- **Primitive = ~200-500 steps**: Not trajectory-level, but discrete action level
- **Multi-phase Execution**: Pick (4 phases), Place (4 phases)
- **OSC Control**: Uses Operational Space Control for smooth motion
- **Feedback**: Returns success status for replanning

**Usage:**
```python
from robosuite.planning import PrimitiveExecutor

# Initialize with robosuite environment
executor = PrimitiveExecutor(
    env=env,
    approach_height=0.15,   # Height above object for approach
    max_steps_per_primitive=500  # Max steps per primitive
)

# Execute primitive
success, steps, obs = executor.execute_primitive(
    primitive="Pick(milk, table)",
    action_params=action_params,  # From planner
    obs=current_obs
)

# Returns:
#   success: True if primitive completed successfully
#   steps: Number of robosuite steps taken
#   obs: Final observation after execution
```

**Pick Primitive Phases:**
1. **Approach**: Move above object (z + approach_height)
2. **Descend**: Move down to grasp height
3. **Grasp**: Close gripper (50 steps)
4. **Lift**: Lift object to lift_height

**Place Primitive Phases:**
1. **Approach**: Move above target (z + approach_height)
2. **Descend**: Move down to place_height
3. **Release**: Open gripper (50 steps)
4. **Retreat**: Lift back up

## Planning Algorithm

### Rejection Sampling (NOT Tree Search)

Points2Plans uses **rejection sampling** for action selection, not tree search:

```python
def plan_next_primitive():
    best_action = None
    best_feasibility = -inf
    
    for i in range(num_samples):  # Default: 50 samples
        # 1. Sample random action around goal
        action = sample_action_near_goal()
        
        # 2. Forward simulate through dynamics model
        predicted_state = model.forward(current_state, action)
        
        # 3. Check binary feasibility
        feasibility = check_goals_satisfied(predicted_state, goals)
        
        # 4. Keep best action
        if feasibility > best_feasibility:
            best_action = action
            best_feasibility = feasibility
        
        # 5. Early exit if found feasible
        if feasibility >= 0.5:
            break
    
    return best_action
```

**Why Rejection Sampling?**
- Fast: O(K) samples vs exponential tree search
- Sufficient: Binary feasibility check (not optimization)
- Closed-loop: Failures corrected by replanning at next primitive

**When Tree Search Used:**
- Fallback only when rejection sampling finds no feasible actions
- Not the primary planning mechanism

## Closed-Loop Execution Flow

```python
# Episode initialization
obs = env.reset()
goals, plans = llm_planner.generate_goals_and_plans(task)  # ONCE
goal_predicates = llm_planner.goals_to_predicates(goals[0], ...)

primitive_plan = plans[0]  # ["Pick(...)", "Place(...)", ...]

# Closed-loop execution
for primitive_name in primitive_plan:
    # 1. Convert observation to model format
    state_dict = state_converter.convert(obs)
    
    # 2. Plan next primitive (REPLAN EACH TIME)
    primitive, action_params, feasibility = dynamics_planner.plan_next_primitive(
        state_dict=state_dict,
        goal_predicates=goal_predicates,
        primitive_plan=[primitive_name, ...]
    )
    
    # 3. Execute primitive (~200-500 steps)
    success, steps, obs = executor.execute_primitive(
        primitive=primitive,
        action_params=action_params,
        obs=obs
    )
    
    # 4. Check success, replan if needed
    if not success:
        print("Primitive failed, replanning from new state...")
        # Loop continues with new observation
```

## Demo

Run Phase 2 demo to see planning + execution working together:

```bash
cd robosuite/planning

# Mock demo (no environment required)
python demo_phase2.py

# With robosuite environment
python demo_phase2.py --with-env --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_1.pth
```

**Demo Output:**
```
[1/5] Initializing components...
  ✓ All components initialized

[2/5] Generating goals from LLM...
  ✓ Goals: [["On(milk, bin)", "On(cereal, bin)", ...]]
  ✓ Plans: [["Pick(milk, table)", "Place(milk, bin)", ...]]

[3/5] Creating mock observation...
  ✓ Mock observation created

[4/5] Converting observation to model format...
  ✓ State dict created

[5/5] Planning next primitive...
  Sampling 50 candidate actions...
  ✓ Found feasible action at sample 23
  ✓ Planned primitive: Pick(milk, table)
  ✓ Feasibility: 0.876
```

## Key Differences from Common Misunderstandings

| Aspect | Common Misconception | Actual Implementation |
|--------|---------------------|----------------------|
| **Execution Loop** | Open-loop: Plan all primitives upfront | Closed-loop: Replan before each primitive |
| **Planning Frequency** | Once per episode | Before each primitive (~every 200-500 steps) |
| **Action Selection** | Tree search optimization | Rejection sampling with binary feasibility |
| **Primitive Granularity** | Trajectory steps (~10 steps) | Discrete actions (~200-500 steps) |
| **Goal Updates** | Change during episode | Generated once, remain constant |
| **Feasibility Check** | Continuous cost optimization | Binary feasibility check (>0.5) |

## Next Steps

Phase 3 will integrate everything into a complete closed-loop controller:
- Full environment integration with robosuite PickPlace task
- Observation extraction (RGB-D → point clouds)
- Complete episode execution with replanning
- Success metrics and evaluation

## Files Created

Phase 2 files:
- `dynamics_model_planner.py`: Dynamics model wrapper with rejection sampling
- `primitive_executor.py`: Primitive to low-level control converter
- `demo_phase2.py`: Demo script showing planning + execution loop
- `PHASE2_README.md`: This documentation

Complete Phase 1+2 files:
- `state_converter.py`: Robosuite obs → model format (Phase 1)
- `llm_task_planner.py`: Task → goals + plans (Phase 1, refactored)
- `__init__.py`: Package exports (updated)

## Testing

Test individual components:

```python
# Test DynamicsModelPlanner
from robosuite.planning import DynamicsModelPlanner
planner = DynamicsModelPlanner("checkpoint.pth", num_samples=10)
# ... (see demo_phase2.py for full example)

# Test PrimitiveExecutor
from robosuite.planning import PrimitiveExecutor
executor = PrimitiveExecutor(env)
success, steps, obs = executor.execute_primitive("Pick(milk, table)", [0, 0, 0.8], obs)
```

## Troubleshooting

**"Could not find object ID for X":**
- Ensure `state_dict['object_names']` is populated
- Implement proper object name → ID mapping in StateConverter

**"Planning failed: checkpoint not found":**
- Verify checkpoint path points to trained model
- Check that training completed successfully

**"Primitive execution failed at approach phase":**
- Check position threshold (may be too strict)
- Verify OSC controller is properly configured
- Check max_steps_per_primitive is sufficient

**"Feasibility always 0.0":**
- Check goal_predicates tensor shape and values
- Verify dynamics model is loaded correctly
- Ensure state_dict format matches training data
