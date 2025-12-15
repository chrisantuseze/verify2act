# Phase 2 Quick Reference

## Files Created

```
robosuite/planning/
├── dynamics_model_planner.py     # 450 lines - Dynamics model wrapper
├── primitive_executor.py         # 300 lines - Primitive execution
├── demo_phase2.py                # 350 lines - Demo script
├── PHASE2_README.md              # Full documentation
├── PHASE2_COMPLETE.md            # Implementation summary
└── PHASE2_QUICK_REFERENCE.md     # This file
```

## Quick Usage

### DynamicsModelPlanner

```python
from robosuite.planning import DynamicsModelPlanner

planner = DynamicsModelPlanner(
    checkpoint_path="Points2Plans/ckpt/checkpoint/cp_1.pth",
    num_samples=50
)

primitive, params, feasibility = planner.plan_next_primitive(
    state_dict=state_dict,
    goal_predicates=goal_predicates,
    primitive_plan=["Pick(milk, table)", "Place(milk, bin)"]
)
```

### PrimitiveExecutor

```python
from robosuite.planning import PrimitiveExecutor

executor = PrimitiveExecutor(env, max_steps_per_primitive=500)

success, steps, obs = executor.execute_primitive(
    primitive="Pick(milk, table)",
    action_params=np.array([0.1, 0.2, 0.8]),
    obs=current_obs
)
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Closed-Loop** | Replan before EACH primitive (not trajectory, not episode) |
| **Primitive** | Discrete action (Pick/Place) = ~200-500 steps |
| **Rejection Sampling** | Sample K actions, check feasibility, return best |
| **Binary Feasibility** | Check if prediction matches goals (>0.5) |
| **Planning Frequency** | Before each primitive (~every 200-500 steps) |

## Execution Flow

```
Episode Start
  ↓
LLM: Generate goals + plans (ONCE)
  ↓
Loop until task complete:
  ├─ Observe current state
  ├─ Convert to model format (StateConverter)
  ├─ Plan next primitive (DynamicsModelPlanner)
  │   ├─ Sample 50 actions
  │   ├─ Forward simulate each
  │   ├─ Check feasibility
  │   └─ Return best action
  ├─ Execute primitive (PrimitiveExecutor)
  │   ├─ Pick: Approach→Grasp→Lift
  │   └─ Place: Approach→Place→Retreat
  └─ Get new observation, repeat
```

## Common Pitfalls

❌ **Wrong:** Plan all primitives upfront (open-loop)
✅ **Right:** Replan before each primitive (closed-loop)

❌ **Wrong:** Replan every trajectory step (~10 steps)
✅ **Right:** Replan every primitive (~200-500 steps)

❌ **Wrong:** Use tree search for all planning
✅ **Right:** Use rejection sampling, tree search only as fallback

❌ **Wrong:** Continuous cost optimization
✅ **Right:** Binary feasibility check (>0.5)

## Run Demo

```bash
cd robosuite/planning
python demo_phase2.py
```

## Next: Phase 3

Phase 3 completes the integration:
- Full robosuite environment setup
- RGB-D → point cloud pipeline
- End-to-end episode execution
- Success evaluation

## Questions?

See:
- `PHASE2_README.md` for detailed documentation
- `PHASE2_COMPLETE.md` for implementation summary
- `demo_phase2.py` for working examples
