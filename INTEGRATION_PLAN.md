# Points2Plans + Robosuite Integration Plan

## Overview

This document outlines the steps to integrate Points2Plans as a planner for your robosuite simulation environment. Since your data collection pipeline already works with Points2Plans' data format, the main task is creating the inference/planning interface.

## Architecture (Corrected)

```
                    ┌──────────────────────────┐
                    │   Natural Language Task  │
                    │ "Put all objects in bin" │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   LLM Module (Once)      │
                    │   Points2Plans/LLM/      │
                    │ - Goal Prediction        │
                    │ - Task Decomposition     │
                    └────────────┬─────────────┘
                                 │
                         Goal Predicates
              [On(milk,bin), On(cereal,bin)]
                                 │
        ┌────────────────────────┴────────────────────────┐
        │         CLOSED-LOOP PLANNING (Per Primitive)    │
        │                                                  │
        │  ┌───────────────────────────────────────────┐ │
        │  │ Robosuite Environment                     │ │
        │  │ - Current state observation               │ │
        │  └──────────────┬────────────────────────────┘ │
        │                 │ RGB-D, object poses          │
        │  ┌──────────────▼────────────────────────────┐ │
        │  │ State Converter (Lightweight)             │ │
        │  │ - Point clouds from RGB-D                 │ │
        │  │ - Object poses (reference)                │ │
        │  │ - One-hot encodings                       │ │
        │  │ NO manual predicate computation!          │ │
        │  └──────────────┬────────────────────────────┘ │
        │                 │ Point clouds + metadata      │
        │  ┌──────────────▼────────────────────────────┐ │
        │  │ Dynamics Model + Decoder                  │ │
        │  │ 1. Encode: Point clouds → Latent          │ │
        │  │ 2. Sample: N candidate actions            │ │
        │  │ 3. Forward: Predict next latent           │ │
        │  │ 4. Decode: Predicates + Feasibility       │ │
        │  │    (decoder does this, not us!)           │ │
        │  └──────────────┬────────────────────────────┘ │
        │                 │ Predicted states             │
        │  ┌──────────────▼────────────────────────────┐ │
        │  │ Feasibility Check (Rejection Sampling)    │ │
        │  │ - Grasp feasibility (from decoder)        │ │
        │  │ - Collision check (geometric)             │ │
        │  │ - Reject infeasible actions               │ │
        │  │ - Return FIRST feasible action            │ │
        │  └──────────────┬────────────────────────────┘ │
        │                 │ Selected primitive action    │
        │  ┌──────────────▼────────────────────────────┐ │
        │  │ Action Executor (Trajectory-Level)        │ │
        │  │ Execute ONE primitive:                    │ │
        │  │ - Pick: approach → grasp → lift           │ │
        │  │ - Place: move → descend → release         │ │
        │  │ (~200-500 robosuite timesteps)            │ │
        │  └──────────────┬────────────────────────────┘ │
        │                 │                              │
        │  ┌──────────────▼────────────────────────────┐ │
        │  │ Goal Check (Compare Predicates)           │ │
        │  │ If goal achieved: DONE                    │ │
        │  │ If not: REPLAN (back to top of loop)      │ │
        │  └───────────────────────────────────────────┘ │
        └──────────────────────────────────────────────────┘
```

## Key Integration Steps

### Step 1: Create State Converter Module

**File**: `robosuite/planning/state_converter.py`

**Purpose**: Convert robosuite observations to Points2Plans input format (LIGHTWEIGHT - decoder handles predicates).

**Key responsibilities**:
- Generate point clouds for each object (reuse your `PointCloudGenerator`)
- Extract object poses and identities
- Build one-hot encodings for object types
- Format into tensors matching Points2Plans' expected input
- **IMPORTANT**: NO manual predicate computation - decoder predicts them!

**Input**: Robosuite environment observations
**Output**: Dictionary with keys:
  - `batch_voxel_list_single`: Point clouds for each object [batch, num_objects, num_points, 3]
  - `batch_one_hot_encoding`: Object type encoding [batch, num_objects, num_types]
  - `batch_6DOF_pose`: Object poses [batch, num_objects, 6] (for reference, not supervision)
  - `batch_edge_attr`: Graph connectivity (edge indices)
  - `batch_num_objects`: Number of objects
  
**NOT NEEDED** (decoder predicts these during inference):
  - ~~`batch_all_obj_pair_relation`~~ - Decoder outputs this!
  - ~~`batch_env_identity`~~ - Decoder outputs this!
  - ~~`batch_grasp_identity`~~ - Decoder outputs this!

### Step 2: Create Points2Plans Planner Interface

**File**: `robosuite/planning/points2plans_planner.py`

**Purpose**: Wrapper around Points2Plans' relational dynamics model for inference.

**Key methods**:
- `__init__(checkpoint_path, config)`: Load pretrained model
- `plan_next_primitive(current_state, goal_predicates)`: Plan next primitive action (closed-loop)
- `_predict_forward(current_latent, action)`: Forward dynamics prediction
- `_check_feasibility(predicted_state, action)`: Feasibility checks

**Core planning approach** (adapted from `base_RD.planner()`):

**Rejection Sampling (NOT cost optimization)**:

1. **Encode current state**:
   - Point clouds → Embedding model → Latent representation
   - Decoder outputs current predicates/feasibility (for reference)

2. **Sample candidate actions** (N = 50-100):
   - Random pick-place combinations
   - Sample placement offsets around target objects

3. **For each candidate action**:
   - Encode action as tensor
   - Run dynamics model forward: current_latent + action → next_latent
   - Decode next state: **decoder predicts** next predicates, poses, feasibility
   - **Feasibility check**:
     - Grasp feasibility: `if grasp_identity[obj] > 0.5: reject`
     - Collision check: geometric overlap detection
     - If BOTH pass: action is feasible!

4. **Return first feasible action**:
   - No cost minimization, just binary feasibility
   - If all samples fail: resample (up to 10 iterations)
   - If still failing: fallback to tree search (advanced)

**Key insight**: This is **rejection sampling**, not optimization!

### Step 3: Create Action Executor Module

**File**: `robosuite/planning/action_executor.py`

**Purpose**: Execute ONE primitive action as a multi-phase robosuite trajectory.

**Key understanding**: 
- **Input**: One Points2Plans primitive (pick OR place)
- **Output**: Final observation after primitive completes
- **Execution**: 200-500 robosuite timesteps per primitive

**Key methods**:
- `execute_primitive(action_dict)`: Main entry point
  - Executes ONE primitive (pick or place)
  - Returns success status and final observation
  - Handles ~200-500 low-level robosuite steps internally

- `_execute_pick(object_id)`: Multi-phase pick execution
  - Phase 1: Approach object (~100 steps)
  - Phase 2: Close gripper (~50 steps)
  - Phase 3: Lift object (~100 steps)
  - Total: ~250 robosuite steps for ONE pick primitive

- `_execute_place(target_pos, offset)`: Multi-phase place execution
  - Phase 1: Move to target (~150 steps)
  - Phase 2: Descend (~50 steps)
  - Phase 3: Open gripper (~50 steps)
  - Total: ~250 robosuite steps for ONE place primitive

**Implementation**: Can reuse/adapt your existing heuristic policy code from `run_pickplace.py`.

**Critical**: Each primitive execution must return to the planner for closed-loop replanning!

### Step 4: Integrate LLM Module (Required - Core Component)

**Source**: Points2Plans already includes this in `Points2Plans/LLM/`

**File**: `robosuite/planning/llm_task_planner.py`

**Purpose**: Wrapper around Points2Plans LLM module for task decomposition and goal generation. This is a **core component** of Points2Plans, not optional.

**Key methods**:
- `generate_goals(task_description, objects, predicates)`: Get goal predicates from natural language
- `generate_task_plan(task_description, objects)`: Get high-level action sequence template

**Implementation**: Use the existing `Points2Plans/LLM/scripts/llm_planner.py` and `Points2Plans/LLM/fm_planning/` modules.

**Usage**: Call at episode start to decompose natural language tasks (e.g., "Put all objects in bin") into structured goals and plans before the dynamics model plans low-level actions.

### Step 5: Create Planning Loop Controller

**File**: `robosuite/planning/planning_controller.py`

**Purpose**: Orchestrate closed-loop planning at primitive boundaries.

**Key understanding**:
- **LLM runs ONCE** at episode start (or subtask boundary)
- **Dynamics model replans** before each primitive
- **Executor runs** entire primitive trajectory (~200-500 steps)
- **Loop repeats** until goal achieved

**Pseudocode**:
```python
class PlanningController:
    def __init__(self, env, planner, llm_planner, state_converter, action_executor):
        self.env = env
        self.planner = planner  # Dynamics model
        self.llm_planner = llm_planner
        self.state_converter = state_converter
        self.action_executor = action_executor
    
    def run_episode(self, task_description, max_primitives=20):
        """
        Args:
            max_primitives: Max number of primitives (NOT trajectory steps!)
                          Each primitive = ~200-500 robosuite steps
        """
        obs = self.env.reset()
        
        # LLM: Generate goal predicates ONCE
        objects = self.state_converter.get_object_list()
        initial_predicates = self._get_initial_scene_description(obs)
        goals, plans = self.llm_planner.generate_goals_and_plans(
            task_description, objects, initial_predicates
        )
        goal_predicates = self.llm_planner.goals_to_predicates(goals[0], ...)
        
        # Closed-loop execution: Plan → Execute → Replan
        for primitive_idx in range(max_primitives):
            print(f"\n=== Primitive {primitive_idx} ===")
            
            # 1. Convert current observation to planner format
            state = self.state_converter.convert(obs)
            
            # 2. Plan next primitive (rejection sampling)
            #    Returns FIRST feasible action
            action = self.planner.plan_next_primitive(state, goal_predicates)
            
            if action is None:
                print("No feasible action found!")
                # Fallback: tree search or failure handling
                break
            
            # 3. Execute primitive as trajectory (~200-500 robosuite steps)
            print(f"Executing: {action['skill']} object {action['object_id']}")
            final_obs, success = self.action_executor.execute_primitive(action)
            
            if not success:
                print("Primitive failed, replanning...")
                # Failure is OK - just replan with new state
                obs = final_obs
                continue
            
            # 4. Update observation for next iteration
            obs = final_obs
            
            # 5. Check if goal achieved
            #    Decode current state to get predicates
            state = self.state_converter.convert(obs)
            current_predicates = self.planner.decode_predicates(state)
            
            if self._goal_satisfied(current_predicates, goal_predicates):
                print("Goal achieved!")
                return True
        
        print("Max primitives reached")
        return False
    
    def _goal_satisfied(self, current, goal, threshold=0.1):
        """Check if current predicates match goal predicates."""
        diff = np.abs(current - goal)
        return np.mean(diff) < threshold
```

**Key points**:
- `max_primitives=20` means max 20 pick/place actions, NOT 20 robosuite steps
- Each iteration = one primitive = one call to dynamics model
- Closed-loop: Replanning happens naturally at each iteration
- No manual predicate computation - decoder does it

### Step 6: Create Example/Demo Script

**File**: `robosuite/run_points2plans_planner.py`

**Purpose**: End-to-end demonstration of the integration.

**Example tasks**:
- Simple: "Put milk in bin"
- Medium: "Put all objects in their respective bins"
- Complex: "Stack objects in a specific order"

## Data Format Compatibility (Corrected)

Since your data collection already works, you have verified compatibility. Here's what we actually need:

### For Inference (What StateConverter Provides):

| Robosuite Data | Points2Plans Input | Source | Required? |
|----------------|-------------------|---------|----------|
| Point clouds per object | `batch_voxel_list_single` | `PointCloudGenerator` | ✅ YES |
| Object poses | `batch_6DOF_pose` | `StateCapture.get_object_states()` | ✅ YES (reference) |
| Object types | `batch_one_hot_encoding` | Metadata | ✅ YES |
| Edge indices | `batch_edge_attr` | Computed from num_objects | ✅ YES |
| Num objects | `batch_num_objects` | Count | ✅ YES |

### NOT Needed for Inference (Decoder Predicts These):

| Data | Training Use | Inference | Why? |
|------|-------------|-----------|------|
| `batch_all_obj_pair_relation` | Ground truth labels | ❌ NOT NEEDED | Decoder outputs this! |
| `batch_env_identity` | Ground truth labels | ❌ NOT NEEDED | Decoder outputs this! |
| `batch_grasp_identity` | Ground truth labels | ❌ NOT NEEDED | Decoder outputs this! |
| `batch_action` | Ground truth labels | ❌ NOT NEEDED | We're planning actions! |

**Key Insight**: During training, you provided ground truth predicates to teach the decoder. During inference, the decoder predicts them - we don't compute them manually!

## Implementation Order (Corrected)

### Phase 1: Core Infrastructure (Days 1-2)
1. ✅ **Lightweight State Converter**
   - Point cloud generation (reuse `PointCloudGenerator`)
   - Object pose extraction
   - One-hot encodings
   - **REMOVE**: Manual predicate computation (decoder does this!)
   - **Simpler than originally thought!**

2. ✅ **LLM Integration**
   - Wrap existing `Points2Plans/LLM/` module
   - Goal generation from natural language
   - Runs ONCE per task

### Phase 2: Planning Logic (Days 3-4)
3. ✅ **Dynamics Model Planner Interface**
   - Load pretrained model
   - Implement rejection sampling (already in `base_RD.planner()`)
   - Feasibility checks: grasp + collision
   - Returns FIRST feasible action
   - **Key**: Trust decoder to predict predicates!

4. ✅ **Action Executor (Primitive-Level)**
   - Execute ONE primitive at a time
   - Multi-phase trajectory execution
   - Return after ~200-500 robosuite steps
   - **Key**: Primitive boundaries for replanning!

### Phase 3: Integration (Days 5-6)
5. ✅ **Closed-Loop Planning Controller**
   - LLM → goal predicates (once)
   - Loop: State → Dynamics → Executor → Check goal
   - Replanning at primitive boundaries
   - **Key**: Closed-loop, not open-loop!

6. ✅ **Demo Script** - End-to-end example

### Phase 4: Advanced Features (Days 7+)
7. ⚪ **Tree Search Fallback**
   - Only when rejection sampling fails repeatedly
   - Multi-step lookahead
   - Optional for most tasks

8. ⚪ **Failure Recovery**
   - Primitive execution monitoring
   - Automatic replanning on failure
   - Error recovery strategies

9. ⚪ **Online Fine-tuning**
   - Collect online experience
   - Fine-tune dynamics model
   - Improve prediction accuracy

## Critical Corrections from Original Plan

### What Changed:

1. **Predicates**: 
   - ❌ **Wrong**: Manually compute predicates in StateConverter
   - ✅ **Correct**: Decoder predicts predicates during inference
   - **Impact**: StateConverter is much simpler!

2. **Execution Loop**:
   - ❌ **Wrong**: Open-loop execution of full plan
   - ✅ **Correct**: Closed-loop replanning at each primitive boundary
   - **Impact**: Natural failure recovery, more robust

3. **Cost Function**:
   - ❌ **Wrong**: Optimize continuous cost to find best action
   - ✅ **Correct**: Binary feasibility checks, return first feasible action
   - **Impact**: Simpler algorithm (rejection sampling)

4. **Step Definition**:
   - ❌ **Wrong**: Steps are robosuite trajectory timesteps
   - ✅ **Correct**: Steps are discrete primitives (pick/place)
   - **Impact**: Each "step" is ~200-500 robosuite timesteps

5. **Tree Search**:
   - ❌ **Wrong**: Always used for planning
   - ✅ **Correct**: Fallback when sampling fails
   - **Impact**: Simpler base implementation

### Key Insights:

- **Trust the trained models**: Decoder knows how to predict predicates
- **Closed-loop is natural**: Just replan after each primitive
- **Feasibility, not optimization**: Binary checks are sufficient
- **Primitives, not steps**: Planning is higher-level than trajectory control

## Technical Considerations

### Model Inference
- **Batch size**: Use batch_size=1 for real-time inference
- **Device**: Load model on GPU for speed (`config.device = 'cuda'`)
- **Latency**: Single forward pass ~10-50ms depending on hardware

### Action Space
Points2Plans uses discrete action primitives:
- **Pick action**: `[skill_id=0, object_id (one-hot), x_offset, y_offset]`
- **Place action**: `[skill_id=0, object_id (one-hot), target_id (one-hot), x_offset, y_offset]`
- **Push action**: `[skill_id=1, object_id (one-hot), x_direction, y_direction]` (if trained)

Map these to your robosuite controller actions.

### State Representation
- **Point clouds**: 128 points per object (downsample if needed)
- **Coordinate frame**: Points2Plans likely uses world frame, ensure consistency
- **Object ordering**: Maintain consistent ordering across time steps

### Error Handling
- **Execution failures**: Gripper miss, object slip → trigger replan
- **Invalid predictions**: Predicted pose out of bounds → resample or use backup
- **Stale observations**: Ensure state converter uses latest sim state

## Testing Strategy

### Unit Tests
- `test_state_converter.py`: Verify tensor shapes and value ranges
- `test_planner_inference.py`: Check model loads and runs
- `test_action_executor.py`: Verify primitive execution

### Integration Tests
- **Simple pick-place**: Single object, single action
- **Multi-object**: Plan sequence of actions
- **With failures**: Test replanning when execution fails

### Validation
- Compare planned trajectories with offline data (should match training distribution)
- Measure success rate on standard tasks
- Profile computational bottlenecks

## File Structure

```
verify2act/
├── robosuite/
│   ├── planning/              # NEW: Planning integration
│   │   ├── __init__.py
│   │   ├── state_converter.py        # Step 1
│   │   ├── points2plans_planner.py   # Step 2
│   │   ├── action_executor.py        # Step 3
│   │   ├── llm_task_planner.py       # Step 4
│   │   └── planning_controller.py    # Step 5
│   ├── run_points2plans_planner.py   # Step 6: Demo script
│   ├── data_capture/
│   │   ├── episode_recorder.py       # Reuse for state extraction
│   │   ├── state_capture.py          # Reuse
│   │   ├── data_formatter.py         # Reuse
│   │   └── ...
│   └── ...
├── Points2Plans/
│   ├── relational_dynamics/
│   │   ├── base_RD.py               # Reference for planning logic
│   │   └── ...
│   └── ...
└── INTEGRATION_PLAN.md              # This file
```

## Configuration Example

Create `robosuite/planning/config/default_planner_config.yaml`:

```yaml
planner:
  checkpoint_path: "Points2Plans/ckpt/checkpoint/cp_1.pth"
  device: "cuda"
  planning_batch_size: 50  # Number of candidate actions to sample
  planning_horizon: 1      # Number of steps to plan ahead
  
state_converter:
  num_points_per_object: 128
  voxel_size: 0.005
  workspace_bounds: [[-0.5, 0.5], [-0.5, 0.5], [0.7, 1.5]]

action_executor:
  max_retries: 3
  grasp_height_threshold: 0.05
  
llm:
  model: "gpt-4"
  api_key_env: "OPENAI_API_KEY"
```

## Next Steps

1. **Start with Step 1**: Create `state_converter.py` by adapting your `EpisodeRecorder` for real-time use
2. **Test incrementally**: Verify each component works before moving to next
3. **Reuse existing code**: Your data collection pipeline has most needed utilities
4. **Profile performance**: Identify bottlenecks early (likely point cloud generation)
5. **Start simple**: Test with single-object tasks before complex multi-step planning

## Questions to Consider

Before implementation:
1. **Which tasks to prioritize?** Start with PickPlace variants you already have
2. **Real-time vs offline?** Real-time needs faster point cloud generation
3. **LLM integration scope?** Optional but valuable for natural language tasks
4. **Failure recovery?** How to handle execution failures and replanning
5. **Evaluation metrics?** Define success criteria (task completion rate, steps, time)

## Resources

- **Points2Plans paper**: https://arxiv.org/pdf/2408.14769
- **Your trained model**: `Points2Plans/ckpt/checkpoint/cp_1.pth`
- **Your data format**: `robosuite/data_capture/dataset/episodes/`
- **Existing heuristic policy**: `robosuite/run_pickplace.py` (for reference)
