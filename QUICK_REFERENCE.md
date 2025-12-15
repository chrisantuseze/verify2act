# Points2Plans + Robosuite Integration: Quick Reference

## TL;DR

Your robosuite data already works with Points2Plans! Now you need to:

1. **Integrate LLM module** (REQUIRED - already in Points2Plans/LLM/)
2. **Wrap the dynamics model** for real-time inference
3. **Convert observations** on-the-fly (reuse your data collection code)
4. **Execute planned actions** with robot controller
5. **Close the loop** with a planning controller (LLM → Planner → Executor)

---

## Key Files to Create

```
robosuite/planning/
├── llm_task_planner.py       # REQUIRED: Wrap Points2Plans/LLM for task decomposition
├── state_converter.py        # Obs → Points2Plans format (reuse data_capture/)
├── points2plans_planner.py   # Dynamics model wrapper for inference
├── action_executor.py        # High-level actions → robot commands
├── planning_controller.py    # Orchestration logic (LLM → Planner → Executor)
└── config/
    └── planner_config.yaml   # Configuration

robosuite/
└── run_points2plans_planner.py  # Demo script
```

---

## Data Flow

```
Task Description (Natural Language)
    ↓
LLM Module (Points2Plans/LLM) [REQUIRED]
    ↓ goals & high-level plans
Robosuite Env
    ↓ obs (robot state, object poses, RGB-D)
StateConverter
    ↓ formatted_state (point clouds, predicates, tensors)
Relational Dynamics Planner
    ↓ planned_action (pick/place primitive to achieve LLM goals)
ActionExecutor
    ↓ robot_commands (joint velocities/positions)
Robosuite Env
```

---

## Core Components

### 1. LLM Task Planner (REQUIRED - Core Component)

**Purpose**: Use Points2Plans' LLM module for task decomposition

**Location**: Already exists in `Points2Plans/LLM/`

**Key methods**:
```python
def generate_goals_and_plans(self, task_description, objects, initial_predicates):
    """Convert natural language → structured goals and plans."""
    # Uses Points2Plans' two-stage LLM approach:
    # 1. Goal Prediction: task → goal predicates
    # 2. Task Planning: task + goals → action sequence
```

**Two-stage approach**:
1. **Goal Prediction**: "Put all objects in bin" → `["On(milk, bin)", "On(cereal, bin)"]`
2. **Task Planning**: Goals → `["Pick(milk, table)", "Place(milk, bin)", ...]`

### 2. StateConverter

**Purpose**: Transform robosuite observations → Points2Plans tensor format

**Reuses**:
- `PointCloudGenerator` - Point cloud generation
- `StateCapture` - Object state extraction
- `DataFormatter` - Relational predicate computation

**Key method**:
```python
def convert(self, obs: Dict) -> Dict[str, torch.Tensor]:
    """Returns Points2Plans state dict with all required tensors."""
```

### 3. Relational Dynamics Planner

**Purpose**: Load pretrained dynamics model and run inference

**Key methods**:
```python
def plan_action(self, state, goal_predicates, num_samples=50):
    """Sample-based planning to find best action."""
    
def _predict_forward(self, current_latent, action, state):
    """Predict next state given action."""
```

**Planning algorithm** (from `base_RD.planner()`):
1. Sample N candidate actions
2. For each: predict next state using dynamics model
3. Evaluate: compute cost = distance to goal
4. Return: action with lowest cost

### 4. ActionExecutor

**Purpose**: Execute high-level primitives with robot

**Key method**:
```python
def execute(self, action: Dict, obs: Dict) -> bool:
    """Execute pick-place or push primitive."""
```

**Can reuse** your heuristic policy from `run_pickplace.py`.

### 5. PlanningController

**Purpose**: Orchestrate full planning loop (LLM → Planner → Executor)

**Key method**:
```python
def run_episode(self, goal_description: str, max_steps: int):
    # Step 1: LLM generates goals from natural language
    goals, plans = llm_planner.generate_goals_and_plans(goal_description, ...)
    goal_predicates = llm_planner.goals_to_predicates(goals[0], ...)
    
    # Step 2: Dynamics model plans actions to achieve goals
    for step in range(max_steps):
        state = state_converter.convert(obs)
        action = planner.plan_action(state, goal_predicates)
        success = executor.execute(action, obs)
        if goal_achieved: break
```

---

## Points2Plans State Format

Based on your successful training run, the model expects:

| Key | Shape | Description |
|-----|-------|-------------|
| `batch_voxel_list_single` | `[1, N, 128, 3]` | Point clouds (N objects, 128 points each) |
| `batch_one_hot_encoding` | `[1, N, K]` | Object type encodings (K types) |
| `batch_6DOF_pose` | `[1, N, 6]` | Object poses (x,y,z, roll,pitch,yaw) |
| `batch_all_obj_pair_relation` | `[1, N, N, 3]` | Pairwise predicates (On, Inside, Graspable) |
| `batch_env_identity` | `[1, N, 3]` | Environment features per object |
| `batch_grasp_identity` | `[1, N, 1]` | Graspability per object |
| `batch_edge_attr` | `[2, E]` | Graph edge indices (E = num edges) |
| `batch_num_objects` | `int` | Number of objects N |

**Note**: Batch dimension = 1 for real-time inference

---

## Action Format

High-level action primitive:

```python
action = {
    'skill': 0,           # 0=pick-place, 1=push
    'object_id': 2,       # Which object to manipulate
    'target_id': 4,       # Where to place (for pick-place)
    'offset': [0.01, -0.02]  # Relative offset (x, y) in meters
}
```

Action encoding for model (as tensor):

```python
# Discrete part: one-hot object selection
[0, 0, 1, 0, 0, 0, ...]  # object_id=2

# Continuous part: offset
[0.01, -0.02]
```

---

## Model Interface

### Loading Model

```python
from relational_dynamics.base_RD import RelationalDynamics
from relational_dynamics.config.base_config import BaseConfig

config = BaseConfig(args, dtype=torch.cuda.FloatTensor)
model = RelationalDynamics(config)
model.load_checkpoint("Points2Plans/ckpt/checkpoint/cp_1.pth")
model.set_model_device(torch.device("cuda"))
```

### Forward Prediction

Based on `base_RD.training()` method:

```python
# 1. Encode state (point clouds → latent)
img_emb = model.emb_model(point_clouds)
one_hot_emb = model.classif_model.one_hot_encoding_embed(one_hot)
current_latent = torch.cat([img_emb, one_hot_emb], dim=-1)

# 2. Encode action
discrete_action = model.classif_model.one_hot_encoding_embed(object_id)
continuous_action = model.classif_model.continuous_action_emb(offset)

# 3. Concatenate state + action
graph_node_action = torch.cat([current_latent, discrete_action, continuous_action], dim=1)

# 4. Run dynamics model (skill-dependent)
if skill == 0:
    next_latent = model.classif_model.graph_dynamics_0(graph_node_action)
else:
    next_latent = model.classif_model.graph_dynamics_1(graph_node_action)

# 5. Decode next state
pred_state = model.classif_model_decoder(next_latent, edge_index)
# pred_state contains: predicates, poses, feasibility
```

---

## LLM Integration (REQUIRED - Core Component)

**Location**: Already exists in `Points2Plans/LLM/`

Points2Plans uses a two-stage LLM approach for task decomposition:

```python
from robosuite.planning.llm_task_planner import LLMTaskPlanner

# Initialize (wraps Points2Plans/LLM module)
llm_planner = LLMTaskPlanner(
    model_config_path="Points2Plans/LLM/configs/models/pretrained/generative/gpt4.yaml",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Two-stage generation
goals, plans = llm_planner.generate_goals_and_plans(
    task_description="Put all objects in the bin",
    objects=["milk", "cereal", "bread", "bin"],
    initial_predicates=["On(milk, table)", "On(cereal, table)"]
)

# Stage 1 Output - Goals: [["On(milk, bin)", "On(cereal, bin)", "On(bread, bin)"]]
# Stage 2 Output - Plans: [["Pick(milk, table)", "Place(milk, bin)", ...]]

# Convert goals to tensor for dynamics model
object_to_id = {"milk": 0, "cereal": 1, "bread": 2, "bin": 3}
goal_predicates = llm_planner.goals_to_predicates(goals[0], object_to_id, 4)
```

**Why it's required**: The dynamics model needs structured goal predicates. The LLM converts natural language tasks into these structured goals.

---

## Configuration

### Planner Config (`planner_config.yaml`)

```yaml
llm:
  model_config: "Points2Plans/LLM/configs/models/pretrained/generative/gpt4.yaml"
  api_key_env: "OPENAI_API_KEY"
  
planner:
  checkpoint_path: "Points2Plans/ckpt/checkpoint/cp_1.pth"
  device: "cuda"
  planning_batch_size: 50
  planning_horizon: 1
  
model:
  max_objects: 10
  node_emb_size: 256
  n_layers: 4
  n_heads: 4
  z_dim: 128
  
state_converter:
  num_points_per_object: 128
  voxel_size: 0.005
  workspace_bounds: 
    x: [-0.5, 0.5]
    y: [-0.5, 0.5]
    z: [0.7, 1.5]
  camera_names: ["frontview", "agentview"]

executor:
  p_gain: 10.0
  r_gain: 5.0
  grasp_duration: 50
  max_retries: 3
```

---

## Example Usage

### Simple Demo

```python
import os
from robosuite import make
from robosuite.planning import PlanningController

# Create environment
env = make("PickPlaceMulti3", robots="Panda", has_renderer=True)

# Initialize planner (includes LLM module)
controller = PlanningController(
    env,
    checkpoint_path="Points2Plans/ckpt/checkpoint/cp_1.pth",
    llm_config_path="Points2Plans/LLM/configs/models/pretrained/generative/gpt4.yaml",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Run episode with natural language task
success = controller.run_episode(
    goal_description="Put all objects in the bin",  # LLM converts this to structured goals
    max_steps=50
)
```

### With Custom Goals

```python
# Define goal predicates manually
goal_predicates = np.zeros((num_objects, num_objects, 3))
goal_predicates[milk_id, bin_id, 0] = 1.0  # On(milk, bin)
goal_predicates[cereal_id, bin_id, 0] = 1.0  # On(cereal, bin)

# Run with custom goals
for step in range(max_steps):
    state = state_converter.convert(obs)
    action = planner.plan_action(state, goal_predicates)
    executor.execute(action, obs)
```

---

## Debugging Tips

### Check State Conversion

```python
state = state_converter.convert(obs)
print("Point clouds shape:", state['batch_voxel_list_single'].shape)
print("Object poses shape:", state['batch_6DOF_pose'].shape)
print("Relations shape:", state['batch_all_obj_pair_relation'].shape)
```

Expected output:
```
Point clouds shape: torch.Size([1, 5, 128, 3])
Object poses shape: torch.Size([1, 5, 6])
Relations shape: torch.Size([1, 5, 5, 3])
```

### Check Model Output

```python
with torch.no_grad():
    predicted_state = planner._predict_forward(latent, action, state)
    print("Predicted predicates:", predicted_state['predicates'].shape)
    print("Predicted poses:", predicted_state['poses'].shape)
```

### Visualize Point Clouds

```python
import open3d as o3d

pcd_list = state['batch_voxel_list_single'][0].cpu().numpy()
for i, points in enumerate(pcd_list):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
```

---

## Performance Metrics

**Expected latency** (on GPU):
- State conversion: ~50-100ms (point cloud generation)
- Planning (50 samples): ~100-200ms (model forward pass)
- Action execution: ~2-5 seconds (motion primitive)

**Total loop**: ~2-5 seconds per action

**Optimization**:
- Cache point cloud embeddings between steps
- Use batch inference for candidate actions
- Reduce point cloud resolution if real-time is critical

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `KeyError: 'batch_voxel_list_single'` | Check state_converter.convert() returns all keys |
| `RuntimeError: CUDA out of memory` | Reduce num_points or planning_batch_size |
| `Tensor size mismatch` | Ensure batch dimension = 1 for inference |
| Actions fail to execute | Check action feasibility in action sampling |
| Planning is slow | Use GPU, reduce num_samples, cache embeddings |
| Predictions are poor | Verify state format matches training data |

---

## File Locations

**Your trained model**: `Points2Plans/ckpt/checkpoint/cp_1.pth`  
**Training data**: `robosuite/data_capture/dataset/episodes/`  
**Existing data collection**: `robosuite/data_capture/episode_recorder.py`  
**Heuristic policy reference**: `robosuite/run_pickplace.py`

---

## Next Actions

1. ✅ Read `INTEGRATION_PLAN.md` for architecture overview
2. ✅ Read `IMPLEMENTATION_ROADMAP.md` for detailed code templates
3. ⬜ Explore existing `Points2Plans/LLM/` module and examples
4. ⬜ Create `LLMTaskPlanner` wrapper for Points2Plans LLM
5. ⬜ Start with `StateConverter` implementation
6. ⬜ Test dynamics model loading and inference
7. ⬜ Implement `ActionExecutor` (can reuse heuristic policy)
8. ⬜ Connect all in `PlanningController` (LLM → Planner → Executor)
9. ⬜ Test on simple task with natural language input
10. ⬜ Expand to multi-object tasks

---

## Key Insights

1. **LLM module already exists** - Points2Plans/LLM/ is a required core component
2. **Your data already works** - Training succeeded, so format is correct
3. **Reuse existing code** - Data collection has most utilities you need
4. **Two-stage planning** - LLM (task→goals) → Dynamics (goals→actions)
5. **Start simple** - Test components independently before integration
6. **Model is trained** - No retraining needed, just inference
7. **Natural language tasks** - LLM enables flexible task specification

---

## Resources

- **Integration Plan**: `INTEGRATION_PLAN.md` (architecture, design decisions)
- **Implementation Roadmap**: `IMPLEMENTATION_ROADMAP.md` (code templates, examples)
- **Points2Plans Paper**: https://arxiv.org/pdf/2408.14769
- **Your Training Run**: See terminal output from previous command
