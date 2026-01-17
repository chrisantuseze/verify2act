# Points2Plans Robosuite Integration - Complete Project Reference

**Status**: ✅ Production Ready  
**Last Updated**: January 13, 2026  
**Format Compliance**: 100% Points2Plans Compatible

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Data Collection Pipeline](#3-data-collection-pipeline)
4. [Planning System](#4-planning-system)
5. [Data Format Specification](#5-data-format-specification)
6. [Configuration Reference](#6-configuration-reference)
7. [Command Reference](#7-command-reference)
8. [Implementation Details](#8-implementation-details)
9. [Testing & Verification](#9-testing--verification)
10. [Troubleshooting Guide](#10-troubleshooting-guide)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. Executive Summary

### Project Overview

This project integrates the Points2Plans planning framework with robosuite simulation environments, enabling:

1. **Automated Data Collection**: Batch collection of robot manipulation episodes in Points2Plans format
2. **LLM-Based Task Planning**: Natural language task instructions → symbolic goal/plan generation
3. **Dynamics Model Planning**: Learned dynamics models for primitive-level action planning
4. **Closed-Loop Execution**: Real-time replanning with collision detection and multi-step lookahead

### Key Achievements

| Component | Status | Description |
|-----------|--------|-------------|
| Data Collection | ✅ Complete | 4-phase pipeline with 100% format compliance |
| LLM Task Planner | ✅ Complete | YAML-based prompt configuration with few-shot learning |
| Dynamics Planner | ✅ Complete | Multi-step lookahead + collision detection |
| Closed-Loop Controller | ✅ Complete | Real-time execution with replanning |
| Points2Plans Alignment | ✅ Complete | Matches original paper's algorithm |

### Core Files

**Data Collection (`data_capture/`):**
- `batch_collect.py` - Automated multi-episode collection
- `episode_recorder.py` - Episode recording with key timestep detection
- `state_capture.py` - Robot & object state extraction
- `data_formatter.py` - Points2Plans format conversion
- `metadata_extractor.py` - Object metadata extraction
- `inspect_dataset.py` - Quality assurance tools

**Planning (`planning/`):**
- `llm_task_planner.py` - LLM-based goal/plan generation
- `dynamics_model_planner.py` - Dynamics-based action planning
- `closed_loop_controller.py` - Execution orchestration
- `primitive_executor.py` - Low-level action execution
- `collision_checker.py` - 2D collision detection
- `state_converter.py` - State format conversion

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                    │
│              "Stack all cubes on the table"                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM TASK PLANNER                                  │
│  • Parse natural language instruction                                │
│  • Generate symbolic goals: [Stacked(cubeA, cubeB), ...]            │
│  • Generate action plan: [Pick(cubeA), Place(cubeA, cubeB), ...]    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 CLOSED-LOOP CONTROLLER                               │
│  For each primitive in plan:                                         │
│    1. Get current state from environment                            │
│    2. Call dynamics planner for action parameters                   │
│    3. Execute action via primitive executor                          │
│    4. Check success & replan if needed                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 DYNAMICS MODEL PLANNER                               │
│  • Encode current scene (point clouds → latent)                     │
│  • Multi-step lookahead (2-3 primitives)                            │
│  • Rejection sampling with collision checking                        │
│  • Return best action parameters                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PRIMITIVE EXECUTOR                                  │
│  • Pick(object): Move to object → Grasp                             │
│  • Place(object, target): Move to target → Release                  │
│  • Convert parameters to motor commands                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ROBOSUITE ENVIRONMENT                             │
│  • Execute motor commands                                            │
│  • Return observations (robot state, object poses, contacts)        │
│  • Provide reward signal                                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Environment State → State Capture → Point Cloud Generation → Dynamics Encoder
                                                                    ↓
                                                            Latent Representation
                                                                    ↓
                                                            Action Sampling
                                                                    ↓
                                                            Forward Simulation
                                                                    ↓
                                                            Feasibility Check
                                                            (Goals + Collisions)
                                                                    ↓
                                                            Best Action Selection
                                                                    ↓
                                                            Primitive Execution
```

---

## 3. Data Collection Pipeline

### Overview

The data collection pipeline captures robot manipulation episodes in Points2Plans format. It consists of 4 phases, all complete and production-ready.

### Phase 1: State Capture ✅

**Purpose**: Extract robot and object states from robosuite environment

**Captured Data**:
- **Robot State** (8 arrays):
  - `joint_position`: (T, 7) - 7-DoF joint positions
  - `joint_velocity`: (T, 7) - Joint velocities
  - `joint_torque`: (T, 7) - Joint torques
  - `target_joint_position`: (T, 7) - Target positions
  - `ee_position`: (T, 3) - End-effector XYZ
  - `ee_orientation`: (T, 4) - End-effector quaternion
  - `ee_velocity`: (T, 3) - End-effector velocity
  - `target_ee_discrete`: (T, 3) - Discrete EE targets

- **Object State** (per object):
  - `position`: (T, 3) - Object center position
  - `orientation`: (T, 4) - Object quaternion

- **Metadata**:
  - Contact events per timestep
  - Behavior labels (grasp/release/none)
  - Hidden/occlusion labels

**File**: `state_capture.py` (~260 lines)

### Phase 2: Point Cloud Integration ✅

**Purpose**: Generate 3D point clouds for each object

**Generated Data**:
- `point_cloud_N`: (T, 128, 3) - Raw point cloud for object N
- `point_cloud_Nsampling`: (T, 128, 3) - Sampled point cloud
- `point_cloud_Nsampling_noise`: (T, 128, 3) - Noisy variant

**Features**:
- Multi-camera fusion (frontview, agentview, eye-in-hand `robot0_eye_in_hand`)
- Geometry-based object segmentation
- Configurable point count (default: 128)
- Optional noise injection

**Dependencies**: `open3d` (optional - falls back to placeholders if not installed)

### Phase 3: Data Packaging ✅

**Purpose**: Convert captured data to Points2Plans format

**Output Structure**:
```python
episode = (data_dict, attrs_dict)  # Pickle tuple
```

**File**: `data_formatter.py` (~200 lines)

### Phase 4: Batch Collection ✅

**Purpose**: Automated multi-episode collection with quality assurance

**Features**:
- Integration with `HeuristicStackPolicy` from `run_stack.py`
- Progress tracking and statistics
- Automatic error recovery with retry
- Metadata generation (JSON)
- Key timestep detection mode

**File**: `batch_collect.py` (~380 lines)

### Key Timestep Recording Mode

Instead of recording all timesteps and subsampling later, the system can record only key state transitions:

**Key Timesteps**:
1. Initial state (T0, behavior='none')
2. After grasp (transition TO grasp with valid object)
3. After release (transition FROM release with valid object)

**Benefits**:
- Memory efficient (~5 timesteps vs ~500)
- More reliable detection (real-time context)
- Cleaner action sequences
- Faster (no post-processing)

**Configuration**:
```python
recorder = EpisodeRecorder(env, key_timesteps_only=True)
```

---

## 4. Planning System

### LLM Task Planner

**Purpose**: Convert natural language instructions to symbolic goals and plans

**Architecture**:
```
Task Instruction + Objects + Initial State
            ↓
    BehaviorPromptManager (YAML config)
            ↓
    GPT-4 / Claude API
            ↓
    Goals: [On(cubeA, cubeB), On(cubeC, cubeA)]
    Plans: [Pick(cubeA), Place(cubeA, cubeB), Pick(cubeC), Place(cubeC, cubeA)]
```

**Configuration Files**:
```
planning/configs/prompts/
├── system/
│   ├── goal_prediction_v1.yaml
│   └── task_planning_goal_condition_v1.yaml
├── examples/
│   ├── example_1.yaml (single placement)
│   ├── example_2.yaml (two-object stack)
│   └── example_3.yaml (multi-object stack)
└── tasks/
    └── stack_task.yaml
```

**Supported Predicates**:
- `On(obj, surface)` - Object is on surface/object
- `Stacked(obj1, obj2)` - obj1 is stacked on obj2
- `Grasped(obj)` - Object is currently grasped

**Supported Actions**:
- `Pick(object, location)` - Pick object from location
- `Place(object, target)` - Place object on target

**Usage**:
```python
from planning.llm_task_planner import LLMTaskPlanner

planner = LLMTaskPlanner(
    prompt_config_path="planning/configs/prompts/tasks/stack_task.yaml",
    use_examples=True  # Enable few-shot learning
)

goals, plans = planner.generate_goals_and_plans(
    task_description="Stack all cubes",
    objects=["cubeA", "cubeB", "cubeC", "table"],
    initial_predicates=["On(cubeA, table)", "On(cubeB, table)", "On(cubeC, table)"]
)
```

### Dynamics Model Planner

**Purpose**: Use learned dynamics model to plan primitive-level actions

**Key Features**:
1. **Scene Encoding**: Point clouds → latent representation
2. **Action Sampling**: Sample candidate action parameters
3. **Forward Simulation**: Predict next state via dynamics model
4. **Feasibility Check**: Goal matching + collision detection
5. **Multi-Step Lookahead**: Simulate 2-3 primitives ahead

**Configuration**:
```python
planner = DynamicsModelPlanner(
    checkpoint_path="Points2Plans/ckpt/checkpoint/cp_1.pth",
    lookahead_depth=2,           # 1=greedy, 2-3=multi-step
    enable_collision_checking=True,
    num_samples=50,
    x_collision=0.05,            # Collision box half-width
    y_collision=0.05
)
```

### Multi-Step Lookahead (Phase 1 Alignment) ✅

**Problem Solved**: Single-step (greedy) planning can make short-sighted decisions that block future actions.

**Solution**: Simulate 2-3 primitives ahead before deciding on current action.

**Algorithm**:
```python
for each_sample in range(50):
    # Build action sequence for next N primitives
    action_sequence = [sample_action(primitive) for primitive in plan[:lookahead_depth]]
    
    # Simulate entire sequence
    terminal_state = forward_simulate_sequence(current_state, action_sequence)
    
    # Check terminal state feasibility
    feasibility = check_goals(terminal_state) * check_collisions(terminal_state)
    
    if feasibility > threshold:
        return action_sequence[0]  # Execute first action only
```

**Performance**:
| Depth | Time/Primitive | Use Case |
|-------|---------------|----------|
| 1 | ~1-2s | Simple tasks, speed priority |
| 2 | ~2-4s | Multi-step (recommended default) |
| 3 | ~4-8s | Complex dependencies |

### Collision Detection (Phase 2 Alignment) ✅

**Purpose**: Prevent actions that would cause object-object collisions

**Algorithm**:
```python
for each pair of objects (i, j):
    # Skip if vertically separated (stacked objects)
    if abs(z_i - z_j) > z_threshold:
        continue
    
    # 2D AABB collision check in XY plane
    bbox_i = get_2d_bbox(position_i, x_collision, y_collision)
    bbox_j = get_2d_bbox(position_j, x_collision, y_collision)
    
    if boxes_overlap(bbox_i, bbox_j):
        collision_detected = True
```

**Integration**:
```python
def _check_feasibility(predicted_state, goal_predicates):
    goal_feasibility = check_goal_match(predicted_state, goal_predicates)
    collision_feasibility = collision_checker.check_scene_collisions(predicted_state)
    return goal_feasibility * collision_feasibility
```

### Closed-Loop Controller

**Purpose**: Orchestrate execution with real-time replanning

**Flow**:
```python
for primitive in plan:
    while not primitive_complete:
        # Get current state
        state = capture_state(env)
        
        # Plan action parameters
        action_params = dynamics_planner.plan_next_primitive(
            state, goal_predicates, remaining_plan
        )
        
        # Execute action
        success = primitive_executor.execute(primitive, action_params)
        
        # Check and replan if needed
        if not success:
            remaining_plan = llm_planner.replan(...)
```

---

## 5. Data Format Specification

### Episode Structure

```python
import pickle

# Save
with open('episode.pkl', 'wb') as f:
    pickle.dump((data_dict, attrs_dict), f)

# Load
with open('episode.pkl', 'rb') as f:
    data_dict, attrs_dict = pickle.load(f)
```

### data_dict (32 keys)

#### Robot State (8 keys)

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `joint_position` | `(T, 7)` | float64 | Joint positions over time |
| `joint_velocity` | `(T, 7)` | float64 | Joint velocities |
| `joint_torque` | `(T, 7)` | float64 | Joint torques |
| `target_joint_position` | `(T, 7)` | float64 | Target joint positions |
| `target_ee_discrete` | `(T, 3)` | float64 | Discrete EE targets |
| `ee_position` | `(T, 3)` | float64 | End-effector position |
| `ee_orientation` | `(T, 4)` | float32 | End-effector quaternion |
| `ee_velocity` | `(T, 3)` | float64 | End-effector velocity |

#### Camera Data (5 keys - Placeholders)

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `rgb` | `(T, 480, 640, 3)` | uint8 | RGB images |
| `depth` | `(T, 480, 640)` | float32 | Depth maps |
| `segmentation` | `(T, 480, 640)` | int32 | Segmentation masks |
| `projection_matrix` | `(T, 4, 4)` | float64 | Camera projection |
| `view_matrix` | `(T, 4, 4)` | float64 | Camera view transform |

#### Object State (1 key, nested dict)

```python
'objects': {
    'block_01': {
        'position': (T, 3),      # float64
        'orientation': (T, 4)    # float64
    },
    'block_02': {...},
    ...
}
```

#### Point Clouds (3 keys per object)

For each object N:
| Key | Shape | Type |
|-----|-------|------|
| `point_cloud_N` | `(T, 128, 3)` | float64 |
| `point_cloud_Nsampling` | `(T, 128, 3)` | float64 |
| `point_cloud_Nsampling_noise` | `(T, 128, 3)` | float64 |

#### Metadata (3 keys)

| Key | Type | Description |
|-----|------|-------------|
| `contact` | list | Contact events per timestep |
| `behavior` | list | Behavior labels (grasp/release/none) |
| `hidden_label` | `(T, N)` int64 | Occlusion flags |

### attrs_dict (9 keys)

#### Robot Metadata (4 keys)

| Key | Type | Example |
|-----|------|---------|
| `robot_joint_names` | list[str] | `['robot0_joint1', ..., 'robot0_joint7']` |
| `robot_link_names` | list[str] | `['panda_link0', ..., 'panda_hand']` |
| `n_arm_joints` | int | `7` |
| `n_ee_joints` | int | `2` (or 0 for Panda gripper) |

#### Segmentation (2 keys)

```python
'segmentation_labels': {
    'block_01': 'table',
    'block_02': 'cubeA_main',
    'block_03': 'cubeB_main'
}

'segmentation_ids': {
    'block_01': 0,
    'block_02': 1,
    'block_03': 2
}
```

#### Episode Metadata (3 keys)

| Key | Type | Description |
|-----|------|-------------|
| `objects` | dict | Object properties (extents, mass, static) |
| `sudo_action_list` | list | Action history |
| `behavior_params` | dict | Behavior parameters |

### Common Access Patterns

```python
import pickle

with open('episode.pkl', 'rb') as f:
    data_dict, attrs_dict = pickle.load(f)

# Get dimensions
T = len(data_dict['joint_position'])  # Number of timesteps
N = len(data_dict['objects'])          # Number of objects

# Robot state at timestep t
joint_pos_t = data_dict['joint_position'][t]    # (7,)
ee_pos_t = data_dict['ee_position'][t]          # (3,)

# Object trajectories
obj_pos = data_dict['objects']['block_02']['position']      # (T, 3)
obj_quat = data_dict['objects']['block_02']['orientation']  # (T, 4)

# Point cloud at timestep t
pc_t = data_dict['point_cloud_1'][t]              # (128, 3)

# Actions
actions = attrs_dict['sudo_action_list']          # List of action dicts
```

---

## 6. Configuration Reference

### Environment Options

| Environment | Cubes | Stacking Sequence | Timesteps |
|-------------|-------|-------------------|-----------|
| `Stack` | 2 | A → B | 250-350 |
| `Stack3` | 3 | A → B, C → A | 350-450 |
| `Stack4` | 4 | A → B, C → A, D → C | 400-500 |

### Batch Collection Parameters

```bash
mjpython batch_collect.py \
    --env Stack4 \              # Environment (Stack, Stack3, Stack4)
    --num-episodes 100 \        # Number of episodes
    --output-dir ./dataset \    # Output directory
    --max-timesteps 1000 \      # Max steps per episode
    --max-retries 3 \           # Retry attempts on failure
    --num-points 128 \          # Points per object cloud
    --cameras frontview agentview robot0_eye_in_hand  # Camera names
```

### Point Cloud Settings

| Setting | Points | Speed | File Size |
|---------|--------|-------|-----------|
| Low | 64 | Fast | Small |
| Medium (default) | 128 | Balanced | Medium |
| High | 256 | Slow | Large |

### Dynamics Planner Parameters

```python
DynamicsModelPlanner(
    checkpoint_path="path/to/cp_1.pth",  # Required
    lookahead_depth=2,                    # 1-3 (default: 2)
    enable_collision_checking=True,       # Default: True
    num_samples=50,                       # Rejection sampling count
    x_collision=0.05,                     # Collision box X (meters)
    y_collision=0.05,                     # Collision box Y (meters)
    movement_idx=None,                    # Movement type index
    num_objects=3                         # Number of manipulable objects
)
```

### LLM Planner Parameters

```python
LLMTaskPlanner(
    model_config_path=None,               # Optional model config
    prompt_config_path="path/to/task.yaml",  # YAML prompt config
    api_key=None,                         # OpenAI API key (or env var)
    use_examples=True                     # Enable few-shot learning
)
```

---

## 7. Command Reference

### Data Collection

```bash
# Quick test (5 episodes)
mjpython batch_collect.py --env Stack --num-episodes 5 --output-dir ./test

# Development dataset (100 episodes)
mjpython batch_collect.py --env Stack --num-episodes 100 --output-dir ./dev_dataset

# Production dataset (1000 episodes, Stack4)
mjpython batch_collect.py --env Stack4 --num-episodes 1000 --output-dir ./production
```

### Dataset Validation

```bash
# Validate all episodes
mjpython inspect_dataset.py ./dataset --validate

# Compute statistics
mjpython inspect_dataset.py ./dataset --stats

# Visualize statistics
mjpython inspect_dataset.py ./dataset --visualize --save-viz stats.png

# Inspect specific episode
mjpython inspect_dataset.py ./dataset --inspect 0

# Full inspection
mjpython inspect_dataset.py ./dataset --validate --stats --visualize
```

### Format Verification

```bash
# Static format check (no simulation)
python data_capture/verify_format_alignment.py

# Inspect saved episode
python data_capture/verify_saved_format.py path/to/episode.pkl

# Test recording pipeline
mjpython data_capture/episode_recorder.py
```

### Planning Demo

```bash
# Run closed-loop planning with defaults
mjpython demo_phase3.py --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_1.pth

# With lookahead depth
mjpython demo_phase3.py --checkpoint path/to/cp_1.pth --lookahead-depth 2

# Compare planning strategies
mjpython demo_phase3.py --lookahead-depth 1  # Greedy
mjpython demo_phase3.py --lookahead-depth 2  # 2-step (recommended)
mjpython demo_phase3.py --lookahead-depth 3  # 3-step
```

### Headless Execution

```bash
# For servers without display
xvfb-run -a mjpython demo_phase3.py --lookahead-depth 2
```

---

## 8. Implementation Details

### File Structure

```
robosuite/
├── data_capture/
│   ├── batch_collect.py         # Batch collection (380 lines)
│   ├── episode_recorder.py      # Episode recording (385 lines)
│   ├── state_capture.py         # State extraction (260 lines)
│   ├── data_formatter.py        # Format conversion (200 lines)
│   ├── metadata_extractor.py    # Metadata extraction (180 lines)
│   ├── inspect_dataset.py       # QA tools (450 lines)
│   ├── verify_format_alignment.py
│   ├── verify_saved_format.py
│   └── dataset/                 # Collected data
│
├── planning/
│   ├── llm_task_planner.py      # LLM planning
│   ├── dynamics_model_planner.py # Dynamics planning
│   ├── closed_loop_controller.py # Execution control
│   ├── primitive_executor.py     # Action execution
│   ├── collision_checker.py      # Collision detection (380 lines)
│   ├── state_converter.py        # State conversion
│   ├── demo_phase2.py           # LLM demo
│   ├── demo_phase3.py           # Full pipeline demo
│   ├── configs/                 # YAML configurations
│   └── prompts/                 # Prompt templates
│
└── run_stack.py                 # Heuristic policy
```

### Key Dependencies

**Required**:
- `robosuite` - MuJoCo simulation
- `numpy` - Array operations
- `torch` - PyTorch for dynamics model
- `pickle` - Serialization

**Optional**:
- `open3d` - Point cloud generation
- `openai` - GPT-4 API access
- `matplotlib` - Visualization

### Integration Points

**With HeuristicStackPolicy (run_stack.py)**:
```python
from run_stack import HeuristicStackPolicy, create_environment

env = create_environment("Stack3")
policy = HeuristicStackPolicy(env)
recorder = EpisodeRecorder(env, key_timesteps_only=True)

obs = env.reset()
recorder.start_episode()

while not done:
    action, _ = policy.step()
    obs, reward, done, info = env.step(action)
    recorder.record_step(action, obs)
    policy.obs = obs

data_dict, attrs_dict = recorder.end_episode()
recorder.save_episode("./episodes")
```

**With Points2Plans Dynamics Model**:
```python
from planning.dynamics_model_planner import DynamicsModelPlanner

planner = DynamicsModelPlanner(
    checkpoint_path="Points2Plans/ckpt/checkpoint/cp_1.pth",
    lookahead_depth=2,
    enable_collision_checking=True
)

primitive, params, feasibility = planner.plan_next_primitive(
    state_dict=current_state,
    goal_predicates=goal_tensor,
    primitive_plan=["Pick(cubeA, table)", "Place(cubeA, cubeB)"]
)
```

---

## 9. Testing & Verification

### Unit Tests

```bash
# Test collision detection
python planning/test_collision_integration.py

# Test lookahead
python planning/test_lookahead.py
```

### Format Verification Checklist

- ✅ 32 data_dict keys present
- ✅ 9 attrs_dict keys present
- ✅ Robot state arrays: correct shapes (T, 7), (T, 3), (T, 4)
- ✅ Object states as numpy arrays (T, 3), (T, 4)
- ✅ Point clouds: (T, 128, 3) per object
- ✅ Metadata lists match timestep count

### Test Results Summary

**Batch Collection (3 episodes)**:
```
✅ Total Episodes: 3
✅ Successful: 3 (100% success rate)
✅ Failed: 0
✅ Total Timesteps: 885
✅ Avg Timesteps/Episode: 295.0
✅ Dataset Size: 23.7 MB (7.9 MB/episode)
```

**Collision Detection**:
```
✓ Non-colliding objects (0.2m separation)
✓ Colliding objects (0.03m separation)
✓ Vertical separation (stacked objects)
```

**Multi-Step Lookahead**:
```
✓ Depth parameter validation (1-3)
✓ Automatic clamping (>3 → 3)
✓ Forward sequence simulation
✓ Backward compatibility (depth=1)
```

---

## 10. Troubleshooting Guide

### Critical Implementation Details

> ⚠️ **IMPORTANT**: These are critical implementation details discovered during integration. Getting these wrong will cause the model to fail silently.

#### Predicate Index Mapping (Points2Plans Training Format)

The predicate indices in the training data follow this **specific order** (defined in `Points2Plans/relational_dynamics/dataloader/dataloader.py`):

| Index | Predicate | Description |
|-------|-----------|-------------|
| 0 | Left | A is to the left of B |
| 1 | Right | A is to the right of B |
| 2 | Below | A is below B |
| 3 | Above | A is above B |
| 4 | Front | A is in front of B |
| 5 | Behind | A is behind B |
| **6** | **On/Contact** | **A is on/touching B (most common for stacking)** |
| 7 | Boundary | A is near boundary of B |
| **8** | **Inside** | **A is inside B (for container tasks)** |

**Common Mistake**: Assuming `On` is at index 0 and `Inside` is at index 1. This causes the model to output near-zero confidences for predicates like `On(cubeA, table)`.

**Affected Files**:
- `planning/closed_loop_controller.py` - `_predicates_to_strings()` function
- `planning/llm_task_planner.py` - `_predicate_type_to_idx()` function

```python
# CORRECT mapping in llm_task_planner.py
predicate_map = {
    'left': 0, 'right': 1, 'below': 2, 'above': 3,
    'front': 4, 'behind': 5,
    'on': 6,      # ← NOT 0!
    'stacked': 6, # ← Same as 'on'
    'boundary': 7,
    'inside': 8,  # ← NOT 1!
}
```

#### MuJoCo Segmentation Channel Order

MuJoCo's segmentation rendering returns a 2-channel image `[H, W, 2]`:

| Channel | Contents | Values |
|---------|----------|--------|
| **Channel 0** | **Geom ID** | Unique ID per geometry (0, 1, 2, ..., 91, etc.) |
| Channel 1 | Geom Type | Type category (usually 5 for visual geoms) |

**Common Mistake**: Assuming channel 0 is "geom type" and channel 1 is "geom ID". This causes all objects except the table to have zero point clouds.

**Affected File**: `robosuite/utils/camera_utils.py` - `render_camera()` function

```python
# CORRECT channel selection
seg = seg[..., 0]  # ← Channel 0 contains geom IDs
```

#### Object Name Formats (and matching)

Two naming schemes appear together, so name matching must be permissive:

| Component | Name Format | Example |
|-----------|-------------|---------|
| MuJoCo bodies / metadata | `{name}_main` | `cubeA_main`, `cubeB_main` |
| PointCloudGenerator output (segmentation) | `{name}` | `cubeA`, `cubeB` |
| Training data objects | `block_N` | `block_1`, `block_2` |

**Current approach (works):** do *not* filter the segmented clouds by object_names, then map segmented names to metadata keys with lowercase + substring matching (e.g., `cubeA` ↔ `cubeA_main`). This prevents valid cube clouds from being dropped.

**Common Mistake**: Filtering `generate_segmented(..., object_names=[metadata keys])` when metadata uses `_main` while segmentation uses clean names → cubes get discarded and only the table remains.

**Affected File**: `data_capture/episode_recorder.py` - `_capture_point_clouds()` and `_map_segmented_point_clouds()`

#### Contact Capture (MuJoCo Simulation Reference)

When capturing contacts from MuJoCo, two critical steps are required:

1. **Use fresh `env.sim` reference**: After `env.reset()`, any cached `sim` reference becomes stale. Always access `self.env.sim` directly in `capture_contacts()`.

2. **Call `sim.forward()` before reading contacts**: The contact buffer (`sim.data.ncon`) is only populated after a physics forward pass. Without this, `ncon=0` even when objects are in contact.

```python
# CORRECT contact capture pattern
def capture_contacts(self):
    sim = self.env.sim          # ← Fresh reference, not cached self.sim
    sim.forward()               # ← Required to populate contact buffer
    
    for contact_id in range(sim.data.ncon):
        contact = sim.data.contact[contact_id]
        # ... process contact
```

**Common Mistake**: Caching `self.sim = env.sim` during `__init__` and reusing it after `env.reset()`. This causes `ncon=0` because the cached reference points to stale simulation data.

**Affected File**: `data_capture/state_capture.py` - `capture_contacts()` function

---

### Data Collection Issues

**Point clouds all zeros**:
- Cause 1: `open3d` not installed → Solution: `pip install open3d`
- Cause 2: Wrong segmentation channel in `camera_utils.py` → Solution: Use `seg[..., 0]` (channel 0 = geom ID)
- Cause 3: Object name mismatch between StateConverter and PointCloudGenerator → Solution: Use clean names (e.g., `cubeA` not `cubeA_main`)

**Only table has point clouds, cubes are zeros**:
- Cause 1: Segmentation mask channel misuse → Fix: `seg = seg[..., 0]` (geom ID channel)
- Cause 2: Object name filter too strict (metadata uses `_main`, segmentation returns clean names) → Fix: do not filter `generate_segmented` by `object_names`; rely on substring/lowercase mapping (`cubeA` ↔ `cubeA_main`).

**Episode too large**:
- Cause: Long episodes or high-res camera data
- Solution: Reduce `--max-timesteps` or `--num-points`

**Collection fails frequently**:
- Cause: Policy gets stuck
- Solution: Increase `--max-retries`

**Out of disk space**:
- Cause: Dataset too large
- Solution: Use `--num-points 64` for smaller files

### Planning Issues

**Predicate predictions all zeros or near-zero**:
- Cause 1: Model trained on data with empty point clouds → Solution: Re-collect training data with fixed segmentation channel
- Cause 2: Wrong predicate index mapping → Solution: Use index 6 for "On", index 8 for "Inside" (see Critical Implementation Details above)
- Cause 3: Hardcoded checkpoint path in `base_RD.py` → Solution: Remove hardcoded path in `load_checkpoint()` function

**LLM returns malformed response**:
- Cause: Missing few-shot examples
- Solution: Set `use_examples=True` in LLMTaskPlanner

**Dynamics planner times out**:
- Cause: Too many samples or deep lookahead
- Solution: Reduce `num_samples` or `lookahead_depth`

**Collision detection too strict**:
- Cause: Large collision boxes
- Solution: Reduce `x_collision` and `y_collision` values

### Environment Issues

**MuJoCo license error**:
- Solution: Install mujoco >= 2.1.0 (free license)

**Display error (headless server)**:
- Solution: Use `xvfb-run -a` prefix for commands

**Import errors**:
- Solution: Run from `robosuite/` directory

---

## 11. Implementation Roadmap

### Completed Phases

#### Data Collection Pipeline

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | State Capture (robot, objects, contacts) |
| Phase 2 | ✅ Complete | Point Cloud Integration |
| Phase 3 | ✅ Complete | Data Packaging (Points2Plans format) |
| Phase 4 | ✅ Complete | Batch Collection & QA |

#### Planning Alignment with Points2Plans

| Phase | Status | Description |
|-------|--------|-------------|
| LLM Planner | ✅ Complete | YAML-based prompts, few-shot learning |
| Phase 1 | ✅ Complete | Multi-Step Lookahead (2-3 primitives) |
| Phase 2 | ✅ Complete | Collision Detection (2D AABB) |
| Phase 3 | ⏸️ Deferred | Batch Template Evaluation (LLM approach preferred) |

### Future Enhancements (Optional)

1. **Adaptive lookahead depth** - Increase when stuck
2. **Beam search** - Keep top-K sequences
3. **3D collision checking** - For complex geometries
4. **Real camera capture** - Enable in data_formatter.py
5. **Custom predicates** - Add to system prompts
6. **Parallel collection** - Multiple instances

---

## Quick Reference Card

### Essential Commands

```bash
# Collect 100 episodes
mjpython batch_collect.py --env Stack4 --num-episodes 100

# Validate dataset
mjpython inspect_dataset.py ./dataset --validate --stats

# Run planning demo
mjpython demo_phase3.py --lookahead-depth 2

# Verify format
python data_capture/verify_saved_format.py episode.pkl
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--env` | Stack | Environment type |
| `--num-episodes` | 10 | Episodes to collect |
| `--num-points` | 128 | Points per cloud |
| `--lookahead-depth` | 2 | Planning lookahead |
| `--max-retries` | 3 | Retry on failure |

### Performance Expectations

| Environment | Episodes/Hour | Size/Episode |
|-------------|---------------|--------------|
| Stack | 9-12 | ~7-8 MB |
| Stack3 | 7-9 | ~7-8 MB |
| Stack4 | 6-8 | ~7-8 MB |

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| Nov 22, 2025 | 1.0 | Initial Phase 4 completion |
| Nov 24, 2025 | 1.1 | Format alignment update |
| Dec 22, 2025 | 2.0 | Planning alignment (lookahead + collision) |
| Dec 27, 2025 | 2.1 | Key timestep recording mode |
| Jan 9, 2026 | 3.0 | Consolidated reference document |
| Jan 13, 2026 | 3.1 | Added critical implementation details: predicate index mapping, segmentation channel order, object name formats |

---

**This document serves as the single source of truth for implementing and maintaining the Points2Plans robosuite integration.**
