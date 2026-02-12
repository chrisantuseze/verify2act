# Critic Data Collection Implementation Plan

**Date**: February 8, 2026  
**Status**: Implementation Phase  
**Goal**: Hook `CriticDataCollector` into Points2Plans dynamics model for training data generation

---

## Overview

This plan details how to integrate the critic data collection system with the existing Points2Plans closed-loop planning pipeline to generate training samples for the critic model.

---

## Architecture

```
Episode Execution Loop
    ↓
DynamicsModelPlanner.plan_next_primitive()
    ↓
[HOOK] Capture: z_t, a_t, z_next, metadata
    ↓
DynamicsModelDataCollector (wrapper)
    ↓
Extract embeddings:
  - Action embedding (from dynamics model internals)
  - Predicate embedding (from goal predicates)
  - Plan summary (from remaining primitives)
    ↓
CriticDataCollector.add_*_trajectory()
    ↓
Save to disk (pickle)
```

---

## Phase 1: Foundation (This Implementation)

### 1.1 Create Embedding Extractors

**File**: `robosuite/planning/critic_embedding_utils.py`

Utility functions to extract consistent embeddings from dynamics model:

- `extract_action_embedding(dynamics_model, action_params, obj_id, target_id, state_dict)` → [64-dim]
  - Concatenate discrete action + continuous action embeddings
  - Match what's fed to the graph dynamics model
  
- `extract_predicate_embedding(goal_predicates, obj_id, target_id)` → [128-dim]
  - Extract target predicate vector for (obj, target) pair
  - Project to fixed dimension via simple MLP or flatten+pad
  
- `extract_plan_summary(primitive_plan, current_step)` → [128-dim]
  - Encode remaining primitives as feature vector
  - Simple encoding: [num_remaining, primitive_type_one_hot, ...]

**Dimensions**: Measure actual dimensions from dynamics model, then define projection layers.

### 1.2 Create Wrapper Class

**File**: `robosuite/planning/dynamics_model_data_collector.py`

```python
class DynamicsModelDataCollector:
    """Wraps CriticDataCollector with dynamics-model-specific logic."""
    
    def __init__(self, dynamics_planner, save_dir)
    
    def start_episode(self, goal_predicates, primitive_plan)
        # Initialize episode buffer
    
    def record_step(self, state_dict, action_params, obj_id, target_id, next_state_dict)
        # Extract embeddings and append to buffer
    
    def end_episode(self, success: bool, failure_step: int = None)
        # Add trajectory to CriticDataCollector
        # Apply labeling based on success/failure
    
    def save_dataset(self)
        # Call CriticDataCollector.balance_dataset() and save
```

**Key Logic**:
- Maintains per-episode buffer of raw data
- On episode end, extracts all embeddings in batch
- Handles success/failure labeling automatically
- Manages hard negative generation

### 1.3 Modify DynamicsModelPlanner

**File**: `robosuite/planning/dynamics_model_planner.py`

Add optional data collection mode:

```python
class DynamicsModelPlanner:
    def __init__(self, ..., data_collector=None):
        self.data_collector = data_collector  # Optional
    
    def plan_next_primitive(self, ...):
        # Existing logic
        ...
        
        # [NEW] Record step if collector enabled
        if self.data_collector is not None:
            self.data_collector.record_step(
                state_dict=state_dict,
                action_params=best_params,
                obj_id=obj_id,
                target_id=target_id,
                next_state_dict=predicted_next_state
            )
        
        return best_action, best_params, best_feasibility, failure_analysis
```

### 1.4 Create Collection Script

**File**: `robosuite/planning/collect_critic_data.py`

Standalone script to run episodes and collect data:

```python
# Parse args: --num_episodes, --save_dir, --checkpoint, --task
# Initialize environment, planner, collector
# Run episodes in loop:
#   - Reset environment
#   - Get LLM plan
#   - Execute primitives with data collection
#   - Label success/failure
#   - Save periodically (every 50 episodes)
```

### 1.5 Add Flag to demo_phase3.py

**File**: `robosuite/planning/demo_phase3.py`

Add `--collect_data` and `--data_save_dir` arguments:

```python
if args.collect_data:
    collector = DynamicsModelDataCollector(
        dynamics_planner=planner,
        save_dir=args.data_save_dir
    )
    planner.data_collector = collector
```

---

## Phase 2: Validation (After Phase 1)

1. **Run 10 test episodes**: 5 success, 5 failure
2. **Inspect collected data**:
   - Check dimensions match critic expectations
   - Verify labels are correct
   - Visualize embedding distributions
3. **Fix bugs** in extraction logic

---

## Phase 3: Scale-Up (After Validation)

1. **Collect 500-1000 episodes**:
   - Multiple tasks (stacking, sorting, placement)
   - Various difficulty levels
   - Different object configurations
2. **Balance dataset**:
   - Generate hard negatives
   - Target 1:1 pos/neg ratio
3. **Split dataset**: 70% train, 15% val, 15% test
4. **Save final dataset**: `data/critic_phase1.pkl`

---

## Implementation Details

### Action Embedding (64-dim)

**Source**: `_simulate_one_step()` in dynamics_model_planner.py, lines 585-600

```python
# What we need to capture:
discrete_action = classif_model.one_hot_encoding_embed([obj_id])  # Shape: [1, D]
continuous_action = classif_model.continuous_action_emb(action_params[:2])  # Shape: [1, C]
action_embedding = torch.cat([discrete_action, continuous_action], dim=-1)  # [1, D+C]
```

**Strategy**: Add method to extract this concatenated embedding before forward pass.

### Predicate Embedding (128-dim)

**Source**: `goal_predicates` [N, N, 9] tensor

**Strategy**: 
```python
# For primitive "Place(A, B)", extract goal_predicates[A, B, :]
target_predicate_vector = goal_predicates[obj_id, target_id, :]  # [9]
# Project to 128-dim via learned MLP or pad/repeat
predicate_embed = simple_projection(target_predicate_vector)  # [128]
```

**Initial implementation**: Repeat/tile to 128-dim (simple, no learning)
**Future**: Add learned projection layer trained alongside critic

### Plan Summary (128-dim)

**Source**: `primitive_plan` list of strings

**Strategy**:
```python
# Encode as feature vector
features = [
    len(primitive_plan),  # Number of remaining primitives
    current_step_index,   # Position in plan
    *encode_primitive_types(primitive_plan),  # One-hot for primitive types
]
plan_summary = pad_or_project(features, target_dim=128)
```

**Initial implementation**: Hand-crafted features
**Future**: Use sentence embeddings (BERT) for richer representation

### Ground Truth Labels

**Phase 1 Strategy** (Bootstrap):
- Use `_check_feasibility()` output as pseudo-label
- Label predicate head based on feasibility score:
  - score ≥ 0.5 → label = 1 (predicate satisfied)
  - score < 0.5 → label = 0 (predicate violated)
- Set feasibility and non-interference labels to 1 (not used in Phase 1)

**Phase 2 Strategy** (Refinement):
- Execute subset of trajectories in real sim
- Verify predicate satisfaction with ground truth state
- Use for validation set labeling

**Phase 3 Strategy** (Full training):
- Add feasibility labels (collision detection)
- Add non-interference labels (object stability)

### Data Collection Frequency

**Option A: Every sampled candidate** (inside rejection sampling loop)
- Pros: Lots of data (50 candidates per primitive)
- Cons: Mostly negative samples, biased distribution

**Option B: Only executed actions** (after best action selected)
- Pros: Balanced distribution, less storage
- Cons: Fewer samples per episode

**Recommendation**: **Option B** for Phase 1
- Collect only the actions that are actually executed
- Generate hard negatives via augmentation (as designed in CriticDataCollector)

---

## File Checklist

- [ ] `verify2act/critic/DATA_COLLECTION_PLAN.md` (this file)
- [ ] `robosuite/planning/critic_embedding_utils.py` (new)
- [ ] `robosuite/planning/dynamics_model_data_collector.py` (new)
- [ ] `robosuite/planning/dynamics_model_planner.py` (modify)
- [ ] `robosuite/planning/demo_phase3.py` (modify)
- [ ] `robosuite/planning/collect_critic_data.py` (new)

---

## Success Criteria

After implementation:
1. ✅ Can run `collect_critic_data.py --num_episodes 10`
2. ✅ Generates `data/critic_samples.pkl` with expected format
3. ✅ All embeddings have correct dimensions (64, 128, 128)
4. ✅ Labels match success/failure outcomes
5. ✅ Can load data and pass to `CriticDataCollector`

---

## Timeline

- **Phase 1 Implementation**: 2-3 hours
- **Phase 2 Validation**: 1 hour
- **Phase 3 Scale-up**: Run overnight (4-8 hours compute)

**Total**: ~1 day of dev + overnight compute

---

## Next Steps After Data Collection

1. Train Phase 1 critic (predicate head only)
2. Evaluate on validation set
3. Calibrate thresholds
4. Integrate trained critic into planning loop
5. Measure improvement in plan success rate

---

## Notes

- Start with simple embedding strategies (pad/repeat)
- Can add learned projection layers later
- Focus on correct dimensions and data flow first
- Validate on small dataset before scaling up
