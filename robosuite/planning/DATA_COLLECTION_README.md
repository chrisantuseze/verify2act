# Critic Data Collection Implementation

**Status**: ✅ Complete and Ready for Testing  
**Date**: February 8, 2026

---

## 📦 What Was Implemented

### 1. **Embedding Extraction Utilities** ([critic_embedding_utils.py](critic_embedding_utils.py))

Extracts consistent embeddings from the dynamics model for critic training:

- **`EmbeddingExtractor` class**:
  - `extract_action_embedding()`: Extracts 64-dim action embeddings (discrete + continuous)
  - `extract_predicate_embedding()`: Extracts 128-dim predicate embeddings from goal predicates
  - `extract_plan_summary()`: Extracts 128-dim plan summary from remaining primitives
  
**Key Features**:
- Matches actual embeddings fed to dynamics model
- Handles dimension mismatches via padding/truncation
- Simple strategy for Phase 1 (can add learned projections later)

### 2. **Data Collector Wrapper** ([dynamics_model_data_collector.py](dynamics_model_data_collector.py))

Wraps `CriticDataCollector` with dynamics-model-specific logic:

- **`DynamicsModelDataCollector` class**:
  - `start_episode()`: Initialize episode buffer
  - `record_step()`: Record each planning step
  - `end_episode()`: Process trajectory and label success/failure
  - `save_dataset()`: Balance and save collected data
  
**Key Features**:
- Episode-level buffering
- Automatic embedding extraction at episode end
- Success/failure labeling
- Hard negative generation
- Periodic saving

### 3. **Integration with Closed-Loop Controller** ([closed_loop_controller.py](closed_loop_controller.py))

Added data collection hooks into the existing planning pipeline:

**Modifications**:
- Added `self.data_collector` attribute (optional)
- Call `start_episode()` when episode begins
- Call `record_step()` when each primitive is planned
- Call `end_episode()` when episode completes
- Added `_parse_primitive_for_collection()` helper method

### 4. **Standalone Collection Script** ([collect_critic_data.py](collect_critic_data.py))

Dedicated script for running data collection:

```bash
# Collect 50 episodes on Stack3
xvfb-run -a python collect_critic_data.py --num-episodes 50 --task Stack3

# With custom checkpoint
xvfb-run -a python collect_critic_data.py \
    --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_99.pth \
    --num-episodes 100 \
    --save-interval 25
```

**Features**:
- Runs multiple episodes automatically
- Periodic saving (every N episodes)
- Progress reporting
- Statistics summary

### 5. **Command-Line Flags for demo_phase3.py**

Added optional data collection to existing demo script:

```bash
# Enable data collection
xvfb-run -a python demo_phase3.py \
    --collect-data \
    --data-save-dir ./data/critic \
    --num-episodes 10
```

New arguments:
- `--collect-data`: Enable data collection
- `--data-save-dir`: Directory to save collected data

### 6. **Test Script** ([test_data_collection.py](test_data_collection.py))

Validates all components work together:

```bash
python test_data_collection.py
```

Tests:
- Embedding extraction
- Data collector
- Saving/loading datasets

---

## 🚀 Quick Start

### Step 1: Verify Installation

```bash
cd /home/scratch1/cheze/verify2act/robosuite/planning
python test_data_collection.py
```

Expected output: "ALL TESTS PASSED! ✓"

### Step 2: Collect Small Test Dataset (10 episodes)

```bash
xvfb-run -a python collect_critic_data.py \
    --num-episodes 10 \
    --task Stack3 \
    --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_99.pth \
    --save-dir ./data/critic_test
```

### Step 3: Inspect Collected Data

```python
import pickle
with open('./data/critic_test/critic_data_final.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Positive samples: {len(data['positive_samples'])}")
print(f"Negative samples: {len(data['negative_samples'])}")

# Check dimensions
sample = data['positive_samples'][0]
print(f"z_t shape: {sample['z_t'].shape}")  # Should be (256,)
print(f"a_t shape: {sample['a_t'].shape}")  # Should be (64,)
print(f"z_next shape: {sample['z_next'].shape}")  # Should be (256,)
print(f"predicate_embed shape: {sample['predicate_embed'].shape}")  # Should be (128,)
print(f"plan_summary shape: {sample['plan_summary'].shape}")  # Should be (128,)
```

### Step 4: Collect Full Dataset (500-1000 episodes)

```bash
# Run overnight
nohup xvfb-run -a python collect_critic_data.py \
    --num-episodes 500 \
    --task Stack3 \
    --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_99.pth \
    --save-dir ./data/critic_phase1 \
    --save-interval 50 \
    > collect_data.log 2>&1 &
```

---

## 📊 Data Format

Each sample in the dataset contains:

```python
{
    'z_t': np.ndarray,              # State embedding (256,)
    'a_t': np.ndarray,              # Action embedding (64,)
    'z_next': np.ndarray,           # Next state embedding (256,)
    'predicate_embed': np.ndarray,  # Target predicate embedding (128,)
    'plan_summary': np.ndarray,     # Remaining plan embedding (128,)
    'label_predicate': int,         # 0 or 1 (Phase 1 only)
    'label_feas': int,              # 0 or 1 (unused in Phase 1)
    'label_nonint': int,            # 0 or 1 (unused in Phase 1)
    'source': str,                  # "successful_trajectory", "failure_predicate", etc.
    'step_idx': int,                # Step index in episode
}
```

Dataset structure:
```python
{
    'positive_samples': List[Dict],  # Successful steps
    'negative_samples': List[Dict],  # Failed steps
    'config': {
        'latent_dim': 256,
        'action_dim': 64,
        'predicate_embed_dim': 128,
        'plan_summary_dim': 128,
    }
}
```

---

## 🔧 Implementation Details

### Action Embedding (64-dim)

**What it captures**:
- Discrete action: One-hot encoding of which object to move
- Continuous action: Embedding of (dx, dy) movement

**Source**: Extracted from `_simulate_one_step()` in dynamics_model_planner.py

### Predicate Embedding (128-dim)

**What it captures**:
- Target predicate relationship for (object, target) pair
- Extracted from goal_predicates[obj_id, target_id, :]

**Strategy**: 
- Extract 9-dim predicate vector
- Pad/project to 128-dim (simple for Phase 1)
- Can add learned projection later

### Plan Summary (128-dim)

**What it captures**:
- Number of remaining primitives
- Current position in plan
- Primitive type distribution
- Next primitive type

**Strategy**:
- Hand-crafted features for Phase 1
- Can use sentence embeddings (BERT) later

### Ground Truth Labels

**Phase 1 Strategy** (Bootstrap):
- Use `_check_feasibility()` output as pseudo-label
- Label predicate head based on feasibility score:
  - score ≥ 0.5 → label = 1 (predicate satisfied)
  - score < 0.5 → label = 0 (predicate violated)

---

## 📝 Next Steps

### Immediate (Testing)

1. ✅ Run test script to verify components work
2. ⏳ Collect 10-episode test dataset
3. ⏳ Inspect data dimensions and labels
4. ⏳ Fix any bugs in extraction logic

### Short-term (Data Collection)

1. ⏳ Collect 500-1000 episodes on Stack3
2. ⏳ Balance dataset (hard negatives)
3. ⏳ Split into train/val/test (70/15/15)
4. ⏳ Save final dataset

### Medium-term (Training)

1. ⏳ Train Phase 1 critic (predicate head only)
2. ⏳ Evaluate on validation set
3. ⏳ Calibrate thresholds
4. ⏳ Integrate trained critic into planning loop

---

## 🐛 Troubleshooting

### Issue: Dimension mismatch errors

**Solution**: Check that embeddings match expected dimensions:
- Action: 64-dim
- Predicate: 128-dim
- Plan summary: 128-dim
- State latent: 256-dim

Run `test_data_collection.py` to verify.

### Issue: Not collecting any data

**Solution**: Ensure data collector is attached:
```python
controller.planner.data_collector = data_collector
```

Or use `collect_critic_data.py` which handles this automatically.

### Issue: All samples labeled as negative

**Solution**: Check `feasibility_threshold` in dynamics planner. If model is undertrained, lower threshold (e.g., 0.3 instead of 0.5).

---

## 📚 Files Created/Modified

**New Files**:
- `verify2act/critic/DATA_COLLECTION_PLAN.md` - Implementation plan
- `robosuite/planning/critic_embedding_utils.py` - Embedding extractors
- `robosuite/planning/dynamics_model_data_collector.py` - Data collector wrapper
- `robosuite/planning/collect_critic_data.py` - Standalone collection script
- `robosuite/planning/test_data_collection.py` - Test script
- `robosuite/planning/DATA_COLLECTION_README.md` - This file

**Modified Files**:
- `robosuite/planning/closed_loop_controller.py` - Added data collection hooks
- `robosuite/planning/demo_phase3.py` - Added --collect-data flag

---

## 📊 Expected Results

After collecting 500 episodes:
- ~2000-5000 positive samples (successful steps)
- ~500-1500 negative samples (failed steps)
- With hard negatives: ~3000-6000 total samples
- Balanced dataset (1:1 ratio)

Success rate: 30-70% depending on task and model quality

---

## 🎓 Citation

Based on:
- Verify2Act Critic Implementation Plan
- Points2Plans Relational Dynamics Model
- PETS/MBPO Ensemble-based Model Learning

---

**Status**: Ready for testing and data collection! 🚀
