# Critic Data Collection - Quick Reference

## 🚀 Quick Start Commands

### Test Installation (5 seconds)
```bash
cd /home/scratch1/cheze/verify2act/robosuite/planning
python test_data_collection.py
```

### Collect Test Dataset (10 episodes, ~5 min)
```bash
xvfb-run -a python collect_critic_data.py \
    --num-episodes 10 \
    --task Stack3 \
    --save-dir ./data/critic_test
```

### Collect Full Dataset (500 episodes, ~4-8 hours)
```bash
nohup xvfb-run -a python collect_critic_data.py \
    --num-episodes 500 \
    --task Stack3 \
    --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_99.pth \
    --save-dir ./data/critic_phase1 \
    --save-interval 50 \
    > collect_data.log 2>&1 &

# Monitor progress
tail -f collect_data.log
```

---

## 📊 Inspect Collected Data

```python
import pickle
import numpy as np

# Load dataset
with open('./data/critic_phase1/critic_data_final.pkl', 'rb') as f:
    data = pickle.load(f)

# Check statistics
print(f"Positive samples: {len(data['positive_samples'])}")
print(f"Negative samples: {len(data['negative_samples'])}")

# Check dimensions
sample = data['positive_samples'][0]
print(f"\nSample dimensions:")
print(f"  z_t:            {sample['z_t'].shape}")           # (256,)
print(f"  a_t:            {sample['a_t'].shape}")           # (64,)
print(f"  z_next:         {sample['z_next'].shape}")        # (256,)
print(f"  predicate_embed: {sample['predicate_embed'].shape}")  # (128,)
print(f"  plan_summary:   {sample['plan_summary'].shape}")  # (128,)

# Check labels
print(f"\nLabels:")
print(f"  label_predicate: {sample['label_predicate']}")
print(f"  label_feas:      {sample['label_feas']}")
print(f"  label_nonint:    {sample['label_nonint']}")
```

---

## 🔧 Command-Line Options

### collect_critic_data.py

```bash
--task              # Task name: Stack3, PickPlace, ClutteredNutAssembly
--checkpoint        # Path to dynamics model checkpoint
--num-episodes      # Number of episodes to collect
--save-dir          # Directory to save data
--save-interval     # Save every N episodes (default: 25)
--max-primitives    # Max primitives per episode (default: 20)
--verbose           # Enable verbose output
```

### Example: Different Tasks

```bash
# Stack3
xvfb-run -a python collect_critic_data.py --task Stack3 --num-episodes 200

# PickPlace
xvfb-run -a python collect_critic_data.py --task PickPlace --num-episodes 200

# ClutteredNutAssembly
xvfb-run -a python collect_critic_data.py --task ClutteredNutAssembly --num-episodes 200
```

---

## 📁 Output Files

Data is saved to `{save_dir}/`:

```
critic_data_ep25.pkl     # Checkpoint at episode 25
critic_data_ep50.pkl     # Checkpoint at episode 50
...
critic_data_final.pkl    # Final dataset with all episodes
```

Each `.pkl` file contains:
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

## 🐛 Common Issues

### Issue: "ModuleNotFoundError: No module named 'critic'"

**Fix**:
```bash
export PYTHONPATH=/home/scratch1/cheze/verify2act:$PYTHONPATH
```

### Issue: "Checkpoint not found"

**Fix**: Specify full path to checkpoint:
```bash
--checkpoint /home/scratch1/cheze/verify2act/Points2Plans/ckpt/checkpoint/cp_99.pth
```

### Issue: All samples labeled negative

**Fix**: Lower feasibility threshold (model might be undertrained):
```python
# In collect_critic_data.py, modify controller creation:
controller = ClosedLoopController(
    ...
    predicate_threshold=0.3,  # Lower from 0.5
)
```

### Issue: Dimension mismatch

**Fix**: Run test script to identify which component has wrong dimensions:
```bash
python test_data_collection.py
```

---

## 📈 Expected Results

After 500 episodes:

- **Episodes**: 500 total
- **Success rate**: 30-70% (varies by task/model)
- **Positive samples**: 2000-5000 (successful steps)
- **Negative samples**: 500-1500 (failed steps)
- **Hard negatives**: Auto-generated to balance
- **Total samples**: 3000-6000 (balanced 1:1)

Sample distribution by source:
```
Positive sources:
  successful_trajectory    2500
  pre_failure             800

Negative sources:
  failure_predicate       600
  hard_negative_predicate 800
  hard_negative_action    400
  hard_negative_noise     200
```

---

## ⏭️ Next Steps After Collection

### 1. Split Dataset (70/15/15)

```python
from verify2act.critic.critic_data_collector import split_dataset
import pickle

# Load full dataset
with open('./data/critic_phase1/critic_data_final.pkl', 'rb') as f:
    data = pickle.load(f)

all_samples = data['positive_samples'] + data['negative_samples']

# Split
train, val, test = split_dataset(all_samples, train_split=0.7, val_split=0.15, test_split=0.15)

# Save splits
with open('./data/critic_phase1/train.pkl', 'wb') as f:
    pickle.dump(train, f)
with open('./data/critic_phase1/val.pkl', 'wb') as f:
    pickle.dump(val, f)
with open('./data/critic_phase1/test.pkl', 'wb') as f:
    pickle.dump(test, f)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```

### 2. Train Critic (Phase 1)

```bash
cd /home/scratch1/cheze/verify2act/verify2act/critic

python train_critic.py \
    --data_path ../data/critic_phase1/train.pkl \
    --val_path ../data/critic_phase1/val.pkl \
    --use_predicate_head \
    --ensemble_size 5 \
    --num_epochs 100 \
    --save_dir ./checkpoints
```

### 3. Evaluate

```bash
python train_critic.py \
    --eval_only \
    --checkpoint ./checkpoints/best_model.pt \
    --test_path ../data/critic_phase1/test.pkl
```

### 4. Integrate with Planner

See `verified_planner.py` for integration example.

---

## 📚 Documentation

- **Implementation Plan**: [DATA_COLLECTION_PLAN.md](../verify2act/critic/DATA_COLLECTION_PLAN.md)
- **User Guide**: [DATA_COLLECTION_README.md](DATA_COLLECTION_README.md)
- **Implementation Summary**: [DATA_COLLECTION_IMPLEMENTATION_SUMMARY.md](../verify2act/critic/DATA_COLLECTION_IMPLEMENTATION_SUMMARY.md)
- **Critic Overview**: [IMPLEMENTATION_COMPLETE.txt](../verify2act/critic/IMPLEMENTATION_COMPLETE.txt)

---

**Ready to collect data!** 🎉

Start with: `xvfb-run -a python collect_critic_data.py --num-episodes 10`
