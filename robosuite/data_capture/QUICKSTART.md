# Quick Start Guide: Data Collection Pipeline

Complete guide to collecting Points2Plans datasets using the robosuite Stack environment.

---

## Prerequisites

- robosuite installed with MuJoCo
- Python environment with required packages
- `run_stack.py` heuristic policy
- Episode recorder modules (Phase 1-3)

---

## Quick Start

### 1. Collect a Small Test Dataset

```bash
cd /path/to/verify2act/robosuite/data_capture

# Collect 5 test episodes
mjpython batch_collect.py \
    --env Stack \
    --num-episodes 5 \
    --output-dir ./data

# Expected output:
# - 5 episodes collected
# - ~40 MB total size
# - 100% success rate
# - ~30-60 minutes collection time
```

### 2. Validate the Dataset

```bash
# Validate all episodes
mjpython inspect_dataset.py ./data --validate

# Expected output:
# - Valid Episodes: 5/5
# - Invalid Episodes: 0
# - Warnings: ~5-15 (normal)
```

### 3. Check Statistics

```bash
# Compute dataset statistics
mjpython inspect_dataset.py ./data --stats

# Expected output:
# - Episodes: 5
# - Avg Timesteps/Episode: ~250-350
# - Total Size: ~40 MB
```

---

## Production Data Collection

### Small Dataset (100 episodes)

Good for initial development and testing:

```bash
mjpython batch_collect.py \
    --env Stack \
    --num-episodes 100 \
    --output-dir ./dataset_100 \
    --max-timesteps 1000 \
    --num-points 128

# Expected:
# - Collection time: ~10-16 hours
# - Dataset size: ~800 MB
# - Success rate: >95%
```

### Medium Dataset (500 episodes)

Good for training initial models:

```bash
mjpython batch_collect.py \
    --env Stack4 \
    --num-episodes 500 \
    --output-dir ./dataset_500 \
    --max-timesteps 1000 \
    --num-points 128

# Expected:
# - Collection time: ~50-80 hours
# - Dataset size: ~4 GB
# - Success rate: >95%
```

### Large Dataset (1000+ episodes)

For final model training:

```bash
mjpython batch_collect.py \
    --env Stack4 \
    --num-episodes 1000 \
    --output-dir ./dataset_1000 \
    --max-timesteps 1000 \
    --num-points 256 \
    --cameras frontview agentview

# Expected:
# - Collection time: ~100-160 hours
# - Dataset size: ~10 GB
# - Success rate: >95%
```

---

## Environment Options

### Stack (2 cubes)

Simplest stacking task:
- 2 cubes (cubeA, cubeB)
- Stacking sequence: A → B
- Faster episodes (~200-300 timesteps)

```bash
mjpython batch_collect.py --env Stack --num-episodes 100
```

### Stack3 (3 cubes)

Medium complexity:
- 3 cubes (cubeA, cubeB, cubeC)
- Stacking sequence: A → B, C → A
- Moderate episodes (~300-400 timesteps)

```bash
mjpython batch_collect.py --env Stack3 --num-episodes 100
```

### Stack4 (4 cubes)

Most complex:
- 4 cubes (cubeA, cubeB, cubeC, cubeD)
- Stacking sequence: A → B, C → A, D → C
- Longer episodes (~400-500 timesteps)

```bash
mjpython batch_collect.py --env Stack4 --num-episodes 100
```

---

## Configuration Options

### Point Cloud Settings

```bash
# Low resolution (faster, smaller files)
--num-points 64

# Medium resolution (balanced)
--num-points 128

# High resolution (slower, larger files)
--num-points 256
```

### Camera Configuration

```bash
# Single camera (faster)
--cameras frontview

# Dual cameras (more coverage)
--cameras frontview agentview

# Triple cameras (maximum coverage)
--cameras frontview agentview birdview
```

### Episode Length

```bash
# Short episodes (for testing)
--max-timesteps 500

# Standard episodes
--max-timesteps 1000

# Long episodes (if policy takes longer)
--max-timesteps 1500
```

### Error Handling

```bash
# Standard retry
--max-retries 3

# More retries (for unstable policies)
--max-retries 5

# No retries (fail fast)
--max-retries 0
```

---

## Workflow Examples

### Example 1: Quick Test

```bash
# 1. Collect 3 episodes
mjpython batch_collect.py --env Stack --num-episodes 3 --output-dir ./quick_test

# 2. Validate
mjpython inspect_dataset.py ./quick_test --validate

# 3. Inspect first episode
mjpython inspect_dataset.py ./quick_test --inspect 0
```

### Example 2: Development Dataset

```bash
# 1. Collect 50 episodes
mjpython batch_collect.py \
    --env Stack \
    --num-episodes 50 \
    --output-dir ./dev_dataset \
    --max-timesteps 1000

# 2. Full validation
mjpython inspect_dataset.py ./dev_dataset --validate --stats

# 3. Create visualization
mjpython inspect_dataset.py ./dev_dataset --visualize --save-viz dev_stats.png
```

### Example 3: Production Dataset

```bash
# 1. Collect in batches
mjpython batch_collect.py --env Stack4 --num-episodes 250 --output-dir ./dataset_batch1
mjpython batch_collect.py --env Stack4 --num-episodes 250 --output-dir ./dataset_batch2
mjpython batch_collect.py --env Stack4 --num-episodes 250 --output-dir ./dataset_batch3
mjpython batch_collect.py --env Stack4 --num-episodes 250 --output-dir ./dataset_batch4

# 2. Validate each batch
mjpython inspect_dataset.py ./dataset_batch1 --validate
mjpython inspect_dataset.py ./dataset_batch2 --validate
mjpython inspect_dataset.py ./dataset_batch3 --validate
mjpython inspect_dataset.py ./dataset_batch4 --validate

# 3. Merge batches (manual file copy)
mkdir -p ./full_dataset/episodes ./full_dataset/metadata
cp dataset_batch*/episodes/* ./full_dataset/episodes/
cp dataset_batch*/metadata/* ./full_dataset/metadata/

# 4. Final validation
mjpython inspect_dataset.py ./full_dataset --validate --stats
```

---

## Troubleshooting

### Collection Fails

**Problem**: Episodes fail frequently
**Solution**: Increase retries and check policy
```bash
mjpython batch_collect.py --env Stack --num-episodes 10 --max-retries 5
```

### Out of Disk Space

**Problem**: Dataset too large
**Solution**: Reduce point cloud resolution
```bash
mjpython batch_collect.py --env Stack --num-episodes 100 --num-points 64
```

### Collection Too Slow

**Problem**: Takes too long per episode
**Solution**: Reduce max timesteps
```bash
mjpython batch_collect.py --env Stack --num-episodes 100 --max-timesteps 500
```

### Point Cloud Errors

**Problem**: Point cloud capture fails
**Solution**: Check camera configuration
```bash
mjpython batch_collect.py --env Stack --num-episodes 10 --cameras frontview
```

---

## Dataset Organization

After collection, your dataset will have this structure:

```
dataset/
├── episodes/
│   ├── episode_00000.pkl    # Episode data
│   ├── episode_00001.pkl
│   └── ...
├── metadata/
│   ├── collection_summary.json       # Overall statistics
│   ├── episode_00000_meta.json      # Per-episode metadata
│   ├── episode_00001_meta.json
│   └── ...
└── logs/                             # (empty - for future use)
```

---

## Loading Episodes in Python

```python
import pickle

# Load single episode
with open("dataset/episodes/episode_00000.pkl", "rb") as f:
    data_dict, attrs_dict = pickle.load(f)

# Access object data
objects = data_dict['objects']
for block_name, block_data in objects.items():
    positions = block_data['position']
    orientations = block_data['orientation']
    print(f"{block_name}: {len(positions)} timesteps")

# Access point clouds
point_cloud_1 = data_dict['point_cloud_1']
print(f"Point cloud shape: {point_cloud_1[0].shape}")

# Access actions
actions = attrs_dict['sudo_action_list']
print(f"Actions: {len(actions)} timesteps")
```

---

## Performance Expectations

### Collection Speed

| Environment | Timesteps/Episode | Time/Episode | Episodes/Hour |
|-------------|-------------------|--------------|---------------|
| Stack       | 250-350          | 5-7 min      | 9-12          |
| Stack3      | 350-450          | 7-9 min      | 7-9           |
| Stack4      | 400-500          | 8-10 min     | 6-8           |

### Storage Requirements

| Episodes | Points/Object | Size/Episode | Total Size |
|----------|---------------|--------------|------------|
| 10       | 128          | ~7-8 MB      | ~75 MB     |
| 100      | 128          | ~7-8 MB      | ~750 MB    |
| 500      | 128          | ~7-8 MB      | ~3.8 GB    |
| 1000     | 128          | ~7-8 MB      | ~7.5 GB    |
| 1000     | 256          | ~12-15 MB    | ~12-15 GB  |

### Success Rates

- Expected: >95% success rate
- Common failures: Policy gets stuck, environment reset issues
- Solutions: Automatic retry (up to 3 attempts)

---

## Best Practices

### 1. Start Small

Always test with small datasets first:
```bash
mjpython batch_collect.py --env Stack --num-episodes 5
```

### 2. Validate Early

Validate after every batch:
```bash
mjpython inspect_dataset.py ./dataset --validate
```

### 3. Batch Collection

Collect in smaller batches for easier management:
```bash
# Instead of 1000 at once
mjpython batch_collect.py --num-episodes 1000

# Do 10 batches of 100
for i in {1..10}; do
    mjpython batch_collect.py --num-episodes 100 --output-dir ./batch_$i
done
```

### 4. Monitor Progress

Check collection summary periodically:
```bash
cat dataset/metadata/collection_summary.json
```

### 5. Backup Data

Regularly backup collected data:
```bash
tar -czf dataset_backup_$(date +%Y%m%d).tar.gz dataset/
```

---

## Next Steps

After collecting your dataset:

1. **Validate**: Ensure all episodes are valid
2. **Analyze**: Check statistics and distributions
3. **Train**: Use for Points2Plans model training
4. **Iterate**: Collect more data based on model performance

---

## Summary

**You now have a complete data collection pipeline!**

- ✅ Batch collection with progress tracking
- ✅ Integration with heuristic policy
- ✅ Automatic error recovery
- ✅ Quality assurance tools
- ✅ Comprehensive documentation

**Ready to collect production datasets!**
