# Phase 4: Batch Collection & Quality Assurance

Automated data collection pipeline with quality assurance tools for Points2Plans dataset generation.

## Overview

Phase 4 provides:
- **Batch Collection**: Automated multi-episode collection using heuristic policy
- **Progress Tracking**: Real-time progress monitoring and statistics
- **Error Recovery**: Automatic retry on failures
- **Quality Assurance**: Dataset validation and inspection tools
- **Metadata Management**: Comprehensive dataset metadata

---

## Components

### 1. Batch Collector (`batch_collect.py`)

Automates collection of multiple episodes using the heuristic stacking policy from `run_stack.py`.

**Features**:
- Multi-episode collection with progress tracking
- Automatic error recovery and retry
- Dataset organization (episodes, metadata, logs)
- Collection statistics and success rates
- Resumable collection sessions

**Usage**:
```bash
# Collect 10 episodes from Stack environment
mjpython batch_collect.py --env Stack --num-episodes 10

# Collect 50 episodes from Stack4 with custom settings
mjpython batch_collect.py \
    --env Stack4 \
    --num-episodes 50 \
    --output-dir ./my_dataset \
    --max-timesteps 1000 \
    --num-points 256

# Available options
mjpython batch_collect.py --help
```

**Arguments**:
- `--env`: Environment name (`Stack`, `Stack3`, `Stack4`)
- `--num-episodes`: Number of episodes to collect (default: 10)
- `--output-dir`: Output directory (default: `./dataset`)
- `--max-timesteps`: Max timesteps per episode (default: 1000)
- `--max-retries`: Retry attempts for failures (default: 3)
- `--num-points`: Points per object point cloud (default: 128)
- `--cameras`: Camera names (default: `frontview agentview`)
- `--quiet`: Suppress verbose output

**Output Structure**:
```
dataset/
├── episodes/
│   ├── episode_00000.pkl
│   ├── episode_00001.pkl
│   └── ...
├── metadata/
│   ├── collection_summary.json
│   ├── episode_00000_meta.json
│   ├── episode_00001_meta.json
│   └── ...
└── logs/
```

---

### 2. Dataset Inspector (`inspect_dataset.py`)

Quality assurance tools for validating and analyzing collected datasets.

**Features**:
- Format validation for all episodes
- Dataset statistics computation
- Visualization of statistics
- Individual episode inspection
- Anomaly detection

**Usage**:
```bash
# Full inspection (validate, stats, visualize)
mjpython inspect_dataset.py ./dataset

# Validate all episodes
mjpython inspect_dataset.py ./dataset --validate

# Compute statistics
mjpython inspect_dataset.py ./dataset --stats

# Create visualization
mjpython inspect_dataset.py ./dataset --visualize --save-viz stats.png

# Inspect specific episode
mjpython inspect_dataset.py ./dataset --inspect 5
```

**Arguments**:
- `dataset_dir`: Path to dataset directory (required)
- `--validate`: Validate all episodes
- `--stats`: Compute dataset statistics
- `--visualize`: Create statistics visualization
- `--inspect N`: Inspect episode N in detail
- `--save-viz PATH`: Save visualization to file

---

## Integration with Heuristic Policy

The batch collector integrates seamlessly with `run_stack.py`:

### Policy Integration Flow

```python
# 1. Create environment
env = create_environment(env_name)

# 2. Create episode recorder
recorder = EpisodeRecorder(env, camera_names=["frontview", "agentview"])

# 3. Reset and start recording
obs = env.reset()
recorder.start_episode()

# 4. Create policy
policy = HeuristicStackPolicy(env)
policy.obs = obs

# 5. Run episode with policy
while not episode_complete:
    action, _ = policy.step()
    obs, reward, done, info = env.step(action)
    recorder.record_step(action, obs)
    policy.obs = obs
    
    if policy.stage == "done":
        episode_complete = True

# 6. Save episode
recorder.end_episode()
recorder.save_episode("./dataset/episodes", "episode_00000")
```

### Supported Environments

The system works with all Stack variants:
- **Stack**: 2 cubes (cubeA → cubeB)
- **Stack3**: 3 cubes (cubeA → cubeB, cubeC → cubeA)
- **Stack4**: 4 cubes (cubeA → cubeB, cubeC → cubeA, cubeD → cubeC)

The policy automatically detects available cubes and creates the stacking sequence.

---

## Example Workflows

### Workflow 1: Collect Small Dataset
```bash
# Collect 10 episodes from Stack environment
mjpython batch_collect.py --env Stack --num-episodes 10

# Inspect results
mjpython inspect_dataset.py ./dataset --stats --visualize
```

### Workflow 2: Collect Large Dataset with Quality Check
```bash
# Collect 100 episodes
mjpython batch_collect.py \
    --env Stack4 \
    --num-episodes 100 \
    --output-dir ./stack4_dataset

# Validate all episodes
mjpython inspect_dataset.py ./stack4_dataset --validate

# Generate statistics report
mjpython inspect_dataset.py ./stack4_dataset --stats --visualize --save-viz report.png

# Inspect any suspicious episodes
mjpython inspect_dataset.py ./stack4_dataset --inspect 42
```

### Workflow 3: Incremental Collection
```bash
# Collect first batch
mjpython batch_collect.py --env Stack --num-episodes 50 --output-dir ./dataset_v1

# Validate
mjpython inspect_dataset.py ./dataset_v1 --validate

# Collect more if needed
mjpython batch_collect.py --env Stack --num-episodes 50 --output-dir ./dataset_v2
```

---

## Collection Statistics

The batch collector tracks and reports:

### Per-Episode Statistics
- Episode duration (seconds)
- Number of timesteps
- Number of objects
- Number of contacts
- File size

### Overall Statistics
- Total episodes collected
- Success rate (%)
- Average timesteps per episode
- Average duration per episode
- Total dataset size (MB)
- Error log

### Example Output
```
============================================================
Collection Complete!
============================================================
Total Episodes: 50
  ✓ Successful: 48
  ✗ Failed: 2
Success Rate: 96.0%

Total Timesteps: 24500
Avg Timesteps/Episode: 510.4
Avg Duration/Episode: 25.3s

Total Duration: 21.1 minutes
Output Directory: ./dataset
============================================================
```

---

## Dataset Format

Each episode is saved as a pickle file containing:

```python
(data_dict, attrs_dict)
```

### `data_dict` (Time-series data)
```python
{
    'block_01': {
        'positions': np.ndarray,      # (T, 3) - xyz positions
        'orientations': np.ndarray,   # (T, 4) - quaternions
        'point_cloud': np.ndarray,    # (T, N, 3) - point clouds
        'hidden_label': int,          # Hidden state label
    },
    'block_02': { ... },
    ...
}
```

### `attrs_dict` (Static metadata)
```python
{
    'block_01': {
        'extents': np.ndarray,        # (3,) - object dimensions
        'fix_base_link': bool,        # Is object static?
    },
    'block_02': { ... },
    ...
    'max_objects': int,               # Maximum objects in dataset
    'actions': np.ndarray,            # (T, action_dim) - actions
}
```

---

## Metadata Files

### Collection Summary (`metadata/collection_summary.json`)
```json
{
  "env_name": "Stack",
  "collection_date": "2025-11-22T14:30:00",
  "duration_seconds": 1266.5,
  "total_episodes": 50,
  "successful_episodes": 48,
  "failed_episodes": 2,
  "total_timesteps": 24500,
  "avg_timesteps_per_episode": 510.4,
  "avg_duration_per_episode": 25.3,
  "success_rate": 0.96,
  "camera_names": ["frontview", "agentview"],
  "num_points": 128,
  "voxel_size": 0.005,
  "error_log": []
}
```

### Episode Metadata (`metadata/episode_00000_meta.json`)
```json
{
  "episode_idx": 0,
  "filepath": "./dataset/episodes/episode_00000.pkl",
  "timestamp": "2025-11-22T14:30:15",
  "statistics": {
    "num_timesteps": 520,
    "num_objects": 3,
    "num_contacts": 45,
    "num_actions": 519,
    "object_names": ["table", "cubeA_main", "cubeB_main"]
  }
}
```

---

## Quality Assurance

### Validation Checks

The inspector validates:
- ✓ Pickle file integrity
- ✓ Tuple structure `(data_dict, attrs_dict)`
- ✓ Dictionary types and keys
- ✓ Object naming convention (`block_*`)
- ✓ Required keys in data_dict
- ✓ Required keys in attrs_dict
- ✓ Array shapes (positions: (T,3), orientations: (T,4))
- ✓ Temporal consistency across objects

### Statistics Visualization

The inspector generates histograms for:
- Timesteps per episode distribution
- Objects per episode distribution
- Episode file sizes distribution
- Dataset summary table

---

## Error Handling

### Automatic Retry

The batch collector automatically retries failed episodes:

```python
# Configure retry behavior
batch_collect.py --max-retries 3
```

### Error Logging

All errors are logged to:
- Console (during collection)
- `metadata/collection_summary.json` (error_log field)

### Common Issues

1. **Environment resets fail**: Increase `--max-retries`
2. **Point cloud capture errors**: Check camera configuration
3. **Policy gets stuck**: Adjust `--max-timesteps`

---

## Performance Tips

### Collection Speed
- Disable rendering: Environment already configured without renderer
- Reduce timesteps: Use `--max-timesteps 500` for shorter episodes
- Parallel collection: Run multiple instances with different output dirs

### Dataset Size
- Reduce points: Use `--num-points 64` for smaller files
- Larger voxels: Adjust voxel_size in code for coarser point clouds
- Fewer cameras: Use single camera if sufficient

### Quality vs Speed
- High quality: `--num-points 256`, both cameras, full episodes
- Fast collection: `--num-points 64`, single camera, shorter episodes

---

## Next Steps

With Phase 4 complete, you can:

1. **Collect Full Dataset**: Run batch collection for desired number of episodes
2. **Validate Quality**: Use inspector to ensure data quality
3. **Train Models**: Use collected data for Points2Plans model training
4. **Iterate**: Collect more data based on model performance

### Recommended Collection Sizes
- **Development**: 10-50 episodes for testing pipelines
- **Small Dataset**: 100-500 episodes for initial training
- **Full Dataset**: 1000-5000 episodes for robust training

---

## Testing

Test the complete pipeline:

```bash
# 1. Collect small test dataset
mjpython batch_collect.py --env Stack --num-episodes 5 --output-dir ./test_dataset

# 2. Validate
mjpython inspect_dataset.py ./test_dataset --validate

# 3. Check statistics
mjpython inspect_dataset.py ./test_dataset --stats

# 4. Inspect first episode
mjpython inspect_dataset.py ./test_dataset --inspect 0
```

---

## Summary

Phase 4 provides a complete automated data collection pipeline:

✅ **Batch Collection**: Automated multi-episode recording  
✅ **Policy Integration**: Seamless integration with heuristic policy  
✅ **Error Recovery**: Automatic retry on failures  
✅ **Progress Tracking**: Real-time statistics and monitoring  
✅ **Quality Assurance**: Validation and inspection tools  
✅ **Metadata Management**: Comprehensive dataset documentation  
✅ **Visualization**: Statistical analysis and plotting  

**Status**: Ready for production data collection!
