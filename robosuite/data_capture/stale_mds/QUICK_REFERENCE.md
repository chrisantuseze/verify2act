# Quick Reference: Points2Plans Data Format

## Format at a Glance

### Episode Structure
```python
episode = (data_dict, attrs_dict)  # Tuple of 2 dicts
```

---

## data_dict Keys (32 total)

### Robot State (8 keys)
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

### Camera Data (5 keys) - Currently Placeholders
| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `rgb` | `(T, 480, 640, 3)` | uint8 | RGB images |
| `depth` | `(T, 480, 640)` | float32 | Depth maps |
| `segmentation` | `(T, 480, 640)` | int32 | Segmentation masks |
| `projection_matrix` | `(T, 4, 4)` | float64 | Camera projection |
| `view_matrix` | `(T, 4, 4)` | float64 | Camera view transform |

### Object State (1 key, nested)
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

### Point Clouds (9+ keys, varies by N objects)
For each object (N=1, 2, 3, ...):
- `point_cloud_N`: `(T, 128, 3)` float64
- `point_cloud_Nsampling`: `(T, 128, 3)` float64
- `point_cloud_Nsampling_noise`: `(T, 128, 3)` float64

### Metadata (3 keys)
| Key | Type | Description |
|-----|------|-------------|
| `contact` | list | Contact events per timestep |
| `behavior` | list | Behavior labels per timestep |
| `hidden_label` | `(T, N)` int64 | Occlusion flags |

---

## attrs_dict Keys (9 total)

### Robot Metadata (4 keys)
| Key | Type | Example |
|-----|------|---------|
| `robot_joint_names` | list[str] | `['robot0_joint1', ..., 'robot0_joint7']` |
| `robot_link_names` | list[str] | `['panda_link0', ..., 'panda_hand']` |
| `n_arm_joints` | int | `7` |
| `n_ee_joints` | int | `2` (or 0 for special grippers) |

### Segmentation Info (2 keys)
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

### Episode Metadata (3 keys)
| Key | Type | Description |
|-----|------|-------------|
| `objects` | dict | Object properties (extents, mass, static) |
| `sudo_action_list` | list | Action history |
| `behavior_params` | dict | Behavior parameters |

---

## Common Access Patterns

### Load Episode
```python
import pickle

with open('episode.pkl', 'rb') as f:
    data_dict, attrs_dict = pickle.load(f)

T = len(data_dict['joint_position'])  # Number of timesteps
N = len(data_dict['objects'])          # Number of objects
```

### Access Robot State
```python
# Joint state at timestep t
joint_pos_t = data_dict['joint_position'][t]    # (7,)
joint_vel_t = data_dict['joint_velocity'][t]    # (7,)

# End-effector state at timestep t
ee_pos_t = data_dict['ee_position'][t]          # (3,)
ee_quat_t = data_dict['ee_orientation'][t]      # (4,)
ee_vel_t = data_dict['ee_velocity'][t]          # (3,)
```

### Access Object State
```python
# Object trajectories
obj_pos = data_dict['objects']['block_02']['position']      # (T, 3)
obj_quat = data_dict['objects']['block_02']['orientation']  # (T, 4)

# Object at specific timestep
obj_pos_t = obj_pos[t]  # (3,)
```

### Access Point Clouds
```python
# Point cloud for object 1 at timestep t
pc_t = data_dict['point_cloud_1'][t]              # (128, 3)
pc_sampled_t = data_dict['point_cloud_1sampling'][t]  # (128, 3)

# With noise
pc_noisy_t = data_dict['point_cloud_1sampling_noise'][t]  # (128, 3)
```

### Access Metadata
```python
# Robot info
joint_names = attrs_dict['robot_joint_names']   # ['robot0_joint1', ...]
n_joints = attrs_dict['n_arm_joints']           # 7

# Object info
obj_extents = attrs_dict['objects']['block_02']['extents']  # [0.04, 0.04, 0.04]

# Segmentation
seg_label = attrs_dict['segmentation_labels']['block_02']  # 'cubeA_main'
seg_id = attrs_dict['segmentation_ids']['block_02']        # 1
```

---

## Data Dimensions Reference

For Panda robot with Stack task (3 objects):

| Component | Dimension | Note |
|-----------|-----------|------|
| `T` | Variable | Number of timesteps (e.g., 300) |
| `n_joints` | 7 | Panda arm DoF |
| `n_ee` | 0-2 | Gripper DoF (special for Panda) |
| `N_objects` | 3 | table + 2 cubes |
| `points_per_obj` | 128 | Configurable |
| `img_height` | 480 | Camera resolution |
| `img_width` | 640 | Camera resolution |

---

## Verification Checklist

✅ **Required data_dict keys**: 26+ (including point clouds)  
✅ **Required attrs_dict keys**: 9  
✅ **Robot state**: All joint and EE data present  
✅ **Object state**: Position/orientation as arrays `(T, 3/4)`  
✅ **Point clouds**: `(T, 128, 3)` per object  
✅ **Metadata**: Robot and object info complete  

---

## Quick Verification Commands

```bash
# Check format without loading full episode
python data_capture/verify_format_alignment.py

# Detailed inspection of saved episode
python data_capture/verify_saved_format.py path/to/episode.pkl

# Test recording
mjpython data_capture/episode_recorder.py
```

---

## Troubleshooting

### Issue: Missing keys
**Solution**: Re-run collection with updated `data_formatter.py`

### Issue: Objects are lists not arrays
**Solution**: Update `data_formatter.py` lines 154-157 to convert objects to arrays

### Issue: Point clouds all zeros
**Solution**: Install `open3d` or verify `POINTCLOUD_AVAILABLE` flag

### Issue: Camera data all zeros
**Solution**: Expected - using placeholders. Update `data_formatter.py` lines 86-99 to enable real capture.

---

## Performance Notes

**Typical Episode**:
- Timesteps: 200-400
- File size: 15-25 MB
- Collection time: 10-20 seconds (without camera capture)
- Collection time: 100-400 seconds (with camera capture)

**Recommendation**: Use placeholder camera data unless specifically needed.

---

**Last Updated**: November 24, 2025  
**Version**: Points2Plans v1.0 (100% compliant)
