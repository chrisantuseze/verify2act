# Key Timestep Recording Mode

**Date:** December 27, 2025  
**Feature:** Direct key timestep recording during rollout (instead of post-processing subsampling)

---

## Problem Statement

Previously, the system would:
1. Record ALL timesteps during rollout (~500-600 timesteps)
2. Post-process via `subsample_to_key_states()` to extract key states
3. This led to issues:
   - Large memory/storage overhead
   - Subsampling logic had to infer transitions after the fact
   - Object detection issues (object_id=None) caused incomplete action sequences

## Solution: Key Timestep Recording Mode

Instead of recording everything and subsampling later, now we can **record only key timesteps directly during rollout**.

### Key Changes

#### 1. `EpisodeRecorder.__init__()` - New Parameter
```python
def __init__(self, env, camera_names=None, voxel_size=0.005, num_points=128,
             key_timesteps_only: bool = False):
```

**New parameter:**
- `key_timesteps_only`: If `True`, only save key timesteps during rollout

#### 2. New State Tracking Variables
```python
# Key timestep tracking (for key_timesteps_only mode)
self.prev_skill_type = None
self.in_grasp_sequence = False
self.grasp_start_timestep = None
self.current_grasped_object = None
```

#### 3. New Method: `_record_key_timestep_if_needed()`
This method intelligently decides **during rollout** whether the current timestep is a key state:

**Key timesteps are:**
1. **First timestep after grasp begins** (transition TO grasp with valid object)
2. **Last timestep after release completes** (transition FROM release with valid object)

**Logic:**
```python
# Detect grasp START
if skill_type == 'grasp' and prev_skill_type != 'grasp':
    if object_id is not None:
        save_timestep()  # This is a key state

# Detect release END
elif prev_skill_type == 'release' and skill_type != 'release':
    if current_grasped_object is not None:
        save_timestep()  # This is a key state
        create_pickplace_action()
```

#### 4. Modified `record_step()`
Now branches based on mode:
```python
def record_step(self, action, obs):
    self.current_timestep += 1
    
    if self.key_timesteps_only:
        self._record_key_timestep_if_needed(action, obs)
    else:
        self._capture_timestep_state(action, obs)
```

#### 5. Modified `end_episode()`
Handles edge case where episode ends during release:
```python
# Handle edge case: episode ends during release sequence
if self.key_timesteps_only and self.prev_skill_type == 'release' and self.current_grasped_object is not None:
    # Capture final state and create final action
```

#### 6. Modified `save_episode()`
Recognizes when data is already in key format:
```python
if self.key_timesteps_only:
    # Save directly as subsampled format (no post-processing needed)
    key_file = output_path / f"{base_name}_subsampled.pkl"
```

#### 7. Updated `batch_collect.py`
Enable key timestep mode by default:
```python
recorder = EpisodeRecorder(
    env, 
    camera_names=self.camera_names,
    num_points=self.num_points,
    voxel_size=self.voxel_size,
    key_timesteps_only=True  # ← New parameter
)
```

---

## Expected Output Format

For **Stack3 task** (2 cubes to stack):

### Before (Post-processing subsampling):
- Records: ~500-600 timesteps
- Subsamples to: 3-4 key states (but incomplete due to object detection issues)
- Actions: 1 pick-place operation (missing second one)

### After (Direct key timestep recording):
- Records: **5 key timesteps directly**
- Actions: **2 pick-place operations**

**Expected timestep sequence:**
```
T0: Initial state (behavior='none')
T1: After grasp first cube (behavior='grasp', object_id=3)
T2: After place first cube (behavior='release', object_id=3)
T3: After grasp second cube (behavior='grasp', object_id=1)
T4: After place second cube (behavior='release', object_id=1)
```

**Expected actions:**
```
Action 0: ['pickplace', object_id=3, position_delta]
Action 1: ['pickplace', object_id=1, position_delta]
```

---

## Benefits

1. **Memory Efficient**: Only stores ~5 timesteps instead of ~500
2. **More Reliable**: Detects key states in real-time with full context
3. **Cleaner Logic**: No need for post-processing inference
4. **Better Object Tracking**: Uses `current_grasped_object` to maintain object identity across grasp-release sequences
5. **Faster**: No subsampling computation needed

---

## Testing

Run batch collection to test:
```bash
mjpython data_capture/batch_collect.py \
    --num-episodes 2 \
    --output-dir data_capture/dataset/stack_v3
```

Then verify:
```bash
python3 ./data_capture/verify_dataset.py data_capture/dataset/stack_v3/episodes/
```

**Expected verification output:**
- Timesteps: 5 ✓
- Actions: 2 ✓
- All behaviors captured correctly ✓

---

## Debugging

Key timestep mode prints debug messages:
```
[KEY] T129: Grasp START (object 3)
[KEY] T245: Release END (object 3)
[KEY] T403: Grasp START (object 1)
[KEY] T520: Release END (object 1)
```

This helps verify transitions are detected correctly.

---

## Backward Compatibility

The old mode still works:
```python
recorder = EpisodeRecorder(env, key_timesteps_only=False)  # Old behavior
recorder.save_episode(output_dir, save_subsampled=True)    # Post-processing
```

---

## Next Steps

1. Test with Stack3 task
2. Verify 2 pick-place operations are captured
3. Check object_id consistency throughout sequences
4. Confirm episode completes both stacking operations
