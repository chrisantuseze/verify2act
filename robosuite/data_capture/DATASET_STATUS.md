# Dataset Format Verification Report

**Date:** December 27, 2025  
**Task:** Stack3 (stacking 2 cubes)  
**Dataset Location:** `robosuite/data_capture/dataset/stack_v2/episodes/`

---

## Expected Format (Points2Plans Standard)

For a **Stack3 task with 2 stacking operations** (e.g., stack cubeA on cubeB, then stack cubeC on cubeA):

### Expected Structure:
- **Number of timesteps:** 5
- **Number of actions:** 2

### Expected Timestep Breakdown:
1. **T0:** Initial state (behavior='none')
2. **T1:** After grasping first cube (behavior='grasp')
3. **T2:** After placing first cube (behavior='release')
4. **T3:** After grasping second cube (behavior='grasp')
5. **T4:** After placing second cube (behavior='release')

### Expected Actions:
- **Action 0:** ['pickplace', object_id_1, position_delta_1]
- **Action 1:** ['pickplace', object_id_2, position_delta_2]

### Reference - Points2Plans Original Dataset:
**File:** `Points2Plans/Training/cupboard_00/demo_000001.pickle`
- Timesteps: 5
- Actions: 4
- All behaviors: 'TeleportObject' (task-specific)

---

## Current Dataset Status

### Episode 0:
- **Number of timesteps:** 3 ❌ (Expected: 5)
- **Number of actions:** 1 ❌ (Expected: 2)

**Behavior sequence:**
- T0: none
- T1: grasp
- T2: release

**Actions:**
- Action 0: ['pickplace', '3', [-0.036, -0.006, -0.056]]

**Status:** ❌ **Incomplete** - Only completed 1 of 2 stacking operations

---

### Episode 1:
- **Number of timesteps:** 4 ❌ (Expected: 5)
- **Number of actions:** 1 ❌ (Expected: 2)

**Behavior sequence:**
- T0: none
- T1: grasp
- T2: grasp
- T3: release

**Actions:**
- Action 0: ['pickplace', '2', [-0.002, -0.004, 0.048]]

**Status:** ❌ **Incomplete** - Only completed 1 of 2 stacking operations

---

## Issues Identified

### 1. ✅ Subsampling Logic - FIXED
The subsampling logic correctly captures:
- Initial state (T0)
- First grasp timestep (transition TO grasp)
- Last release timestep (transition FROM release)

**Code location:** `episode_recorder.py` → `subsample_to_key_states()`

### 2. ❌ Episode Completion - ISSUE
**Problem:** Episodes are not completing both stacking operations before termination.

**Expected behavior:**
- Stack cubeA onto cubeB (Operation 1) ✓ COMPLETES
- Stack cubeC onto cubeA (Operation 2) ❌ DOES NOT COMPLETE

**Possible causes:**
- Episode timeout before second operation completes
- Heuristic policy fails during second stacking attempt
- Task success condition triggered early after first stack

---

## Next Steps

### To achieve correct format:
1. **Increase episode timeout** or optimize policy speed
2. **Debug why second stacking operation doesn't complete**
3. **Verify task completion logic** in Stack3 environment

### Expected complete episode:
```
Episode structure:
  Timesteps: 5
  Actions: 2
  
Timesteps:
  T0: Initial state (behavior='none')
  T1: After grasp cubeA (behavior='grasp')
  T2: After place cubeA on cubeB (behavior='release')
  T3: After grasp cubeC (behavior='grasp')
  T4: After place cubeC on cubeA (behavior='release')

Actions:
  Action 0: ['pickplace', cubeA_id, position_delta]
  Action 1: ['pickplace', cubeC_id, position_delta]
```

---

## Summary

| Metric | Expected | Current (Ep 0) | Current (Ep 1) | Status |
|--------|----------|----------------|----------------|---------|
| Timesteps | 5 | 3 | 4 | ❌ |
| Actions | 2 | 1 | 1 | ❌ |
| Subsampling | Correct | Correct | Correct | ✅ |
| Episode Completion | 2 stacks | 1 stack | 1 stack | ❌ |

**Overall Status:** Subsampling logic is working correctly, but episodes need to complete both stacking operations.
