#!/usr/bin/env python
"""
Dataset Verification Script

Verifies that collected episodes match the Points2Plans format:
- 5 timesteps for Stack3 task (initial + 2 grasp + 2 release)
- 2 actions (2 pick-place operations)
- Correct behavior sequence

Usage:
    python verify_dataset.py data_capture/dataset/stack_v2/episodes/
    python verify_dataset.py data_capture/dataset/stack_v2/episodes/episode_00000_subsampled.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path


def verify_episode(filepath):
    """Verify a single episode file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        obs = data[0]
        metadata = data[1]
        
        # Expected format for Stack3 (2 stacking operations)
        expected_timesteps = 5
        expected_actions = 2
        expected_behaviors = ['release', 'grasp', 'release', 'grasp', 'release']
        
        # Get actual values
        actual_timesteps = obs['joint_position'].shape[0]
        actual_actions = len(metadata['sudo_action_list'])
        actual_behaviors = [obs['behavior'][i] for i in range(actual_timesteps)]
        
        # Check if valid
        is_valid = (actual_timesteps == expected_timesteps and 
                   actual_actions == expected_actions)
        
        return {
            'filepath': filepath,
            'valid': is_valid,
            'timesteps': actual_timesteps,
            'expected_timesteps': expected_timesteps,
            'actions': actual_actions,
            'expected_actions': expected_actions,
            'behaviors': actual_behaviors,
            'expected_behaviors': expected_behaviors,
            'action_list': metadata['sudo_action_list']
        }
    
    except Exception as e:
        return {
            'filepath': filepath,
            'valid': False,
            'error': str(e)
        }


def print_verification_result(result):
    """Pretty print verification result."""
    filepath = Path(result['filepath'])
    filename = filepath.name
    
    print(f"\n{'='*70}")
    print(f"File: {filename}")
    print(f"{'='*70}")
    
    if 'error' in result:
        print(f"❌ ERROR: {result['error']}")
        return
    
    # Overall status
    if result['valid']:
        print("✅ VALID - Matches Points2Plans format!")
    else:
        print("❌ INVALID - Does not match expected format")
    
    print()
    
    # Timesteps
    status = "✅" if result['timesteps'] == result['expected_timesteps'] else "❌"
    print(f"{status} Timesteps: {result['timesteps']} (expected: {result['expected_timesteps']})")
    
    # Actions
    status = "✅" if result['actions'] == result['expected_actions'] else "❌"
    print(f"{status} Actions: {result['actions']} (expected: {result['expected_actions']})")
    
    print()
    
    # Behavior sequence
    print("Behavior sequence:")
    behaviors_match = result['behaviors'] == result['expected_behaviors']
    for i, (actual, expected) in enumerate(zip(result['behaviors'], 
                                                 result['expected_behaviors'][:len(result['behaviors'])])):
        match = "✅" if actual == expected else "❌"
        print(f"  {match} T{i}: {actual:12s} (expected: {expected})")
    
    # If sequence is shorter than expected
    if len(result['behaviors']) < len(result['expected_behaviors']):
        for i in range(len(result['behaviors']), len(result['expected_behaviors'])):
            print(f"  ❌ T{i}: MISSING       (expected: {result['expected_behaviors'][i]})")
    
    print()
    
    # Actions details
    print("Actions:")
    for i, action in enumerate(result['action_list']):
        skill = action[0] if len(action) > 0 else 'unknown'
        obj_id = action[1] if len(action) > 1 else 'unknown'
        print(f"  Action {i}: skill={skill}, object_id={obj_id}")
    
    print(f"{'='*70}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_dataset.py <path_to_episodes_dir_or_file>")
        print("\nExamples:")
        print("  python verify_dataset.py data_capture/dataset/stack_v2/episodes/")
        print("  python verify_dataset.py data_capture/dataset/stack_v2/episodes/episode_00000_subsampled.pkl")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    # Collect episode files
    if input_path.is_dir():
        episode_files = sorted(input_path.glob("*_subsampled.pkl"))
        if not episode_files:
            print(f"❌ No subsampled episode files found in {input_path}")
            sys.exit(1)
    elif input_path.is_file():
        episode_files = [input_path]
    else:
        print(f"❌ Path not found: {input_path}")
        sys.exit(1)
    
    # Verify each episode
    print(f"\n{'='*70}")
    print(f"DATASET VERIFICATION")
    print(f"{'='*70}")
    print(f"Found {len(episode_files)} episode(s) to verify")
    
    results = []
    for filepath in episode_files:
        result = verify_episode(filepath)
        results.append(result)
        print_verification_result(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    total = len(results)
    valid = sum(1 for r in results if r.get('valid', False))
    invalid = total - valid
    
    print(f"Total episodes: {total}")
    print(f"✅ Valid: {valid}")
    print(f"❌ Invalid: {invalid}")
    print(f"Success rate: {valid/total*100:.1f}%")
    
    if valid == total:
        print("\n🎉 All episodes are valid!")
        return 0
    else:
        print(f"\n⚠️  {invalid} episode(s) need to be recollected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
