"""
Verification script to check Points2Plans format compatibility.

This script loads a saved episode and verifies it matches the expected format.
"""

import pickle
import numpy as np
from pathlib import Path


def verify_episode_format(filepath: str):
    """
    Verify that an episode file matches Points2Plans format.
    
    Args:
        filepath: Path to the pickle file
    """
    print(f"Verifying episode: {filepath}\n")
    
    # Load episode
    with open(filepath, 'rb') as f:
        episode = pickle.load(f)
    
    # Check top-level structure
    assert isinstance(episode, tuple), "Episode should be a tuple"
    assert len(episode) == 2, "Episode should have 2 elements (data, attrs)"
    
    data, attrs = episode
    
    print("✓ Top-level structure: tuple of (data, attrs)")
    
    # Verify data dict structure
    print("\n=== Data Dictionary ===")
    assert 'objects' in data, "data should have 'objects' key"
    assert 'contact' in data, "data should have 'contact' key"
    assert 'hidden_label' in data, "data should have 'hidden_label' key"
    
    num_objects = len(data['objects'])
    print(f"  Number of objects: {num_objects}")
    
    # Check object data
    for block_name, obj_data in data['objects'].items():
        assert 'position' in obj_data, f"{block_name} should have 'position'"
        assert 'orientation' in obj_data, f"{block_name} should have 'orientation'"
        
        positions = np.array(obj_data['position'])
        orientations = np.array(obj_data['orientation'])
        
        print(f"  {block_name}:")
        print(f"    - positions: {positions.shape}")
        print(f"    - orientations: {orientations.shape}")
    
    # Check point clouds
    print(f"\n=== Point Cloud Data ===")
    for i in range(1, num_objects + 1):
        pc_key = f'point_cloud_{i}'
        assert pc_key in data, f"Missing {pc_key}"
        assert f'{pc_key}sampling' in data, f"Missing {pc_key}sampling"
        assert f'{pc_key}sampling_noise' in data, f"Missing {pc_key}sampling_noise"
        
        pc_shape = data[pc_key].shape
        print(f"  {pc_key}: {pc_shape}")
        assert len(pc_shape) == 3, f"{pc_key} should have shape (timesteps, points, 3)"
        assert pc_shape[2] == 3, f"{pc_key} should have 3D coordinates"
    
    # Check contacts
    print(f"\n=== Contact Data ===")
    num_timesteps = len(data['contact'])
    print(f"  Timesteps: {num_timesteps}")
    total_contacts = sum(len(c) for c in data['contact'])
    print(f"  Total contacts: {total_contacts}")
    
    # Check hidden labels
    print(f"\n=== Hidden Labels ===")
    assert len(data['hidden_label']) == num_timesteps, "hidden_label length should match timesteps"
    print(f"  Shape: {num_timesteps} timesteps × {num_objects} objects")
    
    # Verify attrs dict structure
    print("\n=== Attrs Dictionary ===")
    assert 'objects' in attrs, "attrs should have 'objects' key"
    assert 'sudo_action_list' in attrs, "attrs should have 'sudo_action_list' key"
    
    for block_name, obj_attrs in attrs['objects'].items():
        assert 'extents' in obj_attrs, f"{block_name} should have 'extents'"
        assert 'extents_ranges' in obj_attrs, f"{block_name} should have 'extents_ranges'"
        assert 'fix_base_link' in obj_attrs, f"{block_name} should have 'fix_base_link'"
        assert 'object_type' in obj_attrs, f"{block_name} should have 'object_type'"
        
        print(f"  {block_name}:")
        print(f"    - extents: {obj_attrs['extents']}")
        print(f"    - fix_base_link: {obj_attrs['fix_base_link']}")
        print(f"    - object_type: {obj_attrs['object_type']}")
    
    print(f"\n  Action list: {len(attrs['sudo_action_list'])} actions")
    
    print("\n✓ Episode format verification complete!")
    print("  All required fields present and correctly structured")
    print(f"  Compatible with Points2Plans dataloader format")


if __name__ == "__main__":
    # Verify the test episode
    episode_path = "test_episodes/test_episode_001.pkl"
    
    if Path(episode_path).exists():
        verify_episode_format(episode_path)
    else:
        print(f"Error: Episode file not found: {episode_path}")
        print("Run episode_recorder.py first to generate a test episode")
