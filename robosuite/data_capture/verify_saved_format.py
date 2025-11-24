"""
Verify that saved episode matches Points2Plans format exactly.
"""

import pickle
import sys
from pathlib import Path

def verify_episode_format(filepath):
    """Load and verify Points2Plans format."""
    print("=" * 80)
    print(f"Verifying: {filepath}")
    print("=" * 80)
    
    # Load episode
    with open(filepath, 'rb') as f:
        data_dict, attrs_dict = pickle.load(f)
    
    print(f"\n✓ Episode loaded successfully")
    print(f"  Type: {type(data_dict)}, {type(attrs_dict)}")
    
    # Expected keys
    expected_data_keys = [
        'rgb', 'depth', 'projection_matrix', 'view_matrix', 'segmentation',
        'joint_position', 'joint_velocity', 'joint_torque', 'target_joint_position',
        'target_ee_discrete', 'ee_position', 'ee_orientation', 'ee_velocity',
        'contact', 'objects', 'behavior', 'hidden_label',
        # Point clouds will vary by number of objects
    ]
    
    expected_attrs_keys = [
        'segmentation_labels', 'segmentation_ids', 'objects', 'robot_joint_names',
        'robot_link_names', 'n_arm_joints', 'n_ee_joints', 'sudo_action_list',
        'behavior_params'
    ]
    
    # Check data_dict
    print("\n" + "=" * 80)
    print("DATA_DICT KEYS")
    print("=" * 80)
    
    actual_data_keys = sorted(data_dict.keys())
    print(f"\nFound {len(actual_data_keys)} keys:")
    for key in actual_data_keys:
        if 'point_cloud' in key:
            print(f"  ✓ {key} (shape: {data_dict[key].shape})")
        else:
            status = "✓" if key in expected_data_keys else "?"
            if isinstance(data_dict[key], dict):
                print(f"  {status} {key}: {len(data_dict[key])} items")
            elif hasattr(data_dict[key], 'shape'):
                print(f"  {status} {key}: {data_dict[key].shape}")
            else:
                print(f"  {status} {key}: {type(data_dict[key])}")
    
    # Check attrs_dict
    print("\n" + "=" * 80)
    print("ATTRS_DICT KEYS")
    print("=" * 80)
    
    actual_attrs_keys = sorted(attrs_dict.keys())
    print(f"\nFound {len(actual_attrs_keys)} keys:")
    for key in actual_attrs_keys:
        status = "✓" if key in expected_attrs_keys else "?"
        value = attrs_dict[key]
        if isinstance(value, (list, tuple)):
            if len(value) > 0 and isinstance(value[0], str):
                print(f"  {status} {key}: {len(value)} items (strings)")
            else:
                print(f"  {status} {key}: {len(value)} items")
        elif isinstance(value, dict):
            print(f"  {status} {key}: {len(value)} entries")
        else:
            print(f"  {status} {key}: {value}")
    
    # Detailed inspection of key arrays
    print("\n" + "=" * 80)
    print("DETAILED FORMAT CHECK")
    print("=" * 80)
    
    T = len(data_dict.get('joint_position', []))
    print(f"\nTimesteps (T): {T}")
    
    # Robot state
    print("\n--- Robot State ---")
    if 'joint_position' in data_dict:
        print(f"joint_position:        {data_dict['joint_position'].shape} {data_dict['joint_position'].dtype}")
        print(f"joint_velocity:        {data_dict['joint_velocity'].shape} {data_dict['joint_velocity'].dtype}")
        print(f"joint_torque:          {data_dict['joint_torque'].shape} {data_dict['joint_torque'].dtype}")
        print(f"ee_position:           {data_dict['ee_position'].shape} {data_dict['ee_position'].dtype}")
        print(f"ee_orientation:        {data_dict['ee_orientation'].shape} {data_dict['ee_orientation'].dtype}")
        print(f"ee_velocity:           {data_dict['ee_velocity'].shape} {data_dict['ee_velocity'].dtype}")
    
    # Camera data
    print("\n--- Camera Data (placeholders) ---")
    if 'rgb' in data_dict:
        print(f"rgb:                   {data_dict['rgb'].shape} {data_dict['rgb'].dtype}")
        print(f"depth:                 {data_dict['depth'].shape} {data_dict['depth'].dtype}")
        print(f"segmentation:          {data_dict['segmentation'].shape} {data_dict['segmentation'].dtype}")
        print(f"projection_matrix:     {data_dict['projection_matrix'].shape} {data_dict['projection_matrix'].dtype}")
        print(f"view_matrix:           {data_dict['view_matrix'].shape} {data_dict['view_matrix'].dtype}")
    
    # Object state
    print("\n--- Object State ---")
    if 'objects' in data_dict:
        print(f"objects:               {len(data_dict['objects'])} objects")
        for obj_name in sorted(data_dict['objects'].keys()):
            obj_data = data_dict['objects'][obj_name]
            print(f"  {obj_name}:")
            if isinstance(obj_data['position'], list):
                print(f"    position:          list of {len(obj_data['position'])} timesteps")
                print(f"    orientation:       list of {len(obj_data['orientation'])} timesteps")
            else:
                print(f"    position:          {obj_data['position'].shape} {obj_data['position'].dtype}")
                print(f"    orientation:       {obj_data['orientation'].shape} {obj_data['orientation'].dtype}")
    
    if 'hidden_label' in data_dict:
        print(f"hidden_label:          {data_dict['hidden_label'].shape} {data_dict['hidden_label'].dtype}")
    
    # Point clouds
    print("\n--- Point Clouds ---")
    pc_keys = [k for k in data_dict.keys() if 'point_cloud' in k]
    if pc_keys:
        for pc_key in sorted(pc_keys):
            print(f"{pc_key}: {data_dict[pc_key].shape} {data_dict[pc_key].dtype}")
    else:
        print("No point clouds (open3d not available)")
    
    # Metadata
    print("\n--- Metadata (attrs_dict) ---")
    if 'robot_joint_names' in attrs_dict:
        print(f"robot_joint_names:     {len(attrs_dict['robot_joint_names'])} joints")
        print(f"  {attrs_dict['robot_joint_names']}")
    if 'robot_link_names' in attrs_dict:
        print(f"robot_link_names:      {len(attrs_dict['robot_link_names'])} links")
    if 'n_arm_joints' in attrs_dict:
        print(f"n_arm_joints:          {attrs_dict['n_arm_joints']}")
    if 'n_ee_joints' in attrs_dict:
        print(f"n_ee_joints:           {attrs_dict['n_ee_joints']}")
    if 'segmentation_labels' in attrs_dict:
        print(f"segmentation_labels:   {len(attrs_dict['segmentation_labels'])} objects")
        for k, v in attrs_dict['segmentation_labels'].items():
            print(f"  {k}: {v}")
    
    # Compare with expected format
    print("\n" + "=" * 80)
    print("FORMAT COMPLIANCE")
    print("=" * 80)
    
    missing_data = []
    for key in expected_data_keys:
        if key not in data_dict:
            missing_data.append(key)
    
    missing_attrs = []
    for key in expected_attrs_keys:
        if key not in attrs_dict:
            missing_attrs.append(key)
    
    if not missing_data and not missing_attrs:
        print("\n✅ 100% FORMAT COMPLIANT!")
        print("   All required Points2Plans keys present.")
    else:
        if missing_data:
            print(f"\n⚠️  Missing data_dict keys: {missing_data}")
        if missing_attrs:
            print(f"⚠️  Missing attrs_dict keys: {missing_attrs}")
    
    # Additional keys
    extra_data = set(actual_data_keys) - set(expected_data_keys) - set([k for k in actual_data_keys if 'point_cloud' in k])
    if extra_data:
        print(f"\nℹ️  Extra data_dict keys: {extra_data}")
    
    print("\n" + "=" * 80)
    

if __name__ == "__main__":
    filepath = "test_episodes/test_episode.pkl"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    verify_episode_format(filepath)
