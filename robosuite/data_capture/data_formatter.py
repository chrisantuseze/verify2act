"""
Data Formatting Utilities for Points2Plans Format

This module handles building data and attrs dictionaries in Points2Plans format.
"""

import numpy as np
from typing import Dict, List, Any


class DataFormatter:
    """Formats episode data into Points2Plans structure."""
    
    def __init__(self, object_metadata: Dict, num_points: int = 128, state_capture=None):
        """
        Initialize data formatter.
        
        Args:
            object_metadata: Dictionary of object metadata
            num_points: Number of points per object point cloud
            state_capture: StateCapture instance for robot info
        """
        self.object_metadata = object_metadata
        self.num_points = num_points
        self.state_capture = state_capture
    
    def build_data_dict(self, timestep_data: List[Dict], camera_obs: Dict = None) -> Dict:
        """
        Build the 'data' dictionary in Points2Plans format.
        
        Args:
            timestep_data: List of timestep state dictionaries
            camera_obs: Optional camera observations (rgb, depth, seg)
            
        Returns:
            Data dictionary with time-series information
        """
        object_names = list(self.object_metadata.keys())
        num_objects = len(object_names)
        
        data = {
            # Camera data (placeholders if not provided)
            'rgb': [],
            'depth': [],
            'segmentation': [],
            'projection_matrix': [],
            'view_matrix': [],
            
            # Robot state
            'joint_position': [],
            'joint_velocity': [],
            'joint_torque': [],
            'target_joint_position': [],
            'target_ee_discrete': [],
            'ee_position': [],
            'ee_orientation': [],
            'ee_velocity': [],
            
            # Scene state
            'contact': [],
            'objects': {},
            'behavior': [],
            'hidden_label': [],
        }
        
        # Initialize per-object data
        for obj_idx, obj_name in enumerate(object_names):
            block_name = self._to_block_name(obj_idx)
            data['objects'][block_name] = {
                'position': [],
                'orientation': [],
            }
        
        # Initialize point cloud arrays
        for obj_idx in range(num_objects):
            data[f'point_cloud_{obj_idx + 1}'] = []
            data[f'point_cloud_{obj_idx + 1}sampling'] = []
            data[f'point_cloud_{obj_idx + 1}sampling_noise'] = []
        
        # Extract time-series data
        for timestep_state in timestep_data:
            # Robot state
            robot_state = timestep_state['robot_state']
            data['joint_position'].append(robot_state['joint_pos'])
            data['joint_velocity'].append(robot_state['joint_vel'])
            data['joint_torque'].append(robot_state['joint_torque'])
            data['ee_position'].append(robot_state['eef_pos'])
            data['ee_orientation'].append(robot_state['eef_quat'])
            data['ee_velocity'].append(robot_state['eef_vel'])
            
            # Target positions (use current for now - can be updated by policy)
            data['target_joint_position'].append(robot_state['joint_pos'])
            data['target_ee_discrete'].append(np.zeros(3))  # Placeholder
            
            # Camera data (placeholders)
            data['rgb'].append(np.zeros((480, 640, 3), dtype=np.uint8))
            data['depth'].append(np.zeros((480, 640), dtype=np.float32))
            data['segmentation'].append(np.zeros((480, 640), dtype=np.int32))
            data['projection_matrix'].append(np.eye(4))
            data['view_matrix'].append(np.eye(4))
            
            # Object states
            object_states = timestep_state['object_states']
            for obj_idx, obj_name in enumerate(object_names):
                if obj_name in object_states:
                    state = object_states[obj_name]
                    block_name = self._to_block_name(obj_idx)
                    data['objects'][block_name]['position'].append(state['position'])
                    data['objects'][block_name]['orientation'].append(state['orientation'])
            
            # Contacts
            data['contact'].append(timestep_state['contacts'])
            
            # Behavior (skill type from action)
            action = timestep_state.get('action')
            if action:
                data['behavior'].append(action['skill_type'])
            else:
                data['behavior'].append('none')
            
            # Point clouds and occlusion
            point_clouds = timestep_state.get('point_clouds', {})
            hidden_labels = []
            
            for obj_idx, obj_name in enumerate(object_names):
                # Point cloud for this object
                if point_clouds and obj_name in point_clouds:
                    pts = point_clouds[obj_name]
                    pts_sampled = self._sample_points(pts, self.num_points)
                    
                    data[f'point_cloud_{obj_idx + 1}'].append(pts_sampled)
                    data[f'point_cloud_{obj_idx + 1}sampling'].append(pts_sampled)
                    
                    # Add noise variant
                    noise = np.random.randn(*pts_sampled.shape) * 0.001
                    pts_noisy = pts_sampled + noise
                    data[f'point_cloud_{obj_idx + 1}sampling_noise'].append(pts_noisy)
                    
                    # Occlusion label (0 = visible)
                    num_pts = len(pts)
                    is_hidden = 1 if num_pts < 10 else 0
                    hidden_labels.append(is_hidden)
                else:
                    # No point cloud - use zeros
                    zero_cloud = np.zeros((self.num_points, 3))
                    data[f'point_cloud_{obj_idx + 1}'].append(zero_cloud)
                    data[f'point_cloud_{obj_idx + 1}sampling'].append(zero_cloud)
                    data[f'point_cloud_{obj_idx + 1}sampling_noise'].append(zero_cloud)
                    hidden_labels.append(1)  # Hidden
            
            data['hidden_label'].append(hidden_labels)
        
        # Convert lists to numpy arrays
        for key in ['joint_position', 'joint_velocity', 'joint_torque', 'target_joint_position',
                    'target_ee_discrete', 'ee_position', 'ee_orientation', 'ee_velocity',
                    'rgb', 'depth', 'segmentation', 'projection_matrix', 'view_matrix']:
            data[key] = np.array(data[key])
        
        # Convert object states to arrays
        for block_name in data['objects']:
            data['objects'][block_name]['position'] = np.array(data['objects'][block_name]['position'])
            data['objects'][block_name]['orientation'] = np.array(data['objects'][block_name]['orientation'])
        
        for obj_idx in range(num_objects):
            data[f'point_cloud_{obj_idx + 1}'] = np.array(data[f'point_cloud_{obj_idx + 1}'])
            data[f'point_cloud_{obj_idx + 1}sampling'] = np.array(data[f'point_cloud_{obj_idx + 1}sampling'])
            data[f'point_cloud_{obj_idx + 1}sampling_noise'] = np.array(data[f'point_cloud_{obj_idx + 1}sampling_noise'])
        
        data['hidden_label'] = np.array(data['hidden_label'])
        
        return data
    
    def build_attrs_dict(self, action_history: List[Dict]) -> Dict:
        """
        Build the 'attrs' dictionary in Points2Plans format.
        
        Args:
            action_history: List of action dictionaries
            
        Returns:
            Attributes dictionary with static properties
        """
        attrs = {
            'objects': {},
            'sudo_action_list': [],
            'segmentation_labels': {},
            'segmentation_ids': {},
            'robot_joint_names': [],
            'robot_link_names': [],
            'n_arm_joints': 0,
            'n_ee_joints': 0,
            'behavior_params': {},
        }
        
        # Robot information
        if self.state_capture:
            attrs['robot_joint_names'] = self.state_capture.robot_joint_names
            attrs['robot_link_names'] = self.state_capture.robot_link_names
            attrs['n_arm_joints'] = self.state_capture.n_arm_joints
            attrs['n_ee_joints'] = self.state_capture.n_ee_joints
        
        # Object metadata
        for obj_idx, (obj_name, obj_meta) in enumerate(self.object_metadata.items()):
            block_name = self._to_block_name(obj_idx)
            
            attrs['objects'][block_name] = {
                'extents': obj_meta['extents'],
                'extents_ranges': self._compute_extents_ranges(obj_meta['extents']),
                'fix_base_link': obj_meta['fix_base_link'],
                'object_type': obj_meta['object_type'],
            }
            
            if obj_meta.get('asset_filename'):
                attrs['objects'][block_name]['asset_filename'] = obj_meta['asset_filename']
            
            # Segmentation info
            attrs['segmentation_labels'][block_name] = obj_name
            attrs['segmentation_ids'][block_name] = obj_idx + 1
        
        # Action list
        attrs['sudo_action_list'] = self._build_action_list(action_history)
        
        # Behavior parameters (placeholder)
        attrs['behavior_params'] = {
            'skills': ['grasp', 'release', 'move'],
            'skill_params': {}
        }
        
        return attrs
    
    def _build_action_list(self, action_history: List[Dict]) -> List[List]:
        """Build action list in Points2Plans format."""
        action_list = []
        
        for action_dict in action_history:
            if action_dict is None:
                continue
            
            skill_type = action_dict['skill_type']
            obj_idx = action_dict['object_id']
            
            if obj_idx is not None and 0 <= obj_idx < len(self.object_metadata):
                obj_name = self._to_block_name(obj_idx)
            else:
                obj_name = 'unknown'
            
            continuous_params = action_dict['position_delta'].tolist()
            action_list.append([skill_type, obj_name, continuous_params])
        
        return action_list
    
    def _sample_points(self, points: np.ndarray, num_points: int) -> np.ndarray:
        """Sample or pad point cloud to fixed size."""
        if len(points) == 0:
            return np.zeros((num_points, 3))
        
        if len(points) >= num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            return points[indices]
        else:
            indices = np.random.choice(len(points), num_points, replace=True)
            return points[indices]
    
    def _to_block_name(self, idx: int) -> str:
        """Convert object index to block name (block_01, block_02, ...)."""
        return f"block_{idx + 1:02d}"
    
    def _compute_extents_ranges(self, extents: List[float]) -> List[List[float]]:
        """Compute extents_ranges from extents."""
        return [[e, e] for e in extents]
