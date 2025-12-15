"""
State Converter for Points2Plans Inference

Converts robosuite observations to Points2Plans input format in real-time.

IMPORTANT: This is for INFERENCE ONLY. No manual predicate computation!
- During training: Dataloader computes predicates on-the-fly as supervision
- During inference: Decoder predicts predicates from latent state

This module only provides:
- Point clouds
- Object poses (reference)
- One-hot encodings
- Graph connectivity
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add robosuite paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from robosuite.utils.pointcloud_generator import PointCloudGenerator
from data_capture.metadata_extractor import MetadataExtractor


class StateConverter:
    """
    Lightweight state converter for Points2Plans inference.
    
    NO manual predicate computation - decoder handles that during inference!
    """
    
    def __init__(self, 
                 env,
                 camera_names: Optional[List[str]] = None,
                 num_points: int = 128,
                 voxel_size: float = 0.005,
                 workspace_bounds: Optional[np.ndarray] = None):
        """
        Initialize state converter.
        
        Args:
            env: Robosuite environment instance
            camera_names: Camera names for RGB-D capture
            num_points: Target number of points per object point cloud
            voxel_size: Voxel size for point cloud downsampling (meters)
            workspace_bounds: Workspace bounds for filtering [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        """
        self.env = env
        self.sim = env.sim
        self.camera_names = camera_names or ["frontview", "agentview"]
        self.num_points = num_points
        
        # Default workspace bounds
        if workspace_bounds is None:
            workspace_bounds = np.array([
                [-0.5, 0.5],   # x bounds
                [-0.5, 0.5],   # y bounds
                [0.7, 1.5]     # z bounds
            ])
        self.workspace_bounds = workspace_bounds
        
        # Initialize point cloud generator
        self.pcd_generator = PointCloudGenerator(
            voxel_size=voxel_size,
            bounds=self.workspace_bounds
        )
        
        # Extract object metadata (once at initialization)
        self.metadata_extractor = MetadataExtractor(self.sim)
        self.object_metadata = self.metadata_extractor.extract_all_objects()
        self.num_objects = len(self.object_metadata)
        
        # Create consistent object ordering
        self.object_names = sorted(self.object_metadata.keys())
        self.object_name_to_id = {name: idx for idx, name in enumerate(self.object_names)}
        
        # Build object type to index mapping (for one-hot encoding)
        self.object_types = sorted(list(set(
            meta['type'] for meta in self.object_metadata.values()
        )))
        self.type_to_idx = {obj_type: idx for idx, obj_type in enumerate(self.object_types)}
        
        print(f"StateConverter initialized:")
        print(f"  Objects: {self.num_objects}")
        print(f"  Object types: {self.object_types}")
        print(f"  Points per object: {num_points}")
        print(f"  Cameras: {self.camera_names}")
    
    def convert(self, obs: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Convert current robosuite state to Points2Plans format.
        
        Args:
            obs: Optional observation dict (if None, will get from env)
        
        Returns:
            Dictionary with keys required for Points2Plans inference:
            - batch_voxel_list_single: [1, num_objects, num_points, 3]
            - batch_one_hot_encoding: [1, num_objects, num_types]
            - batch_6DOF_pose: [1, num_objects, 6]
            - batch_edge_attr: [2, num_edges] edge indices
            - batch_num_objects: int
            
        NOTE: NO predicates! Decoder predicts them during forward pass.
        """
        # Get current observation if not provided
        if obs is None:
            obs = self._get_current_obs()
        
        # 1. Generate point clouds for each object
        point_clouds = self._generate_point_clouds()
        
        # 2. Extract object poses
        object_poses = self._extract_object_poses(obs)
        
        # 3. Build one-hot encodings
        one_hot_encodings = self._build_one_hot_encodings()
        
        # 4. Build edge indices (fully connected graph)
        edge_indices = self._build_edge_indices()
        
        # 5. Format as tensors
        state_dict = {
            'batch_voxel_list_single': torch.FloatTensor(point_clouds).unsqueeze(0),  # [1, N, P, 3]
            'batch_one_hot_encoding': torch.FloatTensor(one_hot_encodings).unsqueeze(0),  # [1, N, T]
            'batch_6DOF_pose': torch.FloatTensor(object_poses).unsqueeze(0),  # [1, N, 6]
            'batch_edge_attr': torch.LongTensor(edge_indices),  # [2, E]
            'batch_num_objects': self.num_objects,
        }
        
        return state_dict
    
    def _get_current_obs(self) -> Dict:
        """Get current observation from environment."""
        # Get observation with camera data
        obs = self.env._get_observations(force_update=True)
        return obs
    
    def _generate_point_clouds(self) -> np.ndarray:
        """
        Generate point clouds for each object.
        
        Returns:
            Array of shape [num_objects, num_points, 3]
        """
        point_clouds = []
        
        # Capture RGB-D from all cameras
        all_pcds = []
        for cam_name in self.camera_names:
            rgb, depth = self.pcd_generator.capture_rgbd(self.sim, [cam_name])
            pcd = self.pcd_generator.rgbd_to_pointcloud(rgb[0], depth[0], cam_name)
            all_pcds.append(pcd)
        
        # Merge point clouds from all cameras
        if len(all_pcds) > 1:
            full_pcd = np.vstack(all_pcds)
        else:
            full_pcd = all_pcds[0]
        
        # Segment by object
        segmented = self.pcd_generator.segment_by_objects(
            full_pcd,
            self.sim,
            self.object_metadata
        )
        
        # Resample each object's point cloud to fixed size
        for obj_name in self.object_names:
            obj_pcd = segmented.get(obj_name, None)
            
            if obj_pcd is None or len(obj_pcd) == 0:
                # No points for this object, use zeros
                resampled = np.zeros((self.num_points, 3))
            else:
                # Resample to target size
                resampled = self._resample_point_cloud(obj_pcd, self.num_points)
            
            point_clouds.append(resampled)
        
        return np.array(point_clouds)  # [num_objects, num_points, 3]
    
    def _resample_point_cloud(self, points: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resample point cloud to target size.
        
        Args:
            points: Input points [N, 3]
            target_size: Target number of points
        
        Returns:
            Resampled points [target_size, 3]
        """
        num_points = len(points)
        
        if num_points == 0:
            return np.zeros((target_size, 3))
        
        if num_points >= target_size:
            # Downsample: randomly select
            indices = np.random.choice(num_points, target_size, replace=False)
            return points[indices]
        else:
            # Upsample: random selection with replacement
            indices = np.random.choice(num_points, target_size, replace=True)
            return points[indices]
    
    def _extract_object_poses(self, obs: Dict) -> np.ndarray:
        """
        Extract 6DOF poses for all objects.
        
        Args:
            obs: Observation dictionary
        
        Returns:
            Array of shape [num_objects, 6] containing [x, y, z, roll, pitch, yaw]
        """
        poses = []
        
        for obj_name in self.object_names:
            # Get position
            pos_key = f"{obj_name}_pos"
            if pos_key in obs:
                pos = obs[pos_key]
            else:
                # Try to get from sim directly
                obj_id = self.sim.model.body_name2id(obj_name)
                pos = self.sim.data.body_xpos[obj_id]
            
            # Get orientation (as quaternion, convert to euler)
            quat_key = f"{obj_name}_quat"
            if quat_key in obs:
                quat = obs[quat_key]
            else:
                obj_id = self.sim.model.body_name2id(obj_name)
                quat = self.sim.data.body_xquat[obj_id]
            
            # Convert quaternion to euler angles (roll, pitch, yaw)
            euler = self._quat_to_euler(quat)
            
            # Combine position and orientation
            pose = np.concatenate([pos, euler])
            poses.append(pose)
        
        return np.array(poses)  # [num_objects, 6]
    
    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to euler angles (roll, pitch, yaw).
        
        Args:
            quat: Quaternion [w, x, y, z] or [x, y, z, w]
        
        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        # Check if quaternion is in [w, x, y, z] or [x, y, z, w] format
        # MuJoCo uses [w, x, y, z]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def _build_one_hot_encodings(self) -> np.ndarray:
        """
        Build one-hot encodings for object types.
        
        Returns:
            Array of shape [num_objects, num_types]
        """
        num_types = len(self.object_types)
        encodings = np.zeros((self.num_objects, num_types))
        
        for obj_idx, obj_name in enumerate(self.object_names):
            obj_type = self.object_metadata[obj_name]['type']
            type_idx = self.type_to_idx[obj_type]
            encodings[obj_idx, type_idx] = 1.0
        
        return encodings
    
    def _build_edge_indices(self) -> np.ndarray:
        """
        Build edge indices for fully connected graph (excluding self-loops).
        
        Returns:
            Array of shape [2, num_edges] where num_edges = num_objects * (num_objects - 1)
        """
        from itertools import permutations
        
        nodes = list(range(self.num_objects))
        edges = list(permutations(nodes, 2))  # All pairs excluding self-loops
        edge_indices = np.array(edges).T  # [2, num_edges]
        
        return edge_indices
    
    def get_object_list(self) -> List[str]:
        """Get list of object names in consistent order."""
        return self.object_names.copy()
    
    def get_object_id(self, obj_name: str) -> int:
        """Get object ID from name."""
        return self.object_name_to_id.get(obj_name, -1)
