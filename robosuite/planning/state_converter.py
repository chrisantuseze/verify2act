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
sys.path.insert(0, str(Path(__file__).parent.parent / "robosuite" / "utils"))

from pointcloud_generator import PointCloudGenerator
# from robosuite.utils.pointcloud_generator import PointCloudGenerator

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
                 workspace_bounds: Optional[np.ndarray] = None,
                 max_objects: int = 12,
                 object_filter: Optional[List[str]] = None,
                 training_compatible_one_hot: bool = False,
                 one_hot_seed: Optional[int] = None):
        """
        Initialize state converter.
        
        Args:
            env: Robosuite environment instance
            camera_names: Camera names for RGB-D capture
            num_points: Target number of points per object point cloud
            voxel_size: Voxel size for point cloud downsampling (meters)
            workspace_bounds: Workspace bounds for filtering [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            max_objects: Maximum number of objects for one-hot encoding (must match training config)
            object_filter: Optional list of object name patterns to keep (e.g., ['nut', 'peg'] for assembly tasks)
                          If None, auto-detects based on environment name
            training_compatible_one_hot: If True, mimic Points2Plans training behavior by
                                        assigning each object a randomized embedding slot.
                                        Points2Plans trains with random slot assignment per scene;
                                        enabling this at inference matches that distribution.
            one_hot_seed: Optional int seed for deterministic slot assignment (reproducible runs).
        """
        self.env = env
        self.sim = env.sim
        self.camera_names = camera_names or ["frontview", "agentview"]
        self.num_points = num_points
        self.max_objects = max_objects  # Must match training config for embedding layer
        self.training_compatible_one_hot = training_compatible_one_hot
        self.one_hot_seed = one_hot_seed
        
        # Auto-detect task-specific filter if not provided
        if object_filter is None:
            object_filter = self._get_default_filter_for_task(env)
        self.object_filter = object_filter
        
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
        raw_metadata = self.metadata_extractor.extract_all_objects()
        
        # Normalize object names: strip MuJoCo suffixes like "_main" for cleaner semantics
        # Store mapping from clean names to full MuJoCo names for internal use
        self.mujoco_name_map = {}  # clean_name -> full_mujoco_name
        self.object_metadata = {}  # clean_name -> metadata
        
        for full_name, metadata in raw_metadata.items():
            # Extract base name (e.g., "cubeA_main" -> "cubeA", "Milk_main" -> "Milk")
            clean_name = full_name.split('_')[0] if '_' in full_name else full_name
            self.mujoco_name_map[clean_name] = full_name
            self.object_metadata[clean_name] = metadata
        
        # Apply task-specific filtering if specified
        if self.object_filter:
            print(f"Applying task-specific object filter: {self.object_filter}")
            self._apply_object_filter()
        
        self.num_objects = len(self.object_metadata)
        
        # Use clean names everywhere
        self.object_names = sorted(self.object_metadata.keys())
        if 'table' in self.object_names:
            self.object_names.remove('table')
            self.object_names.insert(0, 'table')
        self.object_name_to_id = {name: idx for idx, name in enumerate(self.object_names)}
        self.object_slot_indices = self._build_object_slot_indices()
        
        # Check if we still exceed max_objects after filtering
        if self.num_objects > self.max_objects:
            print(f"⚠ Warning: After filtering, still have {self.num_objects} objects but model supports max {self.max_objects}")
            print(f"  Consider refining your object filter or increasing max_objects")
        
        print(f"Detected objects (clean names): {self.object_names}")
        print(f"MuJoCo name mapping: {self.mujoco_name_map}")
        
        # Build object type to index mapping (for one-hot encoding)
        self.object_types = sorted(list(set(
            meta.get('object_type', 'unknown') for meta in self.object_metadata.values()
        )))
        self.type_to_idx = {obj_type: idx for idx, obj_type in enumerate(self.object_types)}
        
        print(f"StateConverter initialized:")
        print(f"  Objects: {self.num_objects}")
        print(f"  Max objects (for embedding): {self.max_objects}")
        print(f"  Object types: {self.object_types}")
        print(f"  Points per object: {num_points}")
        print(f"  Cameras: {self.camera_names}")
        if self.training_compatible_one_hot:
            print(f"  Training-compatible one-hot slots: {self.object_slot_indices.tolist()}")
    
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

        state_dict['object_names'] = self.get_object_list()
        
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
        missing_objects = []
        
        # Generate segmented point clouds for all objects
        # Pass clean names since pointcloud_generator returns clean names (e.g., "cubeA" not "cubeA_main")
        object_pcds_raw = self.pcd_generator.generate_segmented(
            self.env,
            self.camera_names,
            object_names=self.object_names  # Use clean names
        )
        
        # Normalize point-cloud keys to clean names (e.g., "cubeA_main" -> "cubeA")
        # to match self.object_names from metadata extraction.
        object_pcds = {}
        for raw_name, pcd in object_pcds_raw.items():
            clean_name = raw_name.split('_')[0] if '_' in raw_name else raw_name
            if clean_name not in object_pcds:
                object_pcds[clean_name] = pcd
                continue

            # If multiple geoms/bodies map to the same clean object name,
            # keep the richer point cloud.
            if len(pcd.points) > len(object_pcds[clean_name].points):
                object_pcds[clean_name] = pcd
        
        # Convert to numpy arrays and resample to fixed size
        for clean_name in self.object_names:
            # Look up by clean name (what pointcloud_generator returns)
            pcd = object_pcds.get(clean_name, None)
            
            if pcd is None or len(pcd.points) == 0:
                # No points for this object, use zeros
                resampled = np.zeros((self.num_points, 3))
                missing_objects.append(clean_name)
            else:
                # Convert Open3D point cloud to numpy
                points = np.asarray(pcd.points)
                # Resample to target size
                resampled = self._resample_point_cloud(points, self.num_points)
            
            point_clouds.append(resampled)
        
        if missing_objects:
            print(f"Warning: Missing segmented points for objects: {missing_objects}")

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
            # Deterministic downsample for stable inference
            indices = np.linspace(0, num_points - 1, target_size, dtype=np.int64)
            return points[indices]
        else:
            # Deterministic upsample by tiling + remainder
            repeats = target_size // num_points
            remainder = target_size % num_points
            tiled = np.tile(points, (repeats, 1))
            if remainder > 0:
                tiled = np.concatenate([tiled, points[:remainder]], axis=0)
            return tiled
    
    def _extract_object_poses(self, obs: Dict) -> np.ndarray:
        """
        Extract 6DOF poses for all objects.
        
        Args:
            obs: Observation dictionary
        
        Returns:
            Array of shape [num_objects, 6] containing [x, y, z, roll, pitch, yaw]
        """
        poses = []
        
        for clean_name in self.object_names:
            full_name = self.mujoco_name_map[clean_name]
            
            # Get position
            pos_key = f"{full_name}_pos"
            if pos_key in obs:
                pos = obs[pos_key]
            else:
                # Try to get from sim directly using MuJoCo name
                obj_id = self.env.sim.model.body_name2id(full_name)
                pos = self.env.sim.data.body_xpos[obj_id]
            
            # Get orientation (as quaternion, convert to euler)
            quat_key = f"{full_name}_quat"
            if quat_key in obs:
                quat = obs[quat_key]
            else:
                obj_id = self.env.sim.model.body_name2id(full_name)
                quat = self.env.sim.data.body_xquat[obj_id]
            
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
        Build one-hot encodings for objects.
        
        Each object gets a unique index in [0, max_objects-1].
        The argmax of each row gives the object's index for the embedding lookup.
        
        This matches training format where:
        - one_hot_encoding shape is [num_objects, max_objects]
        - Each object has exactly one 1 in its row
        - argmax(dim=-1) gives object indices for nn.Embedding lookup
        
        Returns:
            Array of shape [num_objects, max_objects]
        """
        # Create one-hot encoding with max_objects columns
        # Object i gets an embedding slot from self.object_slot_indices.
        encodings = np.zeros((self.num_objects, self.max_objects))
        
        for obj_idx in range(self.num_objects):
            encodings[obj_idx, self.object_slot_indices[obj_idx]] = 1.0
        
        return encodings

    def _build_object_slot_indices(self) -> np.ndarray:
        """Build object->embedding-slot mapping for one-hot encoding."""
        if self.max_objects <= 0:
            raise ValueError(f"max_objects must be > 0, got {self.max_objects}")

        if self.num_objects > self.max_objects:
            raise ValueError(
                f"num_objects ({self.num_objects}) exceeds max_objects ({self.max_objects}). "
                f"Use object_filter to reduce tracked objects or increase model capacity."
            )

        if not self.training_compatible_one_hot:
            return np.arange(self.num_objects, dtype=np.int64)

        all_slots = np.arange(self.max_objects, dtype=np.int64)
        rng = np.random.default_rng(self.one_hot_seed)
        rng.shuffle(all_slots)
        return all_slots[:self.num_objects]
    
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
        """
        Get object ID from name (case-insensitive).
        
        Args:
            obj_name: Object name (e.g., "cubeA", "cubea", "Milk", "milk")
        
        Returns:
            Object ID (index), or -1 if not found
        """
        # Try exact match first
        if obj_name in self.object_name_to_id:
            return self.object_name_to_id[obj_name]
        
        # Try case-insensitive match
        obj_lower = obj_name.lower()
        for name in self.object_names:
            if name.lower() == obj_lower:
                return self.object_name_to_id[name]
        
        # Try partial match
        for name in self.object_names:
            if obj_lower in name.lower() or name.lower() in obj_lower:
                return self.object_name_to_id[name]
        
        return -1
    
    def get_mujoco_name(self, clean_name: str) -> str:
        """
        Get full MuJoCo body name from clean name.
        
        Args:
            clean_name: Clean object name (e.g., "cubeA")
        
        Returns:
            Full MuJoCo name (e.g., "cubeA_main") or clean_name if not found
        """
        return self.mujoco_name_map.get(clean_name, clean_name)
    
    def _get_default_filter_for_task(self, env) -> Optional[List[str]]:
        """
        Get default object filter based on environment/task type.
        
        Args:
            env: Robosuite environment
        
        Returns:
            List of object name patterns to keep, or None for no filtering
        """
        env_name = env.__class__.__name__.lower()
        
        # Task-specific filters
        if 'nut' in env_name or 'assembly' in env_name:
            # NutAssembly tasks: keep nuts and pegs only
            return ['nut', 'peg', 'round-nut', 'square-nut', 'table']
        
        elif 'stack' in env_name:
            # Stacking tasks: keep cubes only
            return ['cube', 'table']
        
        elif 'pickplace' in env_name:
            # PickPlace tasks: keep manipulable objects (not bin/table)
            return ['milk', 'bread', 'cereal', 'can', 'cube', 'object', 'table']
        
        elif 'door' in env_name:
            # Door tasks: keep door and handle
            return ['door', 'handle', 'table']
        
        else:
            # Unknown task: no filtering (keep all objects)
            return None
    
    def _apply_object_filter(self):
        """
        Apply task-specific object filtering.
        
        Keeps only objects whose names match any pattern in self.object_filter.
        Uses case-insensitive substring matching.
        """
        if not self.object_filter:
            return
        
        original_count = len(self.object_metadata)
        original_names = list(self.object_metadata.keys())
        
        # Filter objects
        filtered_metadata = {}
        filtered_mujoco_map = {}
        
        for obj_name in original_names:
            # Check if object matches any filter pattern
            obj_lower = obj_name.lower()
            matches = any(pattern.lower() in obj_lower for pattern in self.object_filter)
            
            if matches:
                filtered_metadata[obj_name] = self.object_metadata[obj_name]
                filtered_mujoco_map[obj_name] = self.mujoco_name_map[obj_name]
        
        # Update state
        self.object_metadata = filtered_metadata
        self.mujoco_name_map = filtered_mujoco_map
        
        filtered_count = len(self.object_metadata)
        removed_count = original_count - filtered_count
        
        if removed_count > 0:
            removed_names = [name for name in original_names if name not in self.object_metadata]
            print(f"  Filtered: {original_count} → {filtered_count} objects ({removed_count} removed)")
            print(f"  Kept: {sorted(self.object_metadata.keys())}")
            print(f"  Removed: {sorted(removed_names)}")
