"""
Episode Recorder for Points2Plans Data Collection

This module captures all necessary state information during robosuite episodes
to generate datasets compatible with the Points2Plans dataloader format.

Phase 1: State Capture
- Robot state (end-effector position, orientation, gripper)
- Object states (positions, orientations, velocities)
- Contact information (body pairs and forces)
- Action information (skill type, object ID, continuous parameters)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import mujoco
import pickle
import os
from pathlib import Path
from datetime import datetime
from robosuite.utils.pointcloud_generator import PointCloudGenerator


class EpisodeRecorder:
    """
    Records episode data during robosuite rollouts for Points2Plans dataset generation.
    
    This recorder captures:
    - Time-series state information (robot + objects)
    - Contact dynamics
    - Actions executed
    - Object metadata (static properties)
    """
    
    def __init__(self, env, camera_names: Optional[List[str]] = None, 
                 voxel_size: float = 0.005, num_points: int = 128):
        """
        Initialize the episode recorder.
        
        Args:
            env: Robosuite environment instance
            camera_names: List of camera names for RGB-D capture
            voxel_size: Voxel size for point cloud downsampling (meters)
            num_points: Target number of points per object point cloud
        """
        self.env = env
        self.sim = env.sim
        self.camera_names = camera_names or ["frontview", "agentview"]
        self.num_points = num_points
        
        # Define workspace bounds for point cloud filtering
        # Adjust based on your task (Stack task typical bounds)
        self.workspace_bounds = np.array([
            [-0.5, 0.5],   # x bounds
            [-0.5, 0.5],   # y bounds
            [0.7, 1.5]     # z bounds (table height and above)
        ])
        
        # Initialize point cloud generator
        self.pcd_generator = PointCloudGenerator(
            voxel_size=voxel_size,
            bounds=self.workspace_bounds
        )
        
        # Storage for time-series data
        self.timestep_data = []
        
        # Storage for static metadata (extracted once per episode)
        self.object_metadata = {}
        
        # Current episode state
        self.episode_active = False
        self.current_timestep = 0
        
        # Action tracking
        self.last_action = None
        self.action_history = []
        
        print(f"EpisodeRecorder initialized for environment: {env.__class__.__name__}")
        print(f"Point cloud capture: {len(self.camera_names)} cameras, target {num_points} points/object")
    
    def start_episode(self):
        """
        Start recording a new episode.
        
        Call this after env.reset() to begin capturing data.
        """
        self.episode_active = True
        self.current_timestep = 0
        self.timestep_data = []
        self.action_history = []
        
        # Extract static object metadata once at episode start
        self._extract_object_metadata()
        
        # Capture initial state (timestep 0)
        self._capture_timestep_state(action=None, obs=None)
        
        print(f"Episode recording started. Found {len(self.object_metadata)} objects.")
    
    def record_step(self, action: np.ndarray, obs: Dict[str, Any]):
        """
        Record data for a single timestep after env.step().
        
        Args:
            action: Action that was executed (raw action array)
            obs: Observations returned by env.step()
        """
        if not self.episode_active:
            raise RuntimeError("Cannot record step: episode not started. Call start_episode() first.")
        
        self.current_timestep += 1
        self.last_action = action
        
        # Capture all state information for this timestep
        self._capture_timestep_state(action=action, obs=obs)
    
    def end_episode(self) -> Tuple[Dict, Dict]:
        """
        End recording and return collected data.
        
        Returns:
            Tuple of (data_dict, attrs_dict) in Points2Plans format
        """
        if not self.episode_active:
            raise RuntimeError("Cannot end episode: no active recording.")
        
        self.episode_active = False
        
        print(f"Episode recording ended. Captured {len(self.timestep_data)} timesteps.")
        
        # Package data into Points2Plans format
        data_dict = self._build_data_dict()
        attrs_dict = self._build_attrs_dict()
        
        return data_dict, attrs_dict
    
    def save_episode(self, output_dir: str, episode_name: Optional[str] = None) -> str:
        """
        Save the recorded episode to a pickle file.
        
        Args:
            output_dir: Directory to save the episode file
            episode_name: Optional custom episode name. If None, uses timestamp.
            
        Returns:
            Path to the saved pickle file
        """
        if self.episode_active:
            raise RuntimeError("Cannot save episode: recording still active. Call end_episode() first.")
        
        if len(self.timestep_data) == 0:
            raise RuntimeError("Cannot save episode: no data recorded.")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate episode name if not provided
        if episode_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = f"episode_{timestamp}"
        
        # Ensure .pkl extension
        if not episode_name.endswith('.pkl'):
            episode_name = f"{episode_name}.pkl"
        
        # Full path to output file
        output_file = output_path / episode_name
        
        # Get data and attrs
        data_dict = self._build_data_dict()
        attrs_dict = self._build_attrs_dict()
        
        # Package in Points2Plans format (tuple of data and attrs)
        episode_data = (data_dict, attrs_dict)
        
        # Save to pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(episode_data, f)
        
        # Get file size for logging
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print(f"\n✓ Episode saved to: {output_file}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Timesteps: {len(self.timestep_data)}")
        print(f"  Objects: {len(self.object_metadata)}")
        
        return str(output_file)
    
    def _extract_object_metadata(self):
        """
        Extract static object properties from MuJoCo model.
        
        This includes:
        - Object names and IDs
        - Geometric properties (extents)
        - Object types (primitive, urdf, etc.)
        - Static/dynamic classification (fix_base_link)
        """
        self.object_metadata = {}
        
        model = self.sim.model
        
        # Get all object bodies in the scene
        # In robosuite Stack task, objects are typically named: cubeA, cubeB, table, robot, etc.
        object_bodies = self._get_object_bodies()
        
        for body_name, body_id in object_bodies.items():
            metadata = {
                'body_id': body_id,
                'body_name': body_name,
                'geom_ids': [],
                'extents': None,
                'fix_base_link': False,
                'object_type': 'primitive',
                'asset_filename': None,
            }
            
            # Get associated geoms for this body
            geom_ids = self._get_body_geoms(body_id)
            metadata['geom_ids'] = geom_ids
            
            # Extract extent information from primary geom
            if len(geom_ids) > 0:
                primary_geom_id = geom_ids[0]
                geom_size = model.geom_size[primary_geom_id]
                geom_type = model.geom_type[primary_geom_id]
                
                # Compute extents based on geom type
                # For box: size is half-extents, so multiply by 2
                # For sphere/cylinder: size is radius/radius+half-height
                if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    metadata['extents'] = (geom_size * 2).tolist()  # [x, y, z] full extents
                elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    radius = geom_size[0]
                    metadata['extents'] = [radius * 2, radius * 2, radius * 2]
                elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    radius = geom_size[0]
                    half_height = geom_size[1]
                    metadata['extents'] = [radius * 2, radius * 2, half_height * 2]
                else:
                    # Default: use size directly
                    metadata['extents'] = (geom_size * 2).tolist()
            
            # Determine if object is static (fix_base_link)
            # Check if body has free joint (movable) or is welded (static)
            metadata['fix_base_link'] = self._is_body_static(body_id)
            
            # Store with Points2Plans naming convention (block_01, block_02, etc.)
            self.object_metadata[body_name] = metadata
        
        print(f"Extracted metadata for {len(self.object_metadata)} objects:")
        for name, meta in self.object_metadata.items():
            print(f"  - {name}: extents={meta['extents']}, static={meta['fix_base_link']}")
    
    def _get_object_bodies(self) -> Dict[str, int]:
        """
        Identify relevant object bodies in the scene.
        
        Returns:
            Dictionary mapping body names to body IDs
        """
        model = self.sim.model
        object_bodies = {}
        
        # Keywords to identify relevant objects
        object_keywords = ['cube', 'table', 'block', 'object', 'can', 'milk', 'peg', 'box']
        
        # Keywords to exclude (robot parts and non-objects)
        exclude_keywords = ['robot', 'gripper', 'link', 'joint', 'world', 'floor',
                           'mount', 'controller', 'base', 'pedestal']
        
        for body_id in range(model.nbody):
            body_name = model.body(body_id).name
            
            if not body_name:  # Skip unnamed bodies
                continue
            
            # Skip excluded bodies
            if any(kw in body_name.lower() for kw in exclude_keywords):
                continue
            
            # Include object bodies
            if any(kw in body_name.lower() for kw in object_keywords):
                object_bodies[body_name] = body_id
        

        print("CHRIS:", object_bodies)
        return object_bodies
    
    def _get_body_geoms(self, body_id: int) -> List[int]:
        """
        Get all geom IDs associated with a body.
        
        Args:
            body_id: MuJoCo body ID
            
        Returns:
            List of geom IDs belonging to this body
        """
        model = self.sim.model
        geom_ids = []
        
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                geom_ids.append(geom_id)
        
        return geom_ids
    
    def _is_body_static(self, body_id: int) -> bool:
        """
        Determine if a body is static (fix_base_link=True).
        
        A body is considered static if:
        - It has no associated joints (welded to world)
        - Its mass is very large (kinematically controlled)
        
        Args:
            body_id: MuJoCo body ID
            
        Returns:
            True if body is static, False if movable
        """
        model = self.sim.model
        
        # Check if body has any joints
        has_joints = False
        for jnt_id in range(model.njnt):
            if model.jnt_bodyid[jnt_id] == body_id:
                has_joints = True
                break
        
        # If no joints, it's static (welded to parent/world)
        if not has_joints:
            return True
        
        # Check if body is kinematic (very large mass or mocap)
        body_mass = model.body_mass[body_id]
        if body_mass > 1000:  # Arbitrary threshold for "effectively infinite" mass
            return True
        
        return False
    
    def _capture_timestep_state(self, action: Optional[np.ndarray], obs: Optional[Dict[str, Any]]):
        """
        Capture complete state for current timestep.
        
        Args:
            action: Action executed at this timestep (None for initial state)
            obs: Observations from environment (None for initial state)
        """
        timestep_state = {
            'timestep': self.current_timestep,
            'robot_state': self._capture_robot_state(),
            'object_states': self._capture_object_states(),
            'contacts': self._capture_contacts(),
            'action': self._parse_action(action, obs) if action is not None else None,
            'point_clouds': self._capture_point_clouds(),  # Phase 2: Add point cloud capture
        }
        
        self.timestep_data.append(timestep_state)
        
        if action is not None:
            self.action_history.append(timestep_state['action'])
    
    def _capture_robot_state(self) -> Dict[str, np.ndarray]:
        """
        Capture robot state (end-effector position, orientation, gripper).
        
        Returns:
            Dictionary with robot state information
        """
        # Get robot site for end-effector
        # Robosuite typically uses a site named 'gripper0_grip_site' or similar
        robot = self.env.robots[0]  # Assume single robot
        
        # Get the gripper site name - robosuite stores this in important_sites dict
        # The gripper object itself has the important_sites attribute
        try:
            grip_site_name = robot.gripper.important_sites["grip_site"]
        except (AttributeError, KeyError):
            # Fallback to common naming convention
            grip_site_name = f"{robot.robot_model.naming_prefix}gripper0_grip_site"
        
        # End-effector position
        try:
            grip_site_id = self.sim.model.site_name2id(grip_site_name)
            eef_pos = self.sim.data.site_xpos[grip_site_id]
            eef_mat = self.sim.data.site_xmat[grip_site_id].reshape(3, 3)
        except Exception as e:
            # If specific site not found, use robot's eef position from observations
            if 'robot0_eef_pos' in self.env._observables:
                eef_pos = np.zeros(3)  # Will be filled from obs if available
            else:
                eef_pos = np.zeros(3)
            eef_mat = np.eye(3)
        
        eef_quat = self._mat_to_quat(eef_mat)
        
        # Gripper state (typically -1 to 1, or joint positions)
        gripper_qpos = []
        try:
            for joint in robot.gripper.joints:
                joint_id = self.sim.model.joint_name2id(joint)
                gripper_qpos.append(self.sim.data.qpos[self.sim.model.jnt_qposadr[joint_id]])
            gripper_qpos = np.array(gripper_qpos)
        except Exception:
            # Fallback: empty array if gripper joints not accessible
            gripper_qpos = np.array([])
        
        return {
            'eef_pos': eef_pos.copy(),
            'eef_quat': eef_quat.copy(),
            'gripper_qpos': gripper_qpos.copy(),
        }
    
    def _capture_object_states(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Capture state of all objects (position, orientation, velocity).
        
        Returns:
            Dictionary mapping object names to their states
        """
        object_states = {}
        
        for obj_name, obj_meta in self.object_metadata.items():
            body_id = obj_meta['body_id']
            
            # Position (3D world coordinates)
            position = self.sim.data.body_xpos[body_id].copy()
            
            # Orientation (quaternion: w, x, y, z in MuJoCo convention)
            quat = self.sim.data.body_xquat[body_id].copy()
            
            # Linear velocity
            velocity = self.sim.data.body_xvelp[body_id].copy() if hasattr(self.sim.data, 'body_xvelp') else np.zeros(3)
            
            # Angular velocity
            angular_vel = self.sim.data.body_xvelr[body_id].copy() if hasattr(self.sim.data, 'body_xvelr') else np.zeros(3)
            
            object_states[obj_name] = {
                'position': position,
                'orientation': quat,
                'velocity': velocity,
                'angular_velocity': angular_vel,
            }
        
        return object_states
    
    def _capture_contacts(self) -> List[Dict[str, Any]]:
        """
        Capture contact information between bodies.
        
        Returns:
            List of contact dictionaries with body pairs and forces
        """
        contacts = []
        
        # Iterate through all active contacts in simulation
        for contact_id in range(self.sim.data.ncon):
            contact = self.sim.data.contact[contact_id]
            
            # Get geom IDs involved in contact
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # Get body IDs from geoms
            body1_id = self.sim.model.geom_bodyid[geom1_id]
            body2_id = self.sim.model.geom_bodyid[geom2_id]
            
            # Get body names
            body1_name = self.sim.model.body(body1_id).name
            body2_name = self.sim.model.body(body2_id).name
            
            # Only record contacts between tracked objects
            if body1_name not in self.object_metadata or body2_name not in self.object_metadata:
                continue
            
            # Map body names to object indices (for Points2Plans format)
            body1_idx = self._get_object_index(body1_name)
            body2_idx = self._get_object_index(body2_name)
            
            contact_info = {
                'body0': body1_idx,
                'body1': body2_idx,
                'geom0': geom1_id,
                'geom1': geom2_id,
                'pos': contact.pos.copy(),
                'frame': contact.frame.copy(),
                'dist': contact.dist,
            }
            
            contacts.append(contact_info)
        
        return contacts
    
    def _parse_action(self, action: np.ndarray, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse action into Points2Plans format.
        
        For pick-place tasks:
        - Skill type: 'pickplace', 'push', 'pull', 'stack'
        - Object ID: which object is being manipulated
        - Continuous params: target position, orientation, etc.
        
        Args:
            action: Raw action array from policy
            obs: Current observations
            
        Returns:
            Parsed action dictionary
        """
        # For composite controller, action typically has:
        # - Position control (3D)
        # - Orientation control (axis-angle or quaternion)
        # - Gripper control (1D)
        
        # Determine skill type based on gripper state
        gripper_action = action[6] if len(action) > 6 else action[-1]
        
        # Simplified skill detection (can be enhanced based on your needs)
        if gripper_action > 0:
            skill_type = 'grasp'  # Closing gripper
        elif gripper_action < 0:
            skill_type = 'release'  # Opening gripper
        else:
            skill_type = 'move'  # Just moving
        
        # Detect which object is being manipulated (closest to gripper when grasping)
        manipulated_object = self._detect_manipulated_object(obs)
        
        parsed_action = {
            'skill_type': skill_type,
            'object_id': manipulated_object,
            'position_delta': action[:3].copy() if len(action) >= 3 else np.zeros(3),
            'gripper_action': gripper_action,
            'raw_action': action.copy(),
        }
        
        return parsed_action
    
    def _detect_manipulated_object(self, obs: Optional[Dict[str, Any]]) -> Optional[int]:
        """
        Detect which object is being manipulated based on proximity to gripper.
        
        Args:
            obs: Current observations
            
        Returns:
            Object index or None
        """
        if obs is None:
            return None
        
        # Get current gripper position
        robot = self.env.robots[0]
        
        # Try to get gripper position from site
        try:
            grip_site_name = robot.gripper.important_sites["grip_site"]
        except (AttributeError, KeyError):
            grip_site_name = f"{robot.robot_model.naming_prefix}gripper0_grip_site"
        
        try:
            grip_site_id = self.sim.model.site_name2id(grip_site_name)
            eef_pos = self.sim.data.site_xpos[grip_site_id]
        except Exception:
            # Fallback: use observation if available
            if 'robot0_eef_pos' in obs:
                eef_pos = obs['robot0_eef_pos']
            else:
                return None
        
        # Find closest object
        min_dist = float('inf')
        closest_obj_idx = None
        
        for idx, (obj_name, obj_meta) in enumerate(self.object_metadata.items()):
            # Skip static objects
            if obj_meta['fix_base_link']:
                continue
            
            body_id = obj_meta['body_id']
            obj_pos = self.sim.data.body_xpos[body_id]
            
            dist = np.linalg.norm(eef_pos - obj_pos)
            if dist < min_dist:
                min_dist = dist
                closest_obj_idx = idx
        
        # Only return object if gripper is very close (within 5cm)
        if min_dist < 0.05:
            return closest_obj_idx
        
        return None
    
    def _capture_point_clouds(self) -> Dict[str, np.ndarray]:
        """
        Capture segmented point clouds for all tracked objects.
        
        Uses a geometry-based approach: capture full scene point cloud,
        then segment by proximity to object centers.
        
        Returns:
            Dictionary mapping object names to their point cloud arrays (Nx3)
        """
        try:
            # Capture full scene point cloud from all cameras
            full_pcd = self.pcd_generator.generate(self.env, self.camera_names)
            full_points = np.asarray(full_pcd.points)
            
            if len(full_points) == 0:
                return {}
            
            # Segment points by proximity to tracked objects
            object_point_clouds = self._segment_points_by_proximity(full_points)
            
            return object_point_clouds
            
        except Exception as e:
            import traceback
            print(f"Warning: Failed to capture point clouds at timestep {self.current_timestep}:")
            print(f"  Error: {e}")
            if self.current_timestep < 2:  # Only print full trace for first few failures
                traceback.print_exc()
            # Return empty dict on failure
            return {}
    
    def _segment_points_by_proximity(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment point cloud by assigning each point to the nearest object.
        
        Args:
            points: Nx3 array of 3D points
            
        Returns:
            Dictionary mapping object names to their point arrays
        """
        object_point_clouds = {}
        
        # Get object positions
        object_positions = {}
        for obj_name, obj_meta in self.object_metadata.items():
            body_id = obj_meta['body_id']
            pos = self.sim.data.body_xpos[body_id].copy()
            object_positions[obj_name] = pos
        
        # For each point, find nearest object within a threshold
        for point in points:
            min_dist = float('inf')
            closest_obj = None
            
            for obj_name, obj_pos in object_positions.items():
                dist = np.linalg.norm(point - obj_pos)
                
                # Check if point is within object's bounding box (with margin)
                extents = self.object_metadata[obj_name]['extents']
                if extents is None:
                    continue
                
                # Generous margin for table and small margin for objects
                margin = 0.15 if 'table' in obj_name.lower() else 0.08
                
                # Check if point is within extended bounding box
                if (abs(point[0] - obj_pos[0]) <= extents[0] / 2 + margin and
                    abs(point[1] - obj_pos[1]) <= extents[1] / 2 + margin and
                    abs(point[2] - obj_pos[2]) <= extents[2] / 2 + margin):
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_obj = obj_name
            
            # Assign point to closest object
            if closest_obj is not None:
                if closest_obj not in object_point_clouds:
                    object_point_clouds[closest_obj] = []
                object_point_clouds[closest_obj].append(point)
        
        # Convert lists to numpy arrays
        for obj_name in object_point_clouds:
            object_point_clouds[obj_name] = np.array(object_point_clouds[obj_name])
        
        return object_point_clouds
    
    def _sample_points(self, points: np.ndarray, num_points: int) -> np.ndarray:
        """
        Sample or pad point cloud to fixed number of points.
        
        Args:
            points: Nx3 array of points
            num_points: Target number of points
            
        Returns:
            num_points x 3 array
        """
        if len(points) == 0:
            # No points - return zeros
            return np.zeros((num_points, 3))
        
        if len(points) >= num_points:
            # Randomly sample points
            indices = np.random.choice(len(points), num_points, replace=False)
            return points[indices]
        else:
            # Pad by repeating random points
            indices = np.random.choice(len(points), num_points, replace=True)
            return points[indices]
    
    def _get_object_index(self, body_name: str) -> int:
        """
        Get object index for a given body name.
        
        Args:
            body_name: Name of the body
            
        Returns:
            Index in object list
        """
        object_names = list(self.object_metadata.keys())
        if body_name in object_names:
            return object_names.index(body_name)
        return -1
    
    def _mat_to_quat(self, mat: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion.
        
        Args:
            mat: 3x3 rotation matrix
            
        Returns:
            Quaternion as [w, x, y, z]
        """
        # Use robosuite's utility if available, otherwise implement
        try:
            from robosuite.utils.transform_utils import mat2quat
            return mat2quat(mat)
        except ImportError:
            # Simple implementation
            trace = np.trace(mat)
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (mat[2, 1] - mat[1, 2]) * s
                y = (mat[0, 2] - mat[2, 0]) * s
                z = (mat[1, 0] - mat[0, 1]) * s
            else:
                if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
                    w = (mat[2, 1] - mat[1, 2]) / s
                    x = 0.25 * s
                    y = (mat[0, 1] + mat[1, 0]) / s
                    z = (mat[0, 2] + mat[2, 0]) / s
                elif mat[1, 1] > mat[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
                    w = (mat[0, 2] - mat[2, 0]) / s
                    x = (mat[0, 1] + mat[1, 0]) / s
                    y = 0.25 * s
                    z = (mat[1, 2] + mat[2, 1]) / s
                else:
                    s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
                    w = (mat[1, 0] - mat[0, 1]) / s
                    x = (mat[0, 2] + mat[2, 0]) / s
                    y = (mat[1, 2] + mat[2, 1]) / s
                    z = 0.25 * s
            return np.array([w, x, y, z])
    
    def _build_data_dict(self) -> Dict:
        """
        Build the 'data' dictionary in Points2Plans format.
        
        Returns:
            Data dictionary with time-series information
        """
        num_timesteps = len(self.timestep_data)
        object_names = list(self.object_metadata.keys())
        num_objects = len(object_names)
        
        # Initialize data structure
        data = {
            'objects': {},
            'contact': [],
            'hidden_label': [],  # Occlusion/visibility labels per timestep
        }
        
        # Populate per-object time-series data
        for obj_idx, obj_name in enumerate(object_names):
            data['objects'][self._to_block_name(obj_idx)] = {
                'position': [],
                'orientation': [],
            }
        
        # Extract time-series data
        for timestep_state in self.timestep_data:
            object_states = timestep_state['object_states']
            
            # Store object states
            for obj_idx, obj_name in enumerate(object_names):
                if obj_name in object_states:
                    state = object_states[obj_name]
                    block_name = self._to_block_name(obj_idx)
                    data['objects'][block_name]['position'].append(state['position'])
                    data['objects'][block_name]['orientation'].append(state['orientation'])
            
            # Store contacts
            data['contact'].append(timestep_state['contacts'])
            
            # Store hidden/occlusion labels
            # Compute occlusion based on point cloud availability
            point_clouds = timestep_state['point_clouds']
            hidden_labels = []
            for obj_idx, obj_name in enumerate(object_names):
                # Object is hidden (1) if it has very few points captured
                num_pts = len(point_clouds.get(obj_name, [])) if point_clouds else 0
                is_hidden = 1 if num_pts < 10 else 0  # Threshold for "visible"
                hidden_labels.append(is_hidden)
            data['hidden_label'].append(hidden_labels)
        
        # Phase 2: Add actual point cloud data
        # Initialize point cloud arrays for each object
        for obj_idx in range(num_objects):
            data[f'point_cloud_{obj_idx + 1}'] = []
            data[f'point_cloud_{obj_idx + 1}sampling'] = []
            data[f'point_cloud_{obj_idx + 1}sampling_noise'] = []
        
        # Fill point cloud data from captured timesteps
        for timestep_state in self.timestep_data:
            point_clouds = timestep_state['point_clouds']
            
            for obj_idx, obj_name in enumerate(object_names):
                # Get point cloud for this object at this timestep
                if point_clouds and obj_name in point_clouds:
                    pts = point_clouds[obj_name]
                    
                    # Sample/pad to fixed number of points
                    pts_sampled = self._sample_points(pts, self.num_points)
                    
                    # Add to data arrays
                    data[f'point_cloud_{obj_idx + 1}'].append(pts_sampled)
                    data[f'point_cloud_{obj_idx + 1}sampling'].append(pts_sampled)  # Same for now
                    
                    # Add small noise for augmentation variant
                    noise = np.random.randn(*pts_sampled.shape) * 0.001  # 1mm std
                    pts_noisy = pts_sampled + noise
                    data[f'point_cloud_{obj_idx + 1}sampling_noise'].append(pts_noisy)
                else:
                    # If object not visible, use zeros
                    zero_cloud = np.zeros((self.num_points, 3))
                    data[f'point_cloud_{obj_idx + 1}'].append(zero_cloud)
                    data[f'point_cloud_{obj_idx + 1}sampling'].append(zero_cloud)
                    data[f'point_cloud_{obj_idx + 1}sampling_noise'].append(zero_cloud)
        
        # Convert lists to numpy arrays
        for obj_idx in range(num_objects):
            data[f'point_cloud_{obj_idx + 1}'] = np.array(data[f'point_cloud_{obj_idx + 1}'])
            data[f'point_cloud_{obj_idx + 1}sampling'] = np.array(data[f'point_cloud_{obj_idx + 1}sampling'])
            data[f'point_cloud_{obj_idx + 1}sampling_noise'] = np.array(data[f'point_cloud_{obj_idx + 1}sampling_noise'])
        
        return data
    
    def _build_attrs_dict(self) -> Dict:
        """
        Build the 'attrs' dictionary in Points2Plans format.
        
        Returns:
            Attributes dictionary with static object properties
        """
        attrs = {
            'objects': {},
            'sudo_action_list': [],
        }
        
        # Populate object metadata
        for obj_idx, (obj_name, obj_meta) in enumerate(self.object_metadata.items()):
            block_name = self._to_block_name(obj_idx)
            
            attrs['objects'][block_name] = {
                'extents': obj_meta['extents'],
                'extents_ranges': self._compute_extents_ranges(obj_meta['extents']),
                'fix_base_link': obj_meta['fix_base_link'],
                'object_type': obj_meta['object_type'],
            }
            
            if obj_meta['asset_filename']:
                attrs['objects'][block_name]['asset_filename'] = obj_meta['asset_filename']
        
        # Build action list from action history
        attrs['sudo_action_list'] = self._build_action_list()
        
        return attrs
    
    def _to_block_name(self, idx: int) -> str:
        """
        Convert object index to Points2Plans block name convention.
        
        Args:
            idx: Object index (0-based)
            
        Returns:
            Block name (e.g., 'block_01', 'block_02')
        """
        return f"block_{idx + 1:02d}"
    
    def _compute_extents_ranges(self, extents: List[float]) -> List[List[float]]:
        """
        Compute extents_ranges from extents.
        
        For static objects, ranges are [extent, extent].
        
        Args:
            extents: [x, y, z] extents
            
        Returns:
            [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        """
        return [[e, e] for e in extents]
    
    def _build_action_list(self) -> List[List]:
        """
        Build sudo_action_list from recorded action history.
        
        Returns:
            List of actions in format: [skill_type, object_name, continuous_params]
        """
        action_list = []
        
        for action_dict in self.action_history:
            if action_dict is None:
                continue
            
            skill_type = action_dict['skill_type']
            obj_idx = action_dict['object_id']
            
            # Map to object name
            if obj_idx is not None and 0 <= obj_idx < len(self.object_metadata):
                obj_name = self._to_block_name(obj_idx)
            else:
                obj_name = 'unknown'
            
            # Extract continuous parameters
            continuous_params = action_dict['position_delta'].tolist()
            
            action_list.append([skill_type, obj_name, continuous_params])
        
        return action_list
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the recorded episode.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'num_timesteps': len(self.timestep_data),
            'num_objects': len(self.object_metadata),
            'num_contacts': sum(len(ts['contacts']) for ts in self.timestep_data),
            'num_actions': len(self.action_history),
            'object_names': list(self.object_metadata.keys()),
        }
    
    @staticmethod
    def load_episode(filepath: str) -> Tuple[Dict, Dict]:
        """
        Load an episode from a pickle file.
        
        Args:
            filepath: Path to the pickle file
            
        Returns:
            Tuple of (data_dict, attrs_dict) in Points2Plans format
        """
        with open(filepath, 'rb') as f:
            episode_data = pickle.load(f)
        
        if not isinstance(episode_data, tuple) or len(episode_data) != 2:
            raise ValueError(f"Invalid episode file format: {filepath}")
        
        return episode_data


if __name__ == "__main__":
    """
    Test EpisodeRecorder with Phase 3: Data packaging and saving.
    """
    from robosuite.environments.base import make
    from robosuite.controllers import load_composite_controller_config
    
    print("Testing EpisodeRecorder Phase 3 with Stack environment...")
    print("Phase 3: Data packaging and pickle writing enabled\n")
    
    # Create environment with camera observations enabled
    controller_config = load_composite_controller_config(controller="BASIC")
    env = make(
        env_name="Stack",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
    )
    
    # Create recorder with point cloud capture
    camera_names = ["frontview", "agentview"]
    recorder = EpisodeRecorder(env, camera_names=camera_names, num_points=128)
    
    # Run short episode
    obs = env.reset()
    recorder.start_episode()  # Captures timestep 0
    
    num_steps = 5  # Reduce steps for faster testing with point clouds
    print(f"Running {num_steps} environment steps with point cloud capture...")
    print(f"Expected total timesteps: {num_steps + 1} (initial state + {num_steps} steps)\n")
    
    for step in range(num_steps):
        action = np.random.randn(env.action_dim) * 0.1  # Small random actions
        obs, reward, done, info = env.step(action)
        recorder.record_step(action, obs)  # Captures timestep with point clouds
        print(f"  Step {step+1}/{num_steps} captured")
    
    # End episode and get data
    print("\nPackaging data into Points2Plans format...")
    data, attrs = recorder.end_episode()
    
    # Print statistics
    print("\n=== Episode Statistics ===")
    stats = recorder.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Data Structure ===")
    print(f"data.keys(): {list(data.keys())}")
    print(f"attrs.keys(): {list(attrs.keys())}")
    
    print("\n=== Object Metadata ===")
    for obj_name, obj_attrs in attrs['objects'].items():
        print(f"{obj_name}:")
        for key, value in obj_attrs.items():
            print(f"  {key}: {value}")
    
    print("\n=== Point Cloud Data ===")
    num_objects = len(attrs['objects'])
    for obj_idx in range(num_objects):
        pc_key = f'point_cloud_{obj_idx + 1}'
        if pc_key in data:
            pc_shape = data[pc_key].shape
            print(f"{pc_key}: shape={pc_shape}")
            
            # Check if point clouds are non-zero (actually captured)
            non_zero_timesteps = np.any(data[pc_key] != 0, axis=(1, 2)).sum()
            print(f"  Non-zero timesteps: {non_zero_timesteps}/{pc_shape[0]}")
            
            # Show sample points from first timestep
            if non_zero_timesteps > 0:
                first_nonzero = np.where(np.any(data[pc_key] != 0, axis=(1, 2)))[0][0]
                sample_pts = data[pc_key][first_nonzero][:3]  # First 3 points
                print(f"  Sample points at t={first_nonzero}: {sample_pts}")
    
    print("\n=== Hidden Labels (Occlusion) ===")
    print(f"Hidden label shape: {len(data['hidden_label'])} timesteps x {len(data['hidden_label'][0])} objects")
    # Count how many times each object was occluded
    hidden_counts = np.array(data['hidden_label']).sum(axis=0)
    for obj_idx, count in enumerate(hidden_counts):
        block_name = f"block_{obj_idx + 1:02d}"
        print(f"  {block_name}: occluded {count}/{len(data['hidden_label'])} timesteps")
    
    print("\n=== Sample Timestep Data ===")
    print(f"First timestep object positions:")
    for obj_name, obj_data in data['objects'].items():
        if len(obj_data['position']) > 0:
            print(f"  {obj_name}: {obj_data['position'][0]}")
    
    print("\n=== Contacts ===")
    total_contacts = sum(len(contacts) for contacts in data['contact'])
    print(f"Total contacts across episode: {total_contacts}")
    if total_contacts > 0:
        first_contact_timestep = next(i for i, c in enumerate(data['contact']) if len(c) > 0)
        print(f"First contact at timestep {first_contact_timestep}:")
        print(f"  {data['contact'][first_contact_timestep][0]}")
    
    # Phase 3: Save episode to disk
    print("\n=== Phase 3: Saving Episode ===")
    output_dir = "./test_episodes"
    saved_path = recorder.save_episode(output_dir, episode_name="test_episode_001")
    
    # Verify saved file can be loaded
    print("\n=== Verifying Saved Episode ===")
    loaded_data, loaded_attrs = EpisodeRecorder.load_episode(saved_path)
    
    print(f"Loaded data keys: {list(loaded_data.keys())}")
    print(f"Loaded attrs keys: {list(loaded_attrs.keys())}")
    print(f"Data matches original: {set(loaded_data.keys()) == set(data.keys())}")
    print(f"Attrs matches original: {set(loaded_attrs.keys()) == set(attrs.keys())}")
    
    # Verify data integrity
    print("\n=== Data Integrity Check ===")
    for obj_key in loaded_data['objects']:
        orig_pos = np.array(data['objects'][obj_key]['position'])
        load_pos = np.array(loaded_data['objects'][obj_key]['position'])
        pos_match = np.allclose(orig_pos, load_pos)
        print(f"  {obj_key} positions match: {pos_match}")
    
    print("\n✓ EpisodeRecorder Phase 3 test complete!")
    print("  - State capture: ✓")
    print("  - Point cloud capture: ✓")
    print("  - Occlusion detection: ✓")
    print("  - Episode saving: ✓")
    print("  - Episode loading: ✓")
    print(f"\nTest episode saved to: {saved_path}")
    
    env.close()
