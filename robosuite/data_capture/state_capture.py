"""
State Capture Utilities for Robosuite Environments

This module handles capturing robot and object states from simulation.
"""

import numpy as np
from typing import Dict, List, Optional, Any


class StateCapture:
    """Captures robot and object states from MuJoCo simulation."""
    
    def __init__(self, env, object_metadata: Dict):
        """
        Initialize state capture.
        
        Args:
            env: Robosuite environment instance
            object_metadata: Dictionary of object metadata
        """
        self.env = env
        self.sim = env.sim
        self.object_metadata = object_metadata
    
    def capture_robot_state(self) -> Dict[str, np.ndarray]:
        """
        Capture robot end-effector and gripper state.
        
        Returns:
            Dictionary with robot state information
        """
        robot = self.env.robots[0]
        
        # Get gripper site name
        try:
            grip_site_name = robot.gripper.important_sites["grip_site"]
        except (AttributeError, KeyError):
            grip_site_name = f"{robot.robot_model.naming_prefix}gripper0_grip_site"
        
        # Get end-effector pose
        try:
            grip_site_id = self.sim.model.site_name2id(grip_site_name)
            eef_pos = self.sim.data.site_xpos[grip_site_id].copy()
            eef_mat = self.sim.data.site_xmat[grip_site_id].reshape(3, 3)
        except Exception:
            eef_pos = np.zeros(3)
            eef_mat = np.eye(3)
        
        eef_quat = self._mat_to_quat(eef_mat)
        
        # Get gripper joint positions
        gripper_qpos = []
        try:
            for joint in robot.gripper.joints:
                joint_id = self.sim.model.joint_name2id(joint)
                gripper_qpos.append(self.sim.data.qpos[self.sim.model.jnt_qposadr[joint_id]])
            gripper_qpos = np.array(gripper_qpos)
        except Exception:
            gripper_qpos = np.array([])
        
        return {
            'eef_pos': eef_pos,
            'eef_quat': eef_quat,
            'gripper_qpos': gripper_qpos,
        }
    
    def capture_object_states(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Capture states of all tracked objects.
        
        Returns:
            Dictionary mapping object names to their states
        """
        object_states = {}
        
        for obj_name, obj_meta in self.object_metadata.items():
            body_id = obj_meta['body_id']
            
            position = self.sim.data.body_xpos[body_id].copy()
            quat = self.sim.data.body_xquat[body_id].copy()
            velocity = self.sim.data.body_xvelp[body_id].copy() if hasattr(self.sim.data, 'body_xvelp') else np.zeros(3)
            angular_vel = self.sim.data.body_xvelr[body_id].copy() if hasattr(self.sim.data, 'body_xvelr') else np.zeros(3)
            
            object_states[obj_name] = {
                'position': position,
                'orientation': quat,
                'velocity': velocity,
                'angular_velocity': angular_vel,
            }
        
        return object_states
    
    def capture_contacts(self) -> List[Dict[str, Any]]:
        """
        Capture contact information between objects.
        
        Returns:
            List of contact dictionaries
        """
        contacts = []
        
        for contact_id in range(self.sim.data.ncon):
            contact = self.sim.data.contact[contact_id]
            
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            body1_id = self.sim.model.geom_bodyid[geom1_id]
            body2_id = self.sim.model.geom_bodyid[geom2_id]
            
            body1_name = self.sim.model.body(body1_id).name
            body2_name = self.sim.model.body(body2_id).name
            
            # Only record contacts between tracked objects
            if body1_name not in self.object_metadata or body2_name not in self.object_metadata:
                continue
            
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
    
    def detect_manipulated_object(self, obs: Optional[Dict[str, Any]]) -> Optional[int]:
        """
        Detect which object is closest to gripper.
        
        Args:
            obs: Current observations
            
        Returns:
            Object index or None
        """
        if obs is None:
            return None
        
        robot = self.env.robots[0]
        
        try:
            grip_site_name = robot.gripper.important_sites["grip_site"]
        except (AttributeError, KeyError):
            grip_site_name = f"{robot.robot_model.naming_prefix}gripper0_grip_site"
        
        try:
            grip_site_id = self.sim.model.site_name2id(grip_site_name)
            eef_pos = self.sim.data.site_xpos[grip_site_id]
        except Exception:
            if 'robot0_eef_pos' in obs:
                eef_pos = obs['robot0_eef_pos']
            else:
                return None
        
        # Find closest non-static object
        min_dist = float('inf')
        closest_obj_idx = None
        
        for idx, (obj_name, obj_meta) in enumerate(self.object_metadata.items()):
            if obj_meta['fix_base_link']:
                continue
            
            body_id = obj_meta['body_id']
            obj_pos = self.sim.data.body_xpos[body_id]
            
            dist = np.linalg.norm(eef_pos - obj_pos)
            if dist < min_dist:
                min_dist = dist
                closest_obj_idx = idx
        
        # Only return if very close (within 5cm)
        if min_dist < 0.05:
            return closest_obj_idx
        
        return None
    
    def _get_object_index(self, body_name: str) -> int:
        """Get index of object in metadata dict."""
        object_names = list(self.object_metadata.keys())
        if body_name in object_names:
            return object_names.index(body_name)
        return -1
    
    def _mat_to_quat(self, mat: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]."""
        try:
            from robosuite.utils.transform_utils import mat2quat
            return mat2quat(mat)
        except ImportError:
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
