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
        
        # Cache robot info
        self.robot = env.robots[0]
        self.robot_joint_names = list(self.robot.robot_joints)
        self.robot_link_names = self._get_robot_link_names()
        self.n_arm_joints = len(self.robot_joint_names)
        
        # Get gripper joints count
        if hasattr(self.robot, 'gripper'):
            gripper = self.robot.gripper
            if hasattr(gripper, 'joints'):
                self.n_ee_joints = len(gripper.joints)
            elif isinstance(gripper, dict) and 'joints' in gripper:
                self.n_ee_joints = len(gripper['joints'])
            else:
                # Count gripper DoF from actuators
                self.n_ee_joints = len([j for j in self.robot.robot_joints if 'gripper' in j or 'finger' in j])
        else:
            self.n_ee_joints = 0
        
        # Contact tracking for manipulation detection
        self.contact_history = {}  # obj_idx -> consecutive contact count
        self.last_gripper_width = None
    
    def _get_robot_link_names(self) -> List[str]:
        """Get list of robot link names."""
        try:
            link_names = []
            for i in range(self.sim.model.nbody):
                body_name = self.sim.model.body_id2name(i)
                if body_name and self.robot.robot_model.naming_prefix in body_name:
                    link_names.append(body_name)
            return link_names
        except Exception:
            return []
    
    def capture_robot_state(self) -> Dict[str, np.ndarray]:
        """
        Capture complete robot state including joints and end-effector.
        
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
        
        # Get end-effector velocity (linear)
        try:
            eef_vel = self.sim.data.get_site_xvelp(grip_site_name).copy()
        except Exception:
            eef_vel = np.zeros(3)
        
        # Get joint positions, velocities, and torques
        try:
            # Arm joints
            arm_joint_ids = [self.sim.model.joint_name2id(joint) for joint in robot.robot_joints]
            joint_pos = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[jid]] for jid in arm_joint_ids])
            joint_vel = np.array([self.sim.data.qvel[self.sim.model.jnt_dofadr[jid]] for jid in arm_joint_ids])
            
            # Torques (control)
            joint_torque = self.sim.data.ctrl[:len(arm_joint_ids)].copy()
            
            # Gripper joints
            gripper_joint_ids = [self.sim.model.joint_name2id(joint) for joint in robot.gripper.joints]
            gripper_qpos = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[jid]] for jid in gripper_joint_ids])
        except Exception as e:
            # Fallback to simple observation-based approach
            joint_pos = self.env._joint_positions if hasattr(self.env, '_joint_positions') else np.zeros(7)
            joint_vel = self.env._joint_velocities if hasattr(self.env, '_joint_velocities') else np.zeros(7)
            joint_torque = np.zeros(7)
            gripper_qpos = np.array([])
        
        return {
            'eef_pos': eef_pos,
            'eef_quat': eef_quat,
            'eef_vel': eef_vel,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'joint_torque': joint_torque,
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
        Detect which object is being manipulated using contact persistence and gripper state.
        
        In cluttered environments, the gripper may brush against multiple objects before
        grasping the target. This method uses:
        1. Contact persistence - counts consecutive contacts per object
        2. Gripper closure state - prioritizes contacts when gripper is closing
        3. Proximity as fallback - for approach phase before contact
        
        Args:
            obs: Current observations
            
        Returns:
            Object index of manipulated object, or None
        """
        if obs is None:
            return None
        
        robot = self.env.robots[0]
        robot_prefix = robot.robot_model.naming_prefix
        
        # Get gripper state
        gripper_width = self._get_gripper_width()
        is_closing = False
        if self.last_gripper_width is not None:
            is_closing = gripper_width < self.last_gripper_width - 0.001  # Threshold for closing detection
        self.last_gripper_width = gripper_width
        
        # Get current contacts
        current_contacts = self._get_gripper_contacts()
        
        # Update contact history
        for obj_idx in list(self.contact_history.keys()):
            if obj_idx in current_contacts:
                self.contact_history[obj_idx] += 1
            else:
                # Decay contact count for objects no longer in contact
                self.contact_history[obj_idx] = max(0, self.contact_history[obj_idx] - 2)
                if self.contact_history[obj_idx] == 0:
                    del self.contact_history[obj_idx]
        
        for obj_idx in current_contacts:
            if obj_idx not in self.contact_history:
                self.contact_history[obj_idx] = 1
        
        # Determine manipulated object based on contact persistence and gripper state
        if self.contact_history:
            # If gripper is closing, prioritize current contacts
            if is_closing and current_contacts:
                # Among current contacts, choose the one with highest persistence
                best_obj = max(current_contacts, key=lambda x: self.contact_history.get(x, 0))
                return best_obj
            
            # Otherwise, return object with most persistent contact (>= 3 consecutive frames)
            persistent_contacts = {obj: count for obj, count in self.contact_history.items() if count >= 3}
            if persistent_contacts:
                # Return object with highest contact count
                best_obj = max(persistent_contacts.items(), key=lambda x: x[1])[0]
                return best_obj
        
        # Fallback: proximity check for approach phase (before any contact)
        if not current_contacts:
            try:
                grip_site_name = robot.gripper.important_sites["grip_site"]
            except (AttributeError, KeyError):
                grip_site_name = f"{robot_prefix}gripper0_grip_site"
            
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
            
            # Only return if very close (within 8cm for approach detection)
            if min_dist < 0.08:
                return closest_obj_idx
        
        return None
    
    def _get_gripper_contacts(self) -> List[int]:
        """
        Get list of object indices that are in contact with the gripper.
        
        Returns:
            List of object indices in contact with gripper
        """
        robot = self.env.robots[0]
        robot_prefix = robot.robot_model.naming_prefix
        contacted_objects = []
        
        for contact_id in range(self.sim.data.ncon):
            contact = self.sim.data.contact[contact_id]
            
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            body1_id = self.sim.model.geom_bodyid[geom1_id]
            body2_id = self.sim.model.geom_bodyid[geom2_id]
            
            body1_name = self.sim.model.body(body1_id).name
            body2_name = self.sim.model.body(body2_id).name
            
            # Check if one body is gripper and other is tracked object
            is_gripper_contact = False
            object_name = None
            
            if (robot_prefix in body1_name and ('gripper' in body1_name or 'finger' in body1_name)):
                is_gripper_contact = True
                object_name = body2_name
            elif (robot_prefix in body2_name and ('gripper' in body2_name or 'finger' in body2_name)):
                is_gripper_contact = True
                object_name = body1_name
            
            if is_gripper_contact and object_name in self.object_metadata:
                obj_idx = self._get_object_index(object_name)
                if obj_idx >= 0 and obj_idx not in contacted_objects:
                    contacted_objects.append(obj_idx)
        
        return contacted_objects
    
    def _get_gripper_width(self) -> float:
        """
        Get current gripper width/opening.
        
        Returns:
            Gripper width in meters (approximate)
        """
        try:
            robot = self.env.robots[0]
            gripper = robot.gripper
            
            # Try to get gripper joint positions
            if hasattr(gripper, 'joints'):
                gripper_joint_ids = [self.sim.model.joint_name2id(joint) for joint in gripper.joints]
                gripper_qpos = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[jid]] for jid in gripper_joint_ids])
                # Return sum as approximate width (works for most parallel grippers)
                return np.sum(np.abs(gripper_qpos))
            
            # Fallback: try to get from observation
            if hasattr(self.env, '_gripper_qpos'):
                return np.sum(np.abs(self.env._gripper_qpos))
            
            return 0.0
        except Exception:
            return 0.0
    
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
