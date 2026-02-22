"""
Primitive Executor for Robosuite

Executes high-level primitives (Pick, Place) as low-level robosuite control sequences.

A "primitive" = one discrete action (Pick or Place) that takes ~200-500 robosuite steps.
This is NOT trajectory-level control - primitives are the planning granularity.

Execution loop:
1. Receive primitive action from DynamicsModelPlanner (e.g., "Place(milk, bin)")
2. Convert to robosuite OSC controller commands
3. Execute primitive until completion (~200-500 steps)
4. Return success status

This runs AFTER planning, BEFORE next observation.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import time


class PrimitiveExecutor:
    """
    Executes high-level Pick/Place primitives in robosuite.
    
    Converts primitive actions (e.g., "Pick(milk, table)") into low-level
    OSC (Operational Space Control) sequences for robosuite.
    
    Each primitive execution:
    - Takes ~200-500 robosuite simulation steps
    - Returns when primitive completes (or fails)
    - Provides success feedback for replanning
    """
    
    # Constants
    P_GAIN = 5.0  # Proportional gain for position control
    
    # Height offsets
    GRIP_OFFSET = 0.0      # Gripper offset for grasping
    OBJ_OFFSET = 0.03      # Z-offset above objects before grasping
    STACK_OFFSET = 0.02    # Small additional height for safety during stacking
    SAFE_Z_OFFSET = 0.1    # Safe height for lifting and moving
    
    # Cube dimensions for precise stacking
    CUBE_HEIGHTS = {
        "cubeA": 0.02,   # Red cube
        "cubeB": 0.025,  # Green cube (slightly larger)
        "cubeC": 0.018,  # Blue cube (smaller)
        "cubeD": 0.02,   # Dark cube
    }
    
    # Counter thresholds
    GRASP_DURATION = 50
    RELEASE_DURATION = 50
    
    def __init__(self,
                 env,
                 max_steps_per_primitive: int = 500,
                 position_threshold: float = 0.01):
        """
        Initialize primitive executor.
        
        Args:
            env: Robosuite environment instance
            max_steps_per_primitive: Maximum steps per primitive execution
            position_threshold: Distance threshold for reaching target (meters)
        """
        self.env = env
        self.max_steps_per_primitive = max_steps_per_primitive
        self.position_threshold = position_threshold
        
        # Track execution state
        self.total_steps = 0
        
        print(f"Primitive Executor initialized:")
        print(f"  Max steps per primitive: {max_steps_per_primitive}")
        print(f"  Position threshold: {position_threshold}m")
    
    def execute_primitive(self,
                         primitive: str,
                         action_params: np.ndarray,
                         obs: Dict) -> Tuple[bool, int, Dict]:
        """
        Execute a single primitive action.
        
        Args:
            primitive: Primitive action string (e.g., "Pick(milk, table)" or "Place(milk, bin)")
            action_params: Low-level action parameters from planner (target position)
            obs: Current robosuite observation dict
        
        Returns:
            success: Whether primitive executed successfully
            steps: Number of steps taken
            final_obs: Final observation after execution
        """
        # Parse primitive type
        action_type = primitive.split('(')[0].strip()
        
        if action_type.lower() == 'pick':
            return self._execute_pick(primitive, action_params, obs)
        elif action_type.lower() == 'place':
            return self._execute_place(primitive, action_params, obs)
        else:
            print(f"Unknown primitive type: {action_type}")
            return False, 0, obs
    
    def _parse_primitive(self, primitive: str) -> Tuple[str, str]:
        """
        Parse primitive string to extract object and target names.
        
        Args:
            primitive: Primitive string (e.g., "Pick(milk, table)")
            
        Returns:
            Tuple of (object_name, target_name)
        """
        content = primitive.split('(')[1].split(')')[0]
        parts = [p.strip() for p in content.split(',')]
        return parts[0], parts[1]
    
    def _execute_pick(self,
                     primitive: str,
                     action_params: np.ndarray,
                     obs: Dict) -> Tuple[bool, int, Dict]:
        """
        Execute Pick primitive.
        
        Pick sequence:
        1. Move above object (approach)
        2. Lower to object (descend)
        3. Close gripper (grasp)
        4. Lift object (lift)
        """
        print(f"\n=== Executing: {primitive} ===")
        
        # Parse primitive to get object and target names
        object_name, target_name = self._parse_primitive(primitive)
        
        # Get object position from observation (fallback to action_params)
        object_pos = self._get_object_position(obs, object_name) # @Chris: Ordinarily, this should be action_params, but we can also try to get it from obs for more accuracy since we don't care about best action prediction.
        if object_pos is None:
            object_pos = action_params
            print(f"  Warning: Using planner params as object position")
        
        print(f"  Object: {object_name}, target: {target_name}, object position: {object_pos}, action params: {action_params}")
        
        steps = 0
        
        # Phase 1: Move above object
        print(f"  Phase 1: Move above {object_name}")
        desired = object_pos + np.array([0, 0, self.OBJ_OFFSET])
        success, phase_steps, obs = self._move_to_position(desired, obs, gripper_value=-1)
        steps += phase_steps
        if not success:
            print(f"  ✗ Pick failed at approach phase")
            return False, steps, obs
        
        # Phase 2: Lower to grasp position
        print(f"  Phase 2: Lower to {object_name}")
        desired = object_pos + np.array([0, 0, self.GRIP_OFFSET])
        success, phase_steps, obs = self._move_to_position(desired, obs, gripper_value=-1, position_threshold=0.005)
        steps += phase_steps
        if not success:
            print(f"  ✗ Pick failed at descend phase")
            return False, steps, obs
        
        # Phase 3: Close gripper
        print(f"  Phase 3: Grasp {object_name}")
        phase_steps, obs = self._actuate_gripper(obs, gripper_value=1)
        steps += phase_steps
        
        # Update object position after grasping
        object_pos = self._get_object_position(obs, object_name)
        
        # Phase 4: Lift object
        print(f"  Phase 4: Lift {object_name}")
        desired = object_pos + np.array([0, 0, self.SAFE_Z_OFFSET])
        success, phase_steps, obs = self._move_to_position(desired, obs, gripper_value=1)
        steps += phase_steps
        
        print(f"  ✓ Pick completed in {steps} steps")
        return True, steps, obs
    
    def _execute_place(self,
                      primitive: str,
                      action_params: np.ndarray,
                      obs: Dict) -> Tuple[bool, int, Dict]:
        """
        Execute Place primitive.
        
        Place sequence:
        1. Move above target location (approach)
        2. Lower to placement height (descend)
        3. Open gripper (release)
        4. Retract (lift back up)
        """
        print(f"\n=== Executing: {primitive} ===")
        
        # Parse primitive to get object and target names
        object_name, target_name = self._parse_primitive(primitive)
        
        # Get target position from observation (fallback to action_params)
        target_pos = self._get_object_position(obs, target_name) # @Chris: Ordinarily, this should be action_params, but we can also try to get it from obs for more accuracy since we don't care about best action prediction.
        if target_pos is None:
            target_pos = action_params
            print(f"  Warning: Using planner params as target position")
        
        print(f"  Placing {object_name} onto target: {target_name}, at position: {target_pos}, action params: {action_params}")
        
        steps = 0
        
        # Phase 1: Move above target
        print(f"  Phase 1: Move above {target_name}")
        desired = target_pos + np.array([0, 0, self.SAFE_Z_OFFSET])
        success, phase_steps, obs = self._move_to_position(desired, obs, gripper_value=1)
        steps += phase_steps
        if not success:
            print(f"  ✗ Place failed at approach phase")
            return False, steps, obs
        
        # Phase 2: Lower to placement height (precise stacking calculation)
        print(f"  Phase 2: Lower to {target_name}")
        source_h = self.CUBE_HEIGHTS.get(object_name, 0.02)
        target_h = self.CUBE_HEIGHTS.get(target_name, 0.02)
        desired_z = target_pos[2] + target_h / 2 + source_h / 2 + self.STACK_OFFSET
        desired = np.array([target_pos[0], target_pos[1], desired_z])
        
        success, phase_steps, obs = self._move_to_position(desired, obs, gripper_value=1, position_threshold=0.015)
        steps += phase_steps
        if not success:
            print(f"  ✗ Place failed at descend phase")
            return False, steps, obs
        
        # Phase 3: Open gripper
        print(f"  Phase 3: Release {object_name}")
        phase_steps, obs = self._actuate_gripper(obs, gripper_value=-1)
        steps += phase_steps
        
        # Update target position after releasing
        target_pos = self._get_object_position(obs, target_name)
        
        # Phase 4: Retract
        print(f"  Phase 4: Retract from {target_name}")
        desired = target_pos + np.array([0, 0, self.SAFE_Z_OFFSET + 0.05])
        success, phase_steps, obs = self._move_to_position(desired, obs, gripper_value=-1)
        steps += phase_steps
        
        print(f"  ✓ Place completed in {steps} steps")
        return True, steps, obs
    
    def _move_to_position(self,
                         target_pos: np.ndarray,
                         obs: Dict,
                         gripper_value: int,
                         max_steps: Optional[int] = None,
                         position_threshold: float = 0.01) -> Tuple[bool, int, Dict]:
        """
        Move end-effector to target position using proportional control.
        
        Args:
            target_pos: Target position for end-effector
            obs: Current observation dict
            gripper_value: Gripper command (1 = close, -1 = open)
            max_steps: Maximum steps for this movement
            position_threshold: Distance threshold for reaching target
            
        Returns:
            Tuple of (success, steps_taken, final_obs)
        """
        if max_steps is None:
            max_steps = self.max_steps_per_primitive // 4
        
        steps = 0
        
        for _ in range(max_steps):
            ee_pos = self._get_ee_position(obs)
            error = np.linalg.norm(ee_pos - target_pos)
            
            if error < position_threshold:
                return True, steps, obs
            
            action = np.zeros(self.env.action_dim)
            action[:3] = self._compute_position_action(target_pos, ee_pos)
            action[6] = gripper_value
            
            obs, _, _, _ = self.env.step(action)
            steps += 1
            self.total_steps += 1
        
        return False, steps, obs
    
    def _actuate_gripper(self,
                        obs: Dict,
                        gripper_value: int) -> Tuple[int, Dict]:
        """
        Actuate gripper for specified duration.
        
        Args:
            obs: Current observation dict
            gripper_value: Gripper command (1 = close, -1 = open)
            
        Returns:
            Tuple of (steps_taken, final_obs)
        """
        duration = self.GRASP_DURATION if gripper_value == 1 else self.RELEASE_DURATION
        steps = 0
        
        for _ in range(duration):
            action = np.zeros(self.env.action_dim)
            action[:3] = 0
            action[6] = gripper_value
            
            obs, _, _, _ = self.env.step(action)
            steps += 1
            self.total_steps += 1
        
        return steps, obs
    
    def _compute_position_action(self,
                                target_pos: np.ndarray,
                                current_pos: np.ndarray) -> np.ndarray:
        """
        Compute proportional control action for position.
        
        Args:
            target_pos: Target position
            current_pos: Current position
            
        Returns:
            Position control action
        """
        error = target_pos - current_pos
        return error * self.P_GAIN
    
    def _get_ee_position(self, obs: Dict) -> np.ndarray:
        """Get end-effector position from observation."""
        # Robosuite observation keys vary by robot
        # Common keys: 'robot0_eef_pos', 'eef_pos', etc.
        
        if 'robot0_eef_pos' in obs:
            return obs['robot0_eef_pos']
        elif 'eef_pos' in obs:
            return obs['eef_pos']
        else:
            # Fallback: try to find any key with 'eef' and 'pos'
            for key in obs.keys():
                if 'eef' in key.lower() and 'pos' in key.lower():
                    return obs[key]
            
            raise KeyError("Could not find end-effector position in observation")
    
    def _get_object_position(self, obs: Dict, object_name: str) -> Optional[np.ndarray]:
        """
        Get object position from observation.
        
        Robosuite observations typically have keys like:
        - 'object0_pos', 'object1_pos', ...
        - 'milk_pos', 'bread_pos', ... (if custom naming)
        """
        # Try direct object name
        if f'{object_name}_pos' in obs:
            return obs[f'{object_name}_pos']
        
        # Try with lowercase
        if f'{object_name.lower()}_pos' in obs:
            return obs[f'{object_name.lower()}_pos']
        
        # Try indexed objects (object0, object1, ...)
        # You may need to maintain object name -> index mapping
        # For now, return None and use action_params as fallback
        
        return None
    
    def reset_stats(self):
        """Reset execution statistics."""
        self.total_steps = 0
    
    def get_stats(self) -> Dict:
        """Get execution statistics."""
        return {
            'total_steps': self.total_steps
        }
