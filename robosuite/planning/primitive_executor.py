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
    
    def __init__(self,
                 env,
                 approach_height: float = 0.15,
                 grasp_height: float = 0.02,
                 lift_height: float = 0.20,
                 place_height: float = 0.05,
                 max_steps_per_primitive: int = 500,
                 position_threshold: float = 0.01,
                 gripper_close_steps: int = 50):
        """
        Initialize primitive executor.
        
        Args:
            env: Robosuite environment instance
            approach_height: Height above object for approach phase
            grasp_height: Height offset for grasping
            lift_height: Height to lift object after grasping
            place_height: Height above target for placing
            max_steps_per_primitive: Maximum steps per primitive execution
            position_threshold: Distance threshold for reaching target (meters)
            gripper_close_steps: Steps to close/open gripper
        """
        self.env = env
        self.approach_height = approach_height
        self.grasp_height = grasp_height
        self.lift_height = lift_height
        self.place_height = place_height
        self.max_steps_per_primitive = max_steps_per_primitive
        self.position_threshold = position_threshold
        self.gripper_close_steps = gripper_close_steps
        
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
    
    def _execute_pick(self,
                     primitive: str,
                     action_params: np.ndarray,
                     obs: Dict) -> Tuple[bool, int, Dict]:
        """
        Execute Pick primitive.
        
        Pick sequence:
        1. Move above object (approach)
        2. Move down to object (descend)
        3. Close gripper (grasp)
        4. Lift object (lift)
        """
        print(f"\n=== Executing: {primitive} ===")
        
        # Extract object and source location from primitive
        # "Pick(milk, table)" -> object_name="milk", source="table"
        content = primitive.split('(')[1].split(')')[0]
        parts = [p.strip() for p in content.split(',')]
        object_name, source_name = parts[0], parts[1]
        
        # Get object position from observation
        # Try to find object position in obs
        object_pos = self._get_object_position(obs, object_name)
        
        if object_pos is None:
            # Use action_params as fallback (planner's target)
            object_pos = action_params
            print(f"  Warning: Using planner params as object position")
        
        steps = 0
        
        # Phase 1: Approach (move above object)
        print(f"  Phase 1: Approach to {object_pos}")
        approach_pos = object_pos.copy()
        approach_pos[2] += self.approach_height
        
        success, phase_steps, obs = self._move_to_position(approach_pos, obs, gripper_open=True)
        steps += phase_steps
        if not success:
            print(f"  ✗ Pick failed at approach phase")
            return False, steps, obs
        
        # Phase 2: Descend (move down to object)
        print(f"  Phase 2: Descend to grasp")
        grasp_pos = object_pos.copy()
        grasp_pos[2] += self.grasp_height
        
        success, phase_steps, obs = self._move_to_position(grasp_pos, obs, gripper_open=True)
        steps += phase_steps
        if not success:
            print(f"  ✗ Pick failed at descend phase")
            return False, steps, obs
        
        # Phase 3: Grasp (close gripper)
        print(f"  Phase 3: Close gripper")
        for _ in range(self.gripper_close_steps):
            action = self._create_osc_action(grasp_pos, gripper_open=False)
            obs, _, _, _ = self.env.step(action)
            steps += 1
            self.total_steps += 1
        
        # Phase 4: Lift (move up with object)
        print(f"  Phase 4: Lift object")
        lift_pos = object_pos.copy()
        lift_pos[2] = self.lift_height
        
        success, phase_steps, obs = self._move_to_position(lift_pos, obs, gripper_open=False)
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
        2. Move down to placement height (descend)
        3. Open gripper (release)
        4. Retreat (lift back up)
        """
        print(f"\n=== Executing: {primitive} ===")
        
        # Extract object and target location from primitive
        # "Place(milk, bin)" -> object_name="milk", target="bin"
        content = primitive.split('(')[1].split(')')[0]
        parts = [p.strip() for p in content.split(',')]
        object_name, target_name = parts[0], parts[1]
        
        # Get target position from observation or use action_params
        target_pos = self._get_object_position(obs, target_name)
        
        if target_pos is None:
            # Use action_params as target position
            target_pos = action_params
            print(f"  Warning: Using planner params as target position")
        
        steps = 0
        
        # Phase 1: Approach (move above target)
        print(f"  Phase 1: Approach to {target_pos}")
        approach_pos = target_pos.copy()
        approach_pos[2] += self.approach_height
        
        success, phase_steps, obs = self._move_to_position(approach_pos, obs, gripper_open=False)
        steps += phase_steps
        if not success:
            print(f"  ✗ Place failed at approach phase")
            return False, steps, obs
        
        # Phase 2: Descend (move down to placement height)
        print(f"  Phase 2: Descend to placement height")
        place_pos = target_pos.copy()
        place_pos[2] += self.place_height
        
        success, phase_steps, obs = self._move_to_position(place_pos, obs, gripper_open=False)
        steps += phase_steps
        if not success:
            print(f"  ✗ Place failed at descend phase")
            return False, steps, obs
        
        # Phase 3: Release (open gripper)
        print(f"  Phase 3: Open gripper")
        for _ in range(self.gripper_close_steps):
            action = self._create_osc_action(place_pos, gripper_open=True)
            obs, _, _, _ = self.env.step(action)
            steps += 1
            self.total_steps += 1
        
        # Phase 4: Retreat (lift back up)
        print(f"  Phase 4: Retreat")
        retreat_pos = target_pos.copy()
        retreat_pos[2] += self.approach_height
        
        success, phase_steps, obs = self._move_to_position(retreat_pos, obs, gripper_open=True)
        steps += phase_steps
        
        print(f"  ✓ Place completed in {steps} steps")
        return True, steps, obs
    
    def _move_to_position(self,
                         target_pos: np.ndarray,
                         obs: Dict,
                         gripper_open: bool,
                         max_steps: Optional[int] = None) -> Tuple[bool, int, Dict]:
        """
        Move end-effector to target position using OSC.
        
        Returns when position reached or max steps exceeded.
        """
        if max_steps is None:
            max_steps = self.max_steps_per_primitive // 4  # Allocate 1/4 of primitive budget per phase
        
        steps = 0
        
        for _ in range(max_steps):
            # Get current end-effector position
            ee_pos = self._get_ee_position(obs)
            
            # Check if reached target
            distance = np.linalg.norm(ee_pos - target_pos)
            if distance < self.position_threshold:
                return True, steps, obs
            
            # Create OSC action towards target
            action = self._create_osc_action(target_pos, gripper_open)
            
            # Step environment
            obs, _, _, _ = self.env.step(action)
            steps += 1
            self.total_steps += 1
        
        # Failed to reach target in time
        return False, steps, obs
    
    def _create_osc_action(self,
                          target_pos: np.ndarray,
                          gripper_open: bool) -> np.ndarray:
        """
        Create OSC action for robosuite.
        
        OSC action format: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        - Position: delta from current to target (or absolute, depending on controller)
        - Orientation: keep constant (0, 0, 0)
        - Gripper: 1 = open, -1 = close
        """
        # For OSC controller, action is typically position delta or absolute position
        # Depending on your controller config, you may need to adjust this
        
        # Absolute position mode (common in robosuite OSC)
        action = np.zeros(7)
        action[:3] = target_pos  # Target position (x, y, z)
        action[3:6] = [0, 0, 0]  # Keep orientation constant
        action[6] = 1.0 if gripper_open else -1.0  # Gripper command
        
        return action
    
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
