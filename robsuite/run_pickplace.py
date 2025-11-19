"""
Clean heuristic policy for PickPlaceMulti environments.
This module implements a state-machine-based policy for picking and placing
multiple objects, with support for both top-down and side grasping strategies.
"""

from robosuite.environments.base import make
from robosuite.controllers import load_composite_controller_config
import numpy as np
from typing import Dict, List, Tuple, Optional


class HeuristicPickPlacePolicy:
    """
    State-machine-based heuristic policy for pick and place tasks.
    Supports both top-down and side grasping for different object types.
    """
    
    # Constants
    SIDE_GRASP_KEYWORDS = ["milk", "cereal", "bottle"]  # Objects that need side grasping
    P_GAIN = 10.0  # Proportional gain for position control
    R_GAIN = 5.0   # Proportional gain for orientation control
    
    # Height offsets
    OBJ_Z_OFFSET_TOP = 0.02   # Default offset for top-down grasping
    OBJ_Z_OFFSET_SIDE = 0.04  # Offset for side grasping
    BIN_Z_OFFSET = 0.05       # Offset above bin when placing
    SAFE_Z_OFFSET = 0.25      # Safe height above bins for movements
    
    # Object-specific height offsets for better grasping
    OBJECT_HEIGHT_OFFSETS = {
        "bread": 0.005,   # Bread is very flat, needs lower approach
        "lemon": 0.0,     # Lemon is small, needs lower approach
        "can": 0.02,      # Can is medium height
        "milk": 0.0,      # Tall objects use side grasp
        "cereal": 0.0,    # Tall objects use side grasp
        "bottle": 0.0,    # Tall objects use side grasp
    }
    
    # Grasp parameters
    MAX_GRASP_ATTEMPTS = 3
    GRASP_HEIGHT_THRESHOLD = 0.05  # Minimum lift to verify grasp (meters)
    
    # Counter thresholds
    GRASP_DURATION = 50
    RELEASE_DURATION = 50
    ALIGN_DURATION = 30
    
    def __init__(self, env):
        """
        Initialize the heuristic policy.
        
        Args:
            env: The robosuite environment instance
        """
        self.env = env
        self.obs = env.reset()
        
        # Setup object tracking
        self._setup_objects()
        
        # Initialize state machine
        self.stage = "move_to_object"
        self.grasp_counter = 0
        self.release_counter = 0
        self.align_counter = 0
        self.grasp_attempts = 0
        self.pre_grasp_obj_pos = None
        self.retract_target = None
        
        # Calculate safe z height
        max_bin_height = max(self.env.bin1_pos[2], self.env.bin2_pos[2])
        self.safe_z_height = max_bin_height + self.SAFE_Z_OFFSET
        
        self._print_initialization_info()
    
    def _setup_objects(self):
        """Setup object tracking and grasp type determination."""
        # Extract object names from observations
        self.object_names = [
            name for name in self.obs.keys() 
            if "_pos" in name and "robot" not in name.lower()
        ]
        
        # Determine grasp type for each object
        self.object_grasp_type = {}
        for obj_name in self.object_names:
            needs_side_grasp = any(
                keyword in obj_name.lower() 
                for keyword in self.SIDE_GRASP_KEYWORDS
            )
            self.object_grasp_type[obj_name] = "side" if needs_side_grasp else "top"
        
        # Create object to target position mapping
        self.object_to_target = {}
        target_placements = self.env.target_bin_placements
        for obj_name, target_pos in zip(self.object_names, target_placements):
            self.object_to_target[obj_name] = target_pos.copy()
        
        # Initialize object queue
        self.objects_to_place = self.object_names.copy()
        self.current_object = self.objects_to_place[0]
        self.current_grasp_type = self.object_grasp_type[self.current_object]
    
    def _print_initialization_info(self):
        """Print initialization information."""
        print(f"\nDetected {len(self.object_names)} objects to place:")
        for obj_name in self.object_names:
            grasp_type = self.object_grasp_type[obj_name]
            target_pos = self.object_to_target[obj_name]
            print(f"  - {obj_name}: {grasp_type} grasp, target: {target_pos}")
        
        print(f"\nSource bin at: {self.env.bin1_pos}")
        print(f"Target bin at: {self.env.bin2_pos}")
        print(f"Safe Z height: {self.safe_z_height}")
        print(f"Currently targeting: {self.current_object} ({self.current_grasp_type} grasp)\n")
    
    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract current end-effector and object positions from observations.
        
        Returns:
            Tuple of (end-effector position, current object position)
        """
        eef_pos = self.obs['robot0_eef_pos']
        obj_pos = self.obs[self.current_object]
        return eef_pos, obj_pos
    
    def get_object_z_offset(self, obj_name: str) -> float:
        """
        Get the appropriate z-offset for grasping a specific object.
        
        Args:
            obj_name: Name of the object (e.g., "bread_pos", "lemon_pos")
            
        Returns:
            Z-offset value for that object type
        """
        # Extract object type from name (e.g., "bread_pos" -> "bread")
        for obj_type, offset in self.OBJECT_HEIGHT_OFFSETS.items():
            if obj_type.lower() in obj_name.lower():
                return offset
        
        # Default to standard top offset if not specified
        return self.OBJ_Z_OFFSET_TOP
    
    def compute_position_action(self, target_pos: np.ndarray, 
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
    
    def stage_move_to_object(self, eef_pos: np.ndarray, 
                            obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Move above object first."""
        action = np.zeros(self.env.action_dim)
        target_pos = obj_pos + np.array([0, 0, self.safe_z_height - obj_pos[2]])
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = -1  # Open gripper
        
        error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        if error < 0.01:
            if self.current_grasp_type == "side":
                next_stage = "align_for_side_grasp"
            else:
                next_stage = "lower_to_object"
            print(f"Stage: move_to_object -> {next_stage}")
        
        return action, next_stage
    
    def stage_align_for_side_grasp(self, eef_pos: np.ndarray, 
                                   obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Align gripper for side grasping."""
        action = np.zeros(self.env.action_dim)
        
        # Position slightly to the side of the object
        side_offset = np.array([0.08, 0, 0])
        target_pos = obj_pos + side_offset + np.array([0, 0, self.OBJ_Z_OFFSET_SIDE])
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        
        # Rotate gripper 90 degrees around y-axis
        if self.align_counter < self.ALIGN_DURATION:
            action[3:6] = np.array([0, 0.1, 0])
        
        action[6] = -1  # Open gripper
        self.align_counter += 1
        
        pos_error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        if pos_error < 0.01 and self.align_counter > self.ALIGN_DURATION:
            next_stage = "approach_from_side"
            self.align_counter = 0
            print(f"Stage: align_for_side_grasp -> {next_stage}")
        
        return action, next_stage
    
    def stage_approach_from_side(self, eef_pos: np.ndarray, 
                                obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Approach object from the side."""
        action = np.zeros(self.env.action_dim)
        target_pos = obj_pos + np.array([0, 0, self.OBJ_Z_OFFSET_SIDE])
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = -1  # Open gripper
        
        error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        if error < 0.01:
            self.pre_grasp_obj_pos = obj_pos.copy()
            next_stage = "grasp"
            print(f"Stage: approach_from_side -> {next_stage} "
                  f"(attempt {self.grasp_attempts + 1}/{self.MAX_GRASP_ATTEMPTS})")
        
        return action, next_stage
    
    def stage_lower_to_object(self, eef_pos: np.ndarray, 
                             obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Lower to object for top-down grasping."""
        action = np.zeros(self.env.action_dim)
        
        # Use object-specific z-offset for better grasping
        z_offset = self.get_object_z_offset(self.current_object)
        target_pos = obj_pos + np.array([0, 0, z_offset])
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = -1  # Open gripper
        
        error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        if error < 0.005:
            self.pre_grasp_obj_pos = obj_pos.copy()
            next_stage = "grasp"
            print(f"Stage: lower_to_object -> {next_stage} "
                  f"(attempt {self.grasp_attempts + 1}/{self.MAX_GRASP_ATTEMPTS}, z_offset={z_offset:.3f})")
        
        return action, next_stage
    
    def stage_grasp(self, eef_pos: np.ndarray, 
                   obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Close gripper to grasp object."""
        action = np.zeros(self.env.action_dim)
        action[:3] = 0
        action[6] = 1  # Close gripper
        
        self.grasp_counter += 1
        next_stage = None
        
        if self.grasp_counter > self.GRASP_DURATION:
            next_stage = "verify_grasp"
            self.grasp_counter = 0
            print(f"Stage: grasp -> {next_stage}")
        
        return action, next_stage
    
    def stage_verify_grasp(self, eef_pos: np.ndarray, 
                          obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Verify that object was successfully grasped."""
        action = np.zeros(self.env.action_dim)
        target_pos = [eef_pos[0], eef_pos[1], eef_pos[2] + 0.1]
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        next_stage = None
        
        # Check if we've lifted enough to verify
        if eef_pos[2] > self.pre_grasp_obj_pos[2] + 0.08:
            obj_height_change = obj_pos[2] - self.pre_grasp_obj_pos[2]
            
            if obj_height_change > self.GRASP_HEIGHT_THRESHOLD:
                # Successful grasp
                print(f"✓ Grasp successful! Object lifted {obj_height_change:.3f}m")
                self.grasp_attempts = 0
                next_stage = "lift_object"
                print(f"Stage: verify_grasp -> {next_stage}")
            else:
                # Failed grasp
                self.grasp_attempts += 1
                print(f"✗ Grasp failed! Object only lifted {obj_height_change:.3f}m")
                
                if self.grasp_attempts < self.MAX_GRASP_ATTEMPTS:
                    print(f"Retrying grasp (attempt {self.grasp_attempts + 1}/{self.MAX_GRASP_ATTEMPTS})...")
                    next_stage = "move_to_object"
                else:
                    print(f"Max grasp attempts reached. Skipping {self.current_object}")
                    self.grasp_attempts = 0
                    next_stage = "skip_object"
                
                print(f"Stage: verify_grasp -> {next_stage}")
        
        return action, next_stage
    
    def stage_lift_object(self, eef_pos: np.ndarray, 
                         obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Lift object to safe height."""
        action = np.zeros(self.env.action_dim)
        target_pos = [eef_pos[0], eef_pos[1], self.safe_z_height]
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        if error < 0.01:
            next_stage = "move_to_bin"
            print(f"Stage: lift_object -> {next_stage}")
        
        return action, next_stage
    
    def stage_move_to_bin(self, eef_pos: np.ndarray, 
                         obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Move object above target bin."""
        action = np.zeros(self.env.action_dim)
        target_placement = self.object_to_target[self.current_object]
        target_pos = target_placement + np.array([0, 0, self.safe_z_height - target_placement[2]])
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        error_xy = np.linalg.norm((target_pos - eef_pos)[:2])
        next_stage = None
        
        if error_xy < 0.01:
            next_stage = "lower_to_bin"
            print(f"Stage: move_to_bin -> {next_stage}")
        
        return action, next_stage
    
    def stage_lower_to_bin(self, eef_pos: np.ndarray, 
                          obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Lower object into target bin."""
        action = np.zeros(self.env.action_dim)
        target_placement = self.object_to_target[self.current_object]
        target_pos = target_placement + np.array([0, 0, self.BIN_Z_OFFSET])
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        if error < 0.025:
            next_stage = "release"
            print(f"Stage: lower_to_bin -> {next_stage}")
        
        return action, next_stage
    
    def stage_release(self, eef_pos: np.ndarray, 
                     obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Release object by opening gripper."""
        action = np.zeros(self.env.action_dim)
        action[6] = -1  # Open gripper
        
        self.release_counter += 1
        next_stage = None
        
        if self.release_counter > self.RELEASE_DURATION:
            next_stage = "retract"
            self.release_counter = 0
            print(f"Stage: release -> {next_stage}")
        
        return action, next_stage
    
    def stage_retract(self, eef_pos: np.ndarray, 
                     obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Retract gripper after releasing object."""
        action = np.zeros(self.env.action_dim)
        
        if self.retract_target is None:
            self.retract_target = eef_pos + np.array([0, 0, 0.1])
        
        action[:3] = self.compute_position_action(self.retract_target, eef_pos)
        action[6] = -1  # Open gripper
        
        error = np.linalg.norm(self.retract_target - eef_pos)
        next_stage = None
        
        if error < 0.01:
            self.retract_target = None
            
            # Remove current object from queue
            if self.current_object in self.objects_to_place:
                self.objects_to_place.remove(self.current_object)
            
            # Move to next object or complete episode
            next_stage = self._handle_next_object()
        
        return action, next_stage
    
    def stage_reset_orientation(self, eef_pos: np.ndarray, 
                               obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Reset gripper orientation from side grasp to top grasp."""
        action = np.zeros(self.env.action_dim)        
        if self.align_counter < self.ALIGN_DURATION:
            action[3:6] = np.array([0, -0.1, 0])  # Rotate back around y-axis
        
        action[6] = -1  # Keep gripper open
        
        self.align_counter += 1
        next_stage = None
        
        if self.align_counter > self.ALIGN_DURATION:
            self.align_counter = 0
            next_stage = "move_to_object"
            print(f"Orientation reset complete. Moving to: {self.current_object} "
                  f"({self.current_grasp_type} grasp)")
            print(f"Stage: reset_orientation -> {next_stage}")
        
        return action, next_stage
    
    def stage_skip_object(self, eef_pos: np.ndarray, 
                         obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Skip current object and move to next."""
        action = np.zeros(self.env.action_dim)
        
        # Remove current object from queue
        if self.current_object in self.objects_to_place:
            self.objects_to_place.remove(self.current_object)
        
        # Move to next object or complete episode
        next_stage = self._handle_next_object()
        
        return action, next_stage
    
    def stage_done(self, eef_pos: np.ndarray, 
                  obj_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Reset environment and start new episode."""
        action = np.zeros(self.env.action_dim)
        
        self.obs = self.env.reset()
        self.objects_to_place = self.object_names.copy()
        self.current_object = self.objects_to_place[0]
        self.current_grasp_type = self.object_grasp_type[self.current_object]
        self.grasp_attempts = 0
        next_stage = "move_to_object"
        
        print(f"\nReset complete. Starting with: {self.current_object} "
              f"({self.current_grasp_type} grasp)")
        
        return action, next_stage
    
    def _handle_next_object(self) -> str:
        """
        Handle transition to next object or episode completion.
        
        Returns:
            Next stage name
        """
        if self.objects_to_place:
            next_object = self.objects_to_place[0]
            next_grasp_type = self.object_grasp_type[next_object]
            
            # Check if we need to reset orientation
            if self.current_grasp_type == "side" and next_grasp_type == "top":
                print(f"\n--- Next object needs top grasp, resetting gripper orientation ---")
                self.current_object = next_object
                self.current_grasp_type = next_grasp_type
                return "reset_orientation"
            else:
                self.current_object = next_object
                self.current_grasp_type = next_grasp_type
                print(f"\n--- Moving to next object: {self.current_object} "
                      f"({self.current_grasp_type} grasp) ---")
                return "move_to_object"
        else:
            print("\n--- All objects placed! Resetting episode. ---")
            return "done"
    
    def step(self) -> Tuple[np.ndarray, bool]:
        """
        Execute one step of the policy.
        
        Returns:
            Tuple of (action, done flag)
        """
        eef_pos, obj_pos = self.get_current_state()
        
        # State machine dispatcher
        stage_handlers = {
            "move_to_object": self.stage_move_to_object,
            "align_for_side_grasp": self.stage_align_for_side_grasp,
            "approach_from_side": self.stage_approach_from_side,
            "lower_to_object": self.stage_lower_to_object,
            "grasp": self.stage_grasp,
            "verify_grasp": self.stage_verify_grasp,
            "lift_object": self.stage_lift_object,
            "move_to_bin": self.stage_move_to_bin,
            "lower_to_bin": self.stage_lower_to_bin,
            "release": self.stage_release,
            "retract": self.stage_retract,
            "reset_orientation": self.stage_reset_orientation,
            "skip_object": self.stage_skip_object,
            "done": self.stage_done,
        }
        
        handler = stage_handlers.get(self.stage)
        if handler is None:
            raise ValueError(f"Unknown stage: {self.stage}")
        
        action, next_stage = handler(eef_pos, obj_pos)
        
        if next_stage is not None:
            self.stage = next_stage
        
        return action, False


def create_environment(env_name: str = "PickPlaceMulti4"):
    """
    Create and configure the robosuite environment.
    
    Args:
        env_name: Name of the environment ("PickPlaceMulti3" or "PickPlaceMulti4")
        
    Returns:
        Configured environment instance
    """
    controller_config = load_composite_controller_config(controller="BASIC")
    
    env = make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=2000,
        ignore_done=True,
    )
    
    return env


def run_heuristic_policy(env_name: str = "PickPlaceMulti4"):
    """
    Run the heuristic pick and place policy.
    
    Args:
        env_name: Name of the environment to run
    """
    print(f"Starting heuristic policy for {env_name}...")
    
    # Create environment
    env = create_environment(env_name)
    
    # Create policy
    policy = HeuristicPickPlacePolicy(env)
    
    # Run policy loop
    try:
        while True:
            action, done = policy.step()
            obs, reward, env_done, info = env.step(action)
            policy.obs = obs  # Update observations
            env.render()
            
            if env_done:
                print("--- ENVIRONMENT REPORTED TASK SUCCESS! ---")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run heuristic pick and place policy"
    )
    parser.add_argument(
        '--env', 
        type=str, 
        default='PickPlaceMulti4',
        choices=['PickPlaceMulti3', 'PickPlaceMulti4'],
        help='Which environment to run (3 or 4 objects)'
    )
    
    args = parser.parse_args()
    run_heuristic_policy(env_name=args.env)
