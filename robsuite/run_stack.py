"""
Clean heuristic policy for Stack environments with multi-cube support.
This module implements a state-machine-based policy for stacking
multiple cubes in sequence, supporting Stack, Stack3, and Stack4 environments.
"""

from robosuite.environments.base import make
from robosuite.controllers import load_composite_controller_config
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional


class HeuristicStackPolicy:
    """
    State-machine-based heuristic policy for multi-cube stacking tasks.
    
    Supports Stack (2 cubes), Stack3 (3 cubes), and Stack4 (4 cubes) environments.
    Builds towers by stacking cubes in sequence: A->B, C->A, D->C.
    """
    
    # Constants
    P_GAIN = 5.0  # Proportional gain for position control
    
    # Height offsets
    GRIP_OFFSET = 0.0  # Gripper offset for grasping
    OBJ_OFFSET = 0.03  # Z-offset above objects before grasping
    STACK_OFFSET = 0.01  # Small additional height for safety during stacking
    SAFE_Z_OFFSET = 0.1  # Safe height for lifting and moving
    
    # Cube dimensions for precise stacking
    CUBE_HEIGHTS = {
        "cubeA": 0.02,   # Red cube
        "cubeB": 0.025,  # Green cube (slightly larger)
        "cubeC": 0.018,  # Blue cube (smaller)
        "cubeD": 0.02,   # Dark cube
    }
    
    # Counter thresholds
    GRASP_DURATION = 20
    RELEASE_DURATION = 20
    
    def __init__(self, env):
        """
        Initialize the heuristic stacking policy.
        
        Args:
            env: The robosuite environment instance
        """
        self.env = env
        self.obs = env.reset()
        
        # Setup stacking sequence
        self._setup_stacking_sequence()
        
        # Initialize state machine
        self.stage = "move_to_cube"
        self.grasp_counter = 0
        self.release_counter = 0
        self.pair_idx = 0
        self.post_grasp_source_pos = None
        self.post_place_target_pos = None
        
        self._print_initialization_info()
    
    def _setup_stacking_sequence(self):
        """Setup the stacking sequence based on available cubes."""
        # Always start with cubeA -> cubeB
        self.stacking_pairs = [("cubeA", "cubeB")]
        
        # Add additional pairs based on available cubes
        if "cubeC_pos" in self.obs:
            self.stacking_pairs.append(("cubeC", "cubeA"))
        if "cubeD_pos" in self.obs:
            self.stacking_pairs.append(("cubeD", "cubeC"))
    
    def _print_initialization_info(self):
        """Print initialization information."""
        print(f"\nDetected {len(self.stacking_pairs)} stacking pairs:")
        for i, (source, target) in enumerate(self.stacking_pairs):
            print(f"  {i+1}. Stack {source} onto {target}")
        
        current_source, current_target = self.stacking_pairs[self.pair_idx]
        print(f"Currently targeting: {current_source} -> {current_target}")
        print(f"Stage: {self.stage}\n")
    
    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract current end-effector and cube positions from observations.
        
        Returns:
            Tuple of (end-effector position, source cube position, target cube position)
        """
        eef_pos = self.obs['robot0_eef_pos']
        
        # Handle case where we've completed all pairs (during done stage)
        if self.pair_idx >= len(self.stacking_pairs):
            # Use first pair as dummy values for done stage
            source_name, target_name = self.stacking_pairs[0]
        else:
            source_name, target_name = self.stacking_pairs[self.pair_idx]
            
        source_pos = self.obs[f"{source_name}_pos"]
        target_pos = self.obs[f"{target_name}_pos"]
        return eef_pos, source_pos, target_pos
    
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
    
    def stage_move_to_cube(self, eef_pos: np.ndarray, source_pos: np.ndarray, 
                          target_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Move above the source cube."""
        action = np.zeros(self.env.action_dim)
        desired = source_pos + np.array([0, 0, self.OBJ_OFFSET])
        
        action[:3] = self.compute_position_action(desired, eef_pos)
        action[6] = -1  # Open gripper
        
        error = np.linalg.norm(desired - eef_pos)
        next_stage = None
        
        if error < 0.01:
            next_stage = "lower_to_cube"
            # Reset counters
            self.grasp_counter = 0
            self.release_counter = 0
            source_name, _ = self.stacking_pairs[self.pair_idx]
            print(f"Stage: move_to_{source_name} -> lower_to_cube")
        
        return action, next_stage
    
    def stage_lower_to_cube(self, eef_pos: np.ndarray, source_pos: np.ndarray, 
                           target_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Lower to grasp the source cube."""
        action = np.zeros(self.env.action_dim)
        desired = source_pos + np.array([0, 0, self.GRIP_OFFSET])
        
        action[:3] = self.compute_position_action(desired, eef_pos)
        action[6] = -1  # Open gripper
        
        error = np.linalg.norm(desired - eef_pos)
        next_stage = None
        
        if error < 0.005:
            next_stage = "grasp"
            source_name, _ = self.stacking_pairs[self.pair_idx]
            print(f"Stage: lower_to_{source_name} -> grasp")
        
        return action, next_stage
    
    def stage_grasp(self, eef_pos: np.ndarray, source_pos: np.ndarray, 
                   target_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Close gripper to grasp the cube."""
        action = np.zeros(self.env.action_dim)
        action[:3] = 0
        action[6] = 1  # Close gripper
        
        self.grasp_counter += 1
        next_stage = None
        
        if self.grasp_counter > self.GRASP_DURATION:
            next_stage = "lift_cube"
            self.grasp_counter = 0
            # Store source position after grasping
            source_name, _ = self.stacking_pairs[self.pair_idx]
            self.post_grasp_source_pos = self.obs[f"{source_name}_pos"]
            print("Stage: grasp -> lift_cube")
        
        return action, next_stage
    
    def stage_lift_cube(self, eef_pos: np.ndarray, source_pos: np.ndarray, 
                       target_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Lift cube to safe height."""
        action = np.zeros(self.env.action_dim)
        
        if self.post_grasp_source_pos is not None:
            desired = self.post_grasp_source_pos + np.array(
                [0, 0, -self.post_grasp_source_pos[2] + target_pos[2] + self.SAFE_Z_OFFSET])
        else:
            desired = source_pos + np.array([0, 0, self.SAFE_Z_OFFSET])
        
        action[:3] = self.compute_position_action(desired, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        error = np.linalg.norm(desired - eef_pos)
        next_stage = None
        
        if error < 0.01:
            next_stage = "move_above_target"
            print("Stage: lift_cube -> move_above_target")
        
        return action, next_stage
    
    def stage_move_above_target(self, eef_pos: np.ndarray, source_pos: np.ndarray, 
                               target_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Move above the target cube."""
        action = np.zeros(self.env.action_dim)
        desired = target_pos + np.array([0, 0, self.SAFE_Z_OFFSET])
        
        action[:3] = self.compute_position_action(desired, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        error = np.linalg.norm(desired - eef_pos)
        next_stage = None
        
        if error < 0.01:
            next_stage = "lower_to_target"
            print("Stage: move_above_target -> lower_to_target")
        
        return action, next_stage
    
    def stage_lower_to_target(self, eef_pos: np.ndarray, source_pos: np.ndarray, 
                             target_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Lower source cube onto target cube with precise height calculation."""
        action = np.zeros(self.env.action_dim)
        
        # Get cube names and heights
        source_name, target_name = self.stacking_pairs[self.pair_idx]
        source_h = self.CUBE_HEIGHTS[source_name]
        target_h = self.CUBE_HEIGHTS[target_name]
        
        # Calculate precise stacking height
        desired_z = target_pos[2] + target_h / 2 + source_h / 2 + self.STACK_OFFSET
        desired = np.array([target_pos[0], target_pos[1], desired_z])
        
        action[:3] = self.compute_position_action(desired, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        error = np.linalg.norm(desired - eef_pos)
        next_stage = None
        
        if error < 0.015:
            next_stage = "release"
            print("Stage: lower_to_target -> release")
        
        return action, next_stage
    
    def stage_release(self, eef_pos: np.ndarray, source_pos: np.ndarray, 
                     target_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Release the cube by opening gripper."""
        action = np.zeros(self.env.action_dim)
        action[:3] = 0
        action[6] = -1  # Open gripper
        
        self.release_counter += 1
        next_stage = None
        
        if self.release_counter > self.RELEASE_DURATION:
            next_stage = "retract"
            self.release_counter = 0
            # Store target position after placing
            _, target_name = self.stacking_pairs[self.pair_idx]
            self.post_place_target_pos = self.obs[f"{target_name}_pos"]
            print("Stage: release -> retract")
        
        return action, next_stage
    
    def stage_retract(self, eef_pos: np.ndarray, source_pos: np.ndarray, 
                     target_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Retract gripper and advance to next stacking pair."""
        action = np.zeros(self.env.action_dim)
        
        if self.post_place_target_pos is not None:
            desired = self.post_place_target_pos + np.array([0, 0, 0.15])
        else:
            desired = target_pos + np.array([0, 0, 0.15])
        
        action[:3] = self.compute_position_action(desired, eef_pos)
        action[6] = -1  # Open gripper
        
        error = np.linalg.norm(desired - eef_pos)
        next_stage = None
        
        if error < 0.01:
            print("Stage: retract -> next")
            # Move to next pair or reset
            self.pair_idx += 1
            if self.pair_idx >= len(self.stacking_pairs):
                next_stage = "done"
                print("All pairs completed. Resetting episode.")
            else:
                next_stage = "move_horizontal_to_next"
                current_source, current_target = self.stacking_pairs[self.pair_idx]
                print(f"\n--- Moving to next pair: {current_source} -> {current_target} ---")
                print("Stage: retract -> move_horizontal_to_next")
        
        return action, next_stage
    
    def stage_move_horizontal_to_next(self, eef_pos: np.ndarray, source_pos: np.ndarray, 
                                    target_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Move horizontally to next source cube while maintaining current height to avoid knocking stacked cubes."""
        action = np.zeros(self.env.action_dim)
        
        # Move to next source cube position but maintain current height
        # This prevents knocking off stacked cubes during horizontal movement
        desired = np.array([source_pos[0], source_pos[1], eef_pos[2]])
        
        action[:3] = self.compute_position_action(desired, eef_pos)
        action[6] = -1  # Open gripper
        
        # Only check horizontal distance since we're maintaining height
        horizontal_error = np.linalg.norm(desired[:2] - eef_pos[:2])
        next_stage = None
        
        if horizontal_error < 0.01:
            next_stage = "move_to_cube"
            source_name, _ = self.stacking_pairs[self.pair_idx]
            print(f"Stage: move_horizontal_to_next -> move_to_{source_name}")
        
        return action, next_stage
    
    def stage_done(self, eef_pos: np.ndarray, source_pos: np.ndarray, 
                  target_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Reset environment and start new episode."""
        action = np.zeros(self.env.action_dim)
        
        self.obs = self.env.reset()
        self._setup_stacking_sequence()
        self.pair_idx = 0
        next_stage = "move_to_cube"
        
        print("\nReset complete. Starting new stacking sequence.")
        
        return action, next_stage
    
    def step(self) -> Tuple[np.ndarray, bool]:
        """
        Execute one step of the stacking policy.
        
        Returns:
            Tuple of (action, done flag)
        """
        eef_pos, source_pos, target_pos = self.get_current_state()
        
        # State machine dispatcher
        stage_handlers = {
            "move_to_cube": self.stage_move_to_cube,
            "lower_to_cube": self.stage_lower_to_cube,
            "grasp": self.stage_grasp,
            "lift_cube": self.stage_lift_cube,
            "move_above_target": self.stage_move_above_target,
            "lower_to_target": self.stage_lower_to_target,
            "release": self.stage_release,
            "retract": self.stage_retract,
            "move_horizontal_to_next": self.stage_move_horizontal_to_next,
            "done": self.stage_done,
        }
        
        handler = stage_handlers.get(self.stage)
        if handler is None:
            raise ValueError(f"Unknown stage: {self.stage}")
        
        action, next_stage = handler(eef_pos, source_pos, target_pos)
        
        if next_stage is not None:
            self.stage = next_stage
        
        return action, False


def create_environment(env_name: str = "Stack4"):
    """
    Create and configure the robosuite stacking environment.
    
    Args:
        env_name: Name of the environment ("Stack", "Stack3", or "Stack4")
        
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
        horizon=1000,
        ignore_done=True,
    )
    
    return env


def run_heuristic_policy(env_name: str = "Stack4"):
    """
    Run the heuristic stacking policy.
    
    Args:
        env_name: Name of the environment to run
    """
    print(f"Starting heuristic stacking policy for {env_name}...")
    
    # Create environment
    env = create_environment(env_name)
    
    # Create policy
    policy = HeuristicStackPolicy(env)
    
    # Run policy loop
    try:
        while True:
            action, done = policy.step()
            obs, reward, env_done, info = env.step(action)
            policy.obs = obs  # Update observations
            env.render()
            
            if env_done:
                print("--- STACKING TASK SUCCESSFUL! ---")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run heuristic multi-cube stacking policy"
    )
    parser.add_argument(
        '--env', 
        type=str, 
        default='Stack4',
        choices=['Stack', 'Stack3', 'Stack4'],
        help='Which stack environment to run (2, 3, or 4 cubes)'
    )
    
    args = parser.parse_args()
    run_heuristic_policy(env_name=args.env)