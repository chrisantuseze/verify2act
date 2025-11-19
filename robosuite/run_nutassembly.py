"""
Clean heuristic policy for NutAssembly environments.
This module implements a state-machine-based policy for picking nuts and
placing them on their corresponding pegs.
"""

from robosuite.environments.base import make
from robosuite.controllers import load_composite_controller_config
import numpy as np
from typing import Dict, List, Tuple, Optional


class HeuristicNutAssemblyPolicy:
    """
    State-machine-based heuristic policy for nut assembly tasks.
    Handles both square and round nuts with appropriate grasping strategies.
    """
    
    # Constants
    P_GAIN = 10.0  # Proportional gain for position control
    R_GAIN = 5.0   # Proportional gain for orientation control
    
    # Height offsets
    NUT_Z_OFFSET = 0.01      # Offset for approaching nut (lower for flat nuts)
    SAFE_Z_OFFSET = 0.15      # Safe height above table for movements
    PEG_Z_OFFSET = 0.08       # Height above peg when placing
    
    # Grasp parameters
    MAX_GRASP_ATTEMPTS = 3
    GRASP_HEIGHT_THRESHOLD = 0.04  # Minimum lift to verify grasp (meters)
    
    # Counter thresholds
    GRASP_DURATION = 50
    RELEASE_DURATION = 50
    ALIGN_DURATION = 40  # For aligning nut over peg
    
    def __init__(self, env):
        """
        Initialize the heuristic policy.
        
        Args:
            env: The robosuite environment instance
        """
        self.env = env
        self.obs = env.reset()
        
        # Setup nut tracking
        self._setup_nuts()
        
        # Initialize state machine
        self.stage = "move_to_nut"
        self.grasp_counter = 0
        self.release_counter = 0
        self.align_counter = 0
        self.grasp_attempts = 0
        self.pre_grasp_nut_pos = None
        self.retract_target = None
        
        # Calculate safe z height
        self.table_z = self.env.table_offset[2]
        self.safe_z_height = self.table_z + self.SAFE_Z_OFFSET

        # Cache peg body ids from simulator (observations don't include peg positions)
        try:
            self.peg_body_ids = {
                0: self.env.sim.model.body_name2id("peg1"),
                1: self.env.sim.model.body_name2id("peg2"),
            }
        except Exception:
            # Fallback: leave unset; we'll handle this in get_current_state
            self.peg_body_ids = None

        # Cache nut handle site ids (so we can aim the gripper at the handle)
        # NutAssembly defines an important_sites['handle'] for each nut; try common naming
        self.nut_handle_site_ids = {}
        try:
            for nut in self.nut_names:
                # Common convention used in the environment: '<NutName>_handle_site'
                candidate = f"{nut}_handle_site"
                try:
                    sid = self.env.sim.model.site_name2id(candidate)
                    self.nut_handle_site_ids[nut] = sid
                except Exception:
                    # If convention differs, don't crash; mark as missing
                    self.nut_handle_site_ids[nut] = None
        except Exception:
            # If sim/model isn't available at init time, leave mapping empty and resolve at runtime
            self.nut_handle_site_ids = {n: None for n in self.nut_names}
        
        self._print_initialization_info()
    
    def _setup_nuts(self):
        """Setup nut tracking and target peg mapping."""
        # Extract nut names from observations
        self.nut_names = []
        for name in self.obs.keys():
            if "nut" in name.lower() and "_pos" in name:
                # Extract base name without _pos suffix
                base_name = name.replace("_pos", "")
                self.nut_names.append(base_name)
        
        # Sort to ensure consistent ordering (SquareNut first, then RoundNut)
        self.nut_names = sorted(self.nut_names)
        
        # Map nuts to their target pegs
        # Square nut (peg 0), Round nut (peg 1)
        self.nut_to_peg = {}
        for nut_name in self.nut_names:
            if "square" in nut_name.lower():
                self.nut_to_peg[nut_name] = 0
            elif "round" in nut_name.lower():
                self.nut_to_peg[nut_name] = 1
        
        # Initialize nut queue
        self.nuts_to_place = self.nut_names.copy()
        self.current_nut = self.nuts_to_place[-1]
        self.current_peg_id = self.nut_to_peg[self.current_nut]
    
    def _print_initialization_info(self):
        """Print initialization information."""
        print(f"\nDetected {len(self.nut_names)} nuts to place:")
        for nut_name in self.nut_names:
            peg_id = self.nut_to_peg[nut_name]
            print(f"  - {nut_name}: peg {peg_id}")
        
        print(f"\nTable height: {self.table_z}")
        print(f"Safe Z height: {self.safe_z_height}")
        print(f"Currently targeting: {self.current_nut} -> peg {self.current_peg_id}\n")

        print(f"\nobs keys: {list(self.obs.keys())}\n")
    
    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract current end-effector, nut, and peg positions from observations.
        
        Returns:
            Tuple of (end-effector position, current nut position, target peg position)
        """
        eef_pos = self.obs['robot0_eef_pos']

        # Prefer the handle site position for the current nut (grasp there), fall back to body pos
        nut_pos = None
        try:
            sid = self.nut_handle_site_ids.get(self.current_nut, None)
            if sid is None:
                # Try to resolve lazily if not cached
                candidate = f"{self.current_nut}_handle_site"
                try:
                    sid = self.env.sim.model.site_name2id(candidate)
                    self.nut_handle_site_ids[self.current_nut] = sid
                except Exception:
                    sid = None

            if sid is not None:
                nut_pos = np.array(self.env.sim.data.site_xpos[sid])
            else:
                # fall back to body position observable
                nut_pos = self.obs[f'{self.current_nut}_pos']
        except Exception:
            # final fallback: body pos from observations
            nut_pos = self.obs[f'{self.current_nut}_pos']
        
        # Get peg position from simulator (observations do not include it)
        peg_pos = None
        try:
            if getattr(self, "peg_body_ids", None) is not None:
                peg_bid = self.peg_body_ids[self.current_peg_id]
            else:
                peg_name = "peg1" if self.current_peg_id == 0 else "peg2"
                peg_bid = self.env.sim.model.body_name2id(peg_name)
            peg_pos = np.array(self.env.sim.data.body_xpos[peg_bid])
        except Exception:
            # Fallback: approximate peg XY positions relative to table
            if self.current_peg_id == 0:
                peg_pos = np.array([0.0, -0.15, self.table_z])
            else:
                peg_pos = np.array([0.0, 0.15, self.table_z])
        
        return eef_pos, nut_pos, peg_pos
    
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
    
    def stage_move_to_nut(self, eef_pos: np.ndarray, 
                         nut_pos: np.ndarray, 
                         peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Move above nut first."""
        action = np.zeros(self.env.action_dim)
        target_pos = nut_pos.copy()
        target_pos[2] = self.safe_z_height
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = -1  # Open gripper
        
        error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        if error < 0.01:
            next_stage = "lower_to_nut"
            print(f"Stage: move_to_nut -> {next_stage}")
        
        return action, next_stage
    
    def stage_lower_to_nut(self, eef_pos: np.ndarray, 
                          nut_pos: np.ndarray,
                          peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Lower to nut for top-down grasping."""
        action = np.zeros(self.env.action_dim)
        target_pos = nut_pos + np.array([0, 0, self.NUT_Z_OFFSET])
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = -1  # Open gripper
        
        error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        if error < 0.005:
            self.pre_grasp_nut_pos = nut_pos.copy()
            next_stage = "grasp"
            print(f"Stage: lower_to_nut -> {next_stage} "
                  f"(attempt {self.grasp_attempts + 1}/{self.MAX_GRASP_ATTEMPTS})")
        
        return action, next_stage
    
    def stage_grasp(self, eef_pos: np.ndarray, 
                   nut_pos: np.ndarray,
                   peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Close gripper to grasp nut."""
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
                          nut_pos: np.ndarray,
                          peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Verify that nut was successfully grasped."""
        action = np.zeros(self.env.action_dim)
        target_pos = eef_pos + np.array([0, 0, 0.1])
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        next_stage = None
        
        # Check if we've lifted enough to verify
        if eef_pos[2] > self.pre_grasp_nut_pos[2] + 0.08:
            nut_height_change = nut_pos[2] - self.pre_grasp_nut_pos[2]
            
            if nut_height_change > self.GRASP_HEIGHT_THRESHOLD:
                # Successful grasp
                print(f"✓ Grasp successful! Nut lifted {nut_height_change:.3f}m")
                self.grasp_attempts = 0
                next_stage = "lift_nut"
                print(f"Stage: verify_grasp -> {next_stage}")
            else:
                # Failed grasp
                self.grasp_attempts += 1
                print(f"✗ Grasp failed! Nut only lifted {nut_height_change:.3f}m")
                
                if self.grasp_attempts < self.MAX_GRASP_ATTEMPTS:
                    print(f"Retrying grasp (attempt {self.grasp_attempts + 1}/{self.MAX_GRASP_ATTEMPTS})...")
                    next_stage = "move_to_nut"
                else:
                    print(f"Max grasp attempts reached. Skipping {self.current_nut}")
                    self.grasp_attempts = 0
                    next_stage = "skip_nut"
                
                print(f"Stage: verify_grasp -> {next_stage}")
        
        return action, next_stage
    
    def stage_lift_nut(self, eef_pos: np.ndarray, 
                      nut_pos: np.ndarray,
                      peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Lift nut to safe height."""
        action = np.zeros(self.env.action_dim)
        target_pos = eef_pos.copy()
        target_pos[2] = self.safe_z_height
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        if error < 0.01:
            next_stage = "move_to_peg"
            print(f"Stage: lift_nut -> {next_stage}")
        
        return action, next_stage
    
    def stage_move_to_peg(self, eef_pos: np.ndarray, 
                         nut_pos: np.ndarray,
                         peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Move nut above target peg."""
        action = np.zeros(self.env.action_dim)
        target_pos = peg_pos.copy()
        target_pos[2] = self.safe_z_height
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        error_xy = np.linalg.norm((target_pos - eef_pos)[:2])
        next_stage = None
        
        if error_xy < 0.01:
            next_stage = "align_over_peg"
            self.align_counter = 0
            print(f"Stage: move_to_peg -> {next_stage}")
        
        return action, next_stage
    
    def stage_align_over_peg(self, eef_pos: np.ndarray, 
                            nut_pos: np.ndarray,
                            peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Align nut precisely over peg before lowering."""
        action = np.zeros(self.env.action_dim)
        target_pos = peg_pos.copy()
        target_pos[2] = self.safe_z_height
        
        # Fine-tune position over peg
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        self.align_counter += 1
        next_stage = None
        
        error_xy = np.linalg.norm((target_pos - eef_pos)[:2])
        
        if error_xy < 0.005 and self.align_counter > self.ALIGN_DURATION:
            next_stage = "lower_to_peg"
            self.align_counter = 0
            print(f"Stage: align_over_peg -> {next_stage}")
        
        return action, next_stage
    
    def stage_lower_to_peg(self, eef_pos: np.ndarray, 
                          nut_pos: np.ndarray,
                          peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Lower nut onto target peg."""
        action = np.zeros(self.env.action_dim)
        target_pos = peg_pos + np.array([0, 0, self.PEG_Z_OFFSET])
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        if error < 0.02:
            next_stage = "release"
            print(f"Stage: lower_to_peg -> {next_stage}")
        
        return action, next_stage
    
    def stage_release(self, eef_pos: np.ndarray, 
                     nut_pos: np.ndarray,
                     peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Release nut by opening gripper."""
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
                     nut_pos: np.ndarray,
                     peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Retract gripper after releasing nut."""
        action = np.zeros(self.env.action_dim)
        
        if self.retract_target is None:
            self.retract_target = eef_pos + np.array([0, 0, 0.15])
        
        action[:3] = self.compute_position_action(self.retract_target, eef_pos)
        action[6] = -1  # Open gripper
        
        error = np.linalg.norm(self.retract_target - eef_pos)
        next_stage = None
        
        if error < 0.01:
            self.retract_target = None
            
            # Remove current nut from queue
            if self.current_nut in self.nuts_to_place:
                self.nuts_to_place.remove(self.current_nut)
            
            # Move to next nut or complete episode
            next_stage = self._handle_next_nut()
        
        return action, next_stage
    
    def stage_skip_nut(self, eef_pos: np.ndarray, 
                      nut_pos: np.ndarray,
                      peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Skip current nut and move to next."""
        action = np.zeros(self.env.action_dim)
        
        # Remove current nut from queue
        if self.current_nut in self.nuts_to_place:
            self.nuts_to_place.remove(self.current_nut)
        
        # Move to next nut or complete episode
        next_stage = self._handle_next_nut()
        
        return action, next_stage
    
    def stage_done(self, eef_pos: np.ndarray, 
                  nut_pos: np.ndarray,
                  peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Reset environment and start new episode."""
        action = np.zeros(self.env.action_dim)
        
        self.obs = self.env.reset()
        self.nuts_to_place = self.nut_names.copy()
        self.current_nut = self.nuts_to_place[0]
        self.current_peg_id = self.nut_to_peg[self.current_nut]
        self.grasp_attempts = 0
        next_stage = "move_to_nut"
        
        print(f"\nReset complete. Starting with: {self.current_nut} -> peg {self.current_peg_id}")
        
        return action, next_stage
    
    def _handle_next_nut(self) -> str:
        """
        Handle transition to next nut or episode completion.
        
        Returns:
            Next stage name
        """
        if self.nuts_to_place:
            self.current_nut = self.nuts_to_place[0]
            self.current_peg_id = self.nut_to_peg[self.current_nut]
            print(f"\n--- Moving to next nut: {self.current_nut} -> peg {self.current_peg_id} ---")
            return "move_to_nut"
        else:
            print("\n--- All nuts placed! Resetting episode. ---")
            return "done"
    
    def step(self) -> Tuple[np.ndarray, bool]:
        """
        Execute one step of the policy.
        
        Returns:
            Tuple of (action, done flag)
        """
        eef_pos, nut_pos, peg_pos = self.get_current_state()
        
        # State machine dispatcher
        stage_handlers = {
            "move_to_nut": self.stage_move_to_nut,
            "lower_to_nut": self.stage_lower_to_nut,
            "grasp": self.stage_grasp,
            "verify_grasp": self.stage_verify_grasp,
            "lift_nut": self.stage_lift_nut,
            "move_to_peg": self.stage_move_to_peg,
            "align_over_peg": self.stage_align_over_peg,
            "lower_to_peg": self.stage_lower_to_peg,
            "release": self.stage_release,
            "retract": self.stage_retract,
            "skip_nut": self.stage_skip_nut,
            "done": self.stage_done,
        }
        
        handler = stage_handlers.get(self.stage)
        if handler is None:
            raise ValueError(f"Unknown stage: {self.stage}")
        
        action, next_stage = handler(eef_pos, nut_pos, peg_pos)
        
        if next_stage is not None:
            self.stage = next_stage
        
        return action, False


def create_environment(env_name: str = "NutAssembly", single_arm: bool = True):
    """
    Create and configure the robosuite NutAssembly environment.
    
    Args:
        env_name: Name of the environment ("NutAssembly", "NutAssemblySingle", 
                  "NutAssemblySquare", or "NutAssemblyRound")
        single_arm: Whether to use single arm robot
        
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


def run_heuristic_policy(env_name: str = "NutAssembly"):
    """
    Run the heuristic nut assembly policy.
    
    Args:
        env_name: Name of the environment to run
    """
    print(f"Starting heuristic policy for {env_name}...")
    
    # Create environment
    env = create_environment(env_name)
    
    # Create policy
    policy = HeuristicNutAssemblyPolicy(env)
    
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
        description="Run heuristic nut assembly policy"
    )
    parser.add_argument(
        '--env', 
        type=str, 
        default='NutAssembly',
        choices=['NutAssembly', 'NutAssemblySingle', 'NutAssemblySquare', 'NutAssemblyRound'],
        help='Which NutAssembly environment to run'
    )
    
    args = parser.parse_args()
    run_heuristic_policy(env_name=args.env)
