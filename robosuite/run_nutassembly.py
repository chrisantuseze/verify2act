"""
Clean heuristic policy for NutAssembly environments.
This module implements a state-machine-based policy for picking nuts and
placing them on their corresponding pegs.
"""

from robosuite.environments.base import make
from robosuite.controllers import load_composite_controller_config
from robosuite.utils import transform_utils as T
import numpy as np
from typing import Dict, List, Tuple, Optional


class HeuristicNutAssemblyPolicy:
    """
    State-machine-based heuristic policy for nut assembly tasks.
    Handles both square and round nuts with appropriate grasping strategies.
    """
    
    # Constants
    P_GAIN = 5.0 #10.0  # Proportional gain for position control
    R_GAIN = 2.0 #5.0   # Proportional gain for orientation control
    
    # Height offsets
    NUT_Z_OFFSET = 0.005      # Offset for approaching nut (lower for flat nuts)
    SAFE_Z_OFFSET = 0.2      # Safe height above table for movements
    PEG_Z_OFFSET = 0.1       # Height above peg when placing
    
    # Grasp parameters
    MAX_GRASP_ATTEMPTS = 3
    GRASP_HEIGHT_THRESHOLD = 0.04  # Minimum lift to verify grasp (meters)
    NUT_EEF_ATTACH_THRESH = 0.04  # Max distance (m) between nut and EEF to consider it attached
    
    # Counter thresholds
    GRASP_DURATION = 50
    RELEASE_DURATION = 50
    ALIGN_DURATION = 40  # For aligning nut over peg
    ORIENTATION_RESET_DURATION = 20  # Steps to spend resetting gripper orientation after release
    # End-effector stagnation detection (if EEF stays within `EEF_STAGNATION_THRESH`
    # meters for `EEF_STAGNATION_MAX_STEPS` steps, reset the episode)
    EEF_STAGNATION_THRESH = 0.002  # meters (2 mm)
    EEF_STAGNATION_MAX_STEPS = 120 #50
    
    def __init__(self, env):
        """
        Initialize the heuristic policy.
        
        Args:
            env: The robosuite environment instance
        """
        self.env = env
        self.obs = env.reset()

        # Stagnation tracking for end-effector
        self.last_eef_pos = None
        self.eef_stagnation_count = 0
        
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
        # Orientation reset counter (used after release before next grasp)
        self.reset_counter = 0
        
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
        self._cache_nut_sites()
        self._print_initialization_info()
    
    def _setup_nuts(self):
        """Setup nut tracking and target peg mapping."""
        # Extract nut names from observations
        self.nut_names = []
        for name in self.obs.keys():
            if "nut" in name.lower() and "_pos" in name and "robot" not in name.lower():
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
        # Randomize order so the first target is chosen randomly
        np.random.shuffle(self.nuts_to_place)
        self.current_nut = self.nuts_to_place[0]
        self.current_peg_id = self.nut_to_peg[self.current_nut]

    def _cache_nut_sites(self):
        """Cache nut handle, center, and radius site IDs."""        
        self.nut_handle_site_ids = {}
        self.nut_center_site_ids = {}
        self.nut_horizontal_radius_site_ids = {}
        
        for nut in self.nut_names:
            # Handle sites
            try:
                self.nut_handle_site_ids[nut] = self.env.sim.model.site_name2id(f"{nut}_handle_site")
            except:
                self.nut_handle_site_ids[nut] = None
            
            # Center sites
            try:
                self.nut_center_site_ids[nut] = self.env.sim.model.site_name2id(f"{nut}_center_site")
            except:
                self.nut_center_site_ids[nut] = None
            
            # Radius sites
            try:
                self.nut_horizontal_radius_site_ids[nut] = self.env.sim.model.site_name2id(f"{nut}_horizontal_radius_site")
            except:
                self.nut_horizontal_radius_site_ids[nut] = None
    
    def get_nut_handle_pos(self, nut_name: str) -> np.ndarray:
        """Get nut handle position, fallback to body position."""
        sid = self.nut_handle_site_ids.get(nut_name)
        if sid is not None:
            try:
                return np.array(self.env.sim.data.site_xpos[sid])
            except:
                pass
        return self.obs[f'{nut_name}_pos']
    
    def get_nut_center(self, nut_name: str) -> np.ndarray:
        """Get nut center position."""
        sid = self.nut_center_site_ids.get(nut_name)
        if sid is not None:
            try:
                return np.array(self.env.sim.data.site_xpos[sid])
            except:
                pass
        return self.obs[f'{nut_name}_pos']
    
    def get_nut_horizontal_radius(self, nut_name: str) -> float:
        """Get nut horizontal radius estimate."""
        sid = self.nut_horizontal_radius_site_ids.get(nut_name)
        if sid is not None:
            try:
                site_pos = np.array(self.env.sim.data.site_xpos[sid])
                center = self.get_nut_center(nut_name)
                return float(np.linalg.norm((site_pos - center)[:2]))
            except:
                pass
        return 0.06
    
    def get_current_eef_quat(self) -> Optional[np.ndarray]:
        """Get current end-effector quaternion."""
        return np.array(self.obs['robot0_eef_quat'])
    
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

    # def get_nut_center(self, nut_name: Optional[str] = None) -> np.ndarray:
    #     """
    #     Return the geometric center of the specified nut. Prefer the object's
    #     `center_site` if present in the MuJoCo model, otherwise fall back to
    #     the observable body position `'<NutName>_pos'`.

    #     Args:
    #         nut_name: Optional nut name; if None uses `self.current_nut`.

    #     Returns:
    #         Numpy array (3,) with the nut center position in world coordinates.
    #     """
    #     if nut_name is None:
    #         nut_name = self.current_nut

    #     # Try cached site id first, else resolve lazily
    #     sid = self.nut_center_site_ids.get(nut_name, None)
    #     if sid is None:
    #         candidate = f"{nut_name}_center_site"
    #         try:
    #             sid = self.env.sim.model.site_name2id(candidate)
    #             self.nut_center_site_ids[nut_name] = sid
    #         except Exception:
    #             sid = None

    #     if sid is not None:
    #         try:
    #             return np.array(self.env.sim.data.site_xpos[sid])
    #         except Exception:
    #             pass

    #     # Fallback to body pos observable
    #     return np.array(self.obs.get(f"{nut_name}_pos"))

    # def get_nut_horizontal_radius(self, nut_name: Optional[str] = None) -> float:
    #     """
    #     Return an estimate of the nut's horizontal radius (distance from center to outermost horizontal marker).
    #     Prefers the `horizontal_radius_site` if present; otherwise returns a conservative default.

    #     Args:
    #         nut_name: Optional nut name; if None uses `self.current_nut`.

    #     Returns:
    #         Float radius in meters (XY-plane).
    #     """
    #     if nut_name is None:
    #         nut_name = self.current_nut

    #     sid = self.nut_horizontal_radius_site_ids.get(nut_name, None)
    #     if sid is None:
    #         candidate = f"{nut_name}_horizontal_radius_site"
    #         try:
    #             sid = self.env.sim.model.site_name2id(candidate)
    #             self.nut_horizontal_radius_site_ids[nut_name] = sid
    #         except Exception:
    #             sid = None

    #     if sid is not None:
    #         try:
    #             site_pos = np.array(self.env.sim.data.site_xpos[sid])
    #             center = self.get_nut_center(nut_name)
    #             return float(np.linalg.norm((site_pos - center)[:2]))
    #         except Exception:
    #             pass

    #     # Conservative fallback radius (meters)
    #     return 0.06

    def get_current_eef_quat(self) -> Optional[np.ndarray]:
        """Return current end-effector quaternion in xyzw order, or None.
        """
        return np.array(self.obs['robot0_eef_quat'])

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_yaw_action(self, nut_name: Optional[str] = None) -> np.ndarray:
        """
        Compute an axis-angle vector [0,0,yaw_delta] that rotates the EEF yaw
        to face the nut handle (yaw-only). Scales by `R_GAIN` and clips.
        """
        if nut_name is None:
            nut_name = self.current_nut

        # Get nut center and handle
        nut_center = self.get_nut_center(nut_name)
        sid = self.nut_handle_site_ids.get(nut_name, None)
        if sid is None:
            try:
                candidate = f"{nut_name}_handle_site"
                sid = self.env.sim.model.site_name2id(candidate)
                self.nut_handle_site_ids[nut_name] = sid
            except Exception:
                sid = None

        if sid is None:
            return np.zeros(3)

        handle_pos = np.array(self.env.sim.data.site_xpos[sid])
        handle_vec = handle_pos - nut_center
        desired_yaw = np.arctan2(handle_vec[1], handle_vec[0])
        current_quat = self.get_current_eef_quat()

        if current_quat is None:
            return np.zeros(3)

        # Get yaw from current quaternion via rotation matrix -> euler
        try:
            cur_mat = T.quat2mat(current_quat)
            _, _, cur_yaw = T.mat2euler(cur_mat)
        except Exception:
            return np.zeros(3)

        yaw_err = self._wrap_angle(desired_yaw - cur_yaw)
        yaw_delta = np.clip(self.R_GAIN * yaw_err, -0.6, 0.6)
        return np.array([0.0, 0.0, yaw_delta])

    def compute_insertion_xy(self, nut_name: Optional[str], peg_pos: np.ndarray) -> np.ndarray:
        """
        Compute an edge-biased insertion XY target for `nut_name` and `peg_pos`.

        Logic:
        - Use the nut's `center_site` if present, else body pos.
        - If a `handle_site` exists, compute the direction away from the handle
          (non-handle direction) and offset the peg by the nut horizontal radius
          along that direction so the nut's opposite edge is above the peg.
        - Otherwise fall back to averaging nut center and peg XY.

        Returns:
            `np.ndarray` shape (2,) with insertion XY.
        """
        nut_center = self.get_nut_center(nut_name)

        sid = self.nut_handle_site_ids.get(nut_name, None)
        if sid is None:
            try:
                candidate = f"{nut_name}_handle_site"
                sid = self.env.sim.model.site_name2id(candidate)
                self.nut_handle_site_ids[nut_name] = sid
            except Exception:
                sid = None

        if sid is not None:
            try:
                handle_pos = np.array(self.env.sim.data.site_xpos[sid])
                handle_vec = handle_pos - nut_center
                handle_dist_xy = np.linalg.norm(handle_vec[:2])
            except Exception:
                handle_vec = np.zeros(3)
                handle_dist_xy = 0.0
        else:
            handle_vec = np.zeros(3)
            handle_dist_xy = 0.0

        if handle_dist_xy > 1e-6:
            non_handle_dir = -handle_vec[:2] / handle_dist_xy
            offset = self.get_nut_horizontal_radius(nut_name) * 0.95
            insertion_xy = peg_pos[:2] - non_handle_dir * offset
        else:
            insertion_xy = np.array([(peg_pos[0] + nut_center[0]) / 2.0,
                                     (peg_pos[1] + nut_center[1]) / 2.0])

        return insertion_xy
    
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
        # Keep gripper oriented toward handle while lowering
        if self.env.action_dim >= 6:
            action[3:6] = self.compute_yaw_action(self.current_nut)

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
            # Verify the nut is still attached to the gripper by checking
            # the distance between the nut and the EEF. If the nut slipped
            # during lift, retry the grasp (or skip after max attempts).
            attach_dist = np.linalg.norm(nut_pos - eef_pos)

            if attach_dist <= self.NUT_EEF_ATTACH_THRESH:
                next_stage = "move_to_peg"
                print(f"Stage: lift_nut -> {next_stage} (attach_dist={attach_dist:.3f}m)")
            else:
                # Treat as failed grasp
                self.grasp_attempts += 1
                print(f"✗ Nut not attached after lift (dist={attach_dist:.3f}m)."
                      f" Attempt {self.grasp_attempts}/{self.MAX_GRASP_ATTEMPTS}")

                if self.grasp_attempts < self.MAX_GRASP_ATTEMPTS:
                    next_stage = "move_to_nut"
                    print(f"Retrying grasp: returning to {next_stage}")
                else:
                    print(f"Max grasp attempts reached. Skipping {self.current_nut}")
                    self.grasp_attempts = 0
                    next_stage = "skip_nut"
        
        return action, next_stage
    
    def stage_move_to_peg(self, eef_pos: np.ndarray, 
                         nut_pos: np.ndarray,
                         peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Move nut above target peg."""
        action = np.zeros(self.env.action_dim)
        insertion_xy = self.compute_insertion_xy(self.current_nut, peg_pos)
        target_pos = peg_pos.copy()
        target_pos[:2] = insertion_xy
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
        insertion_xy = self.compute_insertion_xy(self.current_nut, peg_pos)
        target_pos = peg_pos.copy()
        target_pos[:2] = insertion_xy
        target_pos[2] = self.safe_z_height
        
        # Fine-tune position over peg
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        # While aligning above peg, set gripper orientation to face the handle
        if self.env.action_dim >= 6:
            action[3:6] = self.compute_yaw_action(self.current_nut)
        
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
        insertion_xy = self.compute_insertion_xy(self.current_nut, peg_pos)
        target_pos = peg_pos + np.array([0, 0, self.PEG_Z_OFFSET])
        target_pos[:2] = insertion_xy
        
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
            
            # Move to next nut or complete episode. If there is a next nut,
            # first transition to `reset_orientation` so the gripper recenters
            # before starting the next approach.
            next_stage = self._handle_next_nut()
            if next_stage == "move_to_nut":
                next_stage = "reset_orientation"
        
        return action, next_stage

    def stage_reset_orientation(self, eef_pos: np.ndarray,
                                nut_pos: np.ndarray,
                                peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Reset gripper yaw to a neutral heading before moving to next nut."""
        action = np.zeros(self.env.action_dim)

        # Keep position steady (hover) while we reset orientation
        hover_target = eef_pos.copy()
        hover_target[2] = max(self.safe_z_height, eef_pos[2])
        action[:3] = self.compute_position_action(hover_target, eef_pos)

        # Drive yaw toward neutral (0 rad) using same yaw helper logic
        # but with desired yaw of 0.0
        current_quat = self.get_current_eef_quat()

        if self.env.action_dim >= 6 and current_quat is not None:
            try:
                cur_mat = T.quat2mat(current_quat)
                _, _, cur_yaw = T.mat2euler(cur_mat)
                yaw_err = self._wrap_angle(0.0 - cur_yaw)
                yaw_delta = np.clip(self.R_GAIN * yaw_err, -0.6, 0.6)
                action[3:6] = np.array([0.0, 0.0, yaw_delta])
            except Exception:
                # leave orientation command zero if any failure
                pass

        action[6] = -1  # keep gripper open while resetting orientation

        self.reset_counter += 1
        next_stage = None

        if self.reset_counter > self.ORIENTATION_RESET_DURATION:
            self.reset_counter = 0
            next_stage = "move_to_nut"
            print(f"Stage: reset_orientation -> {next_stage}")

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
        if next_stage == "move_to_nut":
            next_stage = "reset_orientation"
        
        return action, next_stage
    
    def stage_done(self, eef_pos: np.ndarray, 
                  nut_pos: np.ndarray,
                  peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Reset environment and start new episode."""
        action = np.zeros(self.env.action_dim)
        
        self.obs = self.env.reset()
        self.nuts_to_place = self.nut_names.copy()
        # Randomize order on episode reset as well so the first nut varies
        np.random.shuffle(self.nuts_to_place)
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

        # print(f"\nCurrent State {self.current_nut} -- EEF Pos: {eef_pos}, Nut Pos: {nut_pos}, Peg Pos: {peg_pos}")
        # --- EEF stagnation detection ---
        try:
            if self.last_eef_pos is None:
                self.last_eef_pos = eef_pos.copy()
                self.eef_stagnation_count = 0
            else:
                moved_dist = np.linalg.norm(eef_pos - self.last_eef_pos)
                if moved_dist < self.EEF_STAGNATION_THRESH:
                    self.eef_stagnation_count += 1
                else:
                    self.eef_stagnation_count = 0
                    self.last_eef_pos = eef_pos.copy()

            if self.eef_stagnation_count >= self.EEF_STAGNATION_MAX_STEPS:
                print(f"EEF stagnation detected (moved {moved_dist:.6f}m over {self.eef_stagnation_count} steps). Resetting episode.")
                action, next_stage = self.stage_done(eef_pos, nut_pos, peg_pos)
                # stage_done sets next stage to 'move_to_nut'
                self.stage = next_stage
                # reset stagnation tracking
                self.last_eef_pos = None
                self.eef_stagnation_count = 0
                return action, False
        except Exception:
            # If any error in stagnation logic, fail-safe to continue normally
            pass
        
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
            "reset_orientation": self.stage_reset_orientation,
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
