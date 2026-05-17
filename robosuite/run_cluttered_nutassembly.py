"""
Clean heuristic policy for NutAssembly environments.
This module implements a state-machine-based policy for picking nuts and
placing them on their corresponding pegs.
"""

from robosuite.environments.base import make
from robosuite.controllers import load_composite_controller_config
from robosuite.utils import transform_utils as T
from goal_renderer import NutAssemblyGoalRenderer
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


class SubtaskType(Enum):
    """High-level subtask types."""
    REMOVE_OBSTACLE = "remove_obstacle"
    GRASP_TARGET = "grasp_target"
    PLACE_ON_PEG = "place_on_peg"


@dataclass
class Subtask:
    """Represents a high-level subtask in the plan."""
    type: SubtaskType
    object_name: str
    target: Optional[str] = None  # For place operations
    placement_type: str = "peg"  # "peg" or "table" - where to place the object
    
    def __repr__(self):
        if self.target:
            return f"{self.type.value}({self.object_name} -> {self.target})"
        return f"{self.type.value}({self.object_name})"


class SymbolicPlanner:
    """
    Symbolic planner that generates and modifies high-level task plans.
    
    Plans are sequences of Subtask objects. The planner can:
    - Generate initial plans to place all round nuts
    - Check preconditions (is nut graspable?)
    - Insert obstacle removal subtasks when blockages are detected
    - Replan when unexpected obstructions occur during execution
    """
    
    def __init__(self, env):
        self.env = env
        self.obstruction_check_threshold = 0.1  # XY distance threshold (3cm)
        self.z_stack_threshold_min = 0.02  # Min Z diff to consider stacking (2cm)
        self.z_stack_threshold_max = 0.06  # Max Z diff to consider stacking (6cm)
        
    def is_graspable(self, nut_name: str, debug=False) -> Tuple[bool, Optional[str]]:
        """
        Check if a nut is graspable (no other nut on top).
        
        Returns:
            (graspable, blocker_name): True if clear, else False and blocker name
        """
        nut_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[nut_name]]
        nut_z = nut_pos[2]
        
        # Check all other nuts
        all_nuts = self.env.round_nut_names + self.env.square_nut_names
        for other_nut in all_nuts:
            if other_nut == nut_name:
                continue
                
            other_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[other_nut]]
            
            # Check if other nut is above this one
            xy_dist = np.linalg.norm(nut_pos[:2] - other_pos[:2])
            z_diff = other_pos[2] - nut_z

            if debug:
                print(f"  Checking obstruction: {other_nut} at {other_pos}, "
                      f"xy_dist={xy_dist:.3f}, z_diff={z_diff:.3f}")
            
            # Only consider it blocked if:
            # 1. Other nut is close in XY
            # 2. Other nut is ABOVE (positive z_diff)
            # 3. Z difference is within stacking range
            if (xy_dist < self.obstruction_check_threshold and 
                self.z_stack_threshold_min < z_diff < self.z_stack_threshold_max):
                if debug:
                    print(f"  -> {nut_name} IS BLOCKED by {other_nut}")
                return False, other_nut
        
        return True, None
    
    def generate_initial_plan(self) -> List[Subtask]:
        """
        Generate initial plan to place all target-type nuts on the target peg.
        Checks for initial obstructions and inserts clearing subtasks.
        """
        print("\n📝 Generating initial plan...")
        plan = []

        if self.env.current_nut_type == "round":
            target_nuts = self.env.round_nut_names
        else:
            target_nuts = self.env.square_nut_names

        target_peg_id = self.env.nut_type_to_peg[self.env.current_nut_type]

        for target_nut in target_nuts:
            # Check if this target nut is blocked
            graspable, blocker = self.is_graspable(target_nut, debug=True)
            
            # If blocked, recursively check if blocker is also blocked
            blockers_to_remove = []
            current_blocker = blocker
            while current_blocker is not None and current_blocker not in blockers_to_remove:
                blockers_to_remove.append(current_blocker)
                graspable, current_blocker = self.is_graspable(current_blocker, debug=False)
            
            # Add clearing subtasks (top to bottom) - place obstacles on table
            for blocker in blockers_to_remove:
                plan.append(Subtask(SubtaskType.REMOVE_OBSTACLE, blocker, "table", placement_type="table"))
            
            # Add grasp and place subtasks for the target nut
            plan.append(Subtask(SubtaskType.GRASP_TARGET, target_nut))
            plan.append(Subtask(SubtaskType.PLACE_ON_PEG, target_nut, f"peg{target_peg_id}", placement_type="peg"))
        
        return plan
    
    def replan(self, remaining_plan: List[Subtask], failed_subtask: Subtask, 
               failure_reason: str) -> List[Subtask]:
        """
        Modify plan when execution fails.
        
        Args:
            remaining_plan: Subtasks that haven't been executed yet
            failed_subtask: The subtask that just failed
            failure_reason: Why it failed (e.g., "blocked_by_SquareNut1")
        
        Returns:
            Updated plan with new subtasks inserted
        """
        print(f"\n🔄 REPLANNING: {failed_subtask} failed due to: {failure_reason}")
        
        # Parse blocker from failure reason
        if "blocked_by_" in failure_reason:
            blocker = failure_reason.split("blocked_by_")[1]
            
            # Check if blocker is also blocked (chain of obstructions)
            blockers_to_remove = []
            current_blocker = blocker
            while current_blocker is not None:
                blockers_to_remove.append(current_blocker)
                _, current_blocker = self.is_graspable(current_blocker)
            
            # Insert clearing subtasks before retrying the failed subtask
            new_plan = []
            for blocker in blockers_to_remove:
                new_plan.append(Subtask(SubtaskType.REMOVE_OBSTACLE, blocker, "table", placement_type="table"))
            
            # Retry the failed subtask
            new_plan.append(failed_subtask)
            
            # Add remaining plan
            new_plan.extend(remaining_plan)
            
            print(f"📋 New plan length: {len(new_plan)} (inserted {len(blockers_to_remove)} clearing subtasks)")
            return new_plan
        
        # Default: just retry and continue
        return [failed_subtask] + remaining_plan

class HeuristicNutAssemblyPolicy:
    """
    State-machine-based heuristic policy for nut assembly tasks.
    Handles both square and round nuts with appropriate grasping strategies.
    """
    
    # Constants
    P_GAIN = 6.0  # Proportional gain for position control
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
    GRASP_DURATION = 25
    RELEASE_DURATION = 20
    ALIGN_DURATION = 40  # For aligning nut over peg
    PRE_GRASP_ALIGN_DURATION = 15  # Min steps to spend aligning orientation before lowering to nut
    ORIENTATION_RESET_DURATION = 50  # Steps to spend resetting gripper orientation after release
    # End-effector stagnation detection (if EEF stays within `EEF_STAGNATION_THRESH`
    # meters for `EEF_STAGNATION_MAX_STEPS` steps, reset the episode)
    EEF_STAGNATION_THRESH = 0.0015  # meters (2 mm)
    EEF_STAGNATION_MAX_STEPS = 150 #50
    
    def __init__(self, env, data_collection_mode: bool = True):
        """
        Initialize the heuristic policy.
        
        Args:
            env: The robosuite environment instance
            data_collection_mode: If True, disable retries for clean training trajectories
        """
        self.env = env
        self.obs = env.reset()
        self.data_collection_mode = data_collection_mode

        # Initialize planner (used for precondition checking only in reactive mode)
        self.planner = SymbolicPlanner(env)

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
        self.pre_grasp_align_counter = 0  # For pre-grasp orientation alignment
        self.grasp_attempts = 0
        self.pre_grasp_nut_pos = None
        self.retract_target = None
        self.table_placement_target = None  # Cache for table placement location
        self.original_target_nut = None  # Track original target before clearing obstacles
        # Orientation reset counter (used after release before next grasp)
        self.reset_counter = 0
        # Yaw offset between nut and gripper, captured at grasp time
        self.nut_gripper_yaw_offset = None
        
        # Calculate safe z height
        self.table_z = self.env.table_offset[2]
        self.safe_z_height = self.table_z + self.SAFE_Z_OFFSET

        # Cache initial EEF position for reset orientation stage
        self.init_eef_pos = self.obs.get("robot0_eef_pos", None)

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
        print("Detected nuts:", self.nut_names)
        
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
        self.current_nut = self._get_nut_by_type(self.env.current_nut_type)
        self.current_peg_id = self.nut_to_peg[self.current_nut]

        print(f"Currently targeting: {self.current_nut} -> peg {self.current_peg_id}")

    def _get_nut_by_type(self, nut_type: str) -> str:
        """
        Get the first nut name from nuts_to_place that matches the given type.
        
        Args:
            nut_type: The nut type ("round" or "square")
            
        Returns:
            The full nut name (e.g., "roundnut0", "squarenut0")
        """
        for nut_name in self.nuts_to_place:
            if nut_type.lower() in nut_name.lower():
                return nut_name
        # Fallback to first nut if no match found
        return self.nuts_to_place[0] if self.nuts_to_place else None

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
    
    def get_peg_orientation(self, peg_id: int) -> float:
        """Get the yaw orientation of the specified peg."""
        try:
            if hasattr(self, 'peg_body_ids') and self.peg_body_ids is not None:
                peg_bid = self.peg_body_ids[peg_id]
            else:
                peg_name = f"peg{peg_id + 1}"
                peg_bid = self.env.sim.model.body_name2id(peg_name)
            
            # Get peg quaternion and convert to yaw
            peg_quat = self.env.sim.data.body_xquat[peg_bid]
            peg_mat = T.quat2mat(peg_quat)
            _, _, peg_yaw = T.mat2euler(peg_mat)
            return peg_yaw
        except Exception as e:
            print(f"Warning: Could not get peg orientation: {e}")
            return 0.0
    
    def get_nut_orientation(self, nut_name: str) -> float:
        """Get the yaw orientation of the specified nut."""
        try:
            nut_bid = self.env.obj_body_id[nut_name]
            nut_quat = self.env.sim.data.body_xquat[nut_bid]
            nut_mat = T.quat2mat(nut_quat)
            _, _, nut_yaw = T.mat2euler(nut_mat)
            return nut_yaw
        except Exception as e:
            print(f"Warning: Could not get nut orientation: {e}")
            return 0.0

    def get_gripper_yaw(self) -> Optional[float]:
        """Get current gripper yaw (radians) from end-effector quaternion."""
        current_quat = self.get_current_eef_quat()
        if current_quat is None:
            return None
        try:
            cur_mat = T.quat2mat(current_quat)
            _, _, cur_yaw = T.mat2euler(cur_mat)
            return cur_yaw
        except Exception:
            return None
    
    def _find_safe_table_placement(self) -> np.ndarray:
        """Find a safe location on the table to place an obstacle nut."""
        # Define table boundaries (away from pegs and existing nuts)
        table_center = self.env.table_offset[:2]
        
        # Try several candidate positions (wider spacing for better clearance)
        candidates = [
            table_center + np.array([0.2, 0.2]),     # Front right
            table_center + np.array([-0.2, 0.2]),    # Front left
            table_center + np.array([0.2, -0.2]),    # Back right
            table_center + np.array([-0.2, -0.2]),   # Back left
            table_center + np.array([0.25, 0.0]),    # Far right
            table_center + np.array([-0.25, 0.0]),   # Far left
            table_center + np.array([0.0, 0.25]),    # Far front
            table_center + np.array([0.0, -0.25]),   # Far back
        ]
        
        # Find candidate with most clearance from other nuts (excluding current nut being moved)
        best_pos = candidates[0]
        best_min_dist = -1.0
        
        for candidate in candidates:
            # Check distance to all nuts except the one we're currently holding
            min_dist = float('inf')
            for nut_name in self.nut_names:
                # Skip the nut we're currently moving
                if nut_name == self.current_nut:
                    continue
                
                try:
                    # Try with exact case first, then capitalized
                    if f'{nut_name}_pos' in self.obs:
                        nut_pos = self.obs[f'{nut_name}_pos'][:2]
                    else:
                        nut_pos = self.obs[f'{nut_name.capitalize()}_pos'][:2]
                    
                    dist = np.linalg.norm(candidate - nut_pos)
                    min_dist = min(min_dist, dist)
                except KeyError:
                    # Skip if observation not found
                    continue
            
            # Keep candidate with largest minimum distance
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = candidate
        
        print(f"  Table placement: {best_pos}, min clearance: {best_min_dist:.3f}m")
        
        # Return 3D position (xy from candidate, z at table height)
        return np.array([best_pos[0], best_pos[1], self.table_z + 0.02])
    
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
    
    def _is_current_nut_obstacle(self) -> bool:
        """Check if current nut is an obstacle (different type from target) or target nut.
        
        Returns:
            True if current nut is an obstacle that should be placed on table,
            False if it's a target nut that should be placed on peg.
        """
        # Check if current nut type matches the target nut type for this episode
        if self.env.current_nut_type == "roundnut":
            # Targeting round nuts, so square nuts are obstacles
            return "square" in self.current_nut.lower()
        else:  # current_nut_type == "squarenut"
            # Targeting square nuts, so round nuts are obstacles
            return "round" in self.current_nut.lower()
    
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
                nut_pos = self.obs[f'{self.current_nut.capitalize()}_pos']
        except Exception:
            # final fallback: body pos from observations
            nut_pos = self.obs[f'{self.current_nut.capitalize()}_pos']
        
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

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _snap_to_axis_aligned(self, yaw: float) -> float:
        """
        Snap a yaw angle to the nearest multiple of 90° (π/2).
        This keeps the gripper parallel to the table edges regardless of
        small nut perturbations.
        """
        quarter = np.pi / 2
        return round(yaw / quarter) * quarter

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

    def compute_axis_aligned_yaw_action(self, nut_name: Optional[str] = None) -> np.ndarray:
        """
        Like compute_yaw_action but snaps the desired yaw to the nearest 90°
        multiple before computing the error.  This keeps the gripper parallel
        to the table edges for both nut types, even if a nut was nudged during
        approach.
        """
        if nut_name is None:
            nut_name = self.current_nut

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
        raw_yaw = np.arctan2(handle_vec[1], handle_vec[0])
        desired_yaw = self._snap_to_axis_aligned(raw_yaw)  # nearest 0/90/180/270°

        current_quat = self.get_current_eef_quat()
        if current_quat is None:
            return np.zeros(3)

        try:
            cur_mat = T.quat2mat(current_quat)
            _, _, cur_yaw = T.mat2euler(cur_mat)
        except Exception:
            return np.zeros(3)

        yaw_err = self._wrap_angle(desired_yaw - cur_yaw)
        yaw_delta = np.clip(self.R_GAIN * yaw_err, -0.6, 0.6)
        return np.array([0.0, 0.0, yaw_delta])

    def compute_square_alignment_action(self, nut_name: str, peg_id: int) -> np.ndarray:
        """
        Compute orientation action to align square nut hole with square peg.
        This rotates the gripper (and thus the grasped nut) to match the peg's orientation.
        """
        try:
            # Get peg orientation
            peg_yaw = self.get_peg_orientation(peg_id)
            cur_gripper_yaw = self.get_gripper_yaw()
            if cur_gripper_yaw is None:
                return np.zeros(3)

            # If we know the nut↔gripper yaw offset from the grasp, use it.
            # Desired gripper yaw so that: nut_yaw = peg_yaw
            # => gripper_yaw + offset = peg_yaw
            desired_gripper_yaw = peg_yaw - self.nut_gripper_yaw_offset
            gripper_yaw_err = self._wrap_angle(desired_gripper_yaw - cur_gripper_yaw)
            yaw_delta = np.clip(self.R_GAIN * gripper_yaw_err, -0.6, 0.6)
            return np.array([0.0, 0.0, yaw_delta])
        except Exception as e:
            print(f"Warning: Could not compute square alignment: {e}")
            return np.zeros(3)

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
        """Move above nut and pre-align gripper orientation before lowering."""
        action = np.zeros(self.env.action_dim)
        target_pos = nut_pos.copy()
        target_pos[2] = self.safe_z_height
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        
        # Pre-align gripper orientation to face the nut handle before lowering.
        # This ensures the fingers close around the handle for a reliable grasp.
        if self.env.action_dim >= 6:
            action[3:6] = self.compute_yaw_action(self.current_nut)

        action[6] = -1  # Open gripper
        
        pos_error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        # Check if both position and orientation are ready
        if pos_error < 0.01:
            self.pre_grasp_align_counter += 1
            
            # Check orientation alignment
            yaw_aligned = True
            current_quat = self.get_current_eef_quat()
            if current_quat is not None:
                try:
                    # Get desired yaw from nut handle
                    nut_center = self.get_nut_center(self.current_nut)
                    sid = self.nut_handle_site_ids.get(self.current_nut, None)
                    if sid is not None:
                        handle_pos = np.array(self.env.sim.data.site_xpos[sid])
                        handle_vec = handle_pos - nut_center
                        desired_yaw = np.arctan2(handle_vec[1], handle_vec[0])

                        cur_mat = T.quat2mat(current_quat)
                        _, _, cur_yaw = T.mat2euler(cur_mat)
                        yaw_err = abs(self._wrap_angle(desired_yaw - cur_yaw))
                        yaw_aligned = yaw_err < np.radians(10)  # Within 10 degrees
                except Exception:
                    yaw_aligned = True  # If we can't check, proceed anyway
            
            # Proceed only if orientation is aligned or we've waited long enough
            if (yaw_aligned and self.pre_grasp_align_counter > self.PRE_GRASP_ALIGN_DURATION) or \
               (self.pre_grasp_align_counter > self.PRE_GRASP_ALIGN_DURATION * 3):  # Timeout
                next_stage = "lower_to_nut"
                self.pre_grasp_align_counter = 0
                print(f"Stage: move_to_nut -> {next_stage}")
        else:
            # Reset counter if not at position yet
            self.pre_grasp_align_counter = 0
        
        return action, next_stage
    
    def stage_lower_to_nut(self, eef_pos: np.ndarray, 
                          nut_pos: np.ndarray,
                          peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Lower to nut for top-down grasping."""
        action = np.zeros(self.env.action_dim)
        target_pos = nut_pos + np.array([0, 0, self.NUT_Z_OFFSET])
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        # Keep gripper facing the handle while lowering for a reliable grasp.
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
            # self._next_subtask()
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
                # Capture nut↔gripper yaw offset for square alignment
                try:
                    if "square" in self.current_nut.lower():
                        nut_yaw = self.get_nut_orientation(self.current_nut)
                        gripper_yaw = self.get_gripper_yaw()
                        if gripper_yaw is not None:
                            self.nut_gripper_yaw_offset = self._wrap_angle(nut_yaw - gripper_yaw)
                        else:
                            self.nut_gripper_yaw_offset = None
                except Exception:
                    self.nut_gripper_yaw_offset = None
                next_stage = "lift_nut"
                print(f"Stage: verify_grasp -> {next_stage}")
            else:
                # Failed grasp
                self.grasp_attempts += 1
                print(f"✗ Grasp failed! Nut only lifted {nut_height_change:.3f}m")
                
                # In data collection mode, don't retry - end episode immediately
                if self.data_collection_mode:
                    print(f"Data collection mode: Grasp failure detected. Ending episode.")
                    return action, "terminate"  # End episode immediately
                elif self.grasp_attempts < self.MAX_GRASP_ATTEMPTS:
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
                # Check if this is an obstacle (wrong nut type) or target placement
                if self._is_current_nut_obstacle():
                    print(f"✓ Lifted obstacle {self.current_nut}, moving to table placement")
                    # Compute and cache table placement location once
                    self.table_placement_target = self._find_safe_table_placement()
                    next_stage = "move_to_table"
                    self.align_counter = 0
                else:
                    next_stage = "move_to_peg"
                    print(f"Stage: lift_nut -> {next_stage} (attach_dist={attach_dist:.3f}m)")
            else:
                # In data collection mode, don't retry - end episode immediately
                if self.data_collection_mode:
                    print(f"Data collection mode: Nut detachment detected. Ending episode.")
                    return action, "terminate"  # End episode immediately
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
        # Begin rotating to axis-aligned yaw early so less correction is needed
        # in align_over_peg. Square nuts use peg alignment; round nuts snap to 90°.
        if self.env.action_dim >= 6:
            if "square" in self.current_nut.lower():
                action[3:6] = self.compute_square_alignment_action(self.current_nut, self.current_peg_id)
            else:
                action[3:6] = self.compute_axis_aligned_yaw_action(self.current_nut)

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
        
        # Align orientation based on nut type
        if "square" in self.current_nut.lower():
            # For square nuts: align hole with square peg orientation
            action[3:6] = self.compute_square_alignment_action(self.current_nut, self.current_peg_id)
            if self.align_counter % 20 == 0:  # Log periodically
                peg_yaw = self.get_peg_orientation(self.current_peg_id)
                gripper_yaw = self.get_gripper_yaw()
                if gripper_yaw is not None and self.nut_gripper_yaw_offset is not None:
                    nut_yaw = self._wrap_angle(gripper_yaw + self.nut_gripper_yaw_offset)
                else:
                    nut_yaw = self.get_nut_orientation(self.current_nut)
                alignment_err = self._wrap_angle(peg_yaw - nut_yaw)
                print(f"  Square alignment: peg_yaw={np.degrees(peg_yaw):.1f}°, "
                        f"nut_yaw={np.degrees(nut_yaw):.1f}°, error={np.degrees(alignment_err):.1f}°")
        else:
            # For round nuts: snap to nearest 90° so the gripper is axis-aligned
            # with the table/peg axis before insertion.
            action[3:6] = self.compute_axis_aligned_yaw_action(self.current_nut)
        
        self.align_counter += 1
        next_stage = None
        
        error_xy = np.linalg.norm((target_pos - eef_pos)[:2])
        
        # For square nuts, also check orientation alignment before proceeding
        can_proceed = error_xy < 0.005 and self.align_counter > self.ALIGN_DURATION
        if can_proceed and "square" in self.current_nut.lower():
            # Additional check: orientation must be reasonably aligned
            peg_yaw = self.get_peg_orientation(self.current_peg_id)
            gripper_yaw = self.get_gripper_yaw()
            if gripper_yaw is not None and self.nut_gripper_yaw_offset is not None:
                nut_yaw = self._wrap_angle(gripper_yaw + self.nut_gripper_yaw_offset)
            else:
                nut_yaw = self.get_nut_orientation(self.current_nut)
            alignment_err = abs(self._wrap_angle(peg_yaw - nut_yaw))
            if alignment_err > np.radians(15):  # More than 15 degrees off
                can_proceed = False
                if self.align_counter > self.ALIGN_DURATION * 3:  # Timeout after 3x normal duration
                    print(f"  Warning: Square alignment timeout (error={np.degrees(alignment_err):.1f}°), proceeding anyway")
                    can_proceed = True
        
        if can_proceed:
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
        
        # Maintain orientation alignment during lowering.
        if self.env.action_dim >= 6:
            if "square" in self.current_nut.lower():
                action[3:6] = self.compute_square_alignment_action(self.current_nut, self.current_peg_id)
            else:
                action[3:6] = self.compute_axis_aligned_yaw_action(self.current_nut)
        
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
    
    def stage_move_to_table(self, eef_pos: np.ndarray, 
                           nut_pos: np.ndarray,
                           peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Move obstacle nut to a safe location on the table."""
        action = np.zeros(self.env.action_dim)
        
        # Use cached table placement target (computed once in lift_nut stage)
        if self.table_placement_target is None:
            # Fallback: compute if somehow not cached
            self.table_placement_target = self._find_safe_table_placement()
        
        target_pos = self.table_placement_target.copy()
        target_pos[2] = self.safe_z_height  # Move at safe height
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        error_xy = np.linalg.norm((target_pos - eef_pos)[:2])
        next_stage = None
        
        if error_xy < 0.02:  # Slightly relaxed threshold
            print(f"✓ Reached table placement location, lowering obstacle {self.current_nut}")
            next_stage = "lower_to_table"
        
        return action, next_stage
    
    def stage_lower_to_table(self, eef_pos: np.ndarray, 
                            nut_pos: np.ndarray,
                            peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Lower obstacle nut onto table surface."""
        action = np.zeros(self.env.action_dim)
        
        # Use cached table placement target
        if self.table_placement_target is None:
            # Fallback: compute if somehow not cached
            self.table_placement_target = self._find_safe_table_placement()
        
        target_pos = self.table_placement_target.copy()
        target_pos[2] = self.table_z + 0.05  # Just above table surface (5cm)
        
        action[:3] = self.compute_position_action(target_pos, eef_pos)
        action[6] = 1  # Keep gripper closed
        
        error = np.linalg.norm(target_pos - eef_pos)
        next_stage = None
        
        if error < 0.03:  # Slightly relaxed threshold
            print(f"✓ Lowered obstacle {self.current_nut} to table, releasing")
            next_stage = "release"
            self.release_counter = 0
        
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
            # Check if we just placed an obstacle or a target nut
            if self._is_current_nut_obstacle():
                print(f"✓ Obstacle {self.current_nut} cleared, moving to next subtask")
                self.table_placement_target = None  # Clear cache for next obstacle
                
                # Switch back to the original target nut if we have one
                if self.original_target_nut is not None:
                    print(f"  Returning to original target: {self.original_target_nut}")
                    self.current_nut = self.original_target_nut
                    self.current_peg_id = self.nut_to_peg.get(self.current_nut, 0)
                    self.original_target_nut = None  # Clear it after use
                
                next_stage = "reset_orientation"
                self.reset_counter = 0
            else:
                # Just placed a target nut on peg
                # Don't remove from queue - let environment tracking handle it
                # The environment will determine if nut is on peg or off table
                
                # Move to next nut or complete episode. If there is a next nut,
                # first transition to `reset_orientation` so the gripper recenters
                # before starting the next approach.
                next_stage = self._handle_next_nut()
                print(f"Stage: retract -> {next_stage}, {self.current_nut} next")
                if next_stage == "move_to_nut":
                    next_stage = "reset_orientation"
            
            self.retract_target = None
        
        return action, next_stage

    def stage_reset_orientation(self, eef_pos: np.ndarray,
                                nut_pos: np.ndarray,
                                peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Reset gripper position and yaw to neutral pose before moving to next nut."""
        action = np.zeros(self.env.action_dim)

        # Move back toward initial EEF position at a safe height
        if self.init_eef_pos is not None:
            reset_target = self.init_eef_pos.copy()
            reset_target[2] = max(self.safe_z_height, reset_target[2])
        else:
            # Fallback: hover at current XY, safe Z
            print("Warning: init_eef_pos is None, using current position for reset")
            reset_target = eef_pos.copy()
            reset_target[2] = max(self.safe_z_height, eef_pos[2])

        # action[:3] = self.compute_position_action(reset_target, eef_pos)

        # Drive yaw toward neutral (0 rad) and check alignment
        current_quat = self.get_current_eef_quat()
        yaw_ready = True
        
        if current_quat is not None:
            try:
                cur_mat = T.quat2mat(current_quat)
                _, _, cur_yaw = T.mat2euler(cur_mat)
                yaw_err = self._wrap_angle(0.0 - cur_yaw)
                
                # Apply orientation control
                if self.env.action_dim >= 6:
                    yaw_delta = np.clip(self.R_GAIN * yaw_err, -0.6, 0.6)
                    action[3:6] = np.array([0.0, 0.0, yaw_delta])
                
                # Check if aligned
                yaw_ready = abs(yaw_err) < np.radians(5)
            except Exception:
                print("Warning: failed to compute yaw error, proceeding without orientation control")

        action[6] = -1  # Keep gripper open

        self.reset_counter += 1
        next_stage = None

        pos_error = np.linalg.norm(reset_target - eef_pos)
        if (pos_error < 0.01 and yaw_ready) or (self.reset_counter > self.ORIENTATION_RESET_DURATION):
            self.reset_counter = 0
            next_stage = "move_to_nut"
            print(f"Stage: reset_orientation -> {next_stage}")

        return action, next_stage
    
    def stage_skip_nut(self, eef_pos: np.ndarray, 
                      nut_pos: np.ndarray,
                      peg_pos: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Skip current nut and move to next. Don't permanently remove it - may retry later."""
        action = np.zeros(self.env.action_dim)
        
        # Don't remove from queue - it may become graspable after other nuts are moved
        # Just move to next available nut
        print(f"⚠️ Temporarily skipping {self.current_nut}, will retry if still available")
        
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
        # Do NOT reset the environment here when running under the batch
        # data-collection pipeline. Instead, signal that the policy is
        # finished by returning the special 'done' next-stage so the outer
        # collector can end the episode cleanly and call env.reset().
        self.grasp_attempts = 0
        self.nut_gripper_yaw_offset = None
        self.original_target_nut = None
        next_stage = "done"

        return action, next_stage
    
    def _handle_next_nut(self) -> str:
        """
        Handle transition to next nut or episode completion.
        Query environment for available nuts instead of using local queue.
        
        Returns:
            Next stage name
        """
        # Get available nuts from environment (accounts for nuts on pegs and off table)
        available_nuts = self.env.get_available_target_nuts()
        
        if available_nuts:
            # Pick next available nut (prefer nuts we haven't tried recently)
            # If current nut is still available, try a different one first
            if self.current_nut in available_nuts and len(available_nuts) > 1:
                # Try a different nut
                other_nuts = [n for n in available_nuts if n != self.current_nut]
                self.current_nut = other_nuts[0]
            else:
                # Take the first available
                self.current_nut = available_nuts[0]
            
            self.current_peg_id = self.nut_to_peg[self.current_nut]
            self.nut_gripper_yaw_offset = None
            self.table_placement_target = None  # Clear cache for new nut
            print(f"\n--- Moving to next nut: {self.current_nut} -> peg {self.current_peg_id} ---")
            print(f"    Available nuts: {available_nuts}")
            return "move_to_nut"
        else:
            print("\n--- No more available nuts! Episode ending. ---")
            return "done"
    
    def _check_subtask_preconditions(self) -> Tuple[bool, str]:
        """
        Check if subtask can be executed.
        
        Returns:
            (can_execute, failure_reason)
        """
        # Check if object is graspable
        graspable, blocker = self.planner.is_graspable(self.current_nut)
        if not graspable:
            print(f"⚠️ {self.current_nut} is blocked by {blocker}")
            return False, blocker
    
        return True, blocker
    
    def step(self) -> Tuple[np.ndarray, bool]:
        """
        Execute one step of the policy.
        
        Returns:
            Tuple of (action, done flag)
        """
        # Check preconditions: is current nut graspable?
        can_execute, blocker = self._check_subtask_preconditions()
        if not can_execute:
            # Reactively switch to clearing the blocker first
            print(f"⚠️ Switching target: {self.current_nut} blocked by {blocker}")

            if self.data_collection_mode:
                print(f"Data collection mode: Blocked nut detected. Ending episode.")
                return np.zeros(self.env.action_dim), "terminate"  # End episode immediately
            
            # Save the original target we're trying to reach
            if self.original_target_nut is None:
                self.original_target_nut = self.current_nut
                print(f"  Saved original target: {self.original_target_nut}")
            # Switch to the blocker
            self.current_nut = blocker
            self.current_peg_id = self.nut_to_peg.get(self.current_nut, 0)
            self.nut_gripper_yaw_offset = None
            self.table_placement_target = None  # Clear cache when switching
            self.stage = "move_to_nut"
        
        eef_pos, nut_pos, peg_pos = self.get_current_state()

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
                if self.data_collection_mode:
                    print(f"Data collection mode: Stagnation detected. Ending episode.")
                    return np.zeros(self.env.action_dim), "terminate"  # End episode immediately
                
                # Reset episode by transitioning to done stage
                action, next_stage = self.stage_done(eef_pos, nut_pos, peg_pos)
                self.stage = next_stage
                # reset stagnation tracking
                self.last_eef_pos = None
                self.eef_stagnation_count = 0
                # If stage_done signaled overall episode completion, propagate done
                if next_stage == "done":
                    return action, True
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
            "move_to_table": self.stage_move_to_table,
            "lower_to_table": self.stage_lower_to_table,
            "release": self.stage_release,
            "retract": self.stage_retract,
            "reset_orientation": self.stage_reset_orientation,
            "skip_nut": self.stage_skip_nut,
        }
        
        handler = stage_handlers.get(self.stage)
        if handler is None:
            raise ValueError(f"Unknown stage: {self.stage}")
        
        action, next_stage = handler(eef_pos, nut_pos, peg_pos)

        # next_stage == "terminate" means "end episode immediately" (e.g. detachment in
        # data_collection_mode).  Handle it before writing to self.stage so we
        # don't corrupt the stage machine with a boolean value.
        if next_stage == "terminate":
            return action, True

        if next_stage is not None:
            self.stage = next_stage

        if next_stage == "done":
            print("\n✅ Policy reported done, ending episode.")
            return action, True
        
        return action, False


def create_environment(env_name: str = "NutAssembly", 
                      num_round_nuts: int = 2,
                      num_square_nuts: int = 2,
                      initial_stacking_prob: float = 0.0,
                      nut_type_mode: str = "random",
                      has_renderer: bool = True,
                      has_offscreen_renderer: bool = False,
                    #   render_camera: str = "agentview",
                      use_camera_obs: bool = False,
                      horizon: int = 2000):
    """
    Create and configure the robosuite ClutteredNutAssembly environment.
    
    Args:
        env_name: Name of the environment (defaults to "NutAssembly" but creates ClutteredNutAssembly)
        num_round_nuts: Number of round nuts in the scene
        num_square_nuts: Number of square nuts in the scene
        initial_stacking_prob: Probability that nuts start stacked
        nut_type_mode: Which nut type mode to use ("roundnut", "squarenut", "random", or "alternate")
        has_renderer: Enable on-screen rendering
        has_offscreen_renderer: Enable offscreen rendering for cameras
        render_camera: Camera used by on-screen renderer (e.g., agentview)
        use_camera_obs: Enable camera observations
        horizon: Episode horizon
        
    Returns:
        Configured environment instance
    """
    controller_config = load_composite_controller_config(controller="BASIC")
    
    env = make(
        env_name="ClutteredNutAssembly",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        # render_camera=render_camera,
        use_camera_obs=use_camera_obs,
        use_object_obs=True,
        control_freq=20,
        horizon=horizon,
        ignore_done=False,
        num_round_nuts=num_round_nuts,
        num_square_nuts=num_square_nuts,
        nut_type_mode=nut_type_mode,
        initial_stacking_prob=initial_stacking_prob,
    )
    
    return env


def run_heuristic_policy(env_name: str = "NutAssembly", horizon: int = 2000, nut_type_mode: str = "roundnut"):
    """
    Run the heuristic nut assembly policy.
    
    Args:
        env_name: Name of the environment to run
        horizon: Maximum number of steps to run
        nut_type_mode: Which nut type mode to use ("roundnut", "squarenut", "random", or "alternate")
    """
    print(f"Starting heuristic policy for {env_name}...")
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Create environment
    env = create_environment(env_name, horizon=horizon, nut_type_mode=nut_type_mode)

    # Goal renderer — created once; render_goal() is called per episode after
    # HeuristicNutAssemblyPolicy.__init__() resets the env (objects settle).
    # target_nut_type=None so it reads env.current_nut_type each episode.
    goal_renderer = NutAssemblyGoalRenderer(env, camera="agentview", image_size=512)
    
    # Run policy loop
    try:
        for episode in range(10):
            # Create policy (calls env.reset() internally — objects settle here).
            # env.reset() does a hard reset so the MuJoCo model/sim is replaced;
            # flush stale renderers so render_goal() allocates fresh ones.
            policy = HeuristicNutAssemblyPolicy(env)
            goal_renderer.flush_renderers()

            # Render goal image anchored to this episode's settled configuration.
            print("Rendering goal image...")
            goal_rgb = goal_renderer.render_goal()
            if goal_rgb is not None:
                goal_path = Path(f"goal_nut_ep{episode}.png")
                Image.fromarray(goal_rgb).save(str(goal_path))
                print(f"Goal image saved to: {goal_path.resolve()}")
            else:
                print("[Warning] Goal rendering failed; continuing without goal image.")

            print("Starting episode loop...\n")
            step = 0
            while True:
                action, done = policy.step()
                obs, reward, env_done, info = env.step(action)
                policy.obs = obs  # Update observations
                env.render()

                if info.get("success", False):
                    print("--- ENVIRONMENT REPORTED TASK SUCCESS! ---")
                    break

                if done or env_done:
                    print("--- POLICY REPORTED EPISODE COMPLETE, BUT WITH A FAILURE :( ---")
                    break

                step += 1

            print(f"Episode {episode + 1} complete. Total steps: {step + 1}\n")
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
    parser.add_argument(
        '--horizon',
        type=int,
        default=2000,
        help='Maximum number of steps to run'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for environment and policy'
    )
    parser.add_argument(
        '--nut-type-mode',
        type=str,
        default='squarenut',
        choices=['roundnut', 'squarenut', 'random', 'alternate'],
        help='Nut type mode for ClutteredNutAssembly'
    )
    
    args = parser.parse_args()
    run_heuristic_policy(env_name=args.env, horizon=args.horizon, nut_type_mode=args.nut_type_mode)        # Identify nut names from observations