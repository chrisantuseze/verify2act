"""
Dynamics Model Planner for Points2Plans

Wraps the trained Points2Plans relational dynamics model for closed-loop planning.

This is the core planning component that:
1. Takes current state (from StateConverter)
2. Takes goal predicates (from LLMTaskPlanner)
3. Uses rejection sampling to find feasible next primitive action
4. Returns primitive action (e.g., Pick(milk, table), Place(milk, bin))

Planning loop (CLOSED-LOOP):
- LLM generates goals ONCE at episode start
- Dynamics model replans BEFORE EACH PRIMITIVE using rejection sampling
- Execute primitive in robosuite (~200-500 steps)
- Get new observation, repeat
"""

import sys
import os
from pathlib import Path
import copy
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import torch
import copy
from itertools import permutations

# Add Points2Plans to path
points2plans_path = Path(__file__).parent.parent.parent / "Points2Plans"
sys.path.insert(0, str(points2plans_path))

from relational_dynamics.base_RD import RelationalDynamics
from relational_dynamics.config.base_config import BaseConfig
from relational_dynamics.utils import parse_util

# Import collision checker
from collision_checker import CollisionChecker
from predicate_registry import PREDICATE_NAMES
from state_converter import StateConverter


class DynamicsModelPlanner:
    """
    Wrapper for Points2Plans dynamics model for closed-loop primitive planning.
    
    Uses rejection sampling (NOT tree search) to find feasible actions:
    1. Sample K candidate actions around goal locations
    2. Forward simulate each action through dynamics model
    3. Check if predicted state satisfies goal predicates (binary feasibility)
    4. Return first feasible action, or best action if none feasible
    
    This runs BEFORE EACH PRIMITIVE execution (closed-loop).
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 config_args: Optional[Dict] = None,
                 num_samples: int = 50,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 state_converter: StateConverter = None,
                 enable_collision_checking: bool = True,
                 x_collision: float = 0.05,
                 y_collision: float = 0.05,
                 lookahead_depth: int = 1,
                 predicate_threshold: float = 0.3,
                 delta_forward: bool = True,
                 latent_forward: bool = True):
        """
        Initialize dynamics model planner.
        
        Args:
            checkpoint_path: Path to trained model checkpoint 
                           (e.g., Points2Plans/ckpt/checkpoint/cp_1.pth)
            config_args: Optional config overrides (dict)
            num_samples: Number of action samples for rejection sampling (default 50)
            device: Device for model
            state_converter: Optional StateConverter instance for consistent object ID lookup
            enable_collision_checking: Whether to enable collision detection (default True)
            x_collision: Half-width of bounding box in X dimension for collision checking
            y_collision: Half-width of bounding box in Y dimension for collision checking
            lookahead_depth: Number of primitives to simulate ahead (1=greedy, 2-3=multi-step)
            predicate_threshold: Threshold for predicate matching (default 0.3, lowered from 0.5 for undertrained models)
            delta_forward: Whether to use delta forward prediction in dynamics model (default True)
        """
        self.checkpoint_path = checkpoint_path
        self.num_samples = num_samples
        self.device = torch.device(device)
        self.predicate_threshold = predicate_threshold
        self.state_converter = state_converter
        self.enable_collision_checking = enable_collision_checking
        self.lookahead_depth = max(1, min(lookahead_depth, 3))  # Clamp between 1-3
        self.delta_forward = delta_forward
        self.latent_forward = latent_forward
        # Per-channel gain (disabled by default). Use sensitivity sweep to populate indices.
        # Set `per_channel_gain_enabled=True` to enable applying multiplicative gains
        # to specific latent channels (helps amplify predicate-sensitive dims).
        self.per_channel_gain_enabled = False
        # Top channels from the recent sweep (On predicate): [9,77,94,105,72,31,36,83,45,3]
        self.per_channel_gain_indices = [9, 77, 94, 105, 72, 31, 36, 83, 45, 3]
        # Multiplicative gain applied to those channels (start modest, e.g., 5.0)
        self.per_channel_gain_value = 1.0
        # Debug flag: enable detailed diagnostic logging during simulation
        self.debug = True

        self.feasibility_threshold = 0.5  # Threshold for considering an action feasible
        
        # Initialize collision checker
        if self.enable_collision_checking:
            self.collision_checker = CollisionChecker(
                x_collision=x_collision,
                y_collision=y_collision,
                verbose=False  # Set to True for debugging
            )
        
        # Load model
        args = self._create_default_args()
        
        # Override with provided config
        if config_args:
            for key, value in config_args.items():
                setattr(args, key, value)
        
        # Set checkpoint path
        args.checkpoint_path = checkpoint_path
        
        # Create config and load model
        dtype = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
        self.config = BaseConfig(args, dtype=dtype)
        
        self.model = RelationalDynamics(self.config)
        self.model.load_checkpoint(checkpoint_path)
        
        # Move models to device
        self.model.emb_model = self.model.emb_model.to(self.device)
        self.model.classif_model = self.model.classif_model.to(self.device)
        self.model.classif_model_decoder = self.model.classif_model_decoder.to(self.device)
        
        self.model.emb_model.eval()
        self.model.classif_model.eval()
        self.model.classif_model_decoder.eval()

        # Cache embedding table capacity for one-hot IDs used by classif_model.
        # one_hot_encoding_embed is nn.Sequential(nn.Embedding(...)), so the Embedding
        # is at index [0]; num_embeddings is not a property of nn.Sequential itself.
        self.one_hot_num_embeddings = int(self.model.classif_model.one_hot_encoding_embed[0].num_embeddings)

        # Align converter one-hot dimensionality with checkpoint embedding table to
        # avoid out-of-range embedding indices (CUDA device-side assert).
        if self.state_converter is not None and hasattr(self.state_converter, 'max_objects'):
            if self.state_converter.max_objects != self.one_hot_num_embeddings:
                print(
                    f"⚠ One-hot dimension mismatch detected: "
                    f"state_converter.max_objects={self.state_converter.max_objects}, "
                    f"checkpoint embedding size={self.one_hot_num_embeddings}. "
                    f"Auto-aligning converter to checkpoint size."
                )
                self.state_converter.max_objects = self.one_hot_num_embeddings

                if hasattr(self.state_converter, '_build_object_slot_indices'):
                    self.state_converter.object_slot_indices = self.state_converter._build_object_slot_indices()
                    if getattr(self.state_converter, 'training_compatible_one_hot', False):
                        print(
                            f"  Recomputed training-compatible slots: "
                            f"{self.state_converter.object_slot_indices.tolist()}"
                        )

            if hasattr(self.state_converter, 'num_objects'):
                if self.state_converter.num_objects > self.one_hot_num_embeddings:
                    raise ValueError(
                        f"Number of objects ({self.state_converter.num_objects}) exceeds "
                        f"checkpoint embedding capacity ({self.one_hot_num_embeddings}). "
                        f"Reduce tracked objects via object_filter or use a checkpoint trained "
                        f"with larger max_objects."
                    )
        
        print(f"Dynamics Model Planner initialized:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Num samples: {num_samples}")
        print(f"  Device: {device}")
        print(f"  Collision checking: {self.enable_collision_checking}")
        print(f"  Lookahead depth: {self.lookahead_depth} primitive(s)")
        print(f"  delta_forward: {self.delta_forward}")
        print(f"  latent_forward: {self.latent_forward}")
    
    def plan_next_primitive(self,
                           state_dict: Dict,
                           goal_predicates: np.ndarray,
                           primitive_plan: List[str],
                           enable_trajectory_tracking: bool = True) -> Tuple[str, np.ndarray, float, Optional[Dict]]:
        """
        Plan next primitive action using rejection sampling with multi-step lookahead.
        
        Closed-loop planning with lookahead and trajectory tracking:
        1. Take current state observation
        2. Know what goals to achieve (from LLM, remains constant)
        3. Sample K candidate actions for next primitive in plan
        4. Simulate each action through dynamics model (with lookahead_depth primitives)
        5. Track per-step feasibility for failure analysis
        6. Check feasibility: does predicted TERMINAL state match goals?
        7. Return first feasible action, or (best action + failure analysis) if none feasible
        
        Args:
            state_dict: Current state from StateConverter.convert()
                       Contains: batch_voxel_list_single, batch_one_hot_encoding, 
                                batch_6DOF_pose, batch_edge_attr, batch_num_objects
            goal_predicates: Goal predicate tensor [num_objects, num_objects, num_predicates]
                           From LLMTaskPlanner.goals_to_predicates()
            primitive_plan: High-level plan from LLM (e.g., ["Pick(milk, table)", "Place(milk, bin)"])
                          We plan for the NEXT unexecuted primitive
            enable_trajectory_tracking: Whether to track per-step diagnostics (default True)
        
        Returns:
            primitive_action: Next primitive to execute (e.g., "Pick(milk, table)")
            action_params: Low-level action parameters (e.g., target position)
            feasibility: Feasibility score [0, 1]
            failure_analysis: Dict with failure diagnostics (None if feasible action found)
        """
        # Get next primitive from plan
        if not primitive_plan:
            raise ValueError("Primitive plan is empty, nothing to execute")
        
        next_primitive = primitive_plan[0]
        
        # Determine how many primitives to look ahead
        num_primitives_in_plan = len(primitive_plan)
        actual_lookahead = min(self.lookahead_depth, num_primitives_in_plan)
        
        # Parse primitives for lookahead sequence
        lookahead_primitives = []
        for i in range(actual_lookahead):
            prim = primitive_plan[i]
            action_type, obj_name, target_name = self._parse_primitive(prim)
            obj_id = self._get_object_id(obj_name, state_dict)
            target_id = self._get_object_id(target_name, state_dict)
            
            # Handle table case
            if target_id is None and target_name.lower() == "table":
                target_id = obj_id  # Use obj_id as placeholder for table
            
            if obj_id is None:
                print(f"Warning: Object '{obj_name}' not found in primitive {i+1}")
                # Truncate lookahead
                actual_lookahead = i
                break
            
            lookahead_primitives.append((action_type, obj_name, target_name, obj_id, target_id))
        
        if actual_lookahead == 0:
            print(f"Warning: Cannot plan, no valid primitives")
            return next_primitive, np.zeros(3), 0.0, None
        
        # Parse the FIRST primitive (what we'll actually execute).
        # Bug fix: always read action_type / target_name from lookahead_primitives[0]
        # so the loop variable doesn't bleed the last iteration's values when
        # actual_lookahead > 1.
        action_type, _, target_name = lookahead_primitives[0][0], lookahead_primitives[0][1], lookahead_primitives[0][2]
        obj_id, target_id = self.retrieve_ids_from_primitive(next_primitive, state_dict)
        
        # Handle special case: "table" is not a tracked object
        use_table_location = False
        if target_id is None and target_name.lower() == "table":
            use_table_location = True
            table_location = self._poses_to_numpy(state_dict['batch_6DOF_pose'])[obj_id].copy()
            table_location[2] = 0.8 # Assume table height is 0.8m, can be adjusted based on env
        
        # Rejection sampling: sample K candidate actions
        best_action = None
        best_params = None
        best_feasibility = -1.0
        all_trajectories = []  # For failure analysis
        
        lookahead_str = f"{actual_lookahead}-step" if actual_lookahead > 1 else "1-step (greedy)"
        print(f"\n=== Planning for: {next_primitive} ({lookahead_str} lookahead) ===")
        print(f"  Looking ahead: {[p for _, p, _, _, _ in lookahead_primitives[:actual_lookahead]]}")
        print(f"  Object ID: {obj_id}, Target ID: {target_id if not use_table_location else 'table (special)'}")
        print(f"  enable_trajectory_tracking={enable_trajectory_tracking}")
        print(f"  Sampling {self.num_samples} candidate actions...")

        # Get object names for reflection info
        object_names = state_dict.get('object_names', [f'obj_{i}' for i in range(state_dict['batch_num_objects'])])
        
        with torch.no_grad():
            # Encode current state (always on model device)
            node_embedding = self._encode_state(state_dict).to(self.device)

            for sample_idx in range(self.num_samples):
                # ------ Build action sequence for this sample ------
                # First primitive: sample or use table location
                if use_table_location:
                    action_params = self._sample_action_for_table(state_dict, obj_id, action_type)
                else:
                    action_params = self._sample_action(state_dict, obj_id, target_id, action_type)

                first_target = target_id if target_id is not None else obj_id
                action_sequence = [(action_type, action_params, obj_id, first_target)]
                primitive_names = [next_primitive]

                # Subsequent primitives for multi-step lookahead
                for i in range(1, actual_lookahead):
                    fut_action_type, _, _, fut_obj_id, fut_target_id = lookahead_primitives[i]
                    fut_target = fut_target_id if fut_target_id is not None else fut_obj_id
                    fut_params = self._sample_action(state_dict, fut_obj_id, fut_target, fut_action_type)
                    action_sequence.append((fut_action_type, fut_params, fut_obj_id, fut_target))
                    primitive_names.append(primitive_plan[i])

                # ------ Simulate and optionally track per-step diagnostics ------
                num_objects = state_dict['batch_num_objects']
                if enable_trajectory_tracking:
                    _, predicted_state, trajectory = self._forward_simulate_sequence_with_tracking(
                        node_embedding, state_dict, action_sequence, primitive_names,
                        goal_predicates, num_objects,
                    )
                    all_trajectories.append(trajectory)
                else:
                    _, predicted_state = self._forward_simulate_sequence(
                        node_embedding, state_dict, action_sequence, num_objects,
                    )

                # ------ Score terminal state ------
                # TODO: replace _check_feasibility with critic model when available
                feasibility = self._check_feasibility(
                    predicted_state, goal_predicates, num_objects,
                    obj_id=obj_id,
                    target_id=target_id if not use_table_location else None,
                )

                if feasibility > best_feasibility:
                    best_feasibility = feasibility
                    best_action = next_primitive
                    best_params = action_params

                if feasibility >= self.feasibility_threshold:
                    print(f"  ✓ Found feasible {actual_lookahead}-step sequence at sample {sample_idx + 1}")
                    return best_action, best_params, best_feasibility, None

        # No feasible action found – analyse failures for targeted reflection
        print(f"  ⚠ No feasible sequence found. Best feasibility: {best_feasibility:.3f}")

        failure_analysis = None
        if enable_trajectory_tracking and all_trajectories:
            print(f"  📊 Analyzing {len(all_trajectories)} trajectories for failure patterns...")
            failure_analysis = self._analyze_trajectories(all_trajectories, primitive_plan[:actual_lookahead])
            reflection_info = self._generate_reflection_info(
                failure_analysis, primitive_plan, goal_predicates, object_names
            )
            failure_analysis['reflection_info'] = reflection_info

            if failure_analysis['most_common_failure_step'] is not None:
                print(f"  📍 Most common failure at step {failure_analysis['most_common_failure_step'] + 1}: "
                      f"{failure_analysis['most_common_failure_primitive']}")
                print(f"  📍 Failure rate: {reflection_info['failure_rate']:.1%}")
                if reflection_info['top_failure_reasons']:
                    print(f"  📍 Top failure reasons: {reflection_info['top_failure_reasons'][:3]}")

        return best_action, best_params, best_feasibility, failure_analysis

    def retrieve_ids_from_primitive(self, primitive: str, state_dict: Dict) -> Tuple[Optional[int], Optional[int]]:
        """
        Helper function to retrieve object and target IDs from a primitive string.
        
        Args:
            primitive: Primitive string (e.g., "Pick(milk, table)")
            state_dict: State dictionary for object ID lookup
        Returns:
            obj_id: ID of the object to manipulate (or None if not found)
            target_id: ID of the target object/location (or None if not found)
        """

        # Parse the FIRST primitive (what we'll actually execute)
        action_type, obj_name, target_name = self._parse_primitive(primitive)
        obj_id = self._get_object_id(obj_name, state_dict)
        target_id = self._get_object_id(target_name, state_dict)
        return obj_id, target_id
    
    def _encode_state(self, state_dict: Dict, debug: bool = False) -> torch.Tensor:
        """Encode current state observation through PointConv + one-hot embedding.

        Matches training code in base_RD.py:
        - ``voxel_data``: ``[batch, N_obj, N_pts, 3]`` → reshaped for PointConv as ``[B*N_obj, 3, N_pts]``
        - ``img_emb``: reshaped back to ``[batch, N_obj, emb_dim]``
        - one-hot IDs via argmax on dim=2, embedded, then concatenated with img_emb on dim=-1
        """
        voxel_data = state_dict['batch_voxel_list_single']
        if not isinstance(voxel_data, torch.Tensor):
            voxel_data = torch.FloatTensor(voxel_data)

        # Normalise to [batch, N_obj, N_pts, 3]
        if voxel_data.dim() == 3:
            voxel_data = voxel_data.unsqueeze(0)
        elif voxel_data.dim() == 5:          # [batch, timestep, N_obj, N_pts, 3]
            voxel_data = voxel_data[:, 0]
        voxel_data = voxel_data.to(self.device)

        batch_size, num_objects = voxel_data.shape[0], voxel_data.shape[1]

        # PointConv expects [B, C, N] = [B*N_obj, 3, N_pts]
        x = voxel_data.reshape(batch_size * num_objects, voxel_data.shape[2], voxel_data.shape[3])
        x = x.permute(0, 2, 1)
        img_emb = self.model.emb_model(x)                   # [B*N_obj, emb_dim]
        img_emb = img_emb.reshape(batch_size, num_objects, img_emb.shape[-1])

        # One-hot embedding
        one_hot = state_dict['batch_one_hot_encoding']
        if not isinstance(one_hot, torch.Tensor):
            one_hot = torch.FloatTensor(one_hot)
        if one_hot.dim() == 2:
            one_hot = one_hot.unsqueeze(0)

        if one_hot.shape[-1] != self.one_hot_num_embeddings:
            raise ValueError(
                f"Invalid one-hot width: got {one_hot.shape[-1]}, expected "
                f"{self.one_hot_num_embeddings} from checkpoint embedding table."
            )

        object_indices = torch.argmax(one_hot, dim=2).long()
        lo, hi = int(object_indices.min()), int(object_indices.max())
        if lo < 0 or hi >= self.one_hot_num_embeddings:
            raise ValueError(
                f"Invalid object index range for embedding lookup: min={lo}, max={hi}, "
                f"embedding size={self.one_hot_num_embeddings}."
            )
        object_indices = object_indices.to(self.device)

        latent_one_hot = self.model.classif_model.one_hot_encoding_embed(object_indices)
        node_embedding = torch.cat([img_emb, latent_one_hot], dim=-1)

        if debug:
            print(f"[_encode_state] voxel_data={tuple(voxel_data.shape)}")
            print(f"[_encode_state] img_emb={tuple(img_emb.shape)}, "
                  f"min={img_emb.min():.4f} max={img_emb.max():.4f} mean={img_emb.mean():.4f}")
            print(f"[_encode_state] one_hot={tuple(one_hot.shape)}, indices={object_indices.tolist()}")
            print(f"[_encode_state] latent_one_hot={tuple(latent_one_hot.shape)}, "
                  f"min={latent_one_hot.min():.4f} max={latent_one_hot.max():.4f}")
            print(f"[_encode_state] node_embedding={tuple(node_embedding.shape)}, "
                  f"min={node_embedding.min():.4f} max={node_embedding.max():.4f}")

        return node_embedding
    
    def _sample_action(self,
                      state_dict: Dict,
                      obj_id: int,
                      target_id: int,
                      action_type: str) -> np.ndarray:
        """
        Sample action parameters around target location.
        
        For Pick/Place: Sample relative position offset from object to target.
        """
        poses = self._poses_to_numpy(state_dict['batch_6DOF_pose'])

        # Get current positions
        obj_pos = poses[obj_id][:3]
        target_pos = poses[target_id][:3]
        
        # Sample random offset around target
        x_range = 0.05  # 5cm range
        y_range = 0.05
        
        delta_x = np.random.uniform(-x_range/2, x_range/2)
        delta_y = np.random.uniform(-y_range/2, y_range/2)
        
        # Relative action: (dx, dy, dz) from object to target
        action_params = np.array([
            target_pos[0] - obj_pos[0] + delta_x,
            target_pos[1] - obj_pos[1] + delta_y,
            target_pos[2]  # Use target height
        ])
        
        return action_params
    
    def _sample_action_for_table(self,
                                 state_dict: Dict,
                                 obj_id: int,
                                 action_type: str) -> np.ndarray:
        """
        Sample action parameters for table-related actions (e.g., Pick from table).
        
        For Pick(obj, table): Sample small movement (lift slightly above current position)
        """
        poses = self._poses_to_numpy(state_dict['batch_6DOF_pose'])

        # Get current position
        obj_pos = poses[obj_id][:3]
        
        # For Pick from table: small upward movement with small xy variation
        if action_type.lower() == "pick":
            delta_x = np.random.uniform(-0.02, 0.02)  # 2cm variation
            delta_y = np.random.uniform(-0.02, 0.02)
            delta_z = np.random.uniform(0.05, 0.15)  # Lift 5-15cm
            
            action_params = np.array([delta_x, delta_y, delta_z])
        else:
            # For other actions, use small random motion
            action_params = np.random.uniform(-0.05, 0.05, size=3)
        
        return action_params
    
    def _simulate_one_step(self,
                          current_latent: torch.Tensor,
                          previous_pc: torch.Tensor,
                          action_params: np.ndarray,
                          obj_id: int,
                          target_id: int,
                          state_dict: Dict,
                          action_scale: float = 1.0,
                          delta_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate one step of dynamics model forward pass.
        
        Common logic extracted from _forward_simulate* functions.
        
        Args:
            current_latent: Current node embeddings [1, num_objects, embed_dim]
            previous_pc: Current point clouds [1, num_objects, N_points, 3]
            action_params: Action parameters [dx, dy, dz]
            obj_id: Object to move
            target_id: Target object/location
            state_dict: State dictionary
            action_scale: Scaling factor for action embeddings (default 1.0)
            delta_scale: Scaling factor for delta latent (default 1.0)
        
        Returns:
            next_latent: Updated node embeddings
            updated_pc: Updated point clouds
            delta_latent: Raw delta latent for pose decoding
        """
        num_objects = state_dict['batch_num_objects']
        
        # Build discrete action: which object to move
        discrete_action = self.model.classif_model.one_hot_encoding_embed(
            torch.LongTensor([obj_id]).to(self.device)
        )
        
        # Build continuous action: x,y movement deltas
        continuous_action = self.model.classif_model.continuous_action_emb(
            torch.FloatTensor(action_params[:2]).unsqueeze(0).to(self.device)
        )
        
        # Combine discrete + continuous
        current_action_continuous = torch.cat([discrete_action, continuous_action], dim=-1)
        current_action_continuous = current_action_continuous.view(1, 1, -1)
        
        # Build place action: where to place it
        if target_id == obj_id or target_id is None:
            discrete_place_id_tensor = torch.zeros_like(discrete_action)
        else:
            discrete_place_id_tensor = self.model.classif_model.one_hot_encoding_embed(
                torch.LongTensor([target_id]).to(self.device)
            )
        
        current_action = torch.cat([discrete_place_id_tensor, continuous_action], dim=-1)
        current_action = current_action.view(1, 1, -1)
        
        # Apply action scaling
        current_action_continuous_scaled = current_action_continuous * action_scale
        current_action_scaled = current_action * action_scale
        
        # Concatenate: [node embeddings, action embeddings]
        graph_node_action = torch.cat([current_latent, current_action_continuous_scaled, current_action_scaled], dim=1)
        
        # Build padding mask (None when no masking metadata is available)
        src_kp_mask = self._build_src_mask(state_dict)
        
        # Forward through dynamics model
        if src_kp_mask is not None:
            next_latent_raw = self.model.classif_model.graph_dynamics_0(graph_node_action, src_key_padding_mask=src_kp_mask)
        else:
            next_latent_raw = self.model.classif_model.graph_dynamics_0(graph_node_action)
        
        next_latent_raw = next_latent_raw[:, :-2, :]
        
        # Apply delta scaling
        delta_latent = next_latent_raw * delta_scale
        
        if self.debug and delta_scale != 1.0:
            dl = delta_latent.detach().cpu()
            # print(f"[DEBUG delta_scale] applied scale={delta_scale} delta_latent shape={tuple(dl.shape)}, mean={float(dl.mean()):.6f}")
        
        # Apply per-channel gains
        if self.per_channel_gain_enabled and len(self.per_channel_gain_indices) > 0:
            idx = torch.LongTensor(self.per_channel_gain_indices).to(delta_latent.device)
            delta_latent[:, :, idx] = delta_latent[:, :, idx] * float(self.per_channel_gain_value)
    
        # Update latent
        if self.delta_forward:
            next_latent = delta_latent + current_latent
        else:
            next_latent = delta_latent
        
        # Update point clouds with predicted pose changes
        with torch.no_grad():
            if self.delta_forward:
                delta_change_torch = self.model.classif_model_decoder.pose_estimation(delta_latent.to(self.device))
            else:
                delta_change_torch = self.model.classif_model_decoder.pose_estimation(next_latent.to(self.device))
        # delta_change_torch: [batch, num_objects, pose_num=2] -> (dx, dy)

        # Compute z_height_change from 6DOF poses (matches original base_RD.py planner logic).
        # For pick-place: stacking height = z_target + z_obj; change = stacking_height - current_z.
        # Only applied to obj_id (the moved object); all others get zero z-change.
        z_height_change = torch.zeros(delta_change_torch.shape[0], delta_change_torch.shape[1]).to(self.device)
        if target_id is not None and target_id != obj_id and 'batch_6DOF_pose' in state_dict:
            poses_np = self._poses_to_numpy(state_dict['batch_6DOF_pose'])
            z_obj = float(poses_np[obj_id, 2])
            z_target = float(poses_np[target_id, 2])
            place_height = z_target + z_obj  # stacking height (matches original)
            z_height_change[:, obj_id] = place_height - z_obj  # = z_target
        z_height_change = z_height_change.unsqueeze(-1)  # [batch, num_objects, 1]

        # Concatenate in Open3D [x, y, z] order so channels align with previous_pc layout.
        # delta_change_torch = (dx, dy), z_height_change = (dz) -> [dx, dy, dz]
        # Original code used (z, dx, dy) order matching its transposed [3, N] PC storage;
        # here previous_pc is [batch, num_objects, N_points, 3] (xyz), so order must be [dx, dy, dz].
        all_change = torch.cat((delta_change_torch, z_height_change), dim=2)  # [batch, N_obj, 3]
        n_points = previous_pc.shape[2]
        all_change_flatten = all_change.unsqueeze(-1).repeat(1, 1, 1, n_points)  # [batch, N_obj, 3, N_pts]
        all_change_flatten = all_change_flatten.permute(0, 1, 3, 2)  # [batch, N_obj, N_pts, 3]
        updated_pc = previous_pc + all_change_flatten
        
        # Re-encode if latent_forward is False
        if not self.latent_forward:
            tmp_state = dict(state_dict)
            tmp_state['batch_voxel_list_single'] = updated_pc
            next_latent = self._encode_state(tmp_state)
        
        return next_latent, updated_pc, delta_latent

    def _forward_simulate_sequence(self,
                                   initial_node_embedding: torch.Tensor,
                                   state_dict: Dict,
                                   primitive_sequence: List[Tuple[str, np.ndarray, int, int]],
                                   num_objects: int) -> Tuple[torch.Tensor, Dict]:
        """
        Forward simulate a sequence of primitives through the dynamics model.
        
        This implements multi-step lookahead by rolling out 2-3 primitives
        and returning the terminal state for feasibility checking.
        
        Args:
            initial_node_embedding: Initial state embedding [1, N_objects, embed_dim]
            state_dict: Current state dictionary
            primitive_sequence: List of (action_type, action_params, obj_id, target_id)
            num_objects: Number of objects in scene
        
        Returns:
            final_latent: Predicted latent state after sequence [1, N_objects, embed_dim]
            final_decoder_output: Predicted predicates and poses at terminal state
        """
        current_latent = initial_node_embedding
        # Store initial decoder preds (per-edge) for later comparison
        init_pred_sigmoid = None

        # Prepare a mutable copy of point-clouds for re-embedding when needed
        previous_pc = state_dict['batch_voxel_list_single']
        if not isinstance(previous_pc, torch.Tensor):
            previous_pc = torch.Tensor(previous_pc)
        # Ensure batch dimension
        if previous_pc.dim() == 3:
            previous_pc = previous_pc.unsqueeze(0)
        previous_pc = previous_pc.to(self.device).clone()

        previous_pc_center_numpy = self._poses_to_numpy(state_dict['batch_6DOF_pose'])
        
        # Roll out each primitive in sequence.
        # Bug fix: maintain a simulated_state_dict whose 'batch_6DOF_pose' is updated
        # after each step so that z_height_change is computed from simulated positions
        # rather than the original observed positions for steps > 0.
        simulated_state_dict = dict(state_dict)
        simulated_poses = state_dict['batch_6DOF_pose']
        if not isinstance(simulated_poses, torch.Tensor):
            simulated_poses = torch.FloatTensor(simulated_poses)
        simulated_state_dict['batch_6DOF_pose'] = simulated_poses.clone()

        delta_latent_last = None
        for step_idx, (action_type, action_params, obj_id, target_id) in enumerate(primitive_sequence):
            current_latent, previous_pc, delta_latent_last = self._simulate_one_step(
                current_latent, previous_pc, action_params, obj_id, target_id, simulated_state_dict
            )
            # Update simulated poses (xyz only) from the new point-cloud centroids
            # so subsequent steps use post-simulation positions for z_height_change.
            with torch.no_grad():
                new_xyz = previous_pc.mean(dim=2).cpu()  # [1, num_objects, 3]
            simulated_state_dict['batch_6DOF_pose'] = simulated_state_dict['batch_6DOF_pose'].clone()
            simulated_state_dict['batch_6DOF_pose'][0, :, :3] = new_xyz[0]
        
        # Decode final state
        edge_index = self._build_edge_index(num_objects)
        final_decoder_output = self.model.classif_model_decoder(current_latent, edge_index)
        if self.delta_forward and delta_latent_last is not None:
            delta_decoder = self.model.classif_model_decoder(delta_latent_last, edge_index)
            final_decoder_output['predicted_pose'] = delta_decoder.get('predicted_pose')

        return current_latent, final_decoder_output

    def _forward_simulate_sequence_with_tracking(
        self,
        initial_node_embedding: torch.Tensor,
        state_dict: Dict,
        primitive_sequence: List[Tuple[str, np.ndarray, int, int]],
        primitive_names: List[str],
        goal_predicates: np.ndarray,
        num_objects: int
    ) -> Tuple[torch.Tensor, Dict, List[Dict]]:
        """
        Forward simulate a sequence with per-step tracking for failure analysis.
        
        This is the trajectory-tracking version that enables targeted reflection
        by recording diagnostics at each step.
        
        Args:
            initial_node_embedding: Initial state embedding [1, N_objects, embed_dim]
            state_dict: Current state dictionary
            primitive_sequence: List of (action_type, action_params, obj_id, target_id)
            primitive_names: List of primitive strings (e.g., ["Pick(cubeA, table)", "Place(cubeA, cubeB)"])
            goal_predicates: Goal predicate tensor for feasibility checking
            num_objects: Number of objects in scene
        
        Returns:
            final_latent: Predicted latent state after sequence
            final_decoder_output: Predicted predicates and poses at terminal state
            trajectory: List of per-step diagnostic dicts
        """
        current_latent = initial_node_embedding
        trajectory = []

        # initial predicate buffer for per-edge diffs
        init_pred_sigmoid = None

        # Prepare mutable point-cloud copy for re-embedding when needed
        previous_pc = state_dict['batch_voxel_list_single']
        if not isinstance(previous_pc, torch.Tensor):
            previous_pc = torch.Tensor(previous_pc)
        if previous_pc.dim() == 3:
            previous_pc = previous_pc.unsqueeze(0)
        previous_pc = previous_pc.to(self.device).clone()

        previous_pc_center_numpy = self._poses_to_numpy(state_dict['batch_6DOF_pose'])
        
        # Build edge index once (reused for decoding at every step)
        edge_index = self._build_edge_index(num_objects)
        
        # Bug fix: maintain a simulated_state_dict whose 'batch_6DOF_pose' is updated
        # after each step so z_height_change uses post-simulation positions for steps > 0.
        simulated_state_dict = dict(state_dict)
        simulated_poses = state_dict['batch_6DOF_pose']
        if not isinstance(simulated_poses, torch.Tensor):
            simulated_poses = torch.FloatTensor(simulated_poses)
        simulated_state_dict['batch_6DOF_pose'] = simulated_poses.clone()

        delta_latent_last = None
        for step_idx, (action_type, action_params, obj_id, target_id) in enumerate(primitive_sequence):
            current_latent, previous_pc, delta_latent_last = self._simulate_one_step(
                current_latent, previous_pc, action_params, obj_id, target_id, simulated_state_dict
            )
            # Update simulated poses (xyz) from new point-cloud centroids.
            with torch.no_grad():
                new_xyz = previous_pc.mean(dim=2).cpu()  # [1, num_objects, 3]
            simulated_state_dict['batch_6DOF_pose'] = simulated_state_dict['batch_6DOF_pose'].clone()
            simulated_state_dict['batch_6DOF_pose'][0, :, :3] = new_xyz[0]
            
            # Decode intermediate state for tracking
            step_decoder_output = self.model.classif_model_decoder(current_latent, edge_index)
            
            # Check feasibility at this step (using existing _check_feasibility as stand-in critic)
            # @Chris: TODO
            step_feasibility = self._check_feasibility(
                step_decoder_output,
                goal_predicates,
                num_objects,
                obj_id=obj_id,
                target_id=target_id
            )
            
            # Detect failure reasons
            failure_reasons = self._detect_failure_reasons(
                step_decoder_output,
                goal_predicates,
                num_objects,
                obj_id,
                target_id
            )
            
            # Record step info
            step_info = {
                'step': step_idx,
                'primitive': primitive_names[step_idx] if step_idx < len(primitive_names) else f"step_{step_idx}",
                'action_type': action_type,
                'obj_id': obj_id,
                'target_id': target_id,
                'step_score': step_feasibility,
                'failure_reasons': failure_reasons,
                'passed': step_feasibility >= self.feasibility_threshold,
            }
            trajectory.append(step_info)
        
        final_decoder_output = self.model.classif_model_decoder(current_latent, edge_index)
        if self.delta_forward and delta_latent_last is not None:
            delta_decoder = self.model.classif_model_decoder(delta_latent_last, edge_index)
            final_decoder_output['predicted_pose'] = delta_decoder.get('predicted_pose')

        return current_latent, final_decoder_output, trajectory
    
    def _detect_failure_reasons(
        self,
        predicted_state: Dict,
        goal_predicates: np.ndarray,
        num_objects: int,
        obj_id: int,
        target_id: int
    ) -> List[str]:
        """
        Detect specific failure reasons from predicted state.
        
        This provides diagnostic information for targeted reflection.
        """
        failure_reasons = []
        
        pred_edges = predicted_state['pred_sigmoid'].detach().cpu().numpy()
        pred_relations_matrix = self._edges_to_matrix(pred_edges, num_objects)
        
        # Check goal predicate mismatches
        goal_mask = goal_predicates > self.predicate_threshold
        pred_mask = pred_relations_matrix[:, :, :goal_predicates.shape[-1]] > self.predicate_threshold
        
        # Find specific mismatches
        predicate_names = PREDICATE_NAMES
        
        for i in range(num_objects):
            for j in range(num_objects):
                if i == j:
                    continue
                for k in range(min(len(predicate_names), goal_predicates.shape[-1])):
                    if goal_mask[i, j, k] and not pred_mask[i, j, k]:
                        pred_name = predicate_names[k] if k < len(predicate_names) else f"pred_{k}"
                        failure_reasons.append(f"missing_{pred_name}({i},{j})")
        
        # Check collision (if enabled)
        if self.enable_collision_checking and 'predicted_pose' in predicted_state:
            predicted_poses = predicted_state['predicted_pose'].detach().cpu().numpy()
            is_feasible, reason = self.collision_checker.check_predicted_state_collisions(
                predicted_point_clouds=None,
                predicted_poses=predicted_poses,
                target_object_id=target_id,
                placement_height=None
            )
            if not is_feasible:
                failure_reasons.append(f"collision:{reason}")
        
        return failure_reasons
    
    def _analyze_trajectories(
        self,
        all_trajectories: List[List[Dict]],
        primitive_plan: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze failure patterns across all sampled trajectories.
        
        This aggregates trajectory data to identify systematic issues
        for targeted LLM reflection.
        
        Args:
            all_trajectories: List of trajectories, each trajectory is a list of step dicts
            primitive_plan: The original primitive plan
        
        Returns:
            Failure analysis dict with:
            - most_common_failure_step: Which step fails most often
            - failure_step_counts: How many times each step failed
            - failure_reasons: Aggregated failure reasons
            - best_trajectory: Trajectory with highest average score
        """
        failure_analysis = {
            'most_common_failure_step': None,
            'most_common_failure_primitive': None,
            'failure_step_counts': defaultdict(int),
            'failure_reasons': defaultdict(int),
            'problematic_predicates': [],
            'best_trajectory': None,
            'best_terminal_score': -1.0,
            'best_avg_score': -1.0,
            'total_samples': len(all_trajectories),
            'samples_with_failures': 0,
        }
        
        for traj in all_trajectories:
            if not traj:
                continue
                
            # Find first failing step
            found_failure = False
            for step_info in traj:
                if not step_info['passed']:
                    failure_analysis['failure_step_counts'][step_info['step']] += 1
                    for reason in step_info['failure_reasons']:
                        failure_analysis['failure_reasons'][reason] += 1
                        # Extract predicate failures
                        if reason.startswith('missing_'):
                            failure_analysis['problematic_predicates'].append(reason)
                    found_failure = True
                    break
            
            if found_failure:
                failure_analysis['samples_with_failures'] += 1
            
            # Track best trajectory (by average score)
            avg_score = sum(s['step_score'] for s in traj) / len(traj) if traj else 0
            terminal_score = traj[-1]['step_score'] if traj else 0
            
            if terminal_score > failure_analysis['best_terminal_score']:
                failure_analysis['best_terminal_score'] = terminal_score
                failure_analysis['best_trajectory'] = traj
                failure_analysis['best_avg_score'] = avg_score
        
        # Identify most problematic step
        if failure_analysis['failure_step_counts']:
            most_common_step = max(
                failure_analysis['failure_step_counts'],
                key=failure_analysis['failure_step_counts'].get
            )
            failure_analysis['most_common_failure_step'] = most_common_step
            if most_common_step < len(primitive_plan):
                failure_analysis['most_common_failure_primitive'] = primitive_plan[most_common_step]
        
        # Convert defaultdicts to regular dicts for cleaner output
        failure_analysis['failure_step_counts'] = dict(failure_analysis['failure_step_counts'])
        failure_analysis['failure_reasons'] = dict(failure_analysis['failure_reasons'])
        
        return failure_analysis
    
    def _generate_reflection_info(
        self,
        failure_analysis: Dict[str, Any],
        primitive_plan: List[str],
        goal_predicates: np.ndarray,
        object_names: List[str]
    ) -> Dict[str, Any]:
        """
        Generate structured information for LLM reflection.
        
        This prepares all the data needed for the LLM to understand
        what went wrong and suggest corrections.
        
        Args:
            failure_analysis: Output from _analyze_trajectories
            primitive_plan: The original plan
            goal_predicates: Goal predicates
            object_names: List of object names in scene
        
        Returns:
            Reflection info dict ready for LLM prompt construction
        """
        failed_step = failure_analysis['most_common_failure_step']
        failed_primitive = failure_analysis['most_common_failure_primitive']
        
        # Get top failure reasons
        sorted_reasons = sorted(
            failure_analysis['failure_reasons'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Get unique problematic predicates
        unique_problematic = list(set(failure_analysis['problematic_predicates']))[:5]
        
        # Compute failure rate
        failure_rate = (
            failure_analysis['samples_with_failures'] / failure_analysis['total_samples']
            if failure_analysis['total_samples'] > 0 else 1.0
        )
        
        reflection_info = {
            'original_plan': primitive_plan,
            'failed_step_index': failed_step,
            'failed_step_number': failed_step + 1 if failed_step is not None else None,
            'failed_primitive': failed_primitive,
            'failure_rate': failure_rate,
            'failure_count_at_step': (
                failure_analysis['failure_step_counts'].get(failed_step, 0)
                if failed_step is not None else 0
            ),
            'total_samples': failure_analysis['total_samples'],
            'top_failure_reasons': sorted_reasons,
            'problematic_predicates': unique_problematic,
            'best_score_achieved': failure_analysis['best_terminal_score'],
            'steps_before_failure': failed_step if failed_step is not None else len(primitive_plan),
            'object_names': object_names,
            'suggestions': self._generate_suggestions(failure_analysis, primitive_plan),
        }
        
        return reflection_info
    
    def _generate_suggestions(
        self,
        failure_analysis: Dict[str, Any],
        primitive_plan: List[str]
    ) -> List[str]:
        """
        Generate actionable suggestions based on failure analysis.
        """
        suggestions = []
        failed_step = failure_analysis['most_common_failure_step']
        
        if failed_step is not None:
            # Check for collision failures
            collision_reasons = [r for r in failure_analysis['failure_reasons'] if 'collision' in r]
            if collision_reasons:
                suggestions.append("Consider clearing the target location before placement")
                suggestions.append("Check if objects are blocking the path")
            
            # Check for predicate failures
            predicate_failures = [r for r in failure_analysis['failure_reasons'] if r.startswith('missing_')]
            if predicate_failures:
                suggestions.append("The target predicate may not be achievable with current action")
                if 'missing_On' in str(predicate_failures) or 'missing_Above' in str(predicate_failures):
                    suggestions.append("Check if prerequisite placement steps are needed")
            
            # General suggestions based on step
            if failed_step == 0:
                suggestions.append("The first step is failing - check if the initial state allows this action")
            elif failed_step > 0:
                suggestions.append(f"Steps 1-{failed_step} may be succeeding but step {failed_step + 1} fails")
                suggestions.append("Consider reordering actions or adding intermediate steps")
        
        return suggestions

    # ------------------------------------------------------------------
    # Low-level helpers shared across simulation / decoding methods
    # ------------------------------------------------------------------

    def _build_edge_index(self, num_objects: int) -> torch.Tensor:
        """Return a fully-connected directed edge index for *num_objects* nodes.

        Produces every ordered pair (i, j) with i != j, packed into a
        ``[2, num_edges]`` LongTensor on self.device.
        """
        edge_list = list(permutations(range(num_objects), 2))
        return torch.LongTensor(np.array(edge_list).T).to(self.device)

    def _edges_to_matrix(self, pred_edges: np.ndarray, num_objects: int) -> np.ndarray:
        """Reshape flat edge predictions to a dense [N, N, P] relation matrix.

        Args:
            pred_edges: ``[batch, num_edges, num_predicates]`` array from decoder.
            num_objects: Number of scene objects (N).
        Returns:
            Dense array of shape ``[num_objects, num_objects, num_predicates]``.
        """
        num_predicates = pred_edges.shape[-1]
        matrix = np.zeros((num_objects, num_objects, num_predicates), dtype=pred_edges.dtype)
        edge_idx = 0
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    matrix[i, j] = pred_edges[0, edge_idx]
                    edge_idx += 1
        return matrix

    def _poses_to_numpy(self, poses) -> np.ndarray:
        """Return poses as a ``[num_objects, 6]`` float32 numpy array.

        Handles torch Tensors, batched ``[1, N, 6]`` arrays, and plain numpy.
        """
        if isinstance(poses, torch.Tensor):
            poses = poses.detach().cpu().numpy()
        else:
            poses = np.asarray(poses)
        if poses.ndim == 3:            # [1, num_objects, 6]
            poses = poses[0]
        return poses.astype(np.float32)

    def _build_src_mask(self, state_dict: Dict, extra_tokens: int = 2) -> Optional[torch.Tensor]:
        """Build ``src_key_padding_mask`` for the dynamics transformer.

        Returns a 1-D BoolTensor (length = num_objects + extra_tokens) where
        ``True`` marks padded / inactive positions, or *None* when no masking
        information is available.

        Args:
            state_dict: State dict that may contain ``batch_env_identity``.
            extra_tokens: Number of action tokens appended to the node sequence
                (default 2 for the two action embeddings used by graph_dynamics_0).
        """
        env_id = state_dict.get('batch_env_identity')
        if env_id is None:
            return None
        try:
            if not torch.is_tensor(env_id):
                env_id = torch.as_tensor(env_id)
            if env_id.dim() < 2:
                return None
            # env_id: [batch, num_objects, ...]; padding flag is channel 0 == -1
            node_pad = (env_id[:, :, 0] == -1).to(self.device)  # [batch, num_objects]
            # Extend with False for the action tokens (they are never padding)
            action_pad = torch.zeros(
                node_pad.shape[0], extra_tokens, dtype=torch.bool, device=self.device
            )
            return torch.cat([node_pad, action_pad], dim=1)[0]   # collapse batch dim
        except Exception:
            return None

    def _forward_simulate(self,
                         node_embedding: torch.Tensor,
                         state_dict: Dict,
                         action_params: np.ndarray,
                         obj_id: int,
                         target_id: int) -> Dict:
        """
        Forward simulate action through dynamics model.
        
        Returns predicted next state (relations, poses, etc.).
        """
        num_objects = state_dict['batch_num_objects']
        
        # Prepare point clouds
        previous_pc = state_dict['batch_voxel_list_single']
        if not isinstance(previous_pc, torch.Tensor):
            previous_pc = torch.Tensor(previous_pc)
        if previous_pc.dim() == 3:
            previous_pc = previous_pc.unsqueeze(0)
        previous_pc = previous_pc.to(self.device).clone()
        
        # Simulate one step - use same default scale (1.0) as _forward_simulate_sequence
        # for consistency. The previous hardcoded 3x was an undocumented override that
        # caused different behaviour depending on whether trajectory tracking was enabled.
        next_latent, updated_pc, delta_latent = self._simulate_one_step(
            node_embedding, previous_pc, action_params, obj_id, target_id, state_dict
        )
        
        # Decode predicted state
        edge_index = self._build_edge_index(num_objects)
        decoder_output = self.model.classif_model_decoder(next_latent, edge_index)
        if self.delta_forward:
            # Decode pose from delta latent to match training behavior
            delta_decoder = self.model.classif_model_decoder(delta_latent, edge_index)
            decoder_output['predicted_pose'] = delta_decoder.get('predicted_pose')
        
        return decoder_output
    
    def _check_feasibility(self,
                          predicted_state: Dict,
                          goal_predicates: np.ndarray,
                          num_objects: int,
                          obj_id: int = None,
                          target_id: int = None) -> float:
        """
        Check if predicted state matches goal predicates and has no collisions.
        
        Returns feasibility score [0, 1].
        """
        # 1. Check goal matching
        pred_edges = predicted_state['pred_sigmoid'].detach().cpu().numpy()
        pred_relations_matrix = self._edges_to_matrix(pred_edges, num_objects)
        
        # print(f"    Predicted relations matrix: {pred_relations_matrix}, Goal predicates: {goal_predicates}")
        # print(f"    Goal predicates in _check_feasibility: {goal_predicates > self.predicate_threshold}")

        # Compare with goals: check if predicted relations match goal relations
        # For each goal predicate that should be 1, check if prediction > threshold
        goal_mask = goal_predicates > self.predicate_threshold
        pred_mask = pred_relations_matrix[:, :, :goal_predicates.shape[-1]] > self.predicate_threshold
        matches = np.logical_and(
            goal_mask,
            pred_mask
        )
        # print(f"    Matches: {matches}, Goal mask: {goal_mask}, Pred mask: {pred_mask}")
        # print(f"    Predicted relations: {pred_relations_matrix[:, :, :goal_predicates.shape[-1]]}")
        
        # Goal feasibility = proportion of goals satisfied
        if goal_mask.sum() == 0:
            goal_feasibility = 1.0  # No goals, always feasible
        else:
            goal_feasibility = matches.sum() / goal_mask.sum()
        
        # 2. Check collision feasibility (new)
        collision_feasibility = 1.0
        if self.enable_collision_checking:
            # Extract predicted poses from decoder output
            if 'predicted_pose' in predicted_state:
                predicted_poses = predicted_state['predicted_pose'].detach().cpu().numpy()
            else:
                # If no predicted poses, skip collision check
                predicted_poses = None
            
            # Check collisions using point clouds or poses
            if predicted_poses is not None:
                # Simple collision check on predicted positions
                is_feasible, reason = self.collision_checker.check_predicted_state_collisions(
                    predicted_point_clouds=None,  # We'll use poses instead
                    predicted_poses=predicted_poses,
                    target_object_id=target_id,
                    placement_height=None
                )
                
                if not is_feasible:
                    collision_feasibility = 0.0
                    # Optionally print reason for debugging
                    # print(f"    Collision detected: {reason}")
        
        # Combined feasibility: both must pass
        # If collision detected, overall feasibility is 0 regardless of goal match
        total_feasibility = goal_feasibility * collision_feasibility
        
        return float(total_feasibility)
    
    def _parse_primitive(self, primitive: str) -> Tuple[str, str, str]:
        """
        Parse primitive string.
        
        Examples:
            "Pick(milk, table)" -> ("Pick", "milk", "table")
            "Place(milk, bin)" -> ("Place", "milk", "bin")
        """
        if '(' not in primitive or ')' not in primitive:
            raise ValueError(f"Invalid primitive format: {primitive}")
        
        action_type = primitive.split('(')[0].strip()
        content = primitive.split('(')[1].split(')')[0]
        parts = [p.strip() for p in content.split(',')]
        
        if len(parts) != 2:
            raise ValueError(f"Primitive must have 2 arguments: {primitive}")
        
        return action_type, parts[0], parts[1]
    
    def _get_object_id(self, obj_name: str, state_dict: Dict) -> Optional[int]:
        """
        Get object ID from name using StateConverter if available, else fallback logic.
        
        Args:
            obj_name: Object name (e.g., "cubeA", "Milk", "table")
            state_dict: State dictionary containing object_names
        
        Returns:
            Object ID (int) or None if not found
        """
        # Use StateConverter's method if available (preferred - consistent lookup)
        if self.state_converter:
            obj_id = self.state_converter.get_object_id(obj_name)
            if obj_id >= 0:
                return obj_id
            # Continue to fallback if not found
        
        # Fallback: search in state_dict object_names
        if 'object_names' not in state_dict:
            print(f"Warning: No object_names in state_dict")
            return None
            
        names = state_dict['object_names']
        
        # 1. Exact match
        if obj_name in names:
            return names.index(obj_name)
        
        # 2. Case-insensitive match
        obj_lower = obj_name.lower()
        for idx, name in enumerate(names):
            if name.lower() == obj_lower:
                return idx
        
        # 3. Partial match
        for idx, name in enumerate(names):
            if obj_lower in name.lower() or name.lower() in obj_lower:
                return idx
        
        return None
    
    def predict_predicates(self, state_dict: Dict, debug: bool = False) -> Optional[np.ndarray]:
        """
        Predict current predicates from state using decoder.
        
        This is used for goal checking: compare predicted predicates with goal.
        
        Args:
            state_dict: Current state from StateConverter
            debug: If True, print debug information about shapes and values
        
        Returns:
            Predicted predicates array [num_objects, num_objects, num_predicates]
            or None if prediction fails
        """
        try:
            # Encode current state
            with torch.no_grad():
                node_embedding = self._encode_state(state_dict, debug=debug)
                
                # Build edge indices for decoder
                num_objects = state_dict['batch_num_objects']
                edge_index = self._build_edge_index(num_objects)
                
                if debug:
                    print(f"[DEBUG predict_predicates] edge_index shape: {edge_index.shape}")
                    print(f"[DEBUG predict_predicates] edge_index max: {edge_index.max()}, min: {edge_index.min()}")
                
                # Decode predicates
                decoder_output = self.model.classif_model_decoder(node_embedding, edge_index)
                pred_sigmoid = decoder_output['pred_sigmoid']  # [batch, num_edges, num_predicates]
                
                if debug:
                    print(f"[DEBUG predict_predicates] pred_sigmoid shape: {pred_sigmoid.shape}")
                    print(f"[DEBUG predict_predicates] pred_sigmoid stats: min={pred_sigmoid.min():.4f}, max={pred_sigmoid.max():.4f}, mean={pred_sigmoid.mean():.4f}")
            
            # Convert to numpy and reshape to matrix form
            pred_edges = pred_sigmoid.detach().cpu().numpy()
            pred_relations_matrix = self._edges_to_matrix(pred_edges, num_objects)
            
            return pred_relations_matrix
        
        except Exception as e:
            print(f"Error predicting predicates: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_default_args(self):
        """Create default args for BaseConfig."""
        # Create parser and parse with explicit minimal args
        parser = parse_util.get_parser()
        
        # Create a temporary dummy training directory (required but not used for inference)
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Provide minimal required arguments
        minimal_args = [
            '--result_dir', temp_dir,
            '--train_dir', temp_dir,
            '--test_dir', temp_dir
        ]
        
        args = parser.parse_args(minimal_args)
        
        # Set reasonable defaults for inference
        args.cuda = torch.cuda.is_available()
        args.batch_size = 1
        args.planning_batch_size = self.num_samples
        args.train = False
        
        return args
