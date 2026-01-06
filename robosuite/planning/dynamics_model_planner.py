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
from typing import Dict, List, Tuple, Optional
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
                 state_converter=None,
                 enable_collision_checking: bool = True,
                 x_collision: float = 0.05,
                 y_collision: float = 0.05,
                 lookahead_depth: int = 1):
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
        """
        self.checkpoint_path = checkpoint_path
        self.num_samples = num_samples
        self.device = torch.device(device)
        self.state_converter = state_converter
        self.enable_collision_checking = enable_collision_checking
        self.lookahead_depth = max(1, min(lookahead_depth, 3))  # Clamp between 1-3
        
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
        
        print(f"Dynamics Model Planner initialized:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Num samples: {num_samples}")
        print(f"  Device: {device}")
        print(f"  Collision checking: {self.enable_collision_checking}")
        print(f"  Lookahead depth: {self.lookahead_depth} primitive(s)")
    
    def plan_next_primitive(self,
                           state_dict: Dict,
                           goal_predicates: np.ndarray,
                           primitive_plan: List[str]) -> Tuple[str, np.ndarray, float]:
        """
        Plan next primitive action using rejection sampling with multi-step lookahead.
        
        Closed-loop planning with lookahead:
        1. Take current state observation
        2. Know what goals to achieve (from LLM, remains constant)
        3. Sample K candidate actions for next primitive in plan
        4. Simulate each action through dynamics model (with lookahead_depth primitives)
        5. Check feasibility: does predicted TERMINAL state match goals?
        6. Return first feasible action (or best if none feasible)
        
        Args:
            state_dict: Current state from StateConverter.convert()
                       Contains: batch_voxel_list_single, batch_one_hot_encoding, 
                                batch_6DOF_pose, batch_edge_attr, batch_num_objects
            goal_predicates: Goal predicate tensor [num_objects, num_objects, num_predicates]
                           From LLMTaskPlanner.goals_to_predicates()
            primitive_plan: High-level plan from LLM (e.g., ["Pick(milk, table)", "Place(milk, bin)"])
                          We plan for the NEXT unexecuted primitive
        
        Returns:
            primitive_action: Next primitive to execute (e.g., "Pick(milk, table)")
            action_params: Low-level action parameters (e.g., target position)
            feasibility: Feasibility score [0, 1]
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
            return next_primitive, np.zeros(3), 0.0
        
        # Parse the FIRST primitive (what we'll actually execute)
        action_type, obj_name, target_name = self._parse_primitive(next_primitive)
        obj_id = self._get_object_id(obj_name, state_dict)
        target_id = self._get_object_id(target_name, state_dict)
        
        # Handle special case: "table" is not a tracked object
        use_table_location = False
        if target_id is None and target_name.lower() == "table":
            use_table_location = True
            poses = state_dict['batch_6DOF_pose']
            if isinstance(poses, torch.Tensor):
                poses = poses.cpu().numpy()
            if poses.ndim == 3:
                poses = poses[0]
            table_location = poses[obj_id].copy()
            table_location[2] = 0.8 # Assume table height is 0.8m, can be adjusted based on env
        
        # Rejection sampling: sample K candidate actions
        best_action = None
        best_params = None
        best_feasibility = -1.0
        
        lookahead_str = f"{actual_lookahead}-step" if actual_lookahead > 1 else "1-step (greedy)"
        print(f"\n=== Planning for: {next_primitive} ({lookahead_str} lookahead) ===")
        print(f"  Looking ahead: {[p for _, p, _, _, _ in lookahead_primitives[:actual_lookahead]]}")
        print(f"  Object ID: {obj_id}, Target ID: {target_id if not use_table_location else 'table (special)'}")
        print(f"  Sampling {self.num_samples} candidate actions...")
        
        with torch.no_grad():
            # Encode current state
            node_embedding = self._encode_state(state_dict)
            
            for sample_idx in range(self.num_samples):
                # Sample action for FIRST primitive
                if use_table_location:
                    action_params = self._sample_action_for_table(
                        state_dict, obj_id, action_type
                    )
                else:
                    action_params = self._sample_action(
                        state_dict, obj_id, target_id, action_type
                    )
                
                # Build action sequence for lookahead
                if actual_lookahead == 1:
                    # Single-step (original behavior)
                    predicted_state = self._forward_simulate(
                        node_embedding, state_dict, action_params, obj_id, 
                        target_id if target_id is not None else obj_id
                    )
                else:
                    # Multi-step lookahead
                    action_sequence = []
                    
                    # First action: use sampled params
                    action_sequence.append((
                        action_type, action_params, obj_id, 
                        target_id if target_id is not None else obj_id
                    ))
                    
                    # Subsequent actions: sample around their targets
                    for i in range(1, actual_lookahead):
                        _, _, _, next_obj_id, next_target_id = lookahead_primitives[i]
                        
                        # Sample action for this future primitive
                        future_action_params = self._sample_action(
                            state_dict, next_obj_id, 
                            next_target_id if next_target_id is not None else next_obj_id,
                            lookahead_primitives[i][0]
                        )
                        
                        action_sequence.append((
                            lookahead_primitives[i][0], future_action_params, 
                            next_obj_id, next_target_id if next_target_id is not None else next_obj_id
                        ))
                    
                    # Simulate entire sequence
                    _, predicted_state = self._forward_simulate_sequence(
                        node_embedding, state_dict, action_sequence, 
                        state_dict['batch_num_objects']
                    )
                
                # Check feasibility of TERMINAL state
                feasibility = self._check_feasibility(
                    predicted_state,
                    goal_predicates,
                    state_dict['batch_num_objects'],
                    obj_id=obj_id,
                    target_id=target_id if not use_table_location else None
                )
                
                # Keep best action
                if feasibility > best_feasibility:
                    best_feasibility = feasibility
                    best_action = next_primitive
                    best_params = action_params
                
                # Early exit if found feasible action
                if feasibility >= 0.5:
                    print(f"  ✓ Found feasible {actual_lookahead}-step sequence at sample {sample_idx + 1}")
                    break
        
        if best_feasibility < 0.5:
            print(f"  ⚠ No feasible sequence found. Best feasibility: {best_feasibility:.3f}")
        
        return best_action, best_params, best_feasibility
    
    def _encode_state(self, state_dict: Dict) -> torch.Tensor:
        """Encode current state observation through PointConv."""
        voxel_data = state_dict['batch_voxel_list_single'][0]
        
        if not isinstance(voxel_data, torch.Tensor):
            voxel_data = torch.FloatTensor(voxel_data)
        
        # Move to device and transpose to PointConv format: [N_objects, N_points, 3] -> [N_objects, 3, N_points]
        voxel_data = voxel_data.to(self.device).permute(0, 2, 1)
        
        # Per-object point cloud encoding
        img_emb = self.model.emb_model(voxel_data)
        
        # One-hot object encoding
        one_hot_encoding = state_dict['batch_one_hot_encoding']
        if not isinstance(one_hot_encoding, torch.Tensor):
            one_hot_encoding = torch.FloatTensor(one_hot_encoding)
        else:
            # Remove batch dimension if present
            if one_hot_encoding.dim() == 3:
                one_hot_encoding = one_hot_encoding[0]
        one_hot_encoding = one_hot_encoding.to(self.device)
        
        latent_one_hot = self.model.classif_model.one_hot_encoding_embed(
            torch.argmax(one_hot_encoding, dim=1)
        )
        
        # Concatenate embeddings
        node_embedding = torch.cat([img_emb, latent_one_hot], dim=1)
        
        if node_embedding.shape[0] != 1:
            node_embedding = node_embedding.view(1, node_embedding.shape[0], node_embedding.shape[1])
        
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
        poses = state_dict['batch_6DOF_pose']
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()
        if poses.ndim == 3:
            poses = poses[0]  # Remove batch dimension
        
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
        poses = state_dict['batch_6DOF_pose']
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()
        if poses.ndim == 3:
            poses = poses[0]  # Remove batch dimension
        
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
        
        # Roll out each primitive in sequence
        for step_idx, (action_type, action_params, obj_id, target_id) in enumerate(primitive_sequence):
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
            
            # Build place action
            if target_id == obj_id or target_id is None:
                discrete_place_id_tensor = torch.zeros_like(discrete_action)
            else:
                discrete_place_id_tensor = self.model.classif_model.one_hot_encoding_embed(
                    torch.LongTensor([target_id]).to(self.device)
                )
            
            current_action = torch.cat([discrete_place_id_tensor, continuous_action], dim=-1)
            current_action = current_action.view(1, 1, -1)
            
            # Concatenate: [node embeddings, action embeddings]
            graph_node_action = torch.cat([current_latent, current_action_continuous, current_action], dim=1)
            
            # Forward through dynamics model (Pick/Place uses graph_dynamics_0)
            next_latent = self.model.classif_model.graph_dynamics_0(graph_node_action)
            
            # Extract predicted node embeddings (exclude action tokens)
            current_latent = next_latent[:, :-2, :]
        
        # Decode final state
        edge_nodes = list(range(num_objects))
        edge_list = list(permutations(edge_nodes, 2))
        edge_index = torch.LongTensor(np.array(edge_list).T).to(self.device)
        
        final_decoder_output = self.model.classif_model_decoder(current_latent, edge_index)
        
        return current_latent, final_decoder_output
    
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
        
        # Build discrete action: which object to move (one-hot encoded)
        discrete_action = self.model.classif_model.one_hot_encoding_embed(
            torch.LongTensor([obj_id]).to(self.device)
        )
        
        # Build continuous action: x,y movement deltas
        continuous_action = self.model.classif_model.continuous_action_emb(
            torch.FloatTensor(action_params[:2]).unsqueeze(0).to(self.device)
        )
        
        # Combine discrete + continuous for "which object, how to move it"
        current_action_continuous = torch.cat([discrete_action, continuous_action], dim=-1)
        current_action_continuous = current_action_continuous.view(1, 1, -1)
        
        # Build place action: where to place it (target object ID + movement)
        if target_id == obj_id:
            # Placing on table/same location
            discrete_place_id_tensor = torch.zeros_like(discrete_action)
        else:
            # Placing on another object
            discrete_place_id_tensor = self.model.classif_model.one_hot_encoding_embed(
                torch.LongTensor([target_id]).to(self.device)
            )
        
        current_action = torch.cat([discrete_place_id_tensor, continuous_action], dim=-1)
        current_action = current_action.view(1, 1, -1)
        
        # Concatenate: [node embeddings, action embeddings]
        graph_node_action = torch.cat([node_embedding, current_action_continuous, current_action], dim=1)
        
        # Forward through dynamics model (Pick/Place uses graph_dynamics_0)
        next_latent = self.model.classif_model.graph_dynamics_0(graph_node_action)
        
        # Extract predicted node embeddings (exclude action tokens)
        next_latent = next_latent[:, :-2, :]
        
        # Decode predicted state
        edge_nodes = list(range(num_objects))
        edge_list = list(permutations(edge_nodes, 2))
        edge_index = torch.LongTensor(np.array(edge_list).T).to(self.device)
        
        decoder_output = self.model.classif_model_decoder(next_latent, edge_index)
        
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
        # 1. Check goal matching (existing logic)
        pred_relations = predicted_state['pred_sigmoid'].detach().cpu().numpy()
        
        # Reshape: [batch, num_edges, num_predicates] -> [num_objects, num_objects, num_predicates]
        # Edge ordering: (0,1), (0,2), ..., (1,0), (1,2), ...
        pred_relations_matrix = np.zeros((num_objects, num_objects, pred_relations.shape[-1]))
        
        edge_idx = 0
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    pred_relations_matrix[i, j, :] = pred_relations[0, edge_idx, :]
                    edge_idx += 1
        
        # Compare with goals: check if predicted relations match goal relations
        # For each goal predicate that should be 1, check if prediction > 0.5
        goal_mask = goal_predicates > 0.5
        matches = np.logical_and(
            goal_mask,
            pred_relations_matrix[:, :, :goal_predicates.shape[-1]] > 0.5
        )
        
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
    
    def predict_predicates(self, state_dict: Dict) -> Optional[np.ndarray]:
        """
        Predict current predicates from state using decoder.
        
        This is used for goal checking: compare predicted predicates with goal.
        
        Args:
            state_dict: Current state from StateConverter
        
        Returns:
            Predicted predicates array [num_objects, num_objects, num_predicates]
            or None if prediction fails
        """
        try:
            # Encode current state
            with torch.no_grad():
                node_embedding = self._encode_state(state_dict)
                
                # Build edge indices for decoder
                num_objects = state_dict['batch_num_objects']
                edge_nodes = list(range(num_objects))
                edge_list = list(permutations(edge_nodes, 2))
                edge_index = torch.LongTensor(np.array(edge_list).T).to(self.device)
                
                # Decode predicates
                decoder_output = self.model.classif_model_decoder(node_embedding, edge_index)
                pred_sigmoid = decoder_output['pred_sigmoid']  # [batch, num_edges, num_predicates]
            
            # Convert to numpy
            pred_relations = pred_sigmoid.detach().cpu().numpy()
            
            # Reshape to matrix form
            pred_relations_matrix = np.zeros((num_objects, num_objects, pred_relations.shape[-1]))
            
            edge_idx = 0
            for i in range(num_objects):
                for j in range(num_objects):
                    if i != j:
                        pred_relations_matrix[i, j, :] = pred_relations[0, edge_idx, :]
                        edge_idx += 1
            
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
