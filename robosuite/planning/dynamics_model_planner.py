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
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize dynamics model planner.
        
        Args:
            checkpoint_path: Path to trained model checkpoint 
                           (e.g., Points2Plans/ckpt/checkpoint/cp_1.pth)
            config_args: Optional config overrides (dict)
            num_samples: Number of action samples for rejection sampling (default 50)
            device: Device for model
        """
        self.checkpoint_path = checkpoint_path
        self.num_samples = num_samples
        self.device = torch.device(device)
        
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
    
    def plan_next_primitive(self,
                           state_dict: Dict,
                           goal_predicates: np.ndarray,
                           primitive_plan: List[str]) -> Tuple[str, np.ndarray, float]:
        """
        Plan next primitive action using rejection sampling.
        
        Closed-loop planning:
        1. Take current state observation
        2. Know what goals to achieve (from LLM, remains constant)
        3. Sample K candidate actions for next primitive in plan
        4. Simulate each action through dynamics model
        5. Check feasibility: does predicted state match goals?
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
        
        # Parse primitive: "Pick(milk, table)" -> ["Pick", "milk", "table"]
        action_type, obj_name, target_name = self._parse_primitive(next_primitive)
        
        # Get object IDs from state
        obj_id = self._get_object_id(obj_name, state_dict)
        target_id = self._get_object_id(target_name, state_dict)
        
        if obj_id is None or target_id is None:
            print(f"Warning: Object not found. obj={obj_name}({obj_id}), target={target_name}({target_id})")
            # Return primitive as-is with zero action
            return next_primitive, np.zeros(3), 0.0
        
        # Rejection sampling: sample K candidate actions
        best_action = None
        best_params = None
        best_feasibility = -1.0
        
        print(f"\n=== Planning for: {next_primitive} ===")
        print(f"  Object ID: {obj_id}, Target ID: {target_id}")
        print(f"  Sampling {self.num_samples} candidate actions...")
        
        with torch.no_grad():
            # Encode current state
            node_embedding = self._encode_state(state_dict)
            
            for sample_idx in range(self.num_samples):
                # Sample action around target location
                action_params = self._sample_action(
                    state_dict,
                    obj_id,
                    target_id,
                    action_type
                )
                
                # Forward simulate through dynamics model
                predicted_state = self._forward_simulate(
                    node_embedding,
                    state_dict,
                    action_params,
                    obj_id
                )
                
                # Check feasibility: does predicted state match goals?
                feasibility = self._check_feasibility(
                    predicted_state,
                    goal_predicates,
                    state_dict['batch_num_objects']
                )
                
                # Keep best action
                if feasibility > best_feasibility:
                    best_feasibility = feasibility
                    best_action = next_primitive
                    best_params = action_params
                
                # Early exit if found feasible action
                if feasibility >= 0.5:
                    print(f"  ✓ Found feasible action at sample {sample_idx + 1}")
                    break
        
        if best_feasibility < 0.5:
            print(f"  ⚠ No feasible action found. Best feasibility: {best_feasibility:.3f}")
        
        return best_action, best_params, best_feasibility
    
    def _encode_state(self, state_dict: Dict) -> torch.Tensor:
        """Encode current state observation through PointConv."""
        voxel_data = state_dict['batch_voxel_list_single'][0]
        
        if not isinstance(voxel_data, torch.Tensor):
            voxel_data = torch.FloatTensor(voxel_data).to(self.device)
        
        # Per-object point cloud encoding
        img_emb = self.model.emb_model(voxel_data)
        
        # One-hot object encoding
        one_hot_encoding = state_dict['batch_one_hot_encoding']
        if not isinstance(one_hot_encoding, torch.Tensor):
            one_hot_encoding = torch.FloatTensor(one_hot_encoding).to(self.device)
        
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
    
    def _forward_simulate(self,
                         node_embedding: torch.Tensor,
                         state_dict: Dict,
                         action_params: np.ndarray,
                         obj_id: int) -> Dict:
        """
        Forward simulate action through dynamics model.
        
        Returns predicted next state (relations, poses, etc.).
        """
        num_objects = state_dict['batch_num_objects']
        
        # Create action tensor
        action_tensor = torch.zeros(1, num_objects, 3).to(self.device)
        action_tensor[0, obj_id, :] = torch.FloatTensor(action_params)
        
        # Build edge index for graph
        nodes = list(range(num_objects))
        edges = list(permutations(nodes, 2))
        edge_index = torch.LongTensor(np.array(edges).T).to(self.device)
        
        # Get action embedding
        if hasattr(self.model.classif_model, 'continuous_action_emb'):
            action_emb = self.model.classif_model.continuous_action_emb(action_tensor)
        else:
            action_emb = action_tensor
        
        # Dynamics forward pass (Pick/Place uses graph_dynamics_0)
        if hasattr(self.model.classif_model, 'graph_dynamics_0'):
            next_embedding = self.model.classif_model.graph_dynamics_0(
                node_embedding,
                action_emb,
                edge_index
            )
        else:
            # Fallback: simple addition
            next_embedding = node_embedding + action_emb
        
        # Decode predicted state
        decoder_output = self.model.classif_model_decoder(next_embedding, edge_index)
        
        return decoder_output
    
    def _check_feasibility(self,
                          predicted_state: Dict,
                          goal_predicates: np.ndarray,
                          num_objects: int) -> float:
        """
        Check if predicted state matches goal predicates (binary feasibility).
        
        Returns feasibility score [0, 1].
        """
        # Get predicted relations from decoder
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
        
        # Feasibility = proportion of goals satisfied
        if goal_mask.sum() == 0:
            return 1.0  # No goals, always feasible
        
        feasibility = matches.sum() / goal_mask.sum()
        
        return float(feasibility)
    
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
        Get object ID from name.
        
        This assumes state_dict has object name mappings.
        You may need to adjust based on your actual state representation.
        """
        # For now, simple heuristic: parse object name if it has ID suffix
        # e.g., "milk_0" -> 0, "bin_1" -> 1
        
        # Try to find in object list if provided
        if 'object_names' in state_dict:
            names = state_dict['object_names']
            if obj_name in names:
                return names.index(obj_name)
        
        # Fallback: try to extract numeric suffix
        if '_' in obj_name:
            try:
                return int(obj_name.split('_')[-1])
            except ValueError:
                pass
        
        # Last resort: match by partial name
        # This is a placeholder - you should implement proper object ID mapping
        # based on your StateConverter's object tracking
        print(f"Warning: Could not find object ID for {obj_name}, returning None")
        return None
    
    def _create_default_args(self):
        """Create default args for BaseConfig."""
        parser = parse_util.get_parser()
        args = parser.parse_args([])
        
        # Set reasonable defaults for inference
        args.cuda = torch.cuda.is_available()
        args.batch_size = 1
        args.planning_batch_size = self.num_samples
        args.train = False
        
        return args
