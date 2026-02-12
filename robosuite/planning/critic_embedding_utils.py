"""
Embedding Extraction Utilities for Critic Data Collection

Provides functions to extract consistent embeddings from the dynamics model
for training the critic model.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class EmbeddingExtractor:
    """Extracts embeddings from dynamics model for critic training."""
    
    def __init__(
        self,
        action_dim: int = 64,
        predicate_embed_dim: int = 128,
        plan_summary_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize embedding extractor.
        
        Args:
            action_dim: Target dimension for action embeddings
            predicate_embed_dim: Target dimension for predicate embeddings
            plan_summary_dim: Target dimension for plan summary embeddings
            device: Device for torch operations
        """
        self.action_dim = action_dim
        self.predicate_embed_dim = predicate_embed_dim
        self.plan_summary_dim = plan_summary_dim
        self.device = torch.device(device)
        
        # Primitive type encoding
        self.primitive_types = {
            'pick': 0,
            'place': 1,
            'push': 2,
            'pull': 3,
        }
    
    def extract_action_embedding(
        self,
        dynamics_model,
        action_params: np.ndarray,
        obj_id: int,
        target_id: Optional[int],
        state_dict: Dict
    ) -> np.ndarray:
        """
        Extract action embedding that matches what's fed to the dynamics model.
        
        This extracts the actual embedding used in _simulate_one_step():
        discrete_action (one-hot obj_id) + continuous_action (dx, dy embedding).
        
        Args:
            dynamics_model: The RelationalDynamics model instance
            action_params: Action parameters [dx, dy, dz]
            obj_id: Object to move
            target_id: Target object (can be None)
            state_dict: State dictionary
        
        Returns:
            Action embedding as numpy array [action_dim]
        """
        with torch.no_grad():
            # Build discrete action: which object to move
            discrete_action = dynamics_model.classif_model.one_hot_encoding_embed(
                torch.LongTensor([obj_id]).to(self.device)
            )  # [1, D_discrete]
            
            # Build continuous action: x,y movement deltas
            continuous_action = dynamics_model.classif_model.continuous_action_emb(
                torch.FloatTensor(action_params[:2]).unsqueeze(0).to(self.device)
            )  # [1, D_continuous]
            
            # Concatenate (this is what's actually fed to the model)
            action_embedding = torch.cat([discrete_action, continuous_action], dim=-1)  # [1, D]
            
            # Convert to numpy
            action_embedding_np = action_embedding.cpu().numpy().squeeze(0)  # [D]
            
            # Project to target dimension if needed
            if action_embedding_np.shape[0] != self.action_dim:
                action_embedding_np = self._project_to_dim(
                    action_embedding_np, 
                    self.action_dim
                )
            
            return action_embedding_np.astype(np.float32)
    
    def extract_predicate_embedding(
        self,
        goal_predicates: np.ndarray,
        obj_id: int,
        target_id: Optional[int],
        num_objects: int
    ) -> np.ndarray:
        """
        Extract predicate embedding from goal predicates.
        
        For primitive "Place(A, B)", this extracts the target predicate
        relationship goal_predicates[A, B, :].
        
        Args:
            goal_predicates: Goal predicate tensor [N, N, num_predicates]
            obj_id: Object ID
            target_id: Target object ID (can be None for table)
            num_objects: Number of objects in scene
        
        Returns:
            Predicate embedding as numpy array [predicate_embed_dim]
        """
        if isinstance(goal_predicates, torch.Tensor):
            goal_predicates = goal_predicates.cpu().numpy()
        
        # Extract target predicate vector
        if target_id is not None and target_id < num_objects:
            # Get predicates for (obj, target) pair
            predicate_vector = goal_predicates[obj_id, target_id, :]  # [num_predicates]
        else:
            # No specific target (e.g., table) - use aggregate goal state
            # Take max across all targets for this object
            predicate_vector = goal_predicates[obj_id, :, :].max(axis=0)  # [num_predicates]
        
        # Project to target dimension
        predicate_embed = self._project_to_dim(predicate_vector, self.predicate_embed_dim)
        
        return predicate_embed.astype(np.float32)
    
    def extract_plan_summary(
        self,
        primitive_plan: List[str],
        current_step: int
    ) -> np.ndarray:
        """
        Extract plan summary embedding from remaining primitives.
        
        Encodes:
        - Number of remaining primitives
        - Current step position
        - Types of remaining primitives (one-hot encoded)
        
        Args:
            primitive_plan: Full primitive plan
            current_step: Current step index in plan
        
        Returns:
            Plan summary embedding as numpy array [plan_summary_dim]
        """
        remaining_primitives = primitive_plan[current_step:]
        num_remaining = len(remaining_primitives)
        
        # Build feature vector
        features = []
        
        # 1. Normalized position in plan
        plan_progress = current_step / max(len(primitive_plan), 1)
        features.append(plan_progress)
        
        # 2. Normalized remaining steps
        remaining_ratio = num_remaining / max(len(primitive_plan), 1)
        features.append(remaining_ratio)
        
        # 3. Absolute counts (normalized)
        features.append(current_step / 10.0)  # Normalize to reasonable range
        features.append(num_remaining / 10.0)
        
        # 4. Primitive type distribution (one-hot counts)
        primitive_type_counts = np.zeros(len(self.primitive_types))
        for prim in remaining_primitives:
            prim_type = self._parse_primitive_type(prim)
            if prim_type in self.primitive_types:
                primitive_type_counts[self.primitive_types[prim_type]] += 1
        
        # Normalize counts
        if primitive_type_counts.sum() > 0:
            primitive_type_counts = primitive_type_counts / primitive_type_counts.sum()
        
        features.extend(primitive_type_counts.tolist())
        
        # 5. Next primitive type (one-hot)
        if num_remaining > 0:
            next_type = self._parse_primitive_type(remaining_primitives[0])
            next_type_onehot = np.zeros(len(self.primitive_types))
            if next_type in self.primitive_types:
                next_type_onehot[self.primitive_types[next_type]] = 1.0
            features.extend(next_type_onehot.tolist())
        else:
            features.extend([0.0] * len(self.primitive_types))
        
        # Convert to numpy
        feature_vector = np.array(features, dtype=np.float32)
        
        # Project to target dimension
        plan_summary = self._project_to_dim(feature_vector, self.plan_summary_dim)
        
        return plan_summary.astype(np.float32)
    
    def _project_to_dim(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Project vector to target dimension via padding or truncation.
        
        Simple strategy for Phase 1. Can be replaced with learned projection later.
        """
        current_dim = vector.shape[0]
        
        if current_dim == target_dim:
            return vector
        elif current_dim < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim, dtype=vector.dtype)
            padded[:current_dim] = vector
            return padded
        else:
            # Truncate (or could use PCA/random projection)
            return vector[:target_dim]
    
    def _parse_primitive_type(self, primitive: str) -> str:
        """
        Parse primitive type from string.
        
        Examples:
            "Pick(milk, table)" -> "pick"
            "Place(milk, bin)" -> "place"
        """
        if '(' in primitive:
            prim_type = primitive.split('(')[0].strip().lower()
        else:
            prim_type = primitive.strip().lower()
        
        return prim_type


def test_embedding_extractor():
    """Test embedding extraction with dummy data."""
    print("Testing EmbeddingExtractor...")
    
    extractor = EmbeddingExtractor()
    
    # Test predicate embedding
    goal_predicates = np.random.rand(5, 5, 9)
    pred_embed = extractor.extract_predicate_embedding(
        goal_predicates, obj_id=0, target_id=1, num_objects=5
    )
    assert pred_embed.shape == (128,), f"Expected (128,), got {pred_embed.shape}"
    print(f"✓ Predicate embedding: {pred_embed.shape}")
    
    # Test plan summary
    primitive_plan = ["Pick(A, table)", "Place(A, B)", "Pick(C, table)"]
    plan_summary = extractor.extract_plan_summary(primitive_plan, current_step=0)
    assert plan_summary.shape == (128,), f"Expected (128,), got {plan_summary.shape}"
    print(f"✓ Plan summary: {plan_summary.shape}")
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_embedding_extractor()
