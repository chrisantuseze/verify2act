"""
Dynamics Model Data Collector

Wrapper around CriticDataCollector that interfaces with the DynamicsModelPlanner
to collect training data for the critic model.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
import copy
import torch

# Add verify2act critic to path
verify2act_path = Path(__file__).parent.parent.parent / "verify2act"
sys.path.insert(0, str(verify2act_path))

from critic.critic_data_collector import CriticDataCollector
from critic_embedding_utils import EmbeddingExtractor


class DynamicsModelDataCollector:
    """
    Wrapper for CriticDataCollector with dynamics-model-specific logic.
    
    Handles:
    - Episode-level buffering of raw data
    - Embedding extraction from dynamics model
    - Success/failure labeling
    - Integration with rejection sampling loop
    """
    
    def __init__(
        self,
        dynamics_planner,
        save_dir: str = "./data/critic",
        latent_dim: int = 256,
        action_dim: int = 64,
        predicate_embed_dim: int = 128,
        plan_summary_dim: int = 128,
        enable_hard_negatives: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize data collector.
        
        Args:
            dynamics_planner: DynamicsModelPlanner instance
            save_dir: Directory to save collected data
            latent_dim: Dimension of state embeddings (from PointConv)
            action_dim: Dimension of action embeddings
            predicate_embed_dim: Dimension of predicate embeddings
            plan_summary_dim: Dimension of plan summary embeddings
            enable_hard_negatives: Whether to generate hard negatives
            verbose: Whether to print collection progress
        """
        self.dynamics_planner = dynamics_planner
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.enable_hard_negatives = enable_hard_negatives
        
        # Initialize base collector
        self.collector = CriticDataCollector(
            latent_dim=latent_dim,
            action_dim=action_dim,
            predicate_embed_dim=predicate_embed_dim,
            plan_summary_dim=plan_summary_dim,
        )
        
        # Initialize embedding extractor
        self.embedding_extractor = EmbeddingExtractor(
            action_dim=action_dim,
            predicate_embed_dim=predicate_embed_dim,
            plan_summary_dim=plan_summary_dim,
            device=dynamics_planner.device,
        )
        
        # Episode-level state
        self.current_episode_buffer = []
        self.current_goal_predicates = None
        self.current_primitive_plan = None
        self.episode_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    def start_episode(
        self,
        goal_predicates: np.ndarray,
        primitive_plan: List[str]
    ):
        """
        Start a new episode and initialize buffer.
        
        Args:
            goal_predicates: Goal predicate tensor [N, N, num_predicates]
            primitive_plan: List of primitive strings
        """
        self.current_episode_buffer = []
        self.current_goal_predicates = goal_predicates
        self.current_primitive_plan = primitive_plan
        
        if self.verbose:
            print(f"\n[DataCollector] Starting episode {self.episode_count + 1}")
            print(f"  Plan: {primitive_plan}")
    
    def record_step(
        self,
        step_idx: int,
        state_dict: Dict,
        action_params: np.ndarray,
        obj_id: int,
        target_id: Optional[int],
        next_state_dict: Optional[Dict] = None,
        feasibility_score: float = None,
    ):
        """
        Record a single step in the current episode.
        
        Args:
            step_idx: Index of current step in plan
            state_dict: Current state dictionary from StateConverter
            action_params: Action parameters [dx, dy, dz]
            obj_id: Object ID being manipulated
            target_id: Target object ID (can be None)
            next_state_dict: Predicted next state (optional, for z_next)
            feasibility_score: Feasibility score for this action (optional)
        """
        if obj_id is None:
            if self.verbose:
                print(f"  [Step {step_idx}] Skipping record: unresolved obj_id")
            return

        # Store raw data in buffer (will extract embeddings at episode end)
        step_data = {
            'step_idx': step_idx,
            'state_dict': copy.deepcopy(state_dict),
            'action_params': action_params,
            'obj_id': obj_id,
            'target_id': target_id,
            'next_state_dict': copy.deepcopy(next_state_dict) if next_state_dict is not None else None,
            'feasibility_score': feasibility_score,
            'execution_success': None,
            'num_steps': None,
        }
        
        self.current_episode_buffer.append(step_data)
        
        if self.verbose:
            print(f"  [Step {step_idx}] Recorded action: obj={obj_id}, target={target_id}")

    def update_last_step_next_state(
        self,
        next_state_dict: Dict,
        execution_success: Optional[bool] = None,
        num_steps: Optional[int] = None,
    ) -> None:
        """
        Update the most recently recorded step with executed next state.

        Args:
            next_state_dict: Observed post-execution state dictionary
            execution_success: Whether primitive execution succeeded
            num_steps: Number of simulator steps taken for execution
        """
        if not self.current_episode_buffer:
            return

        self.current_episode_buffer[-1]['next_state_dict'] = copy.deepcopy(next_state_dict)
        self.current_episode_buffer[-1]['execution_success'] = execution_success
        self.current_episode_buffer[-1]['num_steps'] = num_steps
    
    def end_episode(
        self,
        success: bool,
        failure_step: Optional[int] = None,
        failure_type: str = "predicate"
    ):
        """
        End current episode and process collected data.
        
        Args:
            success: Whether episode succeeded
            failure_step: Index of step where failure occurred (if failed)
            failure_type: Type of failure ("predicate", "feasibility", "noninterference")
        """
        if len(self.current_episode_buffer) == 0:
            print("[DataCollector] Warning: Empty episode buffer, skipping")
            return
        
        # Extract embeddings for all steps
        trajectory = []
        predicate_embeddings = []
        plan_summaries = []
        step_metadata = []
        
        for step_data in self.current_episode_buffer:
            # Extract z_t (current state latent)
            z_t = self._extract_state_latent(step_data['state_dict'])
            
            # Extract a_t (action embedding)
            a_t = self.embedding_extractor.extract_action_embedding(
                dynamics_model=self.dynamics_planner.model,
                action_params=step_data['action_params'],
                obj_id=step_data['obj_id'],
                target_id=step_data['target_id'],
                state_dict=step_data['state_dict'],
            )
            
            # Extract z_next (next state latent)
            if step_data['next_state_dict'] is not None:
                z_next = self._extract_state_latent(step_data['next_state_dict'])
            else:
                # If next state not provided, use z_t as placeholder
                # (will be labeled as failure likely)
                z_next = z_t.copy()
            
            # Extract predicate embedding
            predicate_embed = self.embedding_extractor.extract_predicate_embedding(
                goal_predicates=self.current_goal_predicates,
                obj_id=step_data['obj_id'],
                target_id=step_data['target_id'],
                num_objects=step_data['state_dict']['batch_num_objects'],
            )
            
            # Extract plan summary
            plan_summary = self.embedding_extractor.extract_plan_summary(
                primitive_plan=self.current_primitive_plan,
                current_step=step_data['step_idx'],
            )
            
            trajectory.append({
                'z_t': z_t,
                'a_t': a_t,
                'z_next': z_next,
            })
            predicate_embeddings.append(predicate_embed)
            plan_summaries.append(plan_summary)
            step_metadata.append({
                'state_dict': step_data['state_dict'],
                'next_state_dict': step_data['next_state_dict'],
                'action_params': step_data['action_params'],
                'obj_id': step_data['obj_id'],
                'target_id': step_data['target_id'],
                'feasibility_score': step_data.get('feasibility_score'),
                'execution_success': step_data.get('execution_success'),
                'num_steps': step_data.get('num_steps'),
            })
        
        # Add to collector based on success/failure
        if success:
            self.collector.add_successful_trajectory(
                trajectory=trajectory,
                predicate_embeddings=predicate_embeddings,
                plan_summaries=plan_summaries,
                step_metadata=step_metadata,
            )
            self.success_count += 1
            if self.verbose:
                print(f"  ✓ Episode {self.episode_count + 1} succeeded ({len(trajectory)} steps)")
        else:
            # Determine failure step
            if failure_step is None:
                failure_step = len(trajectory) - 1  # Assume last step failed
            
            self.collector.add_failed_trajectory(
                trajectory=trajectory,
                predicate_embeddings=predicate_embeddings,
                plan_summaries=plan_summaries,
                failure_step=failure_step,
                failure_type=failure_type,
                step_metadata=step_metadata,
            )
            self.failure_count += 1
            if self.verbose:
                print(f"  ✗ Episode {self.episode_count + 1} failed at step {failure_step} ({failure_type})")
        
        # Clear episode buffer
        self.current_episode_buffer = []
        self.current_goal_predicates = None
        self.current_primitive_plan = None
        self.episode_count += 1
    
    def _extract_state_latent(self, state_dict: Dict) -> np.ndarray:
        """
        Extract state latent embedding from state_dict.
        
        This encodes the current state through the dynamics model's PointConv.
        
        Args:
            state_dict: State dictionary from StateConverter
        
        Returns:
            State latent as numpy array [latent_dim]
        """
        with torch.no_grad():
            node_embedding = self.dynamics_planner._encode_state(state_dict, debug=False)
            
            # node_embedding is [batch, num_objects, embed_dim]
            # Flatten to single vector for critic
            # Strategy: Take mean across objects (global scene representation)
            if isinstance(node_embedding, torch.Tensor):
                node_embedding = node_embedding.cpu().numpy()
            
            # Remove batch dimension if present
            if node_embedding.ndim == 3:
                node_embedding = node_embedding[0]  # [num_objects, embed_dim]
            
            # Flatten: mean pool across objects
            state_latent = node_embedding.mean(axis=0)  # [embed_dim]
            
            return state_latent.astype(np.float32)
    
    def save_dataset(self, filename: str = "critic_data.pkl"):
        """
        Save collected dataset to disk.
        
        Args:
            filename: Filename for saved dataset
        """
        save_path = self.save_dir / filename
        
        # Balance dataset (generate hard negatives if enabled)
        if self.enable_hard_negatives:
            self.collector.balance_dataset(
                negative_augmentation=True,
                target_ratio=1.0,
            )
        
        # Save
        self.collector.save_dataset(str(save_path))
        
        if self.verbose:
            print(f"\n[DataCollector] Dataset saved to {save_path}")
            print(f"  Total episodes: {self.episode_count}")
            print(f"  Successes: {self.success_count}")
            print(f"  Failures: {self.failure_count}")
    
    def get_statistics(self) -> Dict:
        """Get collection statistics."""
        stats = self.collector.get_statistics()
        stats['episode_count'] = self.episode_count
        stats['success_count'] = self.success_count
        stats['failure_count'] = self.failure_count
        stats['success_rate'] = self.success_count / max(self.episode_count, 1)
        return stats
    
    def print_statistics(self):
        """Print collection statistics."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("DATA COLLECTION STATISTICS")
        print("=" * 60)
        print(f"Episodes collected:    {stats['episode_count']}")
        print(f"  Successes:           {stats['success_count']}")
        print(f"  Failures:            {stats['failure_count']}")
        print(f"  Success rate:        {stats['success_rate']:.1%}")
        print(f"\nSamples collected:")
        print(f"  Positive:            {stats['num_positive']}")
        print(f"  Negative:            {stats['num_negative']}")
        print(f"  Total:               {stats['total']}")
        print(f"\nPositive sources:")
        for source, count in stats['positive_sources'].items():
            print(f"  {source:20s} {count:5d}")
        print(f"\nNegative sources:")
        for source, count in stats['negative_sources'].items():
            print(f"  {source:20s} {count:5d}")
        print("=" * 60)


if __name__ == "__main__":
    print("DynamicsModelDataCollector utility loaded.")
    print("Use this module by importing it into your planning script.")
