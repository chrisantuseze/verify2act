"""
Verify2Act Critic Data Collection
Utilities for collecting positive and negative samples for critic training.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
from collections import defaultdict


class CriticDataCollector:
    """Collects training data from Points2Plans rollouts."""
    
    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 64,
        predicate_embed_dim: int = 128,
        plan_summary_dim: int = 128,
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.predicate_embed_dim = predicate_embed_dim
        self.plan_summary_dim = plan_summary_dim
        
        self.positive_samples = []
        self.negative_samples = []
    
    def add_successful_trajectory(
        self,
        trajectory: List[Dict],
        predicate_embeddings: List[np.ndarray],
        plan_summaries: List[np.ndarray],
        step_metadata: Optional[List[Dict]] = None,
    ):
        """
        Add positive samples from a successful trajectory.
        
        Args:
            trajectory: List of dicts with z_t, a_t, z_next
            predicate_embeddings: Target predicate embeddings per step
            plan_summaries: Remaining plan summaries per step
        """
        for step_idx, step_data in enumerate(trajectory):
            sample = {
                "z_t": step_data["z_t"],
                "a_t": step_data["a_t"],
                "z_next": step_data["z_next"],
                "predicate_embed": predicate_embeddings[step_idx],
                "plan_summary": plan_summaries[step_idx],
                "label_predicate": 1,  # Successful
                "label_feas": 1,
                "label_nonint": 1,
                "source": "successful_trajectory",
                "step_idx": step_idx,
            }
            if step_metadata is not None and step_idx < len(step_metadata):
                sample.update(step_metadata[step_idx])
            self.positive_samples.append(sample)
    
    def add_failed_trajectory(
        self,
        trajectory: List[Dict],
        predicate_embeddings: List[np.ndarray],
        plan_summaries: List[np.ndarray],
        failure_step: int,
        failure_type: str = "predicate",
        step_metadata: Optional[List[Dict]] = None,
    ):
        """
        Add negative samples from a failed trajectory.
        
        Args:
            trajectory: List of dicts with z_t, a_t, z_next
            predicate_embeddings: Target predicate embeddings per step
            plan_summaries: Remaining plan summaries per step
            failure_step: Index of the step where failure occurred
            failure_type: "predicate", "feasibility", or "noninterference"
        """
        # Label all steps
        for step_idx, step_data in enumerate(trajectory):
            # Steps before failure are positive
            if step_idx < failure_step:
                sample = {
                    "z_t": step_data["z_t"],
                    "a_t": step_data["a_t"],
                    "z_next": step_data["z_next"],
                    "predicate_embed": predicate_embeddings[step_idx],
                    "plan_summary": plan_summaries[step_idx],
                    "label_predicate": 1,
                    "label_feas": 1,
                    "label_nonint": 1,
                    "source": "pre_failure",
                    "step_idx": step_idx,
                }
                if step_metadata is not None and step_idx < len(step_metadata):
                    sample.update(step_metadata[step_idx])
                self.positive_samples.append(sample)
            
            # Failure step is negative
            elif step_idx == failure_step:
                sample = {
                    "z_t": step_data["z_t"],
                    "a_t": step_data["a_t"],
                    "z_next": step_data["z_next"],
                    "predicate_embed": predicate_embeddings[step_idx],
                    "plan_summary": plan_summaries[step_idx],
                    "label_predicate": 0 if failure_type == "predicate" else 1,
                    "label_feas": 0 if failure_type == "feasibility" else 1,
                    "label_nonint": 0 if failure_type == "noninterference" else 1,
                    "source": f"failure_{failure_type}",
                    "step_idx": step_idx,
                }
                if step_metadata is not None and step_idx < len(step_metadata):
                    sample.update(step_metadata[step_idx])
                self.negative_samples.append(sample)
    
    def generate_hard_negatives(
        self,
        positive_sample: Dict,
        perturbation_type: str = "wrong_predicate",
        num_samples: int = 3,
    ) -> List[Dict]:
        """
        Generate hard negative samples by perturbing positive samples.
        
        Args:
            positive_sample: A positive sample dict
            perturbation_type: Type of perturbation
                - "wrong_predicate": Change target predicate
                - "wrong_object": Swap object in action
                - "noise": Add noise to latent states
            num_samples: Number of negatives to generate
        
        Returns:
            List of negative samples
        """
        negatives = []
        
        for _ in range(num_samples):
            sample = positive_sample.copy()
            
            if perturbation_type == "wrong_predicate":
                # Randomize predicate embedding
                sample["predicate_embed"] = np.random.randn(self.predicate_embed_dim).astype(np.float32)
                sample["label_predicate"] = 0
                sample["source"] = "hard_negative_predicate"
            
            elif perturbation_type == "wrong_object":
                # Add noise to action (simulates wrong object selection)
                noise = np.random.randn(self.action_dim) * 0.1
                sample["a_t"] = sample["a_t"] + noise.astype(np.float32)
                sample["label_feas"] = 0
                sample["source"] = "hard_negative_action"
            
            elif perturbation_type == "noise":
                # Add noise to next state (simulates bad prediction)
                noise = np.random.randn(self.latent_dim) * 0.2
                sample["z_next"] = sample["z_next"] + noise.astype(np.float32)
                sample["label_predicate"] = 0
                sample["source"] = "hard_negative_noise"
            
            negatives.append(sample)
        
        return negatives
    
    def balance_dataset(
        self,
        negative_augmentation: bool = True,
        target_ratio: float = 1.0,
    ) -> List[Dict]:
        """
        Balance positive and negative samples.
        
        Args:
            negative_augmentation: If True, generate hard negatives to balance
            target_ratio: Target ratio of negative/positive samples
        
        Returns:
            Balanced dataset
        """
        print(f"Initial: {len(self.positive_samples)} positive, {len(self.negative_samples)} negative")
        
        if negative_augmentation:
            num_needed = int(len(self.positive_samples) * target_ratio) - len(self.negative_samples)
            
            if num_needed > 0:
                print(f"Generating {num_needed} hard negatives...")
                
                # Generate hard negatives from random positive samples
                for _ in range(num_needed):
                    pos_sample = np.random.choice(self.positive_samples)
                    perturbation = np.random.choice([
                        "wrong_predicate",
                        "wrong_object",
                        "noise"
                    ])
                    neg_samples = self.generate_hard_negatives(
                        pos_sample,
                        perturbation_type=perturbation,
                        num_samples=1,
                    )
                    self.negative_samples.extend(neg_samples)
        
        # Combine and shuffle
        all_samples = self.positive_samples + self.negative_samples
        np.random.shuffle(all_samples)
        
        print(f"Final: {len(self.positive_samples)} positive, {len(self.negative_samples)} negative")
        print(f"Total: {len(all_samples)} samples")
        
        return all_samples
    
    def save_dataset(self, save_path: str):
        """Save collected data to disk."""
        data = {
            "positive_samples": self.positive_samples,
            "negative_samples": self.negative_samples,
            "config": {
                "latent_dim": self.latent_dim,
                "action_dim": self.action_dim,
                "predicate_embed_dim": self.predicate_embed_dim,
                "plan_summary_dim": self.plan_summary_dim,
            }
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Saved dataset to {save_path}")
    
    def load_dataset(self, load_path: str):
        """Load collected data from disk."""
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        
        self.positive_samples = data["positive_samples"]
        self.negative_samples = data["negative_samples"]
        
        print(f"Loaded dataset from {load_path}")
        print(f"Positive: {len(self.positive_samples)}, Negative: {len(self.negative_samples)}")
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "num_positive": len(self.positive_samples),
            "num_negative": len(self.negative_samples),
            "total": len(self.positive_samples) + len(self.negative_samples),
            "positive_sources": defaultdict(int),
            "negative_sources": defaultdict(int),
        }
        
        for sample in self.positive_samples:
            stats["positive_sources"][sample["source"]] += 1
        
        for sample in self.negative_samples:
            stats["negative_sources"][sample["source"]] += 1
        
        return stats


def split_dataset(
    data: List[Dict],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train/val/test.
    
    Args:
        data: Full dataset
        train_split, val_split, test_split: Split ratios
        seed: Random seed
    
    Returns:
        (train_data, val_data, test_data)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6
    
    np.random.seed(seed)
    np.random.shuffle(data)
    
    n = len(data)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data


# Example usage
if __name__ == "__main__":
    # Initialize collector
    collector = CriticDataCollector()
    
    # Simulate adding data
    print("Example data collection workflow:")
    
    # Add successful trajectory
    dummy_trajectory = [
        {
            "z_t": np.random.randn(256).astype(np.float32),
            "a_t": np.random.randn(64).astype(np.float32),
            "z_next": np.random.randn(256).astype(np.float32),
        }
        for _ in range(5)
    ]
    
    dummy_predicates = [np.random.randn(128).astype(np.float32) for _ in range(5)]
    dummy_summaries = [np.random.randn(128).astype(np.float32) for _ in range(5)]
    
    collector.add_successful_trajectory(
        dummy_trajectory,
        dummy_predicates,
        dummy_summaries,
    )
    
    # Add failed trajectory
    collector.add_failed_trajectory(
        dummy_trajectory,
        dummy_predicates,
        dummy_summaries,
        failure_step=2,
        failure_type="predicate",
    )
    
    # Balance dataset
    balanced_data = collector.balance_dataset()
    
    # Get statistics
    stats = collector.get_statistics()
    print("\nDataset statistics:")
    print(f"Total samples: {stats['total']}")
    print(f"Positive: {stats['num_positive']}")
    print(f"Negative: {stats['num_negative']}")
