"""
Verify2Act Critic Evaluation
Evaluation scripts for measuring critic performance.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from pathlib import Path

from .critic_config import CriticConfig
from .critic_model import CriticEnsemble
from .critic_inference import CriticInference, FailureReason
from .critic_trainer import compute_calibration_metrics


class CriticEvaluator:
    """Evaluates critic model performance."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: CriticConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device
        self.inference = CriticInference(model, config, device)
    
    def evaluate_head(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        head_name: str,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Evaluate a single head's performance.
        
        Args:
            predictions: Predicted probabilities [N]
            labels: Ground truth labels [N]
            head_name: Name of the head
            threshold: Decision threshold
        
        Returns:
            Dictionary of metrics
        """
        # Binary predictions
        pred_binary = (predictions >= threshold).astype(int)
        
        # Compute metrics
        metrics = {
            "head": head_name,
            "threshold": threshold,
            "accuracy": accuracy_score(labels, pred_binary),
            "precision": precision_score(labels, pred_binary, zero_division=0),
            "recall": recall_score(labels, pred_binary, zero_division=0),
            "f1": f1_score(labels, pred_binary, zero_division=0),
            "auc": roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0,
        }
        
        # Calibration
        calib_metrics = compute_calibration_metrics(predictions, labels)
        metrics.update(calib_metrics)
        
        # Confusion matrix
        cm = confusion_matrix(labels, pred_binary)
        metrics["confusion_matrix"] = cm.tolist()
        
        return metrics
    
    def evaluate_dataset(
        self,
        test_data: List[Dict],
        batch_size: int = 256,
    ) -> Dict:
        """
        Evaluate model on test dataset.
        
        Args:
            test_data: Test dataset
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary of metrics per head
        """
        self.model.eval()
        
        # Collect predictions
        all_predictions = {
            "predicate": [],
            "feas": [],
            "nonint": [],
        }
        
        all_labels = {
            "predicate": [],
            "feas": [],
            "nonint": [],
        }
        
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i + batch_size]
                
                # Prepare batch tensors
                z_t = torch.stack([torch.from_numpy(s["z_t"]) for s in batch]).to(self.device)
                a_t = torch.stack([torch.from_numpy(s["a_t"]) for s in batch]).to(self.device)
                z_next = torch.stack([torch.from_numpy(s["z_next"]) for s in batch]).to(self.device)
                pred_embed = torch.stack([torch.from_numpy(s["predicate_embed"]) for s in batch]).to(self.device)
                plan_sum = torch.stack([torch.from_numpy(s["plan_summary"]) for s in batch]).to(self.device)
                
                # Forward pass
                if isinstance(self.model, CriticEnsemble):
                    outputs = self.model(z_t, a_t, z_next, pred_embed, plan_sum, return_uncertainty=True)
                else:
                    outputs = self.model(z_t, a_t, z_next, pred_embed, plan_sum)
                
                # Store predictions and labels
                if "p_predicate" in outputs:
                    all_predictions["predicate"].extend(outputs["p_predicate"].cpu().numpy())
                    all_labels["predicate"].extend([s["label_predicate"] for s in batch])
                
                if "p_feas" in outputs:
                    all_predictions["feas"].extend(outputs["p_feas"].cpu().numpy())
                    all_labels["feas"].extend([s["label_feas"] for s in batch])
                
                if "p_nonint" in outputs:
                    all_predictions["nonint"].extend(outputs["p_nonint"].cpu().numpy())
                    all_labels["nonint"].extend([s["label_nonint"] for s in batch])
        
        # Evaluate each head
        results = {}
        
        for head_name in ["predicate", "feas", "nonint"]:
            if all_predictions[head_name]:
                preds = np.array(all_predictions[head_name])
                labels = np.array(all_labels[head_name])
                
                results[head_name] = self.evaluate_head(
                    preds, labels, head_name, threshold=0.5
                )
        
        return results
    
    def evaluate_reflection_decisions(
        self,
        test_trajectories: List[List[Dict]],
        ground_truth_failures: List[bool],
    ) -> Dict:
        """
        Evaluate reflection decision quality.
        
        Args:
            test_trajectories: List of trajectories (each is list of step dicts)
            ground_truth_failures: List of bools indicating if each trajectory truly failed
        
        Returns:
            Metrics for reflection decisions
        """
        predicted_failures = []
        
        for traj_data in test_trajectories:
            # Convert to inference format
            trajectory = []
            for step in traj_data:
                trajectory.append({
                    "z_t": torch.from_numpy(step["z_t"]).unsqueeze(0).to(self.device),
                    "a_t": torch.from_numpy(step["a_t"]).unsqueeze(0).to(self.device),
                    "z_next": torch.from_numpy(step["z_next"]).unsqueeze(0).to(self.device),
                    "predicate_embed": torch.from_numpy(step["predicate_embed"]).unsqueeze(0).to(self.device),
                    "plan_summary": torch.from_numpy(step["plan_summary"]).unsqueeze(0).to(self.device),
                    "target_predicate": step.get("target_predicate", ""),
                    "predicted_predicates": step.get("predicted_predicates"),
                })
            
            # Evaluate trajectory
            traj_diag = self.inference.evaluate_trajectory(trajectory)
            predicted_failures.append(traj_diag.should_reflect)
        
        # Compute metrics
        predicted_failures = np.array(predicted_failures).astype(int)
        ground_truth_failures = np.array(ground_truth_failures).astype(int)
        
        metrics = {
            "accuracy": accuracy_score(ground_truth_failures, predicted_failures),
            "precision": precision_score(ground_truth_failures, predicted_failures, zero_division=0),
            "recall": recall_score(ground_truth_failures, predicted_failures, zero_division=0),
            "f1": f1_score(ground_truth_failures, predicted_failures, zero_division=0),
            "confusion_matrix": confusion_matrix(ground_truth_failures, predicted_failures).tolist(),
        }
        
        return metrics
    
    def calibration_plot(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        head_name: str,
        save_path: Optional[str] = None,
        n_bins: int = 10,
    ):
        """
        Generate calibration (reliability) plot.
        
        Args:
            predictions: Predicted probabilities [N]
            labels: Ground truth labels [N]
            head_name: Name of the head
            save_path: Path to save plot
            n_bins: Number of bins
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_confidences.append(predictions[mask].mean())
                bin_accuracies.append(labels[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_confidences.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Actual calibration
        ax.bar(
            bin_confidences,
            bin_accuracies,
            width=1.0 / n_bins,
            alpha=0.7,
            edgecolor='black',
            label='Model predictions',
        )
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Calibration Plot - {head_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved calibration plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_report(self, results: Dict):
        """Print formatted evaluation report."""
        print("\n" + "="*80)
        print("CRITIC EVALUATION REPORT")
        print("="*80)
        
        for head_name, metrics in results.items():
            print(f"\n{head_name.upper()} HEAD:")
            print("-" * 40)
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  AUC:       {metrics['auc']:.4f}")
            print(f"  ECE:       {metrics['ece']:.4f}")
            print(f"  MCE:       {metrics['mce']:.4f}")
            
            if "confusion_matrix" in metrics:
                cm = np.array(metrics["confusion_matrix"])
                print(f"\n  Confusion Matrix:")
                print(f"    TN={cm[0,0]:<6} FP={cm[0,1]:<6}")
                print(f"    FN={cm[1,0]:<6} TP={cm[1,1]:<6}")
        
        print("\n" + "="*80)


def run_full_evaluation(
    model_path: str,
    test_data: List[Dict],
    config: CriticConfig,
    save_dir: Optional[str] = None,
):
    """
    Run complete evaluation pipeline.
    
    Args:
        model_path: Path to trained model checkpoint
        test_data: Test dataset
        config: Critic configuration
        save_dir: Directory to save plots and reports
    """
    # Load model
    model = CriticEnsemble(config.model)
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    
    # Create evaluator
    evaluator = CriticEvaluator(model, config, config.device)
    
    # Evaluate
    print("Evaluating critic model...")
    results = evaluator.evaluate_dataset(test_data)
    
    # Print report
    evaluator.print_report(results)
    
    # Generate calibration plots
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # TODO: Generate plots per head
        # Would need to re-run inference to get per-sample predictions
    
    # Save results
    if save_dir:
        import json
        results_path = Path(save_dir) / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {results_path}")
    
    return results


# Example usage
if __name__ == "__main__":
    from .critic_config import CriticConfig
    
    print("Example evaluation workflow:")
    print("1. Load trained model")
    print("2. Prepare test data")
    print("3. Run evaluation")
    print("4. Generate reports and plots")
