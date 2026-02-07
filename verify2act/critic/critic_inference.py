"""
Verify2Act Critic Inference
Uncertainty computation and reflection decision logic.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .critic_config import CriticConfig, CriticThresholds
from .critic_model import CriticEnsemble, CriticModel


class FailureReason(Enum):
    """Types of failures detected by the critic."""
    PREDICATE_HARD_FAIL = "predicate_hard_fail"
    PREDICATE_UNCERTAINTY_FAIL = "predicate_uncertainty_fail"
    PREDICATE_SOFT_FAIL = "predicate_soft_fail"
    FEASIBILITY_HARD_FAIL = "feasibility_hard_fail"
    FEASIBILITY_UNCERTAINTY_FAIL = "feasibility_uncertainty_fail"
    FEASIBILITY_SOFT_FAIL = "feasibility_soft_fail"
    NONINTERFERENCE_HARD_FAIL = "noninterference_hard_fail"
    NONINTERFERENCE_UNCERTAINTY_FAIL = "noninterference_uncertainty_fail"
    NONINTERFERENCE_SOFT_FAIL = "noninterference_soft_fail"
    NO_FAILURE = "no_failure"


@dataclass
class StepDiagnostics:
    """Diagnostics for a single step in the trajectory."""
    step_idx: int
    
    # Predictions
    p_predicate: Optional[float] = None
    p_feas: Optional[float] = None
    p_nonint: Optional[float] = None
    
    # Uncertainty metrics
    predicate_var: Optional[float] = None
    predicate_entropy: Optional[float] = None
    feas_var: Optional[float] = None
    feas_entropy: Optional[float] = None
    nonint_var: Optional[float] = None
    nonint_entropy: Optional[float] = None
    
    # Target information
    target_predicate: Optional[str] = None
    predicted_predicates: Optional[Dict] = None
    
    # Failure information
    failure_reason: FailureReason = FailureReason.NO_FAILURE
    should_reflect: bool = False


@dataclass
class TrajectoryDiagnostics:
    """Diagnostics for an entire trajectory."""
    steps: List[StepDiagnostics]
    terminal_score: float
    should_reflect: bool
    first_failure_step: Optional[int] = None
    failure_reason: Optional[FailureReason] = None


class CriticInference:
    """
    Handles inference and uncertainty computation for the critic model.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: CriticConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device
        self.thresholds = config.thresholds
        
        self.model.to(device)
        self.model.eval()
    
    def evaluate_step(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_next: torch.Tensor,
        predicate_embed: torch.Tensor,
        plan_summary: torch.Tensor,
        target_predicate: str,
        predicted_predicates: Optional[Dict] = None,
    ) -> StepDiagnostics:
        """
        Evaluate a single step and compute diagnostics.
        
        Args:
            z_t, a_t, z_next, predicate_embed, plan_summary: Input tensors
            target_predicate: String description of target predicate
            predicted_predicates: Predicates decoded from z_next
        
        Returns:
            StepDiagnostics with predictions and failure information
        """
        with torch.no_grad():
            # Get predictions with uncertainty
            if isinstance(self.model, CriticEnsemble):
                if self.config.model.use_mc_dropout:
                    outputs = self.model.forward_mc_dropout(
                        z_t, a_t, z_next, predicate_embed, plan_summary,
                        n_samples=self.config.model.mc_dropout_samples,
                    )
                else:
                    outputs = self.model(
                        z_t, a_t, z_next, predicate_embed, plan_summary,
                        return_uncertainty=True,
                    )
            else:
                # Single model - no uncertainty
                outputs = self.model(z_t, a_t, z_next, predicate_embed, plan_summary)
        
        # Create diagnostics
        diag = StepDiagnostics(step_idx=-1)  # Will be set by caller
        diag.target_predicate = target_predicate
        diag.predicted_predicates = predicted_predicates
        
        # Extract predictions
        if "p_predicate" in outputs:
            diag.p_predicate = outputs["p_predicate"].item()
            if "p_predicate_var" in outputs:
                diag.predicate_var = outputs["p_predicate_var"].item()
            if "p_predicate_entropy" in outputs:
                diag.predicate_entropy = outputs["p_predicate_entropy"].item()
        
        if "p_feas" in outputs:
            diag.p_feas = outputs["p_feas"].item()
            if "p_feas_var" in outputs:
                diag.feas_var = outputs["p_feas_var"].item()
            if "p_feas_entropy" in outputs:
                diag.feas_entropy = outputs["p_feas_entropy"].item()
        
        if "p_nonint" in outputs:
            diag.p_nonint = outputs["p_nonint"].item()
            if "p_nonint_var" in outputs:
                diag.nonint_var = outputs["p_nonint_var"].item()
            if "p_nonint_entropy" in outputs:
                diag.nonint_entropy = outputs["p_nonint_entropy"].item()
        
        # Check for failures
        diag.failure_reason, diag.should_reflect = self._check_failure(diag)
        
        return diag
    
    def _check_failure(self, diag: StepDiagnostics) -> Tuple[FailureReason, bool]:
        """
        Check if step should trigger reflection based on thresholds.
        
        Returns:
            (failure_reason, should_reflect)
        """
        thresholds = self.thresholds
        
        # Check predicate head
        if diag.p_predicate is not None:
            mu = diag.p_predicate
            sigma = np.sqrt(diag.predicate_var) if diag.predicate_var is not None else 0.0
            entropy = diag.predicate_entropy if diag.predicate_entropy is not None else 0.0
            
            # Hard fail
            if mu < thresholds.predicate_hard_fail_mu:
                return FailureReason.PREDICATE_HARD_FAIL, True
            
            # Uncertainty fail
            if (sigma > thresholds.predicate_uncertainty_fail_sigma or
                entropy > thresholds.predicate_uncertainty_fail_entropy):
                return FailureReason.PREDICATE_UNCERTAINTY_FAIL, True
            
            # Soft fail
            if (thresholds.predicate_soft_fail_mu_low <= mu < thresholds.predicate_soft_fail_mu_high and
                sigma > thresholds.predicate_soft_fail_sigma):
                return FailureReason.PREDICATE_SOFT_FAIL, True
        
        # Check feasibility head
        if diag.p_feas is not None:
            mu = diag.p_feas
            sigma = np.sqrt(diag.feas_var) if diag.feas_var is not None else 0.0
            entropy = diag.feas_entropy if diag.feas_entropy is not None else 0.0
            
            # Hard fail
            if mu < thresholds.feasibility_hard_fail_mu:
                return FailureReason.FEASIBILITY_HARD_FAIL, True
            
            # Uncertainty fail
            if (sigma > thresholds.feasibility_uncertainty_fail_sigma or
                entropy > thresholds.feasibility_uncertainty_fail_entropy):
                return FailureReason.FEASIBILITY_UNCERTAINTY_FAIL, True
            
            # Soft fail
            if (thresholds.feasibility_soft_fail_mu_low <= mu < thresholds.feasibility_soft_fail_mu_high and
                sigma > thresholds.feasibility_soft_fail_sigma):
                return FailureReason.FEASIBILITY_SOFT_FAIL, True
        
        # Check non-interference head
        if diag.p_nonint is not None:
            mu = diag.p_nonint
            sigma = np.sqrt(diag.nonint_var) if diag.nonint_var is not None else 0.0
            entropy = diag.nonint_entropy if diag.nonint_entropy is not None else 0.0
            
            # Hard fail
            if mu < thresholds.noninterference_hard_fail_mu:
                return FailureReason.NONINTERFERENCE_HARD_FAIL, True
            
            # Uncertainty fail
            if (sigma > thresholds.noninterference_uncertainty_fail_sigma or
                entropy > thresholds.noninterference_uncertainty_fail_entropy):
                return FailureReason.NONINTERFERENCE_UNCERTAINTY_FAIL, True
            
            # Soft fail
            if (thresholds.noninterference_soft_fail_mu_low <= mu < thresholds.noninterference_soft_fail_mu_high and
                sigma > thresholds.noninterference_soft_fail_sigma):
                return FailureReason.NONINTERFERENCE_SOFT_FAIL, True
        
        return FailureReason.NO_FAILURE, False
    
    def evaluate_trajectory(
        self,
        trajectory_data: List[Dict],
    ) -> TrajectoryDiagnostics:
        """
        Evaluate an entire trajectory and compute diagnostics.
        
        Args:
            trajectory_data: List of dicts with keys:
                - z_t, a_t, z_next, predicate_embed, plan_summary (tensors)
                - target_predicate (str)
                - predicted_predicates (dict, optional)
        
        Returns:
            TrajectoryDiagnostics with step-level and trajectory-level info
        """
        step_diagnostics = []
        
        for step_idx, step_data in enumerate(trajectory_data):
            diag = self.evaluate_step(
                z_t=step_data["z_t"],
                a_t=step_data["a_t"],
                z_next=step_data["z_next"],
                predicate_embed=step_data["predicate_embed"],
                plan_summary=step_data["plan_summary"],
                target_predicate=step_data["target_predicate"],
                predicted_predicates=step_data.get("predicted_predicates"),
            )
            diag.step_idx = step_idx
            step_diagnostics.append(diag)
        
        # Compute terminal score (average of all predicate scores)
        predicate_scores = [d.p_predicate for d in step_diagnostics if d.p_predicate is not None]
        terminal_score = np.mean(predicate_scores) if predicate_scores else 0.0
        
        # Find first failure
        first_failure_step = None
        failure_reason = None
        should_reflect = False
        
        for diag in step_diagnostics:
            if diag.should_reflect:
                should_reflect = True
                if first_failure_step is None:
                    first_failure_step = diag.step_idx
                    failure_reason = diag.failure_reason
                break
        
        return TrajectoryDiagnostics(
            steps=step_diagnostics,
            terminal_score=terminal_score,
            should_reflect=should_reflect,
            first_failure_step=first_failure_step,
            failure_reason=failure_reason,
        )
    
    def aggregate_failure_analysis(
        self,
        all_trajectories: List[TrajectoryDiagnostics],
    ) -> Dict:
        """
        Aggregate failure patterns across multiple trajectories.
        
        Args:
            all_trajectories: List of trajectory diagnostics
        
        Returns:
            Dictionary with aggregated failure statistics
        """
        from collections import defaultdict
        
        analysis = {
            "num_trajectories": len(all_trajectories),
            "num_failed": sum(1 for t in all_trajectories if t.should_reflect),
            "failure_step_counts": defaultdict(int),
            "failure_reason_counts": defaultdict(int),
            "most_common_failure_step": None,
            "most_common_failure_reason": None,
            "average_terminal_score": np.mean([t.terminal_score for t in all_trajectories]),
        }
        
        for traj in all_trajectories:
            if traj.should_reflect and traj.first_failure_step is not None:
                analysis["failure_step_counts"][traj.first_failure_step] += 1
                analysis["failure_reason_counts"][traj.failure_reason.value] += 1
        
        if analysis["failure_step_counts"]:
            analysis["most_common_failure_step"] = max(
                analysis["failure_step_counts"],
                key=analysis["failure_step_counts"].get
            )
        
        if analysis["failure_reason_counts"]:
            analysis["most_common_failure_reason"] = max(
                analysis["failure_reason_counts"],
                key=analysis["failure_reason_counts"].get
            )
        
        return analysis
    
    def generate_reflection_prompt(
        self,
        primitive_plan: List[str],
        failure_analysis: Dict,
        trajectory_diagnostics: Optional[TrajectoryDiagnostics] = None,
    ) -> str:
        """
        Generate a targeted reflection prompt based on failure analysis.
        
        Args:
            primitive_plan: Original plan as list of primitive strings
            failure_analysis: Aggregated failure statistics
            trajectory_diagnostics: Optional single trajectory for detailed context
        
        Returns:
            Reflection prompt string
        """
        failed_step = failure_analysis.get("most_common_failure_step")
        failed_reason = failure_analysis.get("most_common_failure_reason")
        num_failed = failure_analysis.get("num_failed", 0)
        num_total = failure_analysis.get("num_trajectories", 0)
        
        prompt = f"""The plan failed during dynamics model verification.

Original Plan: {primitive_plan}

Failure Analysis:
- {num_failed} out of {num_total} sampled trajectories failed
"""
        
        if failed_step is not None:
            failed_primitive = primitive_plan[failed_step] if failed_step < len(primitive_plan) else "unknown"
            prompt += f"- Most failures occurred at step {failed_step + 1}: \"{failed_primitive}\"\n"
        
        if failed_reason is not None:
            prompt += f"- Primary failure reason: {failed_reason}\n"
        
        # Add specific guidance based on failure type
        if failed_reason and "predicate" in failed_reason:
            prompt += "\nThe target predicate was not satisfied in imagination. Consider:\n"
            prompt += "1. Is there a missing prerequisite step before this action?\n"
            prompt += "2. Is the target object in the correct state for this action?\n"
            prompt += "3. Should the predicate be relaxed or changed?\n"
        
        elif failed_reason and "feasibility" in failed_reason:
            prompt += "\nThe action appears to be infeasible. Consider:\n"
            prompt += "1. Is the object reachable from the current state?\n"
            prompt += "2. Are there collision issues?\n"
            prompt += "3. Should a different primitive be used?\n"
        
        elif failed_reason and "noninterference" in failed_reason:
            prompt += "\nThis action may interfere with future plan execution. Consider:\n"
            prompt += "1. Should objects be manipulated in a different order?\n"
            prompt += "2. Is this action blocking access to other required objects?\n"
            prompt += "3. Should intermediate placement locations be used?\n"
        
        prompt += "\nPlease suggest an alternative approach that addresses these issues."
        
        return prompt
