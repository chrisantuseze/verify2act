"""Reflection mechanism — builds enriched critic failure context.

After the DINOv2DualHeadCritic rejects an imagined trajectory, this module
assembles the failure context that is passed to the VLM planner's reflect():
  1. Trajectory trend analysis  (``classify_failure_pattern``)
  2. Context assembly           (``build_reflection_context``)

Note: the diff-map spatial attribution and Grad-CAM overlay from the
legacy ResNetBetaPRMCritic are intentionally removed — DINOv2 is a ViT
without Conv2d layers, so Grad-CAM does not apply; the diffusion world
model does not expose a latent diff map in pixel space.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Layer 1 — Trajectory trend analysis
# ---------------------------------------------------------------------------

def classify_failure_pattern(
    consistency_scores: List[float],
    proximity_score: Optional[float],
) -> str:
    """Return a human-readable summary of how the rollout quality evolved.

    Parameters
    ----------
    consistency_scores : list of Head-2 cosine similarities for each imagined step
    proximity_score : Head-1 cosine similarity between final frame and goal,
                      or ``None`` if the rollout was aborted early (consistency failure).
    """
    if not consistency_scores:
        return "no imagined frames were evaluated"

    k = len(consistency_scores)

    if proximity_score is None:
        # Early abort path (temporal consistency failure)
        worst_idx = int(np.argmin(consistency_scores))
        worst_val = consistency_scores[worst_idx]
        return (
            f"the world model produced an incoherent transition at step {worst_idx} "
            f"(temporal consistency dropped to {worst_val:.3f}); "
            "this suggests a diffusion artifact rather than a plan error"
        )

    if all(s < 0.3 for s in consistency_scores):
        return (
            f"all {k} imagined frames had low temporal consistency; "
            "the world model may be sampling implausible trajectories"
        )

    if proximity_score < 0.3:
        return (
            f"the plan was physically plausible (mean tc={np.mean(consistency_scores):.3f}) "
            f"but the final imagined state is far from the goal (proximity={proximity_score:.3f}); "
            "the sequence of actions will not reach the target configuration"
        )

    # Intermediate proximity case
    delta = consistency_scores[-1] - consistency_scores[0] if k > 1 else 0.0
    kind = "improving" if delta > 0.05 else ("degrading" if delta < -0.05 else "stable")
    return (
        f"the plan is partially aligned (proximity={proximity_score:.3f}) but insufficient; "
        f"temporal consistency was {kind} across {k} steps "
        f"(scores: {[f'{s:.2f}' for s in consistency_scores]})"
    )


# ---------------------------------------------------------------------------
# Top-level context builder
# ---------------------------------------------------------------------------

def build_reflection_context(
    imagined_state: np.ndarray,
    all_scores: List[Tuple[float, float]],
    consistency_scores: List[float],
    proximity_score: Optional[float],
    failed_step: int,
    full_plan: List[str],
) -> Dict[str, Any]:
    """Assemble the reflection context dict consumed by VLMPlanner.reflect().

    Parameters
    ----------
    imagined_state : [H, W, 3] uint8 numpy array of the last imagined frame.
    all_scores : raw per-step score tuples for backward compatibility logging.
    consistency_scores : list of Head-2 temporal consistency scores (one per step).
    proximity_score : Head-1 goal proximity score for the final frame, or ``None``
                      if the rollout was aborted early.
    failed_step : index into ``full_plan`` where failure was detected.
    full_plan : full list of sub-skill strings generated for this rollout.
    """
    if failed_step < 0 or failed_step >= len(full_plan):
        raise IndexError(
            f"failed_step={failed_step} out of range for plan length {len(full_plan)}"
        )

    failure_pattern = classify_failure_pattern(consistency_scores, proximity_score)

    return {
        "imagined_state":     imagined_state,
        "all_scores":         all_scores,
        "consistency_scores": consistency_scores,
        "proximity_score":    proximity_score,
        "failure_pattern":    failure_pattern,
        "failed_step":        failed_step,
        "failed_action":      full_plan[failed_step],
        "full_plan":          full_plan,
    }

