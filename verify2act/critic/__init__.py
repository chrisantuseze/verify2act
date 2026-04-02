"""Verify2Act critic module — DINOv2 dual-head contrastive critic.

Utilities:
- train_contrastive.py: two-phase contrastive training
- diagnose_dino.py: zero-shot CLS vs patch-mean diagnostic
- calibrate_thresholds.py: post-training theta_p/theta_c calibration
"""

from .model import DINOv2DualHeadCritic
from .losses import InfoNCELoss
from .inference import (
    CriticDecision,
    check_rollout_consistency,
    decide_from_proximity,
)
from verify2act.data_loader import (
    ContrastiveRow,
    ContrastivePairDataset,
    build_contrastive_datasets,
)

__all__ = [
    # Models
    "DINOv2DualHeadCritic",
    # Losses
    "InfoNCELoss",
    # Inference / decisions
    "CriticDecision",
    "check_rollout_consistency",
    "decide_from_proximity",
    # Datasets
    "ContrastiveRow",
    "ContrastivePairDataset",
    "build_contrastive_datasets",
]
