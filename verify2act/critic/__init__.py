"""
Verify2Act Critic Module

A multi-headed critic with uncertainty estimation for verifying
imagined rollouts from dynamics models in robotic manipulation.

Main components:
- CriticModel: Neural network architecture with ensemble uncertainty
- CriticInference: Reflection decision logic and diagnostics
- CriticTrainer: Training and calibration utilities
- VerifiedPlanner: Integration with LLM planner and dynamics model

Quick start:
    >>> from verify2act.critic import CriticConfig, build_critic, CriticInference
    >>> 
    >>> config = CriticConfig()
    >>> model = build_critic(config.model)
    >>> inference = CriticInference(model, config)

For detailed usage, see README_CRITIC.md
"""

__version__ = "0.1.0"

# Core components
from .critic_config import (
    CriticConfig,
    CriticModelConfig,
    CriticTrainingConfig,
    CriticThresholds,
    CalibrationTargets,
)

from .critic_model import (
    CriticModel,
    CriticEnsemble,
    build_critic,
)

from .critic_inference import (
    CriticInference,
    FailureReason,
    StepDiagnostics,
    TrajectoryDiagnostics,
)

from .critic_trainer import (
    CriticTrainer,
    CriticDataset,
    compute_calibration_metrics,
)

from .critic_data_collector import (
    CriticDataCollector,
    split_dataset,
)

from .critic_evaluator import (
    CriticEvaluator,
    run_full_evaluation,
)

from .verified_planner import (
    VerifiedPlanner,
)

__all__ = [
    # Config
    "CriticConfig",
    "CriticModelConfig",
    "CriticTrainingConfig",
    "CriticThresholds",
    "CalibrationTargets",
    
    # Model
    "CriticModel",
    "CriticEnsemble",
    "build_critic",
    
    # Inference
    "CriticInference",
    "FailureReason",
    "StepDiagnostics",
    "TrajectoryDiagnostics",
    
    # Training
    "CriticTrainer",
    "CriticDataset",
    "compute_calibration_metrics",
    
    # Data
    "CriticDataCollector",
    "split_dataset",
    
    # Evaluation
    "CriticEvaluator",
    "run_full_evaluation",
    
    # Integration
    "VerifiedPlanner",
]
