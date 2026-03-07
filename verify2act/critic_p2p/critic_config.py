"""
Verify2Act Critic Configuration
Contains all hyperparameters, thresholds, and configuration for the critic model.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class CriticThresholds:
    """Thresholds for reflection decisions per head."""
    
    # Predicate head thresholds
    predicate_hard_fail_mu: float = 0.35
    predicate_uncertainty_fail_sigma: float = 0.15
    predicate_uncertainty_fail_entropy: float = 0.55
    predicate_soft_fail_mu_low: float = 0.35
    predicate_soft_fail_mu_high: float = 0.55
    predicate_soft_fail_sigma: float = 0.10
    
    # Feasibility head thresholds
    feasibility_hard_fail_mu: float = 0.30
    feasibility_uncertainty_fail_sigma: float = 0.12
    feasibility_uncertainty_fail_entropy: float = 0.50
    feasibility_soft_fail_mu_low: float = 0.30
    feasibility_soft_fail_mu_high: float = 0.55
    feasibility_soft_fail_sigma: float = 0.08
    
    # Non-interference head thresholds
    noninterference_hard_fail_mu: float = 0.40
    noninterference_uncertainty_fail_sigma: float = 0.10
    noninterference_uncertainty_fail_entropy: float = 0.45
    noninterference_soft_fail_mu_low: float = 0.40
    noninterference_soft_fail_mu_high: float = 0.60
    noninterference_soft_fail_sigma: float = 0.07


@dataclass
class CriticModelConfig:
    """Configuration for the critic model architecture."""
    
    # Input dimensions
    latent_dim: int = 256  # Dimension of z_t, z_{t+1}
    action_dim: int = 64   # Dimension of a_t
    predicate_embed_dim: int = 128  # Dimension of predicate embedding
    plan_summary_dim: int = 128  # Dimension of remaining plan summary
    
    # Encoder architecture
    encoder_type: str = "mlp"  # "mlp" or "transformer"
    encoder_hidden_dims: list = field(default_factory=lambda: [512, 512, 256])
    encoder_activation: str = "relu"
    encoder_dropout: float = 0.1
    
    # Head architecture
    head_hidden_dims: list = field(default_factory=lambda: [128, 64])
    head_activation: str = "relu"
    head_dropout: float = 0.1
    
    # Ensemble settings
    ensemble_size: int = 5
    use_mc_dropout: bool = False  # If False, use deep ensemble
    mc_dropout_samples: int = 20
    
    # Active heads (phased implementation)
    use_predicate_head: bool = True
    use_feasibility_head: bool = False  # Phase 2
    use_noninterference_head: bool = False  # Phase 3


@dataclass
class CriticTrainingConfig:
    """Configuration for critic training."""
    
    # Loss weights
    loss_weight_predicate: float = 1.0
    loss_weight_feasibility: float = 0.5
    loss_weight_noninterference: float = 0.5
    
    # Training hyperparameters
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Optimizer
    optimizer: str = "adam"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "cosine", "step", or "none"
    lr_decay_steps: int = 10
    lr_decay_gamma: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    
    # Calibration
    calibration_method: str = "temperature"  # "temperature", "platt", or "none"
    calibration_split: float = 0.2  # Fraction of validation for calibration
    
    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 5


@dataclass
class CalibrationTargets:
    """Target metrics for calibration."""
    
    # Predicate head
    predicate_precision_min: float = 0.70
    predicate_recall_min: float = 0.80
    predicate_ece_max: float = 0.05
    
    # Feasibility head
    feasibility_precision_min: float = 0.75
    feasibility_recall_min: float = 0.85
    
    # Non-interference head
    noninterference_precision_min: float = 0.70
    noninterference_recall_min: float = 0.80
    
    # F-beta for threshold tuning (favor recall)
    f_beta: float = 1.5


@dataclass
class CriticConfig:
    """Master configuration combining all configs."""
    
    model: CriticModelConfig = field(default_factory=CriticModelConfig)
    training: CriticTrainingConfig = field(default_factory=CriticTrainingConfig)
    thresholds: CriticThresholds = field(default_factory=CriticThresholds)
    calibration_targets: CalibrationTargets = field(default_factory=CalibrationTargets)
    
    # Paths
    checkpoint_dir: str = "./checkpoints/critic"
    log_dir: str = "./logs/critic"
    data_dir: str = "./data/critic"
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 42
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "thresholds": self.thresholds.__dict__,
            "calibration_targets": self.calibration_targets.__dict__,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "data_dir": self.data_dir,
            "device": self.device,
            "seed": self.seed,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "CriticConfig":
        """Create config from dictionary."""
        return cls(
            model=CriticModelConfig(**config_dict.get("model", {})),
            training=CriticTrainingConfig(**config_dict.get("training", {})),
            thresholds=CriticThresholds(**config_dict.get("thresholds", {})),
            calibration_targets=CalibrationTargets(**config_dict.get("calibration_targets", {})),
            checkpoint_dir=config_dict.get("checkpoint_dir", "./checkpoints/critic"),
            log_dir=config_dict.get("log_dir", "./logs/critic"),
            data_dir=config_dict.get("data_dir", "./data/critic"),
            device=config_dict.get("device", "cuda"),
            seed=config_dict.get("seed", 42),
        )
