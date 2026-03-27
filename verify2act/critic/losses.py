from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PRMCriticLoss(nn.Module):
    """
    Binary cross-entropy on the Beta mean for discriminative training of the PRM critic.

    Why not Beta NLL?
    -----------------
    Beta NLL admits a degenerate global minimum for binary labels: the model learns
    alpha ≈ beta ≈ 0.5, producing an arcsine-shaped U distribution whose density is
    simultaneously high at *both* y ≈ 0 and y ≈ 1.  This satisfies the loss near
    optimally while providing no discriminative signal (mean = alpha/(alpha+beta) ≈ 0.5
    for every input, AUROC ≈ 0.5).  Beta NLL is not a proper scoring rule for binary
    outcomes expressed as interval labels.

    BCE on mean_feasibility is a proper scoring rule: its unique finite minimum over
    the model parameters is achieved when mean_feasibility equals the true conditional
    probability.  The Beta parameterisation still provides calibrated uncertainty at
    inference time (via alpha + beta, the concentration) — it is just not supervised
    directly during training, which is fine.

    Args:
        label_smoothing: Standard label smoothing applied uniformly to both classes,
            i.e. smoothed_label = label * (1 - ls) + 0.5 * ls.  Default 0.0.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.label_smoothing = label_smoothing

    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        labels: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            alpha: [B, 1] positive concentration (from softplus head).
            beta:  [B, 1] positive concentration (from softplus head).
            labels: [B] binary labels in {0, 1}.
            sample_weight: optional per-sample weights [B].

        Returns:
            Scalar loss.
        """
        mean_p = alpha / (alpha + beta)          # [B, 1], guaranteed in (0, 1)
        targets = labels.float().view(-1, 1)     # [B, 1]

        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = F.binary_cross_entropy(mean_p, targets, reduction="none")  # [B, 1]

        if sample_weight is not None:
            loss = loss * sample_weight.float().view(-1, 1)

        return loss.mean()


class BetaNLLLoss(nn.Module):
    """
    Negative log-likelihood for the Beta distribution.

    NOTE: do NOT use this as the primary training loss for binary classification.
    See PRMCriticLoss for the correct loss.  This class is retained for experiments
    and analysis only.
    """

    def __init__(self, label_smoothing: float = 0.01, clamp_eps: float = 1e-6):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.clamp_eps = clamp_eps

    def _smooth(self, y: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0:
            return y
        s = self.label_smoothing
        return y * (1.0 - 2.0 * s) + s

    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        labels: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        labels = labels.float().view(-1, 1)
        labels = self._smooth(labels)
        labels = labels.clamp(self.clamp_eps, 1.0 - self.clamp_eps)

        dist = torch.distributions.Beta(alpha, beta)
        nll = -dist.log_prob(labels)

        if sample_weight is not None:
            sample_weight = sample_weight.view(-1, 1)
            nll = nll * sample_weight

        return nll.mean()
