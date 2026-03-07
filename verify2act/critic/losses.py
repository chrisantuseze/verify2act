from typing import Optional

import torch
import torch.nn as nn


class BetaNLLLoss(nn.Module):
    """
    Negative log-likelihood for Beta distribution with optional soft-label smoothing.

    Labels are expected as {0,1}. They are smoothed and clamped to avoid log(0).
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
