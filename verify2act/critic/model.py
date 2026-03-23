from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Beta distribution helpers ──────────────────────────────────────────────────

@dataclass
class BetaOutputs:
    alpha: torch.Tensor
    beta: torch.Tensor
    mean_feasibility: torch.Tensor
    uncertainty: torch.Tensor


def beta_mean(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return alpha / (alpha + beta)


def beta_variance(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    denom = (alpha + beta) ** 2 * (alpha + beta + 1.0)
    return (alpha * beta) / denom


# ── CNN tokenizer ──────────────────────────────────────────────────────────────

class CNNTokenizer(nn.Module):
    """
    Maps a VAE latent [B, 4, 64, 64] to a spatial token sequence [B, N, token_dim]
    using two strided convolutions: 64 -> 32 -> 16, so N = 16*16 = 256.
    """

    def __init__(self, in_channels: int = 4, token_dim: int = 128):
        super().__init__()
        mid = token_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1),  # 64->32
            nn.GroupNorm(8, mid),
            nn.GELU(),
            nn.Conv2d(mid, token_dim, kernel_size=3, stride=2, padding=1),   # 32->16
            nn.GroupNorm(8, token_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, 64, 64] -> [B, token_dim, 16, 16] -> [B, 256, token_dim]
        feat = self.conv(x)
        B, C, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2)  # [B, H*W, C]


# ── Transformer building blocks ────────────────────────────────────────────────

class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention + residual + LayerNorm + FFN."""

    def __init__(self, token_dim: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * ffn_mult, token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm and residual
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention + residual + LayerNorm + FFN.

    Queries come from z_t1 tokens; keys/values from z_goal tokens.
    Each z_t1 spatial location attends to all z_goal spatial locations,
    which is exactly the relational reasoning needed for nut-peg alignment.
    """

    def __init__(self, token_dim: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(token_dim)
        self.norm_kv = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * ffn_mult, token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x_t1: torch.Tensor, x_goal: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x_t1)
        kv = self.norm_kv(x_goal)
        attn_out, _ = self.attn(q, kv, kv)
        x = x_t1 + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ── Main critic model ──────────────────────────────────────────────────────────

class SpatialBetaPRMCritic(nn.Module):
    """
    PRM critic with 2-layer transformer encoder operating on VAE latents.

    Pipeline:
        z_t1, z_goal  [B, 4, 64, 64]
            │
        Shared CNN tokenizer  →  z_t1_tokens, z_goal_tokens  [B, 256, token_dim]
            │
        diff_tokens = z_t1_tokens - z_goal_tokens  (spatial difference in token space)
        x_t1 = z_t1_tokens + diff_tokens           (residual: where does z_t1 deviate from goal?)
            │
        Layer 1: SelfAttentionBlock(x_t1)           (z_t1 tokens reason about themselves)
            │
        Layer 2: CrossAttentionBlock(x_t1, z_goal_tokens)  (z_t1 queries against goal)
            │
        Mean-pool over tokens  →  [B, token_dim]
            │
        Dual Beta head  →  alpha, beta, mean_feasibility, uncertainty

    VAE latent shapes:  z_t1 / z_goal  [B, 4, 64, 64]
    N tokens after CNN: 16 * 16 = 256
    """

    def __init__(
        self,
        latent_channels: int = 4,
        token_dim: int = 128,
        n_heads: int = 4,
        ffn_mult: int = 4,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert token_dim % n_heads == 0, "token_dim must be divisible by n_heads"

        # Shared tokenizer used for both z_t1 and z_goal
        self.tokenizer = CNNTokenizer(in_channels=latent_channels, token_dim=token_dim)

        # Layer 1: self-attention within z_t1 token sequence
        self.self_attn = SelfAttentionBlock(
            token_dim=token_dim, n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout
        )

        # Layer 2: cross-attention, z_t1 tokens attend to z_goal tokens
        self.cross_attn = CrossAttentionBlock(
            token_dim=token_dim, n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout
        )

        self.norm_out = nn.LayerNorm(token_dim)

        self.alpha_head = nn.Sequential(
            nn.Linear(token_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(token_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1),
        )

        self.softplus = nn.Softplus()
        self.eps = 1e-6

    def forward(self, z_t1: torch.Tensor, z_goal: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Tokenise both latents with the shared CNN
        t1_tokens = self.tokenizer(z_t1)       # [B, 256, token_dim]
        goal_tokens = self.tokenizer(z_goal)   # [B, 256, token_dim]

        # Use t1 tokens directly; cross-attention will handle the goal relationship.
        # (Avoid the 2*t1 - goal collapse from adding diff back onto t1.)
        x = t1_tokens                          # [B, 256, token_dim]

        # Layer 1: z_t1 tokens reason about themselves
        x = self.self_attn(x)

        # Layer 2: z_t1 tokens attend to goal tokens
        x = self.cross_attn(x, goal_tokens)

        # Pool over spatial tokens → single feature vector
        x = self.norm_out(x)
        x = x.mean(dim=1)                      # [B, token_dim]

        alpha = self.softplus(self.alpha_head(x)) + self.eps   # [B, 1]
        beta  = self.softplus(self.beta_head(x))  + self.eps   # [B, 1]

        return {
            "alpha": alpha,
            "beta": beta,
            "mean_feasibility": beta_mean(alpha, beta),
            "uncertainty": beta_variance(alpha, beta),
        }
