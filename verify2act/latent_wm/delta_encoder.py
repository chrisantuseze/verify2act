"""Delta Encoder / Decoder for the verify2act latent world model.

The ``DeltaEncoder`` mirrors the ``SimpleTokenTransformer`` design from RLA-WM.
It compresses a raw DINO-space feature difference ``F_{t+1} - F_t``  (shape
``[B, num_patches, dino_channels]``) into a small set of compact latent tokens
(shape ``[B, num_latent_tokens, token_dim]``) via a perceiver-style bottleneck:
learnable query tokens attend over the full patch sequence.

The ``DeltaDecoder`` inverts this mapping for pre-training:  given the compact
latent tokens it reconstructs the original difference via learned patch-position
queries that cross-attend to the latent token sequence.

Pre-training workflow
---------------------
1.  Run ``train_encoder.py`` to train Encoder + Decoder jointly with an MSE
    reconstruction loss on ``F_{t+1} - F_t`` samples.
2.  Save the encoder checkpoint.
3.  Load the frozen encoder in ``train_dynamics.py`` and flow-match in the
    low-dimensional latent token space (``[B, num_latent_tokens, token_dim]``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# rla-wm path setup — reuse AttentionBlock from the sibling repo
# ---------------------------------------------------------------------------
_rla_wm_path = Path(__file__).resolve().parent.parent.parent / "rla-wm"
if str(_rla_wm_path) not in sys.path:
    sys.path.insert(0, str(_rla_wm_path))

try:
    from src.models.attention_block import AttentionBlock
    _HAS_ATTN_BLOCK = True
except ImportError:
    _HAS_ATTN_BLOCK = False


# ---------------------------------------------------------------------------
# Fallback AttentionBlock — used only when rla-wm is not on the path
# ---------------------------------------------------------------------------

class _FallbackAttentionBlock(nn.Module):
    """Minimal pre-norm self-attention + FFN block (no conditioning)."""

    def __init__(self, channels: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


def _make_attn_block(channels: int, num_heads: int, mlp_ratio: float, use_fp16: bool):
    if _HAS_ATTN_BLOCK:
        return AttentionBlock(
            channels=channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_fp16=use_fp16,
        )
    return _FallbackAttentionBlock(channels, num_heads, mlp_ratio)


# ---------------------------------------------------------------------------
# Cross-Attention block (decoder only — patches query latent tokens)
# ---------------------------------------------------------------------------

class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + self-attention + FFN.

    ``x`` is the query sequence (patch positions).
    ``kv`` is the key/value sequence (latent tokens from the encoder).
    """

    def __init__(
        self,
        channels: int,
        kv_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm_sa = nn.LayerNorm(channels)
        self.norm_ca = nn.LayerNorm(channels)
        self.norm_ff = nn.LayerNorm(channels)

        self.self_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            channels,
            num_heads,
            kdim=kv_channels,
            vdim=kv_channels,
            batch_first=True,
        )
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )

    def forward(self, x: Tensor, kv: Tensor) -> Tensor:
        # self-attention over patch queries
        h = self.norm_sa(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h
        # cross-attention: patch queries ← latent keys/values
        h = self.norm_ca(x)
        h, _ = self.cross_attn(h, kv, kv, need_weights=False)
        x = x + h
        # FFN
        x = x + self.mlp(self.norm_ff(x))
        return x


# ---------------------------------------------------------------------------
# DeltaEncoder
# ---------------------------------------------------------------------------

class DeltaEncoder(nn.Module):
    """Compress ``F_{t+1} - F_t`` into compact latent tokens.

    Architecture (perceiver-style):
    1.  Project the ``num_patches`` DINO difference tokens to ``model_channels``.
    2.  Prepend ``num_latent_tokens`` learnable query tokens.
    3.  Apply ``num_blocks`` full self-attention blocks over the concatenated sequence.
    4.  Extract the first ``num_latent_tokens`` outputs and project to ``token_dim``.

    This exactly mirrors the ``SimpleTokenTransformer`` (``num_tokens > 0`` branch)
    used as the encoder in RLA-WM.

    Args:
        dino_channels:      Dimensionality of input DINO patch features (e.g. 768).
        model_channels:     Internal transformer width.
        token_dim:          Output dimensionality per latent token.
        num_latent_tokens:  Number of output latent tokens (the bottleneck width).
        num_blocks:         Number of self-attention + FFN layers.
        num_heads:          Attention heads.
        mlp_ratio:          FFN hidden / channels ratio.
        use_fp16:           Run the transformer torso in fp16.
        norm_output:        L2-normalise the output tokens (useful for contrastive loss).
    """

    def __init__(
        self,
        dino_channels: int = 768,
        model_channels: int = 512,
        token_dim: int = 64,
        num_latent_tokens: int = 16,
        num_blocks: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        norm_output: bool = False,
    ):
        super().__init__()
        self.dino_channels = dino_channels
        self.model_channels = model_channels
        self.token_dim = token_dim
        self.num_latent_tokens = num_latent_tokens
        self.norm_output = norm_output
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # Learnable bottleneck query tokens
        self.latent_tokens = nn.Parameter(
            torch.randn(num_latent_tokens, model_channels) * 0.02
        )

        # Project raw DINO diff patches to model width
        self.input_proj = nn.Linear(dino_channels, model_channels)

        # Self-attention backbone
        self.blocks = nn.ModuleList(
            [
                _make_attn_block(model_channels, num_heads, mlp_ratio, use_fp16)
                for _ in range(num_blocks)
            ]
        )

        self.norm_out = nn.LayerNorm(model_channels)
        # Project model_channels → token_dim
        self.out_proj = nn.Linear(model_channels, token_dim)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.normal_(self.latent_tokens, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init output projection for stable training start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------

    def forward(self, delta: Tensor) -> Tensor:
        """Encode a feature difference.

        Args:
            delta: ``(B, num_patches, dino_channels)`` — ``F_{t+1} - F_t``.

        Returns:
            latent: ``(B, num_latent_tokens, token_dim)``
        """
        B = delta.shape[0]
        # Project patches
        h_patches = self.input_proj(delta.float())  # (B, P, C)
        # Broadcast learnable query tokens
        q = self.latent_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, N, C)
        # Concatenate: queries first, patches second
        h = torch.cat([q, h_patches], dim=1)  # (B, N+P, C)
        h = h.to(self.dtype)

        for block in self.blocks:
            h = block(h)

        h = h.float()
        # Extract only the latent query positions
        latent = self.norm_out(h[:, : self.num_latent_tokens])  # (B, N, C)
        latent = self.out_proj(latent)  # (B, N, token_dim)
        if self.norm_output:
            latent = F.normalize(latent, dim=-1)
        return latent


# ---------------------------------------------------------------------------
# DeltaDecoder  (used only during encoder pre-training)
# ---------------------------------------------------------------------------

class DeltaDecoder(nn.Module):
    """Reconstruct ``F_{t+1} - F_t`` from compact latent tokens.

    Architecture:
    1.  Project latent tokens to ``model_channels`` (key/value sequence).
    2.  Use ``num_patches`` learnable patch-position queries (query sequence).
    3.  Apply ``num_blocks`` cross-attention blocks: patches ← latent tokens.
    4.  Project back to ``dino_channels``.

    Only used during the encoder pre-training stage.

    Args:
        token_dim:          Dimensionality of input latent tokens.
        model_channels:     Internal transformer width.
        dino_channels:      Output dimensionality (must match encoder input).
        num_patches:        Number of output patch positions.
        num_blocks:         Number of cross-attention layers.
        num_heads:          Attention heads.
        mlp_ratio:          FFN hidden / channels ratio.
    """

    def __init__(
        self,
        token_dim: int = 64,
        model_channels: int = 512,
        dino_channels: int = 768,
        num_patches: int = 256,
        num_blocks: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_patches = num_patches

        # Project latent tokens to model width (key/value)
        self.latent_proj = nn.Linear(token_dim, model_channels)

        # Learnable spatial patch queries
        self.patch_queries = nn.Parameter(
            torch.randn(num_patches, model_channels) * 0.02
        )

        self.blocks = nn.ModuleList(
            [
                _CrossAttentionBlock(model_channels, model_channels, num_heads, mlp_ratio)
                for _ in range(num_blocks)
            ]
        )

        self.norm_out = nn.LayerNorm(model_channels)
        self.out_proj = nn.Linear(model_channels, dino_channels)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.normal_(self.patch_queries, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, latent: Tensor) -> Tensor:
        """Decode latent tokens back to the DINO feature difference.

        Args:
            latent: ``(B, num_latent_tokens, token_dim)``

        Returns:
            recon: ``(B, num_patches, dino_channels)``
        """
        B = latent.shape[0]
        kv = self.latent_proj(latent.float())  # (B, N, C) — keys/values
        # Broadcast patch position queries
        x = self.patch_queries.unsqueeze(0).expand(B, -1, -1)  # (B, P, C)

        for block in self.blocks:
            x = block(x, kv)

        return self.out_proj(self.norm_out(x))  # (B, P, dino_channels)
