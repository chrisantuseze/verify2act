import sys
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

# Add rla-wm to path so we can import its modules
rla_wm_path = Path(__file__).resolve().parent.parent.parent / "rla-wm"
if str(rla_wm_path) not in sys.path:
    sys.path.append(str(rla_wm_path))

try:
    from src.models.attention_block import AttentionBlock, ModCrossAttentionBlock
    from src.models.sparse_structure_flow import TimestepEmbedder
except ImportError as e:
    print(f"Warning: Could not import rla-wm modules: {e}")

class BaselineRLAWM(nn.Module):
    """
    Baseline RLA-WM architecture adapted for the Verify2Act pipeline.

    Faithfully mirrors the actual RLA-WM design:
    - **Markovian inputs**: conditions on the single current frame $F_t$ only
      (no temporal history).
    - **Weak action grounding**: CLIP token sequence is mean-pooled to a single
      vector and used as an AdaLN/modulation signal to self-attention — no
      cross-attention over the token sequence.
    - **Compact latent flow space**: flow matching operates on
      ``(B, num_latent_tokens, token_dim)`` tokens produced by the shared
      frozen ``DeltaEncoder``, same as the main V2A-WM model.
      The difference between the baseline and V2A-WM is *only* in the
      conditioning stage, not the flow space.
    """
    def __init__(
        self,
        dino_channels: int = 1024,
        clip_channels: int = 512,
        model_channels: int = 1024,
        num_patches: int = 256,
        history_len: int = 3,      # API compat — unused; baseline is Markovian
        num_cond_blocks: int = 4,
        num_flow_blocks: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        # ── Compact latent space (shared with main V2A-WM, from DeltaEncoder) ──
        token_dim: int = 64,
        num_latent_tokens: int = 16,
        # ── Latent normalization (#1) ────────────────────────────────────────
        latent_scale: float = 10.0,
    ):
        super().__init__()
        self.dino_channels = dino_channels
        self.clip_channels = clip_channels
        self.model_channels = model_channels
        self.num_patches = num_patches
        self.token_dim = token_dim
        self.num_latent_tokens = num_latent_tokens
        self.latent_scale = float(latent_scale)
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # 1. Projections
        self.xt_proj = nn.Linear(dino_channels, model_channels)
        
        # To simulate weak action conditioning, we pool the CLIP sequence and project
        self.action_pool_proj = nn.Sequential(
            nn.Linear(clip_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )

        # 2. Stage 1: Conditioning Transformer (Baseline uses AdaLN, not Cross-Attention)
        self.cond_blocks = nn.ModuleList([
            AttentionBlock(
                channels=model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_fp16=use_fp16,
                cond_channels=model_channels,
                use_condition=True,
            ) for _ in range(num_cond_blocks)
        ])

        # 3. Stage 2: Flow Matching Transformer
        # Operates in compact latent token space (token_dim) — same as V2A-WM.
        # The baseline differs from V2A-WM *only* in the conditioning stage.
        self.flow_time_embedder = TimestepEmbedder(model_channels)
        self.flow_latent_proj = nn.Linear(token_dim, model_channels)   # token_dim → model_ch

        self.flow_blocks = nn.ModuleList([
            ModCrossAttentionBlock(
                channels=model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_channels=model_channels,
                use_fp16=use_fp16,
            ) for _ in range(num_flow_blocks)
        ])

        self.flow_norm_out = nn.LayerNorm(model_channels)
        self.flow_out_proj = nn.Linear(model_channels, token_dim)      # model_ch → token_dim

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.zeros_(self.flow_out_proj.weight)
        nn.init.zeros_(self.flow_out_proj.bias)

    def forward_cond(self, xt_history: Tensor, action_tokens: Tensor) -> Tensor:
        # 1. Drop history (Markovian amnesia)
        xt = xt_history[:, -1, :, :] # (B, P, C)
        
        # 2. Weak action conditioning (Pool sequence -> Single vector)
        pooled_action = action_tokens.mean(dim=1) # (B, 512)
        action_h = self.action_pool_proj(pooled_action) # (B, model_channels)
        
        # 3. Process conditioning blocks
        h = self.xt_proj(xt) # (B, P, model_channels)
        for block in self.cond_blocks:
            h = block(h, cond=action_h)
            
        return h # (B, P, model_channels)

    def forward_flow(
        self,
        cond_queries: Tensor,   # (B, P, model_channels) — from forward_cond
        noisy_latent: Tensor,   # (B, num_latent_tokens, token_dim) — compact latent space
        flow_t: Tensor,         # (B,)
    ) -> Tensor:
        """Predict velocity in compact latent token space."""
        flow_h = self.flow_latent_proj(noisy_latent.float())   # (B, N, model_ch)
        t_emb = self.flow_time_embedder(flow_t.float() * 1000) # (B, model_ch)

        for block in self.flow_blocks:
            flow_h = block(flow_h, mod=t_emb, cond=cond_queries)

        flow_h = self.flow_norm_out(flow_h)
        return self.flow_out_proj(flow_h)   # (B, N, token_dim)

    def forward(
        self,
        xt_history: Tensor,
        action_tokens: Tensor,
        noisy_latent: Tensor,   # (B, num_latent_tokens, token_dim)
        flow_t: Tensor,
    ) -> Tensor:
        cond = self.forward_cond(xt_history, action_tokens)
        return self.forward_flow(cond, noisy_latent, flow_t)

    @torch.no_grad()
    def step(
        self,
        xt_history: Tensor,
        action_tokens: Tensor,
        num_steps: int = 5,
    ) -> Tensor:
        """Inference rollout — returns predicted latent tokens.

        Mirrors the V2A-WM ``step()`` interface.  Returns
        ``(B, num_latent_tokens, token_dim)``; caller decodes via
        ``DeltaDecoder`` and adds ``F_t`` to recover ``F_{t+1}``.
        """
        B = xt_history.shape[0]
        device = xt_history.device

        cond = self.forward_cond(xt_history, action_tokens)

        # Sample in compact latent space
        x = torch.randn(
            B, self.num_latent_tokens, self.token_dim,
            device=device, dtype=self.dtype,
        )
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.ones(B, device=device) * (i / num_steps)
            v = self.forward_flow(cond, x, t)
            x = x + v * dt

        # Denormalize: ODE integrated in scaled-down space (÷ latent_scale),
        # so multiply back to get tokens in the original DeltaEncoder output space.
        return (x * self.latent_scale).float()
