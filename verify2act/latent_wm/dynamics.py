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
    # Will fail if dependencies are completely missing, but useful for debugging.


class LatentDynamicsModel(nn.Module):
    """
    Hybrid VLM-Feature World Model: Latent Dynamics Core.
    
    Predicts Residual Latent Actions (\Delta F) using Conditional Flow Matching.
    Incorporates:
    - History Context: [F_{t-2}, F_{t-1}, F_t]
    - Cross-Attention Action Grounding: CLIP action tokens
    - Multi-Scale Spatial Resolution: Handles concatenated DINO features
    """

    def __init__(
        self,
        dino_channels: int = 768,       # Base DINO dim (e.g., 768 for ViT-B). Can be larger if multi-scale.
        clip_channels: int = 512,       # CLIP text embedding dim
        model_channels: int = 1024,     # Internal transformer dim
        num_patches: int = 256,         # e.g., 16x16 grid
        history_len: int = 3,           # F_{t-2}, F_{t-1}, F_t
        num_cond_blocks: int = 4,       # Blocks to process history and action
        num_flow_blocks: int = 6,       # Blocks for flow matching (velocity prediction)
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.dino_channels = dino_channels
        self.clip_channels = clip_channels
        self.model_channels = model_channels
        self.history_len = history_len
        self.num_patches = num_patches
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # 1. Projections for inputs
        self.history_proj = nn.Linear(dino_channels, model_channels)
        self.action_proj = nn.Linear(clip_channels, model_channels)

        # We add learned temporal embeddings to distinguish F_{t-2}, F_{t-1}, F_t
        self.temporal_emb = nn.Parameter(torch.zeros(1, history_len, 1, model_channels))
        # Spatial positional embeddings for the patches
        self.spatial_emb = nn.Parameter(torch.zeros(1, 1, num_patches, model_channels))

        # 2. Stage 1: Conditioning Transformer
        # Processes the history and cross-attends to the CLIP action tokens
        self.cond_blocks = nn.ModuleList([
            ModCrossAttentionBlock(
                channels=model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_channels=model_channels,
                use_fp16=use_fp16,
            ) for _ in range(num_cond_blocks)
        ])
        # Dummy mod embedding for conditioning stage (since ModCrossAttention needs it)
        # We can just use a constant vector or learnable parameter.
        self.cond_mod_emb = nn.Parameter(torch.zeros(1, model_channels))

        # 3. Stage 2: Flow Matching Transformer
        self.flow_time_embedder = TimestepEmbedder(model_channels)
        self.noisy_latent_proj = nn.Linear(dino_channels, model_channels)
        
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
        self.flow_out_proj = nn.Linear(model_channels, dino_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.temporal_emb, std=0.02)
        nn.init.normal_(self.spatial_emb, std=0.02)
        nn.init.zeros_(self.flow_out_proj.weight)
        nn.init.zeros_(self.flow_out_proj.bias)

    def forward_cond(
        self, 
        xt_history: Tensor,    # (B, history_len, num_patches, dino_channels)
        action_tokens: Tensor  # (B, seq_len, clip_channels)
    ) -> Tensor:
        """
        Processes history context and grounds it with the VLM action.
        """
        B = xt_history.shape[0]
        
        # Project inputs
        h = self.history_proj(xt_history)  # (B, H, P, C)
        a = self.action_proj(action_tokens) # (B, Seq, C)

        # Add spatio-temporal embeddings
        h = h + self.temporal_emb + self.spatial_emb
        
        # Flatten history for self-attention
        # (B, H*P, C)
        h = rearrange(h, 'b h p c -> b (h p) c')

        # Dummy modulation for cond blocks
        mod = self.cond_mod_emb.expand(B, -1)

        # Apply blocks (self-attention on history, cross-attention on action tokens)
        for block in self.cond_blocks:
            h = block(h, mod=mod, cond=a)

        return h  # (B, H*P, C)

    def forward_flow(
        self,
        cond_queries: Tensor,  # (B, H*P, C)
        noisy_latent: Tensor,  # (B, num_patches, dino_channels)
        flow_t: Tensor         # (B,) flow timestep [0, 1]
    ) -> Tensor:
        """
        Predicts the velocity of the residual given the conditioning.
        """
        # Project noisy latent
        x = self.noisy_latent_proj(noisy_latent)  # (B, P, C)
        
        # Add spatial embeddings to the noisy latent so it knows where patches are
        x = x + self.spatial_emb.squeeze(1)

        # Timestep embedding for modulation
        t_emb = self.flow_time_embedder(flow_t.float() * 1000)  # (B, C)

        # Flow blocks
        for block in self.flow_blocks:
            x = block(x, mod=t_emb, cond=cond_queries)
            
        x = self.flow_norm_out(x)
        return self.flow_out_proj(x)  # Predicted velocity (B, P, dino_channels)

    def forward(
        self,
        xt_history: Tensor,
        action_tokens: Tensor,
        noisy_latent: Tensor,
        flow_t: Tensor
    ) -> Tensor:
        """Training forward pass."""
        cond = self.forward_cond(xt_history, action_tokens)
        return self.forward_flow(cond, noisy_latent, flow_t)

    @torch.no_grad()
    def step(
        self, 
        xt_history: Tensor, 
        action_tokens: Tensor, 
        num_steps: int = 5
    ) -> Tensor:
        """
        Inference rollout step.
        Uses a simple Euler ODE solver to integrate the predicted velocity 
        and output the final residual \Delta F.
        
        Returns:
            predicted_F_{t+1}: (B, num_patches, dino_channels)
        """
        B = xt_history.shape[0]
        device = xt_history.device
        
        # Precompute conditioning once
        cond = self.forward_cond(xt_history, action_tokens)
        
        # Initialize noisy latent (start from pure noise for standard CFM)
        # In Residual Flow Matching, we map from Noise -> Residual (\Delta F)
        x = torch.randn(B, self.num_patches, self.dino_channels, device=device, dtype=self.dtype)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (i / num_steps)
            # Predict velocity
            v = self.forward_flow(cond, x, t)
            # Euler step
            x = x + v * dt
            
        # x is now the predicted residual \Delta F
        # Return F_t + \Delta F
        F_t = xt_history[:, -1, :, :]
        return F_t + x

