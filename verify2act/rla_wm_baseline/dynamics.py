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
    Uses:
    - Markovian inputs (current frame only, history ignored)
    - Weak Action Grounding (mean-pooled text tokens via AdaLN modulation)
    - Self-Attention for conditioning, ModCrossAttention for flow matching
    """
    def __init__(
        self,
        dino_channels: int = 768,
        clip_channels: int = 512,
        model_channels: int = 1024,
        num_patches: int = 256,
        history_len: int = 3,  # Present for API compatibility, but unused
        num_cond_blocks: int = 4,
        num_flow_blocks: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.dino_channels = dino_channels
        self.clip_channels = clip_channels
        self.model_channels = model_channels
        self.num_patches = num_patches
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
        self.flow_time_embedder = TimestepEmbedder(model_channels)
        self.flow_latent_proj = nn.Linear(dino_channels, model_channels)
        
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

    def forward_flow(self, cond_queries: Tensor, noisy_latent: Tensor, flow_t: Tensor) -> Tensor:
        flow_h = self.flow_latent_proj(noisy_latent)
        t_emb = self.flow_time_embedder(flow_t.float() * 1000)
        
        for block in self.flow_blocks:
            flow_h = block(flow_h, mod=t_emb, cond=cond_queries)
            
        flow_h = self.flow_norm_out(flow_h)
        return self.flow_out_proj(flow_h)

    def forward(self, xt_history: Tensor, action_tokens: Tensor, noisy_latent: Tensor, flow_t: Tensor) -> Tensor:
        cond = self.forward_cond(xt_history, action_tokens)
        return self.forward_flow(cond, noisy_latent, flow_t)

    @torch.no_grad()
    def step(self, xt_history: Tensor, action_tokens: Tensor, num_steps: int = 5) -> Tensor:
        B = xt_history.shape[0]
        device = xt_history.device
        
        cond = self.forward_cond(xt_history, action_tokens)
        
        x = torch.randn(B, self.num_patches, self.dino_channels, device=device, dtype=self.dtype)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (i / num_steps)
            v = self.forward_flow(cond, x, t)
            x = x + v * dt
            
        F_t = xt_history[:, -1, :, :]
        return F_t + x
