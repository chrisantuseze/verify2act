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

    Two-stage flow-matching model that mirrors RLA-WM's architecture while
    adding temporal history conditioning:

    Stage 1 — Conditioning (forward_cond):
        Processes a window of DINO history frames [F_{t-H+1}, ..., F_t]
        with cross-attention to CLIP action tokens.  Output: conditioning
        queries of shape ``(B, H*num_patches, model_channels)``.

    Stage 2 — Flow (forward_flow):
        Operates on the **compact latent token space** produced by a frozen
        ``DeltaEncoder`` (shape ``[B, num_latent_tokens, token_dim]``) rather
        than the raw DINO feature space.  This aligns with RLA-WM's approach
        of encoding ``F_{t+1} - F_t`` through a bottleneck before flow matching.

    Incorporates:
    - History Context: [F_{t-2}, F_{t-1}, F_t] (causal masking optional)
    - Cross-Attention Action Grounding: CLIP action tokens
    - Compact Latent Space: flow matching on DeltaEncoder tokens, not raw DINO
    - Sparsity Regularization: applied in raw DINO space (via decoder, optional)
    """

    def __init__(
        self,
        dino_channels: int = 1024,      # DINOv2 ViT-L/14 — matches RLA-WM (#8)
        clip_channels: int = 512,       # CLIP text embedding dim
        model_channels: int = 1024,     # Conditioning transformer internal dim
        num_patches: int = 256,         # Number of DINO patches (e.g. 16×16)
        history_len: int = 3,           # Frames in the conditioning history window
        num_cond_blocks: int = 4,       # Conditioning transformer depth
        num_flow_blocks: int = 6,       # Flow transformer depth
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        # ── Compact latent space (DeltaEncoder output) ──────────────────────
        token_dim: int = 64,            # Dim of each compact latent token
        num_latent_tokens: int = 16,    # Number of compact latent tokens
        # ── Latent normalization (#1) ────────────────────────────────────────
        latent_scale: float = 10.0,     # Matches RLA-WM latent_scalar_normalization
    ):
        super().__init__()
        self.dino_channels = dino_channels
        self.clip_channels = clip_channels
        self.model_channels = model_channels
        self.history_len = history_len
        self.num_patches = num_patches
        self.token_dim = token_dim
        self.num_latent_tokens = num_latent_tokens
        self.latent_scale = float(latent_scale)  # (#1) stored for step() denorm
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # ── Stage 1: Conditioning Transformer ────────────────────────────────
        # Receives the full temporal history [F_{t-H+1}, ..., F_t] plus CLIP
        # action tokens.  This stage is UNCHANGED from the original design —
        # temporal history conditioning is preserved as a core novelty.

        self.history_proj = nn.Linear(dino_channels, model_channels)
        self.action_proj  = nn.Linear(clip_channels,  model_channels)

        # Learned temporal and spatial positional embeddings
        self.temporal_emb = nn.Parameter(torch.zeros(1, history_len, 1, model_channels))
        self.spatial_emb  = nn.Parameter(torch.zeros(1, 1, num_patches, model_channels))

        self.cond_blocks = nn.ModuleList([
            ModCrossAttentionBlock(
                channels=model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_channels=model_channels,
                use_fp16=use_fp16,
            ) for _ in range(num_cond_blocks)
        ])
        # Learnable modulation seed for the conditioning blocks
        self.cond_mod_emb = nn.Parameter(torch.zeros(1, model_channels))

        # Learnable [START] token substituted for padded history slots
        # (B×H×num_patches×dino_channels — broadcasts over batch and time)
        self.start_token = nn.Parameter(torch.zeros(1, 1, num_patches, dino_channels))

        # ── Stage 2: Flow Matching Transformer ───────────────────────────────
        # Operates in the COMPACT LATENT SPACE produced by the frozen
        # DeltaEncoder.  Input/output shape: (B, num_latent_tokens, token_dim)
        # instead of the raw (B, num_patches, dino_channels) DINO space.
        # This aligns with RLA-WM's two-stage design.

        self.flow_time_embedder = TimestepEmbedder(model_channels)
        # Project compact tokens (token_dim) → model_channels
        self.noisy_latent_proj  = nn.Linear(token_dim, model_channels)

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
        # Output dim is token_dim (compact latent space), NOT dino_channels
        self.flow_out_proj = nn.Linear(model_channels, token_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.temporal_emb, std=0.02)
        nn.init.normal_(self.spatial_emb, std=0.02)
        nn.init.normal_(self.start_token, std=0.02)
        nn.init.zeros_(self.flow_out_proj.weight)
        nn.init.zeros_(self.flow_out_proj.bias)

    def forward_cond(
        self,
        xt_history: Tensor,                          # (B, H, num_patches, dino_channels)
        action_tokens: Tensor,                       # (B, seq_len, clip_channels)
        history_mask: Optional[Tensor] = None,       # (B, H) True=real, False=padded
    ) -> Tensor:
        """
        Processes history context and grounds it with the VLM action.
        If history_mask is provided, padded slots are replaced with a learned [START]
        token and are blocked from self-attention via key_padding_mask.
        """
        B, H = xt_history.shape[:2]

        # 1. Substitute padded history slots with the learnable [START] token
        if history_mask is not None:
            valid = history_mask[:, :, None, None].to(xt_history.device)  # (B, H, 1, 1)
            start = self.start_token.expand(B, H, -1, -1)
            xt_history = torch.where(valid, xt_history, start)

        # 2. Project inputs and add spatio-temporal embeddings
        h = self.history_proj(xt_history)   # (B, H, P, C)
        a = self.action_proj(action_tokens) # (B, Seq, C)
        h = h + self.temporal_emb + self.spatial_emb

        # 3. Flatten history for self-attention: (B, H*P, C)
        h = rearrange(h, 'b h p c -> b (h p) c')

        # 4. Build key_padding_mask for self-attention
        #    history_mask: (B, H); each frame covers num_patches patches.
        #    key_padding_mask: (B, H*P), True = ignore this key token.
        if history_mask is not None:
            patch_valid = history_mask.unsqueeze(-1).expand(-1, -1, self.num_patches)
            patch_valid = rearrange(patch_valid, 'b h p -> b (h p)')
            key_padding_mask = ~patch_valid  # invert: True means "mask out"
        else:
            key_padding_mask = None

        # 5. Dummy modulation for cond blocks
        mod = self.cond_mod_emb.expand(B, -1)

        # 6. Apply cond blocks: self-attention respects the mask
        for block in self.cond_blocks:
            h = block(h, mod=mod, cond=a, key_padding_mask=key_padding_mask)

        return h  # (B, H*P, C)

    def forward_flow(
        self,
        cond_queries: Tensor,  # (B, H*P, model_channels)
        noisy_latent: Tensor,  # (B, num_latent_tokens, token_dim)  ← compact latent space
        flow_t: Tensor,        # (B,) flow timestep in [0, 1]
    ) -> Tensor:
        """Predict the velocity field in the compact latent token space.

        The flow model now operates on the output space of the frozen
        ``DeltaEncoder`` — ``(B, num_latent_tokens, token_dim)`` — rather than
        the raw DINO feature difference space.  The conditioning queries from
        ``forward_cond`` (which embed full temporal history) are passed via
        cross-attention.

        Returns:
            v_pred: ``(B, num_latent_tokens, token_dim)`` — predicted velocity.
        """
        # Project compact latent tokens → model_channels
        x = self.noisy_latent_proj(noisy_latent.float())  # (B, N, C)

        # Timestep modulation embedding
        t_emb = self.flow_time_embedder(flow_t.float() * 1000)  # (B, C)

        x = x.to(self.dtype)
        for block in self.flow_blocks:
            x = block(x, mod=t_emb.to(self.dtype), cond=cond_queries)

        x = x.float()
        x = self.flow_norm_out(x)
        # Output: predicted velocity in token_dim space  (B, N, token_dim)
        return self.flow_out_proj(x)

    def forward(
        self,
        xt_history: Tensor,
        action_tokens: Tensor,
        noisy_latent: Tensor,
        flow_t: Tensor,
        history_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Training forward pass."""
        cond = self.forward_cond(xt_history, action_tokens, history_mask=history_mask)
        return self.forward_flow(cond, noisy_latent, flow_t)

    @torch.no_grad()
    def step(
        self,
        xt_history: Tensor,
        action_tokens: Tensor,
        num_steps: int = 5,
        history_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Inference rollout step — returns predicted latent tokens.

        Runs an Euler ODE integration in the compact latent token space
        (output of ``DeltaEncoder``) rather than the raw DINO feature space.
        The caller is responsible for decoding the predicted tokens back to
        DINO feature space using ``DeltaDecoder`` if ΔF is needed.

        Args:
            xt_history:    ``(B, H, num_patches, dino_channels)``
            action_tokens: ``(B, seq_len, clip_channels)``
            num_steps:     Euler ODE steps.
            history_mask:  ``(B, H)`` bool mask, ``True`` = real frame.

        Returns:
            pred_latent: ``(B, num_latent_tokens, token_dim)`` — predicted
                latent action tokens.  To get ΔF call
                ``DeltaDecoder.forward(pred_latent)``; to get ``F_{t+1}``
                add that to ``xt_history[:, -1]``.
        """
        B = xt_history.shape[0]
        device = xt_history.device

        # Conditioning computed once; temporal history is used here
        cond = self.forward_cond(xt_history, action_tokens, history_mask=history_mask)

        # Initialise in compact latent space (NOT DINO feature space)
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
        # The caller (DeltaDecoder / visualize_wm) expects raw encoder-scale tokens.
        return (x * self.latent_scale).float()

