from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── ImageNet normalisation constants (shared) ──────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════
# DINOv2 Dual-Head Contrastive Critic (primary model)
# ══════════════════════════════════════════════════════════════════════════════

class DINOv2DualHeadCritic(nn.Module):
    """
    Contrastive goal-proximity and temporal-consistency critic.

    Architecture
    ────────────
        Image [B, 3, H, W]  (any size; resized to 224×224 internally)
            │
        DINOv2-B/14 backbone  (frozen initially; fully fine-tuned in phase 2)
            → patch tokens [B, 256, 768]  (256 = 16×16 patches for 224px)
            │
        Mean pool over patch dim → [B, 768]
            │
        ┌───┴───┐
        Head 1  Head 2
        Goal    Temporal
        Prox.   Consist.
        768→256 768→256
        (outputs L2-normalised for cosine similarity)

    Inference
    ─────────
        # Cache goal embedding once per episode
        e_goal = critic.encode(goal_img)                       # [1, 768]

        # For each imagined step:
        e_t  = critic.encode(img_t)                            # [1, 768]
        e_t1 = critic.encode(img_t1)                           # [1, 768]
        consistency = critic.temporal_sim(e_t, e_t1)           # scalar  ← abort if low
        proximity   = critic.goal_sim(e_H, e_goal)             # scalar  ← check at end

    ~86M backbone + ~400k head parameters.
    """

    EMBED_DIM = 768   # DINOv2-B/14 patch embedding dimension
    HEAD_DIM  = 256   # projection head output dimension

    def __init__(self, pretrained: bool = True, head_hidden: Optional[int] = None):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=pretrained,
        )

        # DINOv2 expects ImageNet-normalised input; we receive [-1, 1] images
        self.register_buffer(
            "img_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "img_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

        # ── Projection heads ──────────────────────────────────────────────────
        def _make_head(hidden: Optional[int]) -> nn.Module:
            if hidden is None:
                return nn.Linear(self.EMBED_DIM, self.HEAD_DIM, bias=False)
            return nn.Sequential(
                nn.Linear(self.EMBED_DIM, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.HEAD_DIM, bias=False),
            )

        self.head1 = _make_head(head_hidden)   # goal proximity (Head 1)
        self.head2 = _make_head(head_hidden)   # temporal consistency (Head 2)

        self.freeze_backbone()

    # ── Backbone freeze / unfreeze ─────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (heads remain trainable)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """Unfreeze entire backbone for full fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] image to DINOv2 ImageNet normalisation, resize to 224."""
        x = (x * 0.5 + 0.5).clamp(0.0, 1.0)               # [0, 1]
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.img_mean) / self.img_std

    # ── Public API ─────────────────────────────────────────────────────────────

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Encode a [-1, 1] image to a mean-pooled DINOv2 patch embedding.

        Parameters
        ----------
        img : [B, 3, H, W]

        Returns
        -------
        embed : [B, 768]  (NOT L2-normalised — normalise after projection)
        """
        x = self._normalize(img.float())
        feats = self.backbone.forward_features(x)
        patch_tokens = feats["x_norm_patchtokens"]   # [B, N, 768]
        return patch_tokens.mean(dim=1)               # [B, 768]

    def project(self, embed: torch.Tensor, head: int) -> torch.Tensor:
        """Apply projection head and L2-normalise.

        Parameters
        ----------
        embed : [B, 768]  raw patch-mean embedding from encode()
        head  : 1 (goal proximity) or 2 (temporal consistency)

        Returns
        -------
        proj : [B, 256]  L2-normalised projection embedding
        """
        h = self.head1 if head == 1 else self.head2
        return F.normalize(h(embed), dim=-1)

    def goal_sim(self, embed_frame: torch.Tensor, embed_goal: torch.Tensor) -> torch.Tensor:
        """Cosine similarity via Head 1 (goal proximity).

        Parameters
        ----------
        embed_frame, embed_goal : [B, 768]  outputs of encode()

        Returns
        -------
        sim : [B]  cosine similarity in [-1, 1]
        """
        return (self.project(embed_frame, 1) * self.project(embed_goal, 1)).sum(dim=-1)

    def temporal_sim(self, embed_t: torch.Tensor, embed_t1: torch.Tensor) -> torch.Tensor:
        """Cosine similarity via Head 2 (temporal consistency).

        Parameters
        ----------
        embed_t, embed_t1 : [B, 768]  outputs of encode()

        Returns
        -------
        sim : [B]  cosine similarity in [-1, 1]
        """
        return (self.project(embed_t, 2) * self.project(embed_t1, 2)).sum(dim=-1)

    def forward(
        self,
        img_query: torch.Tensor,
        img_key: torch.Tensor,
        head: int = 1,
    ) -> torch.Tensor:
        """Convenience: encode two images and return cosine similarity.

        Parameters
        ----------
        img_query, img_key : [B, 3, H, W]  in [-1, 1]
        head : 1 (goal proximity) or 2 (temporal consistency)

        Returns
        -------
        sim : [B]
        """
        e_q = self.encode(img_query)
        e_k = self.encode(img_key)
        fn = self.goal_sim if head == 1 else self.temporal_sim
        return fn(e_q, e_k)

