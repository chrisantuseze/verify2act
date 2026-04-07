from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── ImageNet normalisation constants (shared) ──────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Probabilistic embedding container ─────────────────────────────────────────

class ProbEmbedding(NamedTuple):
    """Output of DINOv2DualHeadCritic.encode().

    mu       : [B, 768]  Mean embedding from the backbone.
    log_var1 : [B, 768]  Log-variance for Head 1 (goal proximity).
    log_var2 : [B, 768]  Log-variance for Head 2 (temporal consistency).

    Uncertainty interpretation
    --------------------------
    Each head maintains a separate log-variance network trained alongside the
    InfoNCE loss.  High σ (large log_var) means the backbone embedding is in a
    region of input space where that head is unreliable.  Use
    ``DINOv2DualHeadCritic.goal_sim_with_uncertainty()`` /
    ``temporal_sim_with_uncertainty()`` to propagate this uncertainty through
    Monte-Carlo sampling of the projected similarity.
    """

    mu: torch.Tensor        # [B, 768] backbone mean embedding
    log_var1: torch.Tensor  # [B, 768] log-variance for Head 1
    log_var2: torch.Tensor  # [B, 768] log-variance for Head 2


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
        Mean pool over patch dim → [B, 768]  (μ)
            │
        ┌──────────┬─────────────────┐
        Head 1     Head 2         LogVar Heads 1 & 2
        Goal       Temporal       768→256→768 each
        Prox.      Consist.       (clamped to [−4, 4])
        768→256    768→256        Uncertainty per head
        (L2-normalised)

    Probabilistic Embeddings (Option 2)
    ────────────────────────────────────
    encode() returns a ProbEmbedding(mu, log_var1, log_var2).

    During TRAINING — use sample_embed(mu, log_var) before project() to inject
    reparameterized noise:  z = μ + ε·exp(0.5·log_var),  ε ~ N(0, I)
    Add kl_loss(log_var) * kl_weight to the InfoNCE loss to regularise σ → 1.

    During INFERENCE — call goal_sim_with_uncertainty() or
    temporal_sim_with_uncertainty() with n_samples > 1.  These run n_samples MC
    draws and return (mean_sim, std_sim).  std_sim is the predictive uncertainty:
    high std → critic is unsure, skip replanning.

        # Cache goal embedding once per episode
        emb_goal = critic.encode(goal_img)           # ProbEmbedding

        # For each imagined step:
        emb_t  = critic.encode(img_t)                # ProbEmbedding
        emb_t1 = critic.encode(img_t1)               # ProbEmbedding
        mean_tc, std_tc   = critic.temporal_sim_with_uncertainty(emb_t, emb_t1)
        mean_prox, std_prox = critic.goal_sim_with_uncertainty(emb_H, emb_goal)

    ~86M backbone + ~800k head parameters (projection + log-var heads).
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
        def _make_proj_head(hidden: Optional[int]) -> nn.Module:
            if hidden is None:
                return nn.Linear(self.EMBED_DIM, self.HEAD_DIM, bias=False)
            return nn.Sequential(
                nn.Linear(self.EMBED_DIM, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.HEAD_DIM, bias=False),
            )

        self.head1 = _make_proj_head(head_hidden)   # goal proximity
        self.head2 = _make_proj_head(head_hidden)   # temporal consistency

        # ── Log-variance heads (probabilistic embeddings) ─────────────────
        # Output log σ² in the backbone embedding space [B, 768].
        # Final linear initialised to zero → log_var≈0, σ≈1 at init.
        def _make_logvar_head() -> nn.Module:
            mlp = nn.Sequential(
                nn.Linear(self.EMBED_DIM, 256),
                nn.GELU(),
                nn.Linear(256, self.EMBED_DIM),
            )
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
            return mlp

        self.log_var_head1 = _make_logvar_head()   # uncertainty for Head 1
        self.log_var_head2 = _make_logvar_head()   # uncertainty for Head 2

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

    def encode(self, img: torch.Tensor) -> "ProbEmbedding":
        """Encode a [-1, 1] image to a probabilistic DINOv2 patch embedding.

        Parameters
        ----------
        img : [B, 3, H, W]

        Returns
        -------
        ProbEmbedding
            .mu       : [B, 768]  mean embedding (NOT L2-normalised)
            .log_var1 : [B, 768]  log-variance for Head 1, clamped to [-4, 4]
            .log_var2 : [B, 768]  log-variance for Head 2, clamped to [-4, 4]
        """
        x = self._normalize(img.float())
        feats = self.backbone.forward_features(x)
        mu = feats["x_norm_patchtokens"].mean(dim=1)  # [B, 768]
        log_var1 = self.log_var_head1(mu).clamp(-4.0, 4.0)
        log_var2 = self.log_var_head2(mu).clamp(-4.0, 4.0)
        return ProbEmbedding(mu=mu, log_var1=log_var1, log_var2=log_var2)

    def project(self, embed: torch.Tensor, head: int) -> torch.Tensor:
        """Apply projection head and L2-normalise.

        Parameters
        ----------
        embed : [B, 768]  raw or sampled backbone embedding
        head  : 1 (goal proximity) or 2 (temporal consistency)

        Returns
        -------
        proj : [B, 256]  L2-normalised projection embedding
        """
        h = self.head1 if head == 1 else self.head2
        return F.normalize(h(embed), dim=-1)

    # ── Probabilistic helpers ──────────────────────────────────────────────────

    @staticmethod
    def sample_embed(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε · exp(0.5 · log_var).

        Parameters
        ----------
        mu      : [B, 768]  mean embedding
        log_var : [B, 768]  log-variance (clamped)

        Returns
        -------
        z : [B, 768]  sampled embedding
        """
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_loss(log_var: torch.Tensor) -> torch.Tensor:
        """KL regulariser that pulls σ toward 1 without pulling μ toward 0.

        KL(N(μ, σ²) ‖ N(μ, 1)) = 0.5 · (exp(log_var) − 1 − log_var)

        Returns a scalar averaged over all elements.  Equivalent to 0 when
        log_var == 0 (σ == 1).
        """
        return 0.5 * (log_var.exp() - 1.0 - log_var).mean()

    def goal_sim(self, emb_frame: "ProbEmbedding", emb_goal: "ProbEmbedding") -> torch.Tensor:
        """Deterministic cosine similarity via Head 1 using mean embeddings.

        Suitable for fast evaluation / AUROC computation.  Does NOT sample.
        Use ``goal_sim_with_uncertainty()`` at inference time.

        Parameters
        ----------
        emb_frame, emb_goal : ProbEmbedding  outputs of encode()

        Returns
        -------
        sim : [B]  cosine similarity in [-1, 1]
        """
        return (self.project(emb_frame.mu, 1) * self.project(emb_goal.mu, 1)).sum(dim=-1)

    def temporal_sim(self, emb_t: "ProbEmbedding", emb_t1: "ProbEmbedding") -> torch.Tensor:
        """Deterministic cosine similarity via Head 2 using mean embeddings.

        Suitable for fast evaluation.  Use ``temporal_sim_with_uncertainty()``
        at inference time.

        Parameters
        ----------
        emb_t, emb_t1 : ProbEmbedding  outputs of encode()

        Returns
        -------
        sim : [B]  cosine similarity in [-1, 1]
        """
        return (self.project(emb_t.mu, 2) * self.project(emb_t1.mu, 2)).sum(dim=-1)

    def goal_sim_with_uncertainty(
        self,
        emb_a: "ProbEmbedding",
        emb_b: "ProbEmbedding",
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte-Carlo cosine similarity and uncertainty via Head 1.

        Samples ``n_samples`` embeddings from the probabilistic distribution of
        each input and computes cosine similarity for each pair.  The standard
        deviation over samples is the predictive uncertainty.

        Parameters
        ----------
        emb_a, emb_b : ProbEmbedding
        n_samples    : int  number of MC samples (20 is adequate; 50 for high fidelity)

        Returns
        -------
        mean_sim : [B]  mean cosine similarity
        std_sim  : [B]  std of cosine similarity (predictive uncertainty)
        """
        sims = []
        for _ in range(n_samples):
            z_a = self.sample_embed(emb_a.mu, emb_a.log_var1)
            z_b = self.sample_embed(emb_b.mu, emb_b.log_var1)
            sim = (self.project(z_a, 1) * self.project(z_b, 1)).sum(dim=-1)
            sims.append(sim)
        sims = torch.stack(sims, dim=0)   # [n_samples, B]
        return sims.mean(dim=0), sims.std(dim=0)

    def temporal_sim_with_uncertainty(
        self,
        emb_t: "ProbEmbedding",
        emb_t1: "ProbEmbedding",
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte-Carlo cosine similarity and uncertainty via Head 2.

        Parameters
        ----------
        emb_t, emb_t1 : ProbEmbedding
        n_samples     : int  number of MC samples

        Returns
        -------
        mean_sim : [B]
        std_sim  : [B]  predictive uncertainty
        """
        sims = []
        for _ in range(n_samples):
            z_t  = self.sample_embed(emb_t.mu,  emb_t.log_var2)
            z_t1 = self.sample_embed(emb_t1.mu, emb_t1.log_var2)
            sim  = (self.project(z_t, 2) * self.project(z_t1, 2)).sum(dim=-1)
            sims.append(sim)
        sims = torch.stack(sims, dim=0)   # [n_samples, B]
        return sims.mean(dim=0), sims.std(dim=0)

    def forward(
        self,
        img_query: torch.Tensor,
        img_key: torch.Tensor,
        head: int = 1,
    ) -> torch.Tensor:
        """Convenience: encode two images and return deterministic cosine similarity.

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

