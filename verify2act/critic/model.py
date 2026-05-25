from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# CLIP text encoder is loaded lazily to avoid hard dependency at import time.
# We use openai/clip for the text encoder (768-dim → HEAD_DIM projection).
_CLIP_TEXT_DIM = 512   # CLIP ViT-B/32 text embedding dimension

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

    EMBED_DIM     = 768   # DINOv2-B/14 patch embedding dimension
    HEAD_DIM      = 256   # projection head output dimension
    CLIP_TEXT_DIM = 512   # CLIP ViT-B/32 text embedding dimension

    def __init__(self, pretrained: bool = True, head_hidden: Optional[int] = None, load_backbone: bool = True, dino_channels: int = 1024):
        super().__init__()
        self.dino_channels = dino_channels

        # ── Backbone ──────────────────────────────────────────────────────────
        if load_backbone:
            backbone_name = "dinov2_vitl14" if self.dino_channels == 1024 else "dinov2_vitb14"
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                backbone_name,
                pretrained=pretrained,
            )
        else:
            self.backbone = None

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
                return nn.Linear(self.dino_channels, self.HEAD_DIM, bias=False)
            return nn.Sequential(
                nn.Linear(self.dino_channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.HEAD_DIM, bias=False),
            )

        self.head1 = _make_proj_head(head_hidden)   # goal proximity (visual side)
        self.head2 = _make_proj_head(head_hidden)   # temporal consistency

        # ── CLIP text-goal projection (Head 1 language side) ──────────────
        # Projects a frozen CLIP text embedding (512-dim) into the same
        # HEAD_DIM space as head1, enabling language goals to be scored
        # against DINOv2 predicted terminal states via cosine similarity.
        # This is the only component that bridges the CLIP ↔ DINOv2 spaces.
        self.clip_goal_proj = nn.Sequential(
            nn.Linear(self.CLIP_TEXT_DIM, self.dino_channels),
            nn.GELU(),
            nn.Linear(self.dino_channels, self.HEAD_DIM, bias=False),
        )

        # ── Log-variance heads (probabilistic embeddings) ─────────────────
        # Output log σ² in the backbone embedding space [B, dino_channels].
        # Final linear initialised to zero → log_var≈0, σ≈1 at init.
        def _make_logvar_head() -> nn.Module:
            mlp = nn.Sequential(
                nn.Linear(self.dino_channels, 256),
                nn.GELU(),
                nn.Linear(256, self.dino_channels),
            )
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
            return mlp

        self.log_var_head1 = _make_logvar_head()   # uncertainty for Head 1
        self.log_var_head2 = _make_logvar_head()   # uncertainty for Head 2

        # CLIP model loaded lazily via encode_text_goal()
        self._clip_model = None
        self._clip_tokenizer = None

        self.freeze_backbone()

    # ── Backbone freeze / unfreeze ─────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (heads remain trainable)."""
        if self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """Unfreeze entire backbone for full fine-tuning."""
        if self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad_(True)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] image to DINOv2 ImageNet normalisation, resize to 224."""
        x = (x * 0.5 + 0.5).clamp(0.0, 1.0)               # [0, 1]
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.img_mean) / self.img_std

    # ── CLIP lazy loader ───────────────────────────────────────────────────────

    def _load_clip(self, device: Optional[torch.device] = None) -> None:
        """Load the CLIP ViT-B/32 model and tokenizer on first call."""
        if self._clip_model is not None:
            return
        try:
            import clip as openai_clip
        except ImportError as e:
            raise ImportError(
                "openai-clip is required for language-goal encoding. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            ) from e
        clip_device = device or next(self.parameters()).device
        self._clip_model, _ = openai_clip.load("ViT-B/32", device=clip_device)
        self._clip_model.eval()
        for p in self._clip_model.parameters():
            p.requires_grad_(False)
        self._clip_tokenizer = openai_clip.tokenize

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
        if self.backbone is None:
            raise RuntimeError("Backbone is not loaded. Cannot encode raw images. Use encode_features instead.")
        x = self._normalize(img.float())
        feats = self.backbone.forward_features(x)
        return self.encode_features(feats["x_norm_patchtokens"])

    def encode_features(self, patch_tokens: torch.Tensor) -> "ProbEmbedding":
        """Bypasses the backbone and evaluates predicted DINO patches directly.

        WARNING: The input must be raw DINOv2 patch tokens of shape [B, 256, 768]
        (where 256 is the number of patches and 768 is the embedding dimension).
        Do NOT pass DeltaEncoder compact tokens (which are shape [B, 16, 64]).

        Parameters
        ----------
        patch_tokens : [B, num_patches, 768]  (e.g., from LatentDynamicsModel)

        Returns
        -------
        ProbEmbedding
        """
        mu = patch_tokens.mean(dim=1)  # [B, 768]
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

    def encode_text_goal(self, text: str) -> torch.Tensor:
        """Encode a language goal string into a HEAD_DIM embedding via CLIP + clip_goal_proj.

        The returned embedding is L2-normalised and lies in the same space as
        ``project(emb.mu, head=1)``, so cosine similarity can be used directly.

        Parameters
        ----------
        text : str  natural language goal (e.g. "push the slider right")

        Returns
        -------
        goal_emb : [1, HEAD_DIM]  L2-normalised text-goal embedding
        """
        device = next(self.parameters()).device
        self._load_clip(device)
        with torch.no_grad():
            tokens = self._clip_tokenizer([text], truncate=True).to(device)
            clip_emb = self._clip_model.encode_text(tokens).float()   # [1, 512]
        return F.normalize(self.clip_goal_proj(clip_emb), dim=-1)     # [1, HEAD_DIM]

    def encode_text_goals(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of language goal strings into a HEAD_DIM embedding via CLIP + clip_goal_proj in a single batch.

        Parameters
        ----------
        texts : List[str]  natural language goals

        Returns
        -------
        goal_embs : [N, HEAD_DIM]  L2-normalised text-goal embeddings
        """
        if not texts:
            return torch.empty(0, self.HEAD_DIM, device=next(self.parameters()).device)
        device = next(self.parameters()).device
        self._load_clip(device)
        with torch.no_grad():
            tokens = self._clip_tokenizer(texts, truncate=True).to(device)
            clip_emb = self._clip_model.encode_text(tokens).float()   # [N, 512]
        return F.normalize(self.clip_goal_proj(clip_emb), dim=-1)     # [N, HEAD_DIM]

    def goal_sim_from_text(
        self,
        emb_frame: "ProbEmbedding",
        text_goal: str,
    ) -> torch.Tensor:
        """Deterministic cosine similarity between a predicted DINOv2 frame embedding
        and a language goal, using the CLIP projection head.

        Parameters
        ----------
        emb_frame : ProbEmbedding  output of encode() or encode_features()
        text_goal : str  language instruction

        Returns
        -------
        sim : [B]  cosine similarity in [-1, 1]
        """
        goal_emb = self.encode_text_goal(text_goal)   # [1, HEAD_DIM]
        frame_emb = self.project(emb_frame.mu, 1)     # [B, HEAD_DIM]
        return (frame_emb * goal_emb).sum(dim=-1)     # [B]

    def goal_sim_from_text_with_uncertainty(
        self,
        emb_frame: "ProbEmbedding",
        text_goal: str,
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte-Carlo version of goal_sim_from_text for uncertainty estimation.

        Only the *frame* embedding is sampled (the text goal is deterministic).

        Returns
        -------
        mean_sim : [B]
        std_sim  : [B]  predictive uncertainty
        """
        goal_emb = self.encode_text_goal(text_goal)   # [1, HEAD_DIM]
        sims = []
        for _ in range(n_samples):
            z = self.sample_embed(emb_frame.mu, emb_frame.log_var1)   # [B, 768]
            proj = self.project(z, 1)                                  # [B, HEAD_DIM]
            sims.append((proj * goal_emb).sum(dim=-1))                # [B]
        sims = torch.stack(sims, dim=0)   # [n_samples, B]
        return sims.mean(dim=0), sims.std(dim=0)

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

    def ddp_train_forward(
        self,
        all_imgs: torch.Tensor,
        mask0: torch.Tensor,
        mask1: torch.Tensor,
        has_lang_mask: Optional[torch.Tensor] = None,
        valid_texts: Optional[List[str]] = None,
        use_cached: bool = False,
    ):
        """Unified, DDP-safe forward pass for the contrastive and alignment training step."""
        # 1. Encode all images
        if use_cached:
            all_pe = self.encode_features(all_imgs)
        else:
            all_pe = self.encode(all_imgs)
            
        B = all_imgs.size(0) // 3
        # Split mean embeddings
        mu_anchor   = all_pe.mu[:B]
        mu_positive = all_pe.mu[B:2*B]
        mu_negative = all_pe.mu[2*B:]
        # Split log-variances per head
        lv1_anchor   = all_pe.log_var1[:B]
        lv1_positive = all_pe.log_var1[B:2*B]
        lv1_negative = all_pe.log_var1[2*B:]
        lv2_anchor   = all_pe.log_var2[:B]
        lv2_positive = all_pe.log_var2[B:2*B]
        lv2_negative = all_pe.log_var2[2*B:]
        
        n_gp = int(mask0.sum().item())
        n_tc = int(mask1.sum().item())
        
        # 2. Head 1 Projections
        a0, p0, n0, visual_proj = None, None, None, None
        use_lang = has_lang_mask is not None and int(has_lang_mask.sum().item()) >= 1
        
        if n_gp > 1 or use_lang:
            head1_inputs = []
            if n_gp > 1:
                z_a0 = self.sample_embed(mu_anchor[mask0],   lv1_anchor[mask0])
                z_p0 = self.sample_embed(mu_positive[mask0], lv1_positive[mask0])
                z_n0 = self.sample_embed(mu_negative[mask0], lv1_negative[mask0])
                head1_inputs.extend([z_a0, z_p0, z_n0])
            if use_lang:
                head1_inputs.append(mu_positive[has_lang_mask])
                
            head1_inputs_cat = torch.cat(head1_inputs, dim=0)
            head1_proj_cat = self.project(head1_inputs_cat, head=1)
            
            offset = 0
            if n_gp > 1:
                a0 = head1_proj_cat[offset : offset + n_gp]
                offset += n_gp
                p0 = head1_proj_cat[offset : offset + n_gp]
                offset += n_gp
                n0 = head1_proj_cat[offset : offset + n_gp]
                offset += n_gp
            if use_lang:
                n_lang = int(has_lang_mask.sum().item())
                visual_proj = head1_proj_cat[offset : offset + n_lang]
                
        # 3. Head 2 Projections
        a1, p1, n1 = None, None, None
        if n_tc > 1:
            z_a1 = self.sample_embed(mu_anchor[mask1],   lv2_anchor[mask1])
            z_p1 = self.sample_embed(mu_positive[mask1], lv2_positive[mask1])
            z_n1 = self.sample_embed(mu_negative[mask1], lv2_negative[mask1])
            
            head2_inputs_cat = torch.cat([z_a1, z_p1, z_n1], dim=0)
            head2_proj_cat = self.project(head2_inputs_cat, head=2)
            
            a1 = head2_proj_cat[0 : n_tc]
            p1 = head2_proj_cat[n_tc : 2*n_tc]
            n1 = head2_proj_cat[2*n_tc :]
            
        # 4. Language Goal Projections
        lang_proj = None
        if use_lang and valid_texts is not None:
            lang_proj = self.encode_text_goals(valid_texts)
            
        return {
            "a0": a0, "p0": p0, "n0": n0,
            "a1": a1, "p1": p1, "n1": n1,
            "visual_proj": visual_proj,
            "lang_proj": lang_proj,
            "lv1_anchor": lv1_anchor, "lv1_positive": lv1_positive, "lv1_negative": lv1_negative,
            "lv2_anchor": lv2_anchor, "lv2_positive": lv2_positive, "lv2_negative": lv2_negative,
        }

    def forward(
        self,
        *args,
        mode: str = "default",
        **kwargs,
    ):
        if mode == "default":
            img_query, img_key = args[0], args[1]
            head = kwargs.get("head", 1)
            e_q = self.encode(img_query)
            e_k = self.encode(img_key)
            fn = self.goal_sim if head == 1 else self.temporal_sim
            return fn(e_q, e_k)
        elif mode == "encode":
            return self.encode(*args, **kwargs)
        elif mode == "encode_features":
            return self.encode_features(*args, **kwargs)
        elif mode == "project":
            return self.project(*args, **kwargs)
        elif mode == "encode_text_goals":
            return self.encode_text_goals(*args, **kwargs)
        elif mode == "kl_loss":
            return self.kl_loss(*args, **kwargs)
        elif mode == "ddp_train_step":
            return self.ddp_train_forward(*args, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

