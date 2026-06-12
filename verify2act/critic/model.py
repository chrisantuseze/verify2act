from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class ProbEmbedding(NamedTuple):
    """Probabilistic backbone embedding returned by encode() / encode_features().

    mu       : [B, D]  mean embedding (NOT L2-normalised)
    log_var1 : [B, D]  log-variance for Head 1 (goal proximity), clamped to [-4, 4]
    log_var2 : [B, D]  log-variance for Head 2 (temporal consistency), clamped to [-4, 4]
    """
    mu: torch.Tensor
    log_var1: torch.Tensor
    log_var2: torch.Tensor


class DINOv2DualHeadCritic(nn.Module):
    """Contrastive goal-proximity and temporal-consistency critic built on DINOv2.

    Two projection heads operate on the mean-pooled DINOv2 patch embeddings:
      Head 1 — goal proximity  (visual + optional language alignment)
      Head 2 — temporal consistency

    Each head has a companion log-variance MLP so that encode() returns a
    ProbEmbedding.  Use goal_sim / temporal_sim for fast AUROC-style evaluation,
    or *_with_uncertainty for Monte-Carlo predictive uncertainty at inference time.
    """

    HEAD_DIM      = 256
    CLIP_TEXT_DIM = 512

    def __init__(
        self,
        pretrained: bool = True,
        head_hidden: Optional[int] = None,
        load_backbone: bool = True,
        dino_channels: int = 1024,
    ):
        super().__init__()
        self.dino_channels = dino_channels

        # Backbone
        if load_backbone:
            backbone_name = "dinov2_vitl14" if dino_channels == 1024 else "dinov2_vitb14"
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", backbone_name, pretrained=pretrained
            )
        else:
            self.backbone = None

        self.register_buffer(
            "img_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "img_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

        # Projection heads
        def _make_proj_head(hidden: Optional[int]) -> nn.Module:
            if hidden is None:
                return nn.Linear(dino_channels, self.HEAD_DIM, bias=False)
            return nn.Sequential(
                nn.Linear(dino_channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.HEAD_DIM, bias=False),
            )

        self.head1 = _make_proj_head(head_hidden)  # goal proximity (visual)
        self.head2 = _make_proj_head(head_hidden)  # temporal consistency

        # CLIP text-goal projection: CLIP-dim -> HEAD_DIM (bridges CLIP ↔ DINOv2)
        self.clip_goal_proj = nn.Sequential(
            nn.Linear(self.CLIP_TEXT_DIM, dino_channels),
            nn.GELU(),
            nn.Linear(dino_channels, self.HEAD_DIM, bias=False),
        )

        # Log-variance heads for probabilistic embeddings; zero-init → σ≈1 at start
        def _make_logvar_head() -> nn.Module:
            mlp = nn.Sequential(
                nn.Linear(dino_channels, 256),
                nn.GELU(),
                nn.Linear(256, dino_channels),
            )
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
            return mlp

        self.log_var_head1 = _make_logvar_head()  # uncertainty for Head 1
        self.log_var_head2 = _make_logvar_head()  # uncertainty for Head 2

        # CLIP model loaded lazily
        self._clip_model = None
        self._clip_tokenizer = None

        self.freeze_backbone()

    # Backbone freeze / unfreeze

    def freeze_backbone(self) -> None:
        if self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        if self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad_(True)

    # Internal helpers

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Expect [0,1] images; apply ImageNet normalisation and resize to 224."""
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.img_mean) / self.img_std

    def _load_clip(self, device: Optional[torch.device] = None) -> None:
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

    # Public encode API

    def encode(self, img: torch.Tensor) -> ProbEmbedding:
        """Encode a [0,1] image tensor [B, 3, H, W] into a ProbEmbedding."""
        if self.backbone is None:
            raise RuntimeError("Backbone not loaded. Use encode_features() for cached features.")
        x = self._normalize(img.float())
        feats = self.backbone.forward_features(x)
        return self.encode_features(feats["x_norm_patchtokens"])

    def encode_features(self, patch_tokens: torch.Tensor) -> ProbEmbedding:
        """Mean-pool raw DINOv2 patch tokens [B, num_patches, D] into a ProbEmbedding."""
        mu = patch_tokens.mean(dim=1)
        log_var1 = self.log_var_head1(mu).clamp(-4.0, 4.0)
        log_var2 = self.log_var_head2(mu).clamp(-4.0, 4.0)
        return ProbEmbedding(mu=mu, log_var1=log_var1, log_var2=log_var2)

    def project(self, embed: torch.Tensor, head: int) -> torch.Tensor:
        """Apply projection head and L2-normalise. head: 1=goal proximity, 2=temporal."""
        h = self.head1 if head == 1 else self.head2
        return F.normalize(h(embed), dim=-1)

    # Probabilistic helpers

    @staticmethod
    def sample_embed(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε · exp(0.5 · log_var)."""
        std = (0.5 * log_var).exp()
        return mu + torch.randn_like(std) * std

    @staticmethod
    def kl_loss(log_var: torch.Tensor) -> torch.Tensor:
        """KL(N(μ,σ²) ‖ N(μ,1)) = 0.5·(exp(log_var) − 1 − log_var), averaged."""
        return 0.5 * (log_var.exp() - 1.0 - log_var).mean()

    # Language goal encoding

    def encode_text_goal(self, text: str) -> torch.Tensor:
        """Encode a single language goal string to [1, HEAD_DIM] L2-normalised embedding."""
        device = next(self.parameters()).device
        self._load_clip(device)
        with torch.no_grad():
            tokens = self._clip_tokenizer([text], truncate=True).to(device)
            clip_emb = self._clip_model.encode_text(tokens).float()
        return F.normalize(self.clip_goal_proj(clip_emb), dim=-1)

    def encode_text_goals(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of language goals to [N, HEAD_DIM] L2-normalised embeddings."""
        if not texts:
            return torch.empty(0, self.HEAD_DIM, device=next(self.parameters()).device)
        device = next(self.parameters()).device
        self._load_clip(device)
        with torch.no_grad():
            tokens = self._clip_tokenizer(texts, truncate=True).to(device)
            clip_emb = self._clip_model.encode_text(tokens).float()
        return F.normalize(self.clip_goal_proj(clip_emb), dim=-1)

    # Similarity API (deterministic)

    def goal_sim(self, emb_frame: ProbEmbedding, emb_goal: ProbEmbedding) -> torch.Tensor:
        """Cosine similarity via Head 1 using mean embeddings. Returns [B]."""
        return (self.project(emb_frame.mu, 1) * self.project(emb_goal.mu, 1)).sum(dim=-1)

    def temporal_sim(self, emb_t: ProbEmbedding, emb_t1: ProbEmbedding) -> torch.Tensor:
        """Cosine similarity via Head 2 using mean embeddings. Returns [B]."""
        return (self.project(emb_t.mu, 2) * self.project(emb_t1.mu, 2)).sum(dim=-1)

    def goal_sim_from_text(self, emb_frame: ProbEmbedding, text_goal: str) -> torch.Tensor:
        """Cosine similarity between a visual frame and a language goal. Returns [B]."""
        goal_emb = self.encode_text_goal(text_goal)
        return (self.project(emb_frame.mu, 1) * goal_emb).sum(dim=-1)

    # Similarity API (Monte-Carlo uncertainty)

    def goal_sim_with_uncertainty(
        self,
        emb_a: ProbEmbedding,
        emb_b: ProbEmbedding,
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC cosine similarity and std via Head 1. Returns (mean_sim [B], std_sim [B])."""
        sims = [
            (self.project(self.sample_embed(emb_a.mu, emb_a.log_var1), 1)
             * self.project(self.sample_embed(emb_b.mu, emb_b.log_var1), 1)).sum(dim=-1)
            for _ in range(n_samples)
        ]
        sims = torch.stack(sims, dim=0)
        return sims.mean(dim=0), sims.std(dim=0)

    def temporal_sim_with_uncertainty(
        self,
        emb_t: ProbEmbedding,
        emb_t1: ProbEmbedding,
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC cosine similarity and std via Head 2. Returns (mean_sim [B], std_sim [B])."""
        sims = [
            (self.project(self.sample_embed(emb_t.mu, emb_t.log_var2), 2)
             * self.project(self.sample_embed(emb_t1.mu, emb_t1.log_var2), 2)).sum(dim=-1)
            for _ in range(n_samples)
        ]
        sims = torch.stack(sims, dim=0)
        return sims.mean(dim=0), sims.std(dim=0)

    def goal_sim_from_text_with_uncertainty(
        self,
        emb_frame: ProbEmbedding,
        text_goal: str,
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC cosine similarity between a visual frame and a language goal. Returns (mean [B], std [B])."""
        goal_emb = self.encode_text_goal(text_goal)
        sims = [
            (self.project(self.sample_embed(emb_frame.mu, emb_frame.log_var1), 1) * goal_emb).sum(dim=-1)
            for _ in range(n_samples)
        ]
        sims = torch.stack(sims, dim=0)
        return sims.mean(dim=0), sims.std(dim=0)

    # DDP-safe training forward pass

    def ddp_train_forward(
        self,
        all_imgs: torch.Tensor,
        mask0: torch.Tensor,
        mask1: torch.Tensor,
        has_lang_mask: Optional[torch.Tensor] = None,
        valid_texts: Optional[List[str]] = None,
        use_cached: bool = False,
    ):
        """Unified single-pass forward for contrastive + alignment training (DDP-safe)."""
        all_pe = self.encode_features(all_imgs) if use_cached else self.encode(all_imgs)

        B = all_imgs.size(0) // 3
        mu_a, mu_p, mu_n = all_pe.mu[:B], all_pe.mu[B:2*B], all_pe.mu[2*B:]
        lv1_a, lv1_p, lv1_n = all_pe.log_var1[:B], all_pe.log_var1[B:2*B], all_pe.log_var1[2*B:]
        lv2_a, lv2_p, lv2_n = all_pe.log_var2[:B], all_pe.log_var2[B:2*B], all_pe.log_var2[2*B:]

        n_gp = int(mask0.sum().item())
        n_tc = int(mask1.sum().item())
        use_lang = has_lang_mask is not None and int(has_lang_mask.sum().item()) >= 1

        # Head 1 projections (goal proximity + optional language visual side)
        a0 = p0 = n0 = visual_proj = None
        if n_gp > 1 or use_lang:
            head1_inputs = []
            if n_gp > 1:
                head1_inputs.extend([
                    self.sample_embed(mu_a[mask0], lv1_a[mask0]),
                    self.sample_embed(mu_p[mask0], lv1_p[mask0]),
                    self.sample_embed(mu_n[mask0], lv1_n[mask0]),
                ])
            if use_lang:
                head1_inputs.append(mu_p[has_lang_mask])

            proj_cat = self.project(torch.cat(head1_inputs, dim=0), head=1)
            offset = 0
            if n_gp > 1:
                a0 = proj_cat[offset:offset + n_gp]; offset += n_gp
                p0 = proj_cat[offset:offset + n_gp]; offset += n_gp
                n0 = proj_cat[offset:offset + n_gp]; offset += n_gp
            if use_lang:
                visual_proj = proj_cat[offset:offset + int(has_lang_mask.sum().item())]

        # Head 2 projections (temporal consistency)
        a1 = p1 = n1 = None
        if n_tc > 1:
            proj2_cat = self.project(torch.cat([
                self.sample_embed(mu_a[mask1], lv2_a[mask1]),
                self.sample_embed(mu_p[mask1], lv2_p[mask1]),
                self.sample_embed(mu_n[mask1], lv2_n[mask1]),
            ], dim=0), head=2)
            a1, p1, n1 = proj2_cat[:n_tc], proj2_cat[n_tc:2*n_tc], proj2_cat[2*n_tc:]

        # Language goal projections (CLIP side)
        lang_proj = None
        if use_lang and valid_texts:
            lang_proj = self.encode_text_goals(valid_texts)

        return {
            "a0": a0, "p0": p0, "n0": n0,
            "a1": a1, "p1": p1, "n1": n1,
            "visual_proj": visual_proj,
            "lang_proj": lang_proj,
            "lv1_a": lv1_a, "lv1_p": lv1_p, "lv1_n": lv1_n,
            "lv2_a": lv2_a, "lv2_p": lv2_p, "lv2_n": lv2_n,
            "mask0": mask0, "mask1": mask1,
        }

    def forward(self, *args, mode: str = "default", use_cached: bool = False, **kwargs):
        if mode == "default":
            img_query, img_key = args[0], args[1]
            head = kwargs.get("head", 1)
            encode_fn = self.encode_features if use_cached else self.encode
            e_q, e_k = encode_fn(img_query), encode_fn(img_key)
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
            return self.ddp_train_forward(*args, use_cached=use_cached, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
