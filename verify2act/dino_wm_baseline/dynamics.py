import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat

# Add dino_wm to path
dino_wm_path = Path(__file__).resolve().parent.parent.parent / "dino_wm"
if str(dino_wm_path) not in sys.path:
    sys.path.append(str(dino_wm_path))

try:
    from models.visual_world_model import VWorldModel
    from models.proprio import ProprioceptiveEmbedding
    from models.vit import ViTPredictor
except ImportError as e:
    print(f"Warning: Could not import dino_wm modules: {e}")


class DummyEncoder(nn.Module):
    """
    Dummy visual encoder wrapper.
    Bypasses DINOv2 feature extraction during baseline training/inference
    since raw DINOv2 features are already pre-computed/cached by verify2act.
    """
    def __init__(self, emb_dim: int = 1024):
        super().__init__()
        self.emb_dim = emb_dim
        self.latent_ndim = 2
        self.name = "dummy"
        self.patch_size = 14

    def forward(self, x: Tensor) -> Tensor:
        return x


class BaselineDINOWM(nn.Module):
    """
    DINO-WM baseline wrapper for verify2act.

    Faithfully runs the original DINO-WM sequence dynamics core:
    - Bypasses ODE solver steps (uses direct causal sequence prediction).
    - Concatenates action and proprio embeddings as extra sequence tokens (concat_dim=0).
    - Operates directly in the raw DINOv2 feature space.
    - Projects static CLIP action text embeddings into fixed action embeddings.
    """
    def __init__(
        self,
        dino_channels: int = 1024,      # DINOv2 features (1024 for ViT-L, 768 for ViT-B)
        clip_channels: int = 512,       # CLIP text embedding dimension
        action_dim: int = 64,           # Action dimension for projection
        action_emb_dim: int = 64,       # Action embedding dimension
        proprio_dim: int = 16,          # Proprioceptive dimension
        proprio_emb_dim: int = 16,      # Proprioceptive embedding dimension
        history_len: int = 3,           # Number of context/history steps
        num_patches: int = 256,         # Spatial DINOv2 patch count (16x16)
        depth: int = 6,                 # Transformer layers
        heads: int = 16,                # Attention heads
        mlp_dim: int = 2048,            # Transformer feedforward MLP hidden dim
        concat_dim: int = 0,            # 0 = sequence token concatenation, 1 = channel tiling
    ):
        super().__init__()
        self.dino_channels = dino_channels
        self.clip_channels = clip_channels
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.history_len = history_len
        self.num_patches = num_patches
        self.concat_dim = concat_dim

        # When concat_dim == 0 (token-level concatenation), the visual, proprio,
        # and action sequence tokens must share the same feature size (dino_channels).
        if concat_dim == 0:
            proprio_emb_dim = dino_channels
            action_emb_dim = dino_channels

        # CLIP Text Projection: Maps semantic VLM actions (512) -> action_dim
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_channels, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim)
        )

        # Encoders conforming strictly to original DINO-WM
        self.encoder = DummyEncoder(emb_dim=dino_channels)
        self.proprio_encoder = ProprioceptiveEmbedding(
            num_frames=history_len,
            in_chans=proprio_dim,
            emb_dim=proprio_emb_dim
        )
        self.action_encoder = ProprioceptiveEmbedding(
            num_frames=history_len,
            in_chans=action_dim,
            emb_dim=action_emb_dim
        )


        # Predictor Transformer: ViTPredictor from dino_wm/models/vit.py
        predictor_patches = num_patches
        if concat_dim == 0:
            predictor_patches += 2  # Visual patches + Proprio token + Action token

        predictor_dim = dino_channels
        if concat_dim == 1:
            predictor_dim += (proprio_emb_dim + action_emb_dim)

        self.predictor = ViTPredictor(
            num_patches=predictor_patches,
            num_frames=history_len,
            dim=predictor_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool='mean'
        )

        # Original VWorldModel from dino_wm/models/visual_world_model.py
        self.v_wm = VWorldModel(
            image_size=224,
            num_hist=history_len,
            num_pred=1,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            decoder=None,  # No visual decoder during baseline training/inference
            predictor=self.predictor,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=concat_dim,
            num_action_repeat=1,
            num_proprio_repeat=1,
            train_encoder=False,
            train_predictor=True,
            train_decoder=False
        )

    def _prepare_inputs(self, xt_history: Tensor, action_tokens: Tensor):
        """
        Formats verify2act raw inputs to match DINO-WM's sequence expectations.
        """
        B, T = xt_history.shape[:2]

        # 1. Project CLIP semantic action embeddings and repeat over time steps
        # Mean pool CLIP text token sequence: (B, seq_len, 512) -> (B, 512)
        pooled_clip = action_tokens.mean(dim=1)
        projected_action = self.clip_proj(pooled_clip)  # (B, action_dim)
        projected_action = projected_action.unsqueeze(1).repeat(1, T, 1)  # (B, T, action_dim)

        # 2. Construct dummy proprio inputs (zeros)
        proprio = torch.zeros(B, T, self.proprio_dim, device=xt_history.device)

        obs = {
            "visual": xt_history,
            "proprio": proprio
        }
        return obs, projected_action

    def forward(self, xt_history: Tensor, action_tokens: Tensor):
        """
        Training forward pass.
        Returns:
            z_pred: predicted tokens
            z_tgt: target tokens (from ground-truth F_{t+1})
            loss: MSE feature prediction loss
        """
        # Formulate observation and action tensors
        obs, act = self._prepare_inputs(xt_history, action_tokens)

        # Visual VWorldModel handles the token flattening, predictor forwards, and loss
        z_pred, _, _, loss, loss_components = self.v_wm(obs, act)

        # Extract target features from target observation (sliced at num_pred)
        z_gt = self.v_wm.encode_obs(obs)
        z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.v_wm.num_pred)

        return z_pred, z_tgt, loss

    @torch.no_grad()
    def step(self, xt_history: Tensor, action_tokens: Tensor) -> Tensor:
        """
        Single-step test-time rollout.
        Returns predicted next-state raw DINOv2 features directly: (B, 256, dino_channels)
        """
        obs, act = self._prepare_inputs(xt_history, action_tokens)

        # Encode and predict sequence tokens
        z = self.v_wm.encode(obs, act)
        z_pred = self.v_wm.predict(z[:, -self.history_len:])

        # Extract predicted visual features for the next step
        z_obs_pred, _ = self.v_wm.separate_emb(z_pred)
        # Take the predicted visual features of the next frame: shape (B, 256, dino_channels)
        pred_visual_features = z_obs_pred["visual"][:, -1]

        return pred_visual_features


def slice_trajdict_with_t(dct, start_idx=None, end_idx=None):
    """Slices time dimension in dictionary tensors."""
    new_dct = {}
    for key, value in dct.items():
        if value is None:
            new_dct[key] = None
        else:
            new_dct[key] = value[:, start_idx:end_idx]
    return new_dct
