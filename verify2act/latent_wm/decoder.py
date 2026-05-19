import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add rla-wm to path so we can import its modules
rla_wm_path = Path(__file__).resolve().parent.parent.parent / "rla-wm"
if str(rla_wm_path) not in sys.path:
    sys.path.append(str(rla_wm_path))

try:
    from src.models.dino_to_image_unet_v1 import DinoToImageDecoderV1
except ImportError as e:
    print(f"Warning: Could not import rla-wm modules: {e}")


class FeatureDecoder(nn.Module):
    """
    Visualization wrapper for DinoToImageDecoderV1.
    Used purely for human interpretability to decode latent patch features back to RGB.
    NOT used during the actual MCTS planning loop.
    """

    def __init__(
        self,
        dino_channels: int = 768,
        model_channels: int = 256,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.decoder = DinoToImageDecoderV1(
            in_channels=dino_channels,
            model_channels=model_channels,
            out_channels=3,
            use_fp16=use_fp16,
        )

    def decode(self, patch_tokens: torch.Tensor, patch_hw: tuple[int, int] = (16, 16)) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, num_patches, dino_channels)
            patch_hw: Spatial grid of the patches. Default (16, 16) for 256 patches.

        Returns:
            rgb_images: (B, 3, H, W) where H=16*patch_hw[0], W=16*patch_hw[1]
        """
        # The decoder expects (B, Cam, num_patches, channels)
        # We add a dummy Camera dimension
        x = patch_tokens.unsqueeze(1)  # (B, 1, num_patches, channels)

        # Forward pass
        out = self.decoder(x, patch_hw=patch_hw)  # (B, 1, 3, H, W)

        # Remove dummy Camera dimension
        out = out.squeeze(1)  # (B, 3, H, W)

        # The output is typical unnormalized logits or [-1, 1]. 
        # Typically, visualizers clamp and map to [0, 1] for saving.
        # This will depend on how the decoder was trained, but standard is [-1, 1].
        return out
