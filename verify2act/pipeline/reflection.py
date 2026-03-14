"""Reflection mechanism — builds enriched critic failure context.

Three layers of analysis when the critic rejects an imagined trajectory:
  1. Trajectory trend analysis  (``classify_failure_pattern``)
  2. Spatial attribution         (``get_worst_region``)
  3. Grad-CAM overlay            (``compute_gradcam``, ``make_gradcam_overlay``)

The top-level ``build_reflection_context()`` assembles all three into a
dict that is consumed by ``PromptManager.build_reflect_messages()``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Layer 1 — Trajectory trend analysis
# ---------------------------------------------------------------------------

def classify_failure_pattern(all_scores: List[Tuple[float, float]]) -> str:
    """Return a human-readable summary of how feasibility evolved.

    Parameters
    ----------
    all_scores : list of (mean_feasibility, uncertainty) for steps 0..k
    """
    scores = [s for s, _ in all_scores]
    k = len(scores) - 1

    if all(s < 0.4 for s in scores):
        return (
            "the initial planned action is fundamentally misaligned with the goal "
            "(feasibility was low from step 0)"
        )

    delta = scores[k] - scores[k - 1] if k > 0 else 0
    kind = "sudden" if delta < -0.3 else "gradual"

    return (
        f"the plan was progressing until step {k}, where a {kind} failure occurred "
        f"(scores: {[f'{s:.2f}' for s in scores]})"
    )


# ---------------------------------------------------------------------------
# Layer 2 — Spatial attribution from latent diff map
# ---------------------------------------------------------------------------

def get_worst_region(diff_map: torch.Tensor) -> Tuple[str, Dict[str, float]]:
    """Identify the 3×3 grid cell with highest goal mismatch.

    Parameters
    ----------
    diff_map : [1, 4, 64, 64] tensor  (z_t1 − z_goal)

    Returns
    -------
    worst_label : str    e.g. "center", "top-left"
    grid_scores : dict   mapping region name → mean L2 norm in that cell
    """
    pixel_diff = diff_map.norm(dim=1).squeeze(0)  # [64, 64]

    region_labels = [
        ["top-left",    "top-center",    "top-right"],
        ["middle-left", "center",        "middle-right"],
        ["bottom-left", "bottom-center", "bottom-right"],
    ]

    H, W = pixel_diff.shape
    grid_scores: Dict[str, float] = {}
    for row in range(3):
        for col in range(3):
            r0, r1 = row * H // 3, (row + 1) * H // 3
            c0, c1 = col * W // 3, (col + 1) * W // 3
            label = region_labels[row][col]
            grid_scores[label] = pixel_diff[r0:r1, c0:c1].mean().item()

    worst = max(grid_scores, key=grid_scores.get)
    return worst, grid_scores


# ---------------------------------------------------------------------------
# Layer 3 — Grad-CAM overlay
# ---------------------------------------------------------------------------

def _find_last_conv_module(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last Conv2d module in a model for Grad-CAM hooks."""
    last_conv = None
    for mod in model.modules():
        if isinstance(mod, torch.nn.Conv2d):
            last_conv = mod
    if last_conv is None:
        raise ValueError("No Conv2d module found in critic; cannot compute Grad-CAM.")
    return last_conv

def compute_gradcam(
    critic: torch.nn.Module,
    z_t1: torch.Tensor,
    z_goal: torch.Tensor,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap from the critic's last conv layer.

    Parameters
    ----------
    critic : SpatialBetaPRMCritic (or any critic with Conv2d layers)
    z_t1 : [1, 4, 64, 64]  current/imagined state latent
    z_goal : [1, 4, 64, 64]  goal state latent

    Returns
    -------
    cam : [512, 512] float32 array in [0, 1]
    """
    critic.eval()
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def _save_act(m, inp, out):
        activations["feat"] = out.detach()

    def _save_grad(m, grad_in, grad_out):
        gradients["feat"] = grad_out[0].detach()

    target_layer = _find_last_conv_module(critic)
    hook_a = target_layer.register_forward_hook(_save_act)
    hook_g = target_layer.register_full_backward_hook(_save_grad)

    try:
        critic.zero_grad(set_to_none=True)
        z_t1 = z_t1.detach().requires_grad_(True)
        z_goal = z_goal.detach().requires_grad_(True)
        out = critic(z_t1, z_goal)
        mean_f = out["mean_feasibility"]
        mean_f.sum().backward()
    finally:
        hook_a.remove()
        hook_g.remove()

    if "feat" not in activations or "feat" not in gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    weights = gradients["feat"].mean(dim=(-2, -1), keepdim=True)  # [1, C, 1, 1]
    cam = (weights * activations["feat"]).sum(dim=1, keepdim=True).relu()  # [1, 1, h, w]
    cam = F.interpolate(cam, size=(512, 512), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def make_gradcam_overlay(imagined_img_np: np.ndarray, cam: np.ndarray) -> Image.Image:
    """Blend a Grad-CAM heatmap onto the imagined scene.

    Parameters
    ----------
    imagined_img_np : [512, 512, 3] uint8
    cam : [512, 512] float32 in [0, 1]

    Returns
    -------
    PIL Image with red-hot heatmap overlaid at 50% opacity.
    """
    import matplotlib.cm as cm

    heatmap = cm.hot(cam)[:, :, :3]  # [512, 512, 3] float
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    overlay = (0.5 * imagined_img_np + 0.5 * heatmap_uint8).astype(np.uint8)
    return Image.fromarray(overlay)


# ---------------------------------------------------------------------------
# Top-level context builder
# ---------------------------------------------------------------------------

def build_reflection_context(
    imagined_state: np.ndarray,
    z_t1: torch.Tensor,
    z_goal: torch.Tensor,
    diff_map: torch.Tensor,
    critic: torch.nn.Module,
    all_scores: List[Tuple[float, float]],
    failed_step: int,
    full_plan: List[str],
) -> Dict[str, Any]:
    """Assemble the full reflection context dict.

    This is passed as ``ctx`` to ``PromptManager.build_reflect_messages()``
    and to ``VLMPlanner.reflect()``.
    """
    if failed_step < 0 or failed_step >= len(full_plan):
        raise IndexError(f"failed_step={failed_step} out of range for plan length {len(full_plan)}")
    if failed_step >= len(all_scores):
        raise IndexError(f"failed_step={failed_step} out of range for all_scores length {len(all_scores)}")

    mean_f = all_scores[failed_step][0]
    uncert = all_scores[failed_step][1]

    failure_pattern = classify_failure_pattern(all_scores)
    worst_region, grid_scores = get_worst_region(diff_map)
    cam = compute_gradcam(critic, z_t1, z_goal)
    gradcam_overlay = make_gradcam_overlay(imagined_state, cam)

    return {
        "imagined_state":   imagined_state,
        "gradcam_overlay":  gradcam_overlay,
        "mean_feasibility": mean_f,
        "uncertainty":      uncert,
        "all_scores":       all_scores,
        "failure_pattern":  failure_pattern,
        "worst_region":     worst_region,
        "grid_scores":      grid_scores,
        "failed_step":      failed_step,
        "failed_action":    full_plan[failed_step],
        "full_plan":        full_plan,
    }
