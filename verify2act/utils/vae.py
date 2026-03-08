from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from diffusers import AutoencoderKL


VAE_LATENT_SCALE = 0.18215


def _candidate_subfolders(requested_subfolder: str) -> List[Optional[str]]:
    if requested_subfolder != "auto":
        if requested_subfolder.lower() in {"", "none", "root"}:
            return [None]
        return [requested_subfolder]

    return ["vae", "vae_ema", None]


def load_vae_encoder(
    model_name_or_path: str,
    device: torch.device,
    torch_dtype: torch.dtype = torch.float32,
    subfolder: str = "auto",
    local_files_only: bool = False,
) -> Tuple[AutoencoderKL, str]:
    """
    Load a VAE encoder with robust subfolder resolution.

    Returns:
        (vae_module, resolved_subfolder)
        resolved_subfolder is one of: "vae", "vae_ema", "root", or custom value.
    """
    attempts = []
    candidates = _candidate_subfolders(subfolder)
    model_path = Path(model_name_or_path)

    if subfolder == "auto" and model_path.exists() and model_path.is_dir():
        if (model_path / "config.json").exists():
            candidates = [None] + [c for c in candidates if c is not None]

    for candidate in candidates:
        kwargs = {
            "torch_dtype": torch_dtype,
            "local_files_only": local_files_only,
        }
        if candidate is not None:
            kwargs["subfolder"] = candidate

        try:
            vae = AutoencoderKL.from_pretrained(model_name_or_path, **kwargs).to(device)
            vae.requires_grad_(False)
            vae.eval()
            resolved = candidate if candidate is not None else "root"
            return vae, resolved
        except Exception as exc:
            attempts.append((candidate if candidate is not None else "root", str(exc)))

    tried = "; ".join([f"{name}: {err}" for name, err in attempts])
    raise RuntimeError(
        "Failed to load VAE encoder. "
        f"model={model_name_or_path}, requested_subfolder={subfolder}. Attempts -> {tried}"
    )
