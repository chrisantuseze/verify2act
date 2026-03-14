from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from diffusers import AutoencoderKL


VAE_LATENT_SCALE = 0.18215


def load_vae_decoder(
    decoder_dir: str,
    vae: "AutoencoderKL",
    device: torch.device,
    torch_dtype: torch.dtype = torch.float32,
) -> "AutoencoderKL":
    """
    Load a finetuned VAE decoder and attach it to *vae*, then return the
    updated VAE.  Two checkpoint formats are supported:

    (a) Full diffusers-format VAE directory (contains config.json):
        Produced by ``train_decoder.save_full_vae()``.
        The checkpoint is loaded to extract the finetuned decoder and
        post_quant_conv weights, which are spliced onto *vae*; the frozen
        encoder on *vae* (loaded by ``load_vae_encoder``) is preserved.

    (b) Phase-B per-epoch checkpoint directory (contains decoder_state_dict.pt):
        Only ``vae.decoder`` (and ``vae.post_quant_conv`` when present) are
        overwritten; the frozen encoder on *vae* is left intact.

    Args:
        decoder_dir: Path to the checkpoint directory (string or Path-like).
        vae:         Existing AutoencoderKL whose decoder will be replaced.
                     Ignored when loading a full diffusers VAE (case a).
        device:      Target torch device.
        torch_dtype: Target torch dtype.

    Returns:
        Updated AutoencoderKL with the finetuned decoder, in eval mode with
        gradients disabled.

    Raises:
        FileNotFoundError: If the directory is missing or has an unexpected layout.
    """
    decoder_path = Path(decoder_dir)
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder directory not found: {decoder_path}")

    config_json = decoder_path / "config.json"
    decoder_sd_path = decoder_path / "decoder_state_dict.pt"

    if config_json.exists():
        # (a) Full diffusers-format VAE: load the checkpoint to get the
        # finetuned decoder weights, but splice them onto the existing *vae*
        # so the encoder loaded by load_vae_encoder is preserved.
        full_vae = AutoencoderKL.from_pretrained(
            str(decoder_path), torch_dtype=torch_dtype
        ).to(device)
        vae.decoder.load_state_dict(full_vae.decoder.state_dict())
        vae.post_quant_conv.load_state_dict(full_vae.post_quant_conv.state_dict())
        del full_vae
        vae.decoder.to(device=device, dtype=torch_dtype)
        vae.requires_grad_(False)
        vae.eval()
        return vae

    if decoder_sd_path.exists():
        # (b) Per-epoch checkpoint: patch decoder weights in-place
        decoder_sd = torch.load(decoder_sd_path, map_location=device)
        vae.decoder.load_state_dict(decoder_sd)
        pqc_path = decoder_path / "post_quant_conv_state_dict.pt"
        if pqc_path.exists():
            pqc_sd = torch.load(pqc_path, map_location=device)
            vae.post_quant_conv.load_state_dict(pqc_sd)
        vae.decoder.to(device=device, dtype=torch_dtype)
        vae.requires_grad_(False)
        vae.eval()
        return vae

    raise FileNotFoundError(
        f"--decoder-dir={decoder_path} must contain either config.json "
        "(full diffusers VAE) or decoder_state_dict.pt (per-epoch checkpoint)."
    )


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
