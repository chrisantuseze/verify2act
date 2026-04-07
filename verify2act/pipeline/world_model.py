"""World model abstraction — oracle (sim) and diffusion backends.

The world model predicts the next-state image given a current image and an
action text string.  Two implementations are provided:

- ``OracleWorldModel``: Uses the robosuite simulator as a perfect dynamics
  model via state save/restore (no approximation error).
- ``DiffusionWorldModel``: Uses a finetuned InstructPix2Pix diffusion
  pipeline to imagine the next state from ``(image, action_text)``.
"""

from __future__ import annotations

import abc
import logging
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════


class WorldModelBase(abc.ABC):
    """Abstract interface for a world model.

    ``imagine()`` takes the current RGB observation and an action text
    string and returns the predicted next-state RGB observation.
    """

    @abc.abstractmethod
    def imagine(
        self,
        current_image_np: np.ndarray,
        action_text: str,
    ) -> np.ndarray:
        """Predict the next-state image.

        Parameters
        ----------
        current_image_np : np.ndarray
            ``[H, W, 3]`` uint8 RGB observation of the current state.
        action_text : str
            Semantic action string, e.g. ``"pick round nut"``.

        Returns
        -------
        np.ndarray
            ``[H, W, 3]`` uint8 RGB image of the predicted next state.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
# Oracle world model (uses simulator)
# ═══════════════════════════════════════════════════════════════════════


class OracleWorldModel(WorldModelBase):
    """World model backed by the actual robosuite simulator.

    Saves the simulator state, executes the requested action via the
    heuristic policy, captures the resulting camera frame, then restores
    the original state.  This provides a *perfect* dynamics model and
    serves as the upper-bound reference for the diffusion world model.

    Parameters
    ----------
    env_wrapper : NutAssemblyEnvWrapper
        The wrapped environment instance (must be the *same* instance
        used by the inference loop so that state is coherent).
    """

    def __init__(self, env_wrapper) -> None:
        self.env_wrapper = env_wrapper
        # Rollout-chain state (set by rollout_context; None when not in a chain).
        self._rollout_state: object = None
        self._rollout_obs: object = None
        # Per-step pre-action state for requery retries within a chain.
        self._step_state: object = None
        self._step_obs: object = None

    @contextmanager
    def rollout_context(self):
        """Context manager for chained imagination rollouts.

        On entry the current simulator state is saved.  While inside,
        ``imagine()`` does **not** restore the sim after each action so that
        the rollout chain accumulates: S0 → S1 → S2 → …
        On exit the simulator is always restored to the entry state.
        """
        self._rollout_state = self.env_wrapper.save_state()
        self._rollout_obs = deepcopy(getattr(self.env_wrapper, "_obs", None))
        try:
            yield self
        finally:
            self.env_wrapper.restore_state(self._rollout_state)
            if self._rollout_obs is not None:
                self.env_wrapper._obs = self._rollout_obs
            self._rollout_state = None
            self._rollout_obs = None
            self._step_state = None
            self._step_obs = None

    def rollback_step(self) -> None:
        """Restore the sim to the state *before* the last ``imagine()`` call.

        Used by the requery loop to retry the same action from the same
        pre-action state inside a ``rollout_context``.
        """
        if self._step_state is not None:
            self.env_wrapper.restore_state(self._step_state)
        if self._step_obs is not None:
            self.env_wrapper._obs = self._step_obs

    def _suppress_viewer(self):
        """Prevent ``env.step()`` from changing the on-screen viewer during
        imagination rollouts.

        We keep ``env.viewer`` intact and temporarily replace
        ``env.viewer.update`` with a no-op. This avoids:
        - re-launching extra windows (when ``viewer`` is set to None), and
        - freezing all future updates (when renderer flags get stuck).
        """
        env = self.env_wrapper.env
        viewer = getattr(env, "viewer", None)
        env._saved_viewer_obj = viewer
        env._saved_has_renderer = env.has_renderer

        if viewer is not None and hasattr(viewer, "update"):
            env._saved_viewer_update = viewer.update

            def _noop_update():
                return None

            viewer.update = _noop_update
        else:
            env._saved_viewer_update = None
            # Defensive fallback: if no viewer object is present, disable
            # renderer so env.step() doesn't auto-launch a new window.
            env.has_renderer = False

    def _restore_viewer(self):
        """Restore viewer update behavior after imagination."""
        env = self.env_wrapper.env
        viewer = getattr(env, "_saved_viewer_obj", None)
        saved_update = getattr(env, "_saved_viewer_update", None)
        if viewer is not None and saved_update is not None:
            viewer.update = saved_update
        if hasattr(env, "_saved_viewer_obj"):
            del env._saved_viewer_obj
        if hasattr(env, "_saved_viewer_update"):
            del env._saved_viewer_update
        if hasattr(env, "_saved_has_renderer"):
            env.has_renderer = env._saved_has_renderer
            del env._saved_has_renderer

    def imagine(
        self,
        current_image_np: np.ndarray,
        action_text: str,
    ) -> np.ndarray:
        in_chain = self._rollout_state is not None

        if in_chain:
            # Save the pre-action state so requery retries can restart here.
            self._step_state = self.env_wrapper.save_state()
            self._step_obs = deepcopy(getattr(self.env_wrapper, "_obs", None))
            # Suppress viewer so env.step() doesn't update the on-screen window.
            self._suppress_viewer()
            try:
                self.env_wrapper.execute_action(action_text)
                return self.env_wrapper.read_image()
            finally:
                self._restore_viewer()

        # Standalone call (no active rollout_context): safe save/restore.
        saved_state = self.env_wrapper.save_state()
        saved_obs = deepcopy(getattr(self.env_wrapper, "_obs", None))
        self._suppress_viewer()
        try:
            self.env_wrapper.execute_action(action_text)
            imagined_img = self.env_wrapper.read_image()
        finally:
            self._restore_viewer()
            self.env_wrapper.restore_state(saved_state)
            self.env_wrapper._obs = saved_obs
        return imagined_img


# ═══════════════════════════════════════════════════════════════════════
# Diffusion world model (InstructPix2Pix)
# ═══════════════════════════════════════════════════════════════════════


class DiffusionWorldModel(WorldModelBase):
    """World model backed by a finetuned InstructPix2Pix pipeline.

    Loads the pipeline once at construction time and reuses it for every
    ``imagine()`` call.  Optionally applies LoRA adapters (Phase A) and/or
    a finetuned VAE decoder (Phase B).

    Parameters
    ----------
    pretrained_model : str
        HuggingFace model ID or local path, default ``"timbrooks/instruct-pix2pix"``.
    adapter_dir : str or None
        Path to LoRA adapter directory (Phase A).  ``None`` to skip.
    decoder_dir : str or None
        Path to finetuned VAE decoder checkpoint (Phase B).  ``None`` to skip.
    vae_model : str or None
        Separate VAE encoder source.  Falls back to *pretrained_model* if ``None``.
    vae_subfolder : str
        Subfolder resolution strategy (``"auto"`` tries ``vae`` → ``vae_ema`` → root).
    device : str
        Torch device string.
    torch_dtype : torch.dtype
        Compute dtype (``torch.float16`` is recommended for speed).
    num_inference_steps : int
        Denoising steps during generation.
    image_guidance_scale : float
        InstructPix2Pix image-conditioning strength.
    guidance_scale : float
        Text-conditioning strength.
    seed : int or None
        Optional seed for the generator (reproducibility).
    """

    def __init__(
        self,
        pretrained_model: str = "timbrooks/instruct-pix2pix",
        adapter_dir: Optional[str] = None,
        decoder_dir: Optional[str] = None,
        vae_model: Optional[str] = None,
        vae_subfolder: str = "auto",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        num_inference_steps: int = 30,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> None:
        from diffusers import StableDiffusionInstructPix2PixPipeline
        from verify2act.utils.vae import load_vae_decoder, load_vae_encoder

        self.device = torch.device(device)
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.guidance_scale = guidance_scale
        self._seed = seed

        logger.info("Loading InstructPix2Pix pipeline from %s …", pretrained_model)
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pretrained_model,
            torch_dtype=torch_dtype,
            safety_checker=None,
        ).to(self.device)

        # ── VAE encoder (always frozen) ────────────────────────────────
        vae_src = vae_model or pretrained_model
        vae, resolved = load_vae_encoder(
            model_name_or_path=vae_src,
            device=self.device,
            torch_dtype=torch_dtype,
            subfolder=vae_subfolder,
        )
        self.pipeline.vae = vae
        logger.info("VAE encoder loaded from %s (subfolder=%s)", vae_src, resolved)

        # ── Optional finetuned decoder (Phase B) ──────────────────────
        if decoder_dir is not None:
            self.pipeline.vae = load_vae_decoder(
                decoder_dir=decoder_dir,
                vae=self.pipeline.vae,
                device=self.device,
                torch_dtype=torch_dtype,
            )
            logger.info("Finetuned VAE decoder loaded from %s", decoder_dir)

        # ── Optional LoRA adapters (Phase A) ──────────────────────────
        if adapter_dir is not None:
            from peft import PeftModel

            adapter_path = Path(adapter_dir)
            if not adapter_path.exists():
                raise FileNotFoundError(f"LoRA adapter dir not found: {adapter_path}")
            self.pipeline.unet = PeftModel.from_pretrained(
                self.pipeline.unet, str(adapter_path)
            ).to(self.device)
            logger.info("LoRA adapter loaded from %s", adapter_dir)

    def imagine(
        self,
        current_image_np: np.ndarray,
        action_text: str,
    ) -> np.ndarray:
        img_pil = Image.fromarray(current_image_np).resize((512, 512))

        generator = None
        if self._seed is not None:
            generator = torch.Generator(self.device.type).manual_seed(self._seed)

        with torch.inference_mode():
            out = self.pipeline(
                action_text,
                image=img_pil,
                num_inference_steps=self.num_inference_steps,
                image_guidance_scale=self.image_guidance_scale,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).images[0]

        # Restore to the original spatial resolution.
        h, w = current_image_np.shape[:2]
        if (out.width, out.height) != (w, h):
            out = out.resize((w, h))

        return np.array(out)
