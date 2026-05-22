#!/usr/bin/env python3
"""
Phase A training script for Verify2Act world model.

Trains LoRA adapters on InstructPix2Pix UNet to predict noise in z_{t+1},
conditioned on current-frame latent z_t and action text.

Expected dataset format:
- <dataset_dir>/transitions_subskill.jsonl
- image paths stored in each row under keys `image_t` and `image_t1`
  as paths relative to dataset_dir.
"""

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import sys
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from diffusers import DDPMScheduler, StableDiffusionInstructPix2PixPipeline
from diffusers.training_utils import EMAModel

# Ensure the project root (the directory that contains the `verify2act` package)
# is on sys.path so imports work when running this script directly.
# Location: .../verify2act/verify2act/world_model/train_wm.py
# We need the parent of the outer `verify2act` folder (parents[2]).
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from verify2act.data_loader import WMTransitionDataset
try:
    from verify2act.utils import VAE_LATENT_SCALE, load_vae_encoder
except ImportError:
    from utils import VAE_LATENT_SCALE, load_vae_encoder


def _init_tracker(tracker: str, output_dir: Path, config: dict):
    """Return a lightweight logging object with .log(dict, step) and .close()."""
    if tracker == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        log_dir = output_dir / "tb_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        # Write hparams as text so they show up in the TB UI.
        writer.add_text("hparams", json.dumps(config, indent=2), global_step=0)

        class _TB:
            def log(self, metrics: dict, step: int):
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step=step)
                writer.flush()
            def close(self):
                writer.close()
        return _TB()

    if tracker == "wandb":
        import wandb
        wandb.init(project=config.pop("wandb_project", "verify2act-wm"), config=config)

        class _WB:
            def log(self, metrics: dict, step: int):
                wandb.log(metrics, step=step)
            def close(self):
                wandb.finish()
        return _WB()

    # tracker == "none"
    class _Noop:
        def log(self, metrics: dict, step: int): pass
        def close(self): pass
    return _Noop()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)





def get_dtype(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def compute_snr_weights(
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Min-SNR-γ per-sample loss weights for ε-prediction (Hang et al., 2023).

    weight_t = min(SNR_t, γ) / SNR_t

    This re-balances per-timestep gradient contributions: pure-noise steps
    (SNR→0) and near-clean steps (SNR→∞) are both de-emphasized relative to
    perceptually-important mid-noise timesteps, which improves visual fidelity
    without changing model architecture.

    Returns a [B] tensor of per-sample weights on the same device as
    ``timesteps``.
    """
    alpha = alphas_cumprod[timesteps].float().to(timesteps.device)
    snr = alpha / (1.0 - alpha).clamp(min=1e-8)
    return (torch.clamp(snr, max=gamma) / snr)


def build_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Linear warmup followed by cosine decay to min_lr_ratio * peak_lr."""
    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def evaluate(
    unet,
    vae,
    tokenizer,
    text_encoder,
    noise_scheduler,
    val_loader,
    device,
    latent_scale,
    eval_batches,
    accelerator=None,
):
    unet.eval()
    vae_dtype = next(vae.parameters()).dtype
    loss_sum = torch.tensor(0.0, device=device)
    loss_count = torch.tensor(0.0, device=device)

    autocast_ctx = accelerator.autocast if accelerator is not None else nullcontext

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= eval_batches:
                break

            image_t = batch["image_t"].to(device=device, dtype=vae_dtype)
            image_t1 = batch["image_t1"].to(device=device, dtype=vae_dtype)
            prompts = batch["action_text"]

            tokenized = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(device)

            # z_t: conditioning image – must match IP2P inference
            # (mode, no scaling) so the UNet sees the same magnitude at
            # train and test time.
            z_t = vae.encode(image_t).latent_dist.mode()
            z_t1 = vae.encode(image_t1).latent_dist.sample() * latent_scale
            noise = torch.randn_like(z_t1)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (z_t1.shape[0],),
                device=device,
                dtype=torch.long,
            )

            noisy_z_t1 = noise_scheduler.add_noise(z_t1, noise, timesteps)
            model_input = torch.cat([noisy_z_t1, z_t], dim=1)
            text_emb = text_encoder(input_ids)[0]

            with autocast_ctx():
                noise_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=text_emb,
                ).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            if torch.isfinite(loss):
                loss_sum += loss.detach().float()
                loss_count += 1.0

    if accelerator is not None:
        loss_sum = accelerator.reduce(loss_sum, reduction="sum")
        loss_count = accelerator.reduce(loss_count, reduction="sum")

    unet.train()
    if loss_count.item() <= 0:
        print("[warn] Evaluation produced no finite losses.")
        return math.nan
    return float((loss_sum / loss_count).item())


def save_checkpoint(unet, output_dir: Path, step: int, train_state: Dict):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = ckpt_dir / "unet_lora"
    unet.save_pretrained(adapter_dir)

    with open(ckpt_dir / "train_state.json", "w") as handle:
        json.dump(train_state, handle, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cpu=args.device == "cpu",
    )

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    device = accelerator.device
    if device.type == "cpu" and args.mixed_precision != "no":
        print("[warn] CUDA unavailable, forcing mixed precision to 'no'.")
        args.mixed_precision = "no"

    weight_dtype = get_dtype(args.mixed_precision)

    accelerator.print("=" * 80)
    accelerator.print("VERIFY2ACT PHASE A: UNET LORA TRAINING")
    accelerator.print("=" * 80)
    accelerator.print(f"Dataset:          {args.dataset_dir}")
    accelerator.print(f"Output:           {output_dir}")
    accelerator.print(f"Pretrained model: {args.pretrained_model}")
    accelerator.print(f"Device:           {device}")
    accelerator.print(f"Precision:        {args.mixed_precision}")
    accelerator.print(f"Batch size:       {args.train_batch_size}")
    accelerator.print(f"Grad accum:       {args.gradient_accumulation_steps}")
    accelerator.print(f"LR:               {args.learning_rate}")
    accelerator.print(f"Max steps:        {args.max_steps}")
    accelerator.print(f"Min-SNR γ:        {args.snr_gamma if args.snr_gamma > 0 else 'disabled'}")
    accelerator.print(f"LoRA FF targets:  {args.lora_target_ff}")
    accelerator.print(f"LoRA conv targets:{args.lora_target_conv}")
    accelerator.print(f"EMA:              {'enabled (decay=' + str(args.ema_decay) + ')' if args.use_ema else 'disabled'}")
    accelerator.print(f"Cond dropout:     {args.conditioning_dropout_prob if args.conditioning_dropout_prob > 0 else 'disabled'}")
    accelerator.print(f"Noise offset:     {args.noise_offset if args.noise_offset > 0 else 'disabled'}")

    accelerator.print("\nLoading datasets...")
    if args.dataset_type == "calvin":
        from verify2act.data_loader_calvin import build_calvin_wm_datasets
        train_ds, val_ds = build_calvin_wm_datasets(
            dataset_dir=args.dataset_dir,
            val_frac=args.val_frac,
            seed=args.seed,
            image_size=args.resolution,
        )
        accelerator.print(f"CALVIN Dataset loaded. Train: {len(train_ds)}, Val: {len(val_ds)}")
    else:
        train_ds = WMTransitionDataset(
            dataset_dir=args.dataset_dir,
            image_size=args.resolution,
            split="train",
            val_frac=args.val_frac,
            seed=args.seed,
            transitions_file=args.transitions_file,
        )
        val_ds = WMTransitionDataset(
            dataset_dir=args.dataset_dir,
            image_size=args.resolution,
            split="val",
            val_frac=args.val_frac,
            seed=args.seed,
            transitions_file=args.transitions_file,
        )
        accelerator.print(f"RoboSuite Dataset loaded. Train: {len(train_ds)}, Val: {len(val_ds)}")

    if args.dataset_type != "calvin":
        _sample_weights = train_ds.sample_weights()
        _sampler = WeightedRandomSampler(
            weights=_sample_weights,
            num_samples=len(_sample_weights),
            replacement=True,
        )
        accelerator.print(f"WeightedRandomSampler: {len(_sample_weights)} samples, 3 buckets balanced")
        train_loader = DataLoader(
            train_ds,
            batch_size=args.train_batch_size,
            sampler=_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    accelerator.print("\nLoading InstructPix2Pix components...")
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder.to(device)
    unet = pipeline.unet.to(device)

    vae_model = args.vae_model if args.vae_model else args.pretrained_model
    vae, resolved_subfolder = load_vae_encoder(
        model_name_or_path=vae_model,
        device=device,
        torch_dtype=weight_dtype,
        subfolder=args.vae_subfolder,
        local_files_only=args.local_files_only,
    )
    accelerator.print(
        f"Using VAE encoder from model={vae_model} "
        f"(subfolder={resolved_subfolder}, dtype={weight_dtype}, device={device})"
    )

    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    del pipeline

    # Precompute null text embedding once for use in conditioning dropout.
    with torch.no_grad():
        _null_ids = tokenizer(
            [""],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)
        null_text_emb = text_encoder(_null_ids)[0]  # [1, seq_len, hidden_dim]

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if args.enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            accelerator.print("Enabled xFormers memory-efficient attention.")
        except Exception as err:
            accelerator.print(f"[warn] Could not enable xFormers: {err}")

    _lora_targets = ["to_q", "to_k", "to_v", "to_out.0"]
    if args.lora_target_ff:
        _lora_targets += ["ff.net.0.proj", "ff.net.2"]
    if args.lora_target_conv:
        _lora_targets += ["conv1", "conv2", "proj_in", "proj_out", "conv_shortcut"]

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=_lora_targets,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Unfreeze conv_in (8→320 input projection) for full fine-tuning.
    # This is the only layer that sees the raw 8-channel [noisy_z_t1 | z_t] input
    # and controls how conditioning and target channels are mixed before any
    # downstream LoRA can act. Must be done before EMA init so shadow weights
    # track these parameters from the start.
    # _base_unet = unet.base_model.model
    # for param in _base_unet.conv_in.parameters():
    #     param.requires_grad_(True)
    # _conv_in_param_ids = {id(p) for p in _base_unet.conv_in.parameters()}
    # accelerator.print(
    #     f"conv_in unfrozen: {sum(p.numel() for p in _base_unet.conv_in.parameters())} params "
    #     f"(lr={args.learning_rate * 0.1:.2e})"
    # )

    ema_unet = None
    if args.use_ema:
        _trainable_at_init = [p for p in unet.parameters() if p.requires_grad]
        ema_unet = EMAModel(_trainable_at_init, decay=args.ema_decay)
        ema_unet.to(device)
        accelerator.print(f"EMA enabled (decay={args.ema_decay}, tracking {len(_trainable_at_init)} trainable params)")

    # convert to float32 for stable AdamW optimization, even if the model weights are in lower precision.
    if args.mixed_precision == "fp16" and device.type == "cuda":
        for param in unet.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    # (conv_in differential-LR logic removed; conv_in is frozen / not unfrozen)
    optimizer = AdamW(
        # [
        #     {"params": _lora_params, "lr": args.learning_rate},
        #     {"params": _conv_in_params, "lr": args.learning_rate * 0.1},
        # ],
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.adam_weight_decay,
        eps=1e-8,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(args.warmup_steps, args.max_steps),
    )

    unet, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        unet,
        optimizer,
        train_loader,
        val_loader,
        lr_scheduler,
    )

    trainable_dtype = next(
        (p.dtype for p in accelerator.unwrap_model(unet).parameters() if p.requires_grad),
        next(accelerator.unwrap_model(unet).parameters()).dtype,
    )
    accelerator.print(f"Trainable dtype:  {trainable_dtype}")
    accelerator.print(f"Accelerate MP:    {accelerator.mixed_precision}")
    latent_scale = float(getattr(vae.config, "scaling_factor", VAE_LATENT_SCALE))
    vae_dtype = next(vae.parameters()).dtype

    # ── Tracker ──────────────────────────────────────────────────────────
    tracker = None
    if accelerator.is_main_process and args.tracker != "none":
        tracker = _init_tracker(
            args.tracker,
            output_dir,
            {
                "learning_rate": args.learning_rate,
                "train_batch_size": args.train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_steps": args.max_steps,
                "warmup_steps": args.warmup_steps,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_target_ff": args.lora_target_ff,
                "lora_target_conv": args.lora_target_conv,
                "use_ema": args.use_ema,
                "ema_decay": args.ema_decay,
                "snr_gamma": args.snr_gamma,
                "conditioning_dropout_prob": args.conditioning_dropout_prob,
                "noise_offset": args.noise_offset,
                "change_mask_weight": args.change_mask_weight,
                "change_mask_threshold": args.change_mask_threshold,
                "resolution": args.resolution,
                "dataset_dir": args.dataset_dir,
                "pretrained_model": args.pretrained_model,
                "wandb_project": getattr(args, "wandb_project", "verify2act-wm"),
            },
        )
        accelerator.print(f"Tracker:          {args.tracker}")

    global_step = 0
    loss_window: List[float] = []
    best_val_loss = float("inf")
    history: List[Dict] = []
    _last_eval_step = -1
    _last_save_step = -1

    progress = tqdm(
        total=args.max_steps,
        desc="Training",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    )

    while global_step < args.max_steps:
        for batch in train_loader:
            image_t = batch["image_t"].to(device=device, dtype=vae_dtype, non_blocking=True)
            image_t1 = batch["image_t1"].to(device=device, dtype=vae_dtype, non_blocking=True)
            prompts = batch["action_text"]

            tokenized = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(device)

            with torch.no_grad():
                z_t = vae.encode(image_t).latent_dist.mode()
                z_t1 = vae.encode(image_t1).latent_dist.sample() * latent_scale
                text_emb = text_encoder(input_ids)[0]

            # Compute per-pixel change mask for spatial loss weighting.
            # The mask is 1 where pixels changed significantly between t and t+1,
            # and is downsampled to latent resolution for use in the loss.
            # Images are in [-1, 1]; threshold is converted from [0, 255] scale.
            change_mask_latent = None
            if args.change_mask_weight > 1.0:
                with torch.no_grad():
                    norm_thresh = args.change_mask_threshold / 127.5
                    # Perceptual luminance weights (ITU-R BT.601)
                    LUMA = torch.tensor([0.299, 0.587, 0.114], device=image_t.device).view(1, 3, 1, 1)
                    pixel_diff = ((image_t.float() - image_t1.float()) * LUMA).abs().sum(dim=1, keepdim=True)
                    change_mask = (pixel_diff > norm_thresh).float()
                    lh, lw = z_t1.shape[2], z_t1.shape[3]
                    change_mask_latent = F.interpolate(
                        change_mask, size=(lh, lw), mode="bilinear", align_corners=False
                    ).expand(-1, z_t1.shape[1], -1, -1)

            # IP2P-style 3-region conditioning dropout so the model learns
            # unconditional predictions required by dual CFG at inference.
            if args.conditioning_dropout_prob > 0:
                p = args.conditioning_dropout_prob
                random_p = torch.rand(z_t1.shape[0], device=device)
                # [0, 2p): drop text conditioning
                prompt_mask = (random_p < 2.0 * p).view(-1, 1, 1).to(dtype=text_emb.dtype)
                text_emb = text_emb * (1.0 - prompt_mask) + null_text_emb.to(dtype=text_emb.dtype) * prompt_mask
                # [p, 3p): zero image conditioning
                image_mask = ((random_p >= p) & (random_p < 3.0 * p)).view(-1, 1, 1, 1).to(dtype=z_t.dtype)
                z_t = z_t * (1.0 - image_mask)

            noise = torch.randn_like(z_t1)
            if args.noise_offset > 0:
                noise = noise + args.noise_offset * torch.randn(
                    z_t1.shape[0], z_t1.shape[1], 1, 1, device=device, dtype=noise.dtype,
                )
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (z_t1.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_z_t1 = noise_scheduler.add_noise(z_t1, noise, timesteps)
            model_input = torch.cat([noisy_z_t1, z_t], dim=1)

            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    noise_pred = unet(
                        model_input,
                        timesteps,
                        encoder_hidden_states=text_emb,
                    ).sample
                    # Compute per-element MSE, then apply spatial and temporal weights.
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                    # Spatial weighting: upweight pixels in the changed region.
                    if change_mask_latent is not None:
                        loss = loss * (1.0 + (args.change_mask_weight - 1.0) * change_mask_latent.float())
                    # Reduce spatial dims → [B], then apply per-sample SNR weights.
                    loss = loss.mean(dim=list(range(1, loss.ndim)))  # [B]
                    if args.snr_gamma > 0:
                        snr_w = compute_snr_weights(
                            noise_scheduler.alphas_cumprod,
                            timesteps,
                            args.snr_gamma,
                        )
                        loss = (snr_w * loss).mean()
                    else:
                        loss = loss.mean()

                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    accelerator.print(f"[warn] Non-finite loss at step {global_step + 1}; skipping batch.")
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step_loss = float(loss.item())
                loss_window.append(step_loss)
                global_step += 1
                progress.update(1)
                if ema_unet is not None:
                    ema_unet.step([p for p in accelerator.unwrap_model(unet).parameters() if p.requires_grad])

            if global_step > 0 and global_step % args.log_every == 0:
                avg_loss = float(np.mean(loss_window[-args.log_every :]))
                lr = optimizer.param_groups[0]["lr"]
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
                if tracker is not None:
                    tracker.log({"train/loss": avg_loss, "train/lr": lr}, step=global_step)

            if global_step > 0 and global_step % args.eval_every == 0 and global_step != _last_eval_step:
                _last_eval_step = global_step
                if ema_unet is not None:
                    _eval_params = [p for p in accelerator.unwrap_model(unet).parameters() if p.requires_grad]
                    ema_unet.store(_eval_params)
                    ema_unet.copy_to(_eval_params)
                val_loss = evaluate(
                    unet=unet,
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    noise_scheduler=noise_scheduler,
                    val_loader=val_loader,
                    device=device,
                    latent_scale=latent_scale,
                    eval_batches=args.eval_batches,
                    accelerator=accelerator,
                )
                accelerator.print(f"\n[step {global_step}] val_loss={val_loss:.6f}")
                if tracker is not None:
                    tracker.log({"val/loss": val_loss, "val/best_loss": min(best_val_loss, val_loss)}, step=global_step)

                # Record training history (collected on evaluation steps)
                try:
                    train_loss = float(np.mean(loss_window[-args.log_every :])) if loss_window else float("nan")
                except Exception:
                    train_loss = float("nan")
                history.append({
                    "step": int(global_step),
                    "train_loss": train_loss,
                    "val_loss": float(val_loss) if np.isfinite(val_loss) else None,
                    "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
                })

                if np.isfinite(val_loss) and val_loss < best_val_loss and accelerator.is_main_process:
                    best_val_loss = val_loss
                    save_checkpoint(
                        accelerator.unwrap_model(unet),
                        output_dir,
                        global_step,
                        {
                            "step": global_step,
                            "best_val_loss": best_val_loss,
                            "is_best": True,
                        },
                    )
                    # Always keep a fixed-path "best/" directory so demo_wm.py
                    # can reliably point to the best-val-loss adapter.
                    best_dir = output_dir / "best"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    accelerator.unwrap_model(unet).save_pretrained(best_dir / "unet_lora")
                    with open(best_dir / "best_state.json", "w") as _bfh:
                        json.dump(
                            {"step": global_step, "best_val_loss": best_val_loss},
                            _bfh,
                            indent=2,
                        )
                    accelerator.print(f"Saved best checkpoint at step {global_step} (val_loss={best_val_loss:.6f}) → {best_dir}")
                    _last_save_step = global_step
                if ema_unet is not None:
                    ema_unet.restore([p for p in accelerator.unwrap_model(unet).parameters() if p.requires_grad])
            # Periodic checkpoint: skip if a best-model save already occurred at
            # this step (same directory) or if this step was already saved.
            if (
                global_step > 0
                and global_step % args.save_every == 0
                and global_step != _last_save_step
                and accelerator.is_main_process
            ):
                _last_save_step = global_step
                if ema_unet is not None:
                    _periodic_params = [p for p in accelerator.unwrap_model(unet).parameters() if p.requires_grad]
                    ema_unet.store(_periodic_params)
                    ema_unet.copy_to(_periodic_params)
                save_checkpoint(
                    accelerator.unwrap_model(unet),
                    output_dir,
                    global_step,
                    {
                        "step": global_step,
                        "best_val_loss": best_val_loss,
                        "is_best": False,
                    },
                )
                if ema_unet is not None:
                    ema_unet.restore([p for p in accelerator.unwrap_model(unet).parameters() if p.requires_grad])
                accelerator.print(f"Saved periodic checkpoint at step {global_step}")

            if global_step >= args.max_steps:
                break

    progress.close()

    accelerator.wait_for_everyone()

    final_dir = output_dir / "final"
    if accelerator.is_main_process:
        final_dir.mkdir(parents=True, exist_ok=True)
        if ema_unet is not None:
            ema_unet.copy_to([p for p in accelerator.unwrap_model(unet).parameters() if p.requires_grad])
        accelerator.unwrap_model(unet).save_pretrained(final_dir / "unet_lora")
        # Read best_state to include the best step in the summary
        best_step = None
        best_state_path = output_dir / "best" / "best_state.json"
        if best_state_path.exists():
            with open(best_state_path) as _bsfh:
                best_step = json.load(_bsfh).get("step")
        with open(final_dir / "train_summary.json", "w") as handle:
            json.dump(
                {
                    "max_steps": args.max_steps,
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                    "seed": args.seed,
                    "pretrained_model": args.pretrained_model,
                },
                handle,
                indent=2,
            )

        # Save train history and config (similar to train_prm)
        try:
            with open(output_dir / "train_history.json", "w") as handle:
                json.dump(history, handle, indent=2)
        except Exception:
            accelerator.print("[warn] Could not write train_history.json")

        try:
            with open(output_dir / "train_config.json", "w") as handle:
                json.dump(vars(args), handle, indent=2)
        except Exception:
            accelerator.print("[warn] Could not write train_config.json")

        accelerator.print("\nTraining complete.")
        accelerator.print(f"Final LoRA adapter saved to: {final_dir / 'unet_lora'}")

    if tracker is not None:
        tracker.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet LoRA for Verify2Act world model")

    parser.add_argument("--dataset-dir", type=str, default="robosuite/data_capture_wm/dataset/nut_assembly")
    parser.add_argument("--dataset-type", type=str, default="robosuite", choices=["robosuite", "calvin"],
                        help="Type of dataset loader to use")
    parser.add_argument("--transitions-file", type=str, default="transitions.jsonl",
                        help="JSONL filename inside dataset-dir (e.g. 'transitions.jsonl' or "
                             "'transitions_subskill.jsonl').")
    parser.add_argument("--output-dir", type=str, default="verify2act/output/wm")
    parser.add_argument("--pretrained-model", type=str, default="timbrooks/instruct-pix2pix")
    parser.add_argument(
        "--vae-model",
        type=str,
        default="",
        help="Optional VAE source; if empty uses --pretrained-model.",
    )
    parser.add_argument(
        "--vae-subfolder",
        type=str,
        default="auto",
        help="VAE subfolder to load (e.g. 'vae', 'vae_ema', 'root'). Use 'auto' to resolve automatically.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load VAE only from local cache/files; do not reach HuggingFace Hub.",
    )

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--train-batch-size", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-ff",
        dest="lora_target_ff",
        action="store_true",
        default=True,
        help="Extend LoRA to FF layers (ff.net.0.proj, ff.net.2) in addition to QKV/out (default: enabled).",
    )
    parser.add_argument(
        "--no-lora-target-ff",
        dest="lora_target_ff",
        action="store_false",
        help="Disable extending LoRA to FF layers.",
    )
    parser.add_argument(
        "--lora-target-conv",
        action="store_true",
        help="Extend LoRA to conv layers (conv1, conv2, proj_in, proj_out, conv_shortcut) in ResNet/attention blocks.",
    )
    parser.add_argument(
        "--snr-gamma",
        type=float,
        default=5.0,
        help="Min-SNR-γ loss weighting (Hang et al. 2023). Set 0 to use plain MSE.",
    )
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True,
                        help="Enable EMA weight tracking for eval/checkpoints (default: enabled).")
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false",
                        help="Disable EMA weight tracking.")
    parser.add_argument("--ema-decay", type=float, default=0.9999,
                        help="EMA decay rate (default: 0.9999).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"], default="fp16")

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)

    parser.add_argument("--enable-gradient-checkpointing", action="store_true")
    parser.add_argument("--enable-xformers", action="store_true")

    parser.add_argument("--tracker", type=str, choices=["tensorboard", "wandb", "none"], default="tensorboard",
                        help="Experiment tracker: tensorboard (default), wandb, or none.")
    parser.add_argument("--wandb-project", type=str, default="verify2act-wm",
                        help="W&B project name (only used when --tracker=wandb).")
    parser.add_argument("--conditioning-dropout-prob", type=float, default=0.15,
                        help="IP2P-style 3-region conditioning dropout probability. Set 0 to disable.")
    parser.add_argument("--noise-offset", type=float, default=0.05,
                        help="Noise offset magnitude added during training. Set 0 to disable.")
    parser.add_argument("--change-mask-weight", type=float, default=2.0,
                        help="Spatial loss weight multiplier for changed pixels. 1.0=disabled; e.g. 5.0 puts 5x more "
                             "gradient on pixels that differ between image_t and image_t1, focusing capacity on the "
                             "edited region rather than the static background.")
    parser.add_argument("--change-mask-threshold", type=float, default=15.0,
                        help="Pixel-space difference threshold (0-255 scale) for classifying a pixel as 'changed'. "
                             "Only used when --change-mask-weight > 1.0.")

    return parser.parse_args()

if __name__ == "__main__":
    main()
