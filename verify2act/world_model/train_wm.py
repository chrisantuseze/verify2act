#!/usr/bin/env python3
"""
Phase A training script for Verify2Act world model.

Trains LoRA adapters on InstructPix2Pix UNet to predict noise in z_{t+1},
conditioned on current-frame latent z_t and action text.

Expected dataset format:
- <dataset_dir>/transitions.jsonl
- image paths stored in each row under keys `image_t` and `image_t1`
  as paths relative to dataset_dir.
"""

import argparse
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from diffusers import DDPMScheduler, StableDiffusionInstructPix2PixPipeline

from verify2act.utils.data_loader import WMTransitionDataset
try:
    from verify2act.utils import VAE_LATENT_SCALE, load_vae_encoder
except ImportError:
    from utils import VAE_LATENT_SCALE, load_vae_encoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TransitionRow:
    episode_id: str
    timestep: int
    image_t: str
    image_t1: str
    action_text: str


 


def get_dtype(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def build_lr_lambda(warmup_steps: int):
    def lr_lambda(step: int):
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

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
    weight_dtype,
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

            z_t = vae.encode(image_t).latent_dist.sample() * latent_scale
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

    accelerator.print("\nLoading datasets...")
    train_ds = WMTransitionDataset(
        dataset_dir=args.dataset_dir,
        image_size=args.resolution,
        split="train",
        val_frac=args.val_frac,
        seed=args.seed,
    )
    val_ds = WMTransitionDataset(
        dataset_dir=args.dataset_dir,
        image_size=args.resolution,
        split="val",
        val_frac=args.val_frac,
        seed=args.seed,
    )
    accelerator.print(f"Train samples: {len(train_ds)}")
    accelerator.print(f"Val samples:   {len(val_ds)}")

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

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    if args.mixed_precision == "fp16" and device.type == "cuda":
        for param in unet.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    optimizer = AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.adam_weight_decay,
        eps=1e-8,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(args.warmup_steps),
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

    global_step = 0
    loss_window: List[float] = []
    best_val_loss = float("inf")

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
                z_t = vae.encode(image_t).latent_dist.sample() * latent_scale
                z_t1 = vae.encode(image_t1).latent_dist.sample() * latent_scale
                text_emb = text_encoder(input_ids)[0]

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

            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    noise_pred = unet(
                        model_input,
                        timesteps,
                        encoder_hidden_states=text_emb,
                    ).sample
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

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

            if global_step > 0 and global_step % args.log_every == 0:
                avg_loss = float(np.mean(loss_window[-args.log_every :]))
                lr = optimizer.param_groups[0]["lr"]
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})

            if global_step > 0 and global_step % args.eval_every == 0:
                val_loss = evaluate(
                    unet=unet,
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    noise_scheduler=noise_scheduler,
                    val_loader=val_loader,
                    device=device,
                    latent_scale=latent_scale,
                    weight_dtype=weight_dtype,
                    eval_batches=args.eval_batches,
                    accelerator=accelerator,
                )
                accelerator.print(f"\n[step {global_step}] val_loss={val_loss:.6f}")

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
                    accelerator.print(f"Saved best checkpoint at step {global_step}")

            if global_step > 0 and global_step % args.save_every == 0 and accelerator.is_main_process:
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
                accelerator.print(f"Saved periodic checkpoint at step {global_step}")

            if global_step >= args.max_steps:
                break

    progress.close()

    accelerator.wait_for_everyone()

    final_dir = output_dir / "final"
    if accelerator.is_main_process:
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(unet).save_pretrained(final_dir / "unet_lora")
        with open(final_dir / "train_summary.json", "w") as handle:
            json.dump(
                {
                    "max_steps": args.max_steps,
                    "best_val_loss": best_val_loss,
                    "seed": args.seed,
                    "pretrained_model": args.pretrained_model,
                },
                handle,
                indent=2,
            )

        accelerator.print("\nTraining complete.")
        accelerator.print(f"Final LoRA adapter saved to: {final_dir / 'unet_lora'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet LoRA for Verify2Act world model")

    parser.add_argument("--dataset-dir", type=str, default="robosuite/data_capture_wm/dataset/nut_assembly/episodes", required=True)
    parser.add_argument("--output-dir", type=str, default="verify2act/output/wm", required=True)
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

    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"], default="fp16")

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)

    parser.add_argument("--enable-gradient-checkpointing", action="store_true")
    parser.add_argument("--enable-xformers", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    main()
