#!/usr/bin/env python3
"""
Phase B training script for Verify2Act: VAE Decoder Finetune.

Improves VAE decoder precision on the nut-assembly visual domain so that
decoded images are sharper for small critical objects (nuts, pegs).

Loss:  L = λ_1 * L1(I_gt, I_recon) + λ_2 * LPIPS(I_gt, I_recon)

The VAE *encoder* is kept frozen throughout so that the shared latent
space used by the critic (Phase C) is not disturbed.

Expected dataset: same transitions.jsonl used by Phase A.
Each image_t1 is encoded with the frozen encoder, then decoded with the
trainable decoder, and compared to the ground-truth image_t1.

Usage:
    python verify2act/train_decoder.py \
        --dataset-dir robosuite/data_capture_wm/dataset/nut_assembly/episodes \
        --output-dir  verify2act/output/decoder \
        --max-epochs 5
"""
import sys
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from verify2act.data_loader import WMTransitionDataset
try:
    from verify2act.utils import VAE_LATENT_SCALE, load_vae_encoder
except ImportError:
    from utils import VAE_LATENT_SCALE, load_vae_encoder


# ── Helpers ─────────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


# ── Evaluation ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    vae,
    val_loader,
    device: torch.device,
    latent_scale: float,
    lpips_fn,
    l1_weight: float,
    lpips_weight: float,
    eval_batches: int,
    accelerator=None,
):
    """Compute mean L1 + LPIPS loss on at most *eval_batches* validation batches."""
    vae.eval()
    vae_dtype = next(vae.parameters()).dtype

    loss_sum = torch.tensor(0.0, device=device)
    l1_sum = torch.tensor(0.0, device=device)
    lpips_sum = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= eval_batches:
            break

        image_t1 = batch["image_t1"].to(device=device, dtype=vae_dtype)

        # Encode with frozen encoder → decode with trainable decoder
        z = vae.encode(image_t1).latent_dist.sample() * latent_scale
        recon = vae.decode(z / latent_scale).sample  # [-1, 1]

        loss_l1 = F.l1_loss(recon.float(), image_t1.float())
        loss_lpips = lpips_fn(recon.float(), image_t1.float()).mean()
        loss = l1_weight * loss_l1 + lpips_weight * loss_lpips

        if torch.isfinite(loss):
            loss_sum += loss.detach().float()
            l1_sum += loss_l1.detach().float()
            lpips_sum += loss_lpips.detach().float()
            count += 1.0

    if accelerator is not None:
        loss_sum = accelerator.reduce(loss_sum, reduction="sum")
        l1_sum = accelerator.reduce(l1_sum, reduction="sum")
        lpips_sum = accelerator.reduce(lpips_sum, reduction="sum")
        count = accelerator.reduce(count, reduction="sum")

    vae.train()
    # Re-freeze encoder after switching back to train mode
    raw_vae = accelerator.unwrap_model(vae) if accelerator is not None else vae
    _freeze_encoder(raw_vae)

    if count.item() <= 0:
        return math.nan, math.nan, math.nan
    n = count.item()
    return float(loss_sum / n), float(l1_sum / n), float(lpips_sum / n)


# ── Freeze / unfreeze helpers ───────────────────────────────────────────────────


def _freeze_encoder(vae):
    """Freeze encoder + quant_conv (latent space must not shift)."""
    vae.encoder.requires_grad_(False)
    if hasattr(vae, "quant_conv"):
        vae.quant_conv.requires_grad_(False)


def _unfreeze_decoder(vae):
    """Unfreeze decoder + post_quant_conv."""
    vae.decoder.requires_grad_(True)
    if hasattr(vae, "post_quant_conv"):
        vae.post_quant_conv.requires_grad_(True)


def _get_trainable_params(vae):
    """Return only the decoder parameters that require grad."""
    return [p for p in vae.parameters() if p.requires_grad]


# ── Checkpoint ──────────────────────────────────────────────────────────────────


def save_checkpoint(vae, output_dir: Path, step: int, train_state: Dict):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save only decoder state dict (encoder is frozen/unchanged)
    torch.save(vae.decoder.state_dict(), ckpt_dir / "decoder_state_dict.pt")
    if hasattr(vae, "post_quant_conv"):
        torch.save(vae.post_quant_conv.state_dict(), ckpt_dir / "post_quant_conv_state_dict.pt")
    with open(ckpt_dir / "train_state.json", "w") as f:
        json.dump(train_state, f, indent=2)


def save_full_vae(vae, output_dir: Path):
    """Save the entire VAE (frozen encoder + finetuned decoder) for easy loading."""
    final_dir = output_dir / "final" / "vae"
    final_dir.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(final_dir)


# ── Main ────────────────────────────────────────────────────────────────────────


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
        accelerator.print("[warn] CUDA unavailable, forcing mixed precision to 'no'.")
        args.mixed_precision = "no"

    weight_dtype = get_dtype(args.mixed_precision)

    accelerator.print("=" * 80)
    accelerator.print("VERIFY2ACT PHASE B: VAE DECODER FINETUNE")
    accelerator.print("=" * 80)
    accelerator.print(f"Dataset:           {args.dataset_dir}")
    accelerator.print(f"Output:            {output_dir}")
    accelerator.print(f"VAE model:         {args.vae_model}")
    accelerator.print(f"Device:            {device}")
    accelerator.print(f"Precision:         {args.mixed_precision}")
    accelerator.print(f"Batch size:        {args.batch_size}")
    accelerator.print(f"Grad accum:        {args.gradient_accumulation_steps}")
    accelerator.print(f"LR:                {args.learning_rate}")
    accelerator.print(f"Max steps:         {args.max_steps}")
    accelerator.print(f"L1 weight:         {args.l1_weight}")
    accelerator.print(f"LPIPS weight:      {args.lpips_weight}")

    # ── Datasets ────────────────────────────────────────────────────────────────

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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # ── Model ───────────────────────────────────────────────────────────────────

    accelerator.print("\nLoading VAE...")
    vae_model = args.vae_model if args.vae_model else args.pretrained_model
    # Always load VAE in float32: mixed-precision GradScaler requires FP32
    # parameters; autocast (via accelerator.autocast()) handles FP16 in the
    # forward pass without casting the weights themselves.
    vae, resolved_subfolder = load_vae_encoder(
        model_name_or_path=vae_model,
        device=device,
        torch_dtype=torch.float32,
        subfolder=args.vae_subfolder,
        local_files_only=args.local_files_only,
    )
    accelerator.print(
        f"Loaded VAE from model={vae_model} "
        f"(subfolder={resolved_subfolder}, dtype={weight_dtype})"
    )

    # load_vae_encoder freezes everything + sets eval. Undo for decoder.
    vae.train()
    _freeze_encoder(vae)
    _unfreeze_decoder(vae)

    n_total = sum(p.numel() for p in vae.parameters())
    n_train = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    accelerator.print(f"VAE total params:     {n_total:,}")
    accelerator.print(f"VAE trainable params: {n_train:,} (decoder only)")

    latent_scale = float(getattr(vae.config, "scaling_factor", VAE_LATENT_SCALE))

    # ── LPIPS loss network ──────────────────────────────────────────────────────

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.requires_grad_(False)
    lpips_fn.eval()

    # ── Optimizer ───────────────────────────────────────────────────────────────

    trainable_params = _get_trainable_params(vae)
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # Keep LR constant to match Phase B spec
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )

    # ── Accelerate prepare ──────────────────────────────────────────────────────

    vae, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_loader, val_loader, lr_scheduler,
    )
    # Re-derive trainable params from the (possibly wrapped) model so that
    # clip_grad_norm_ and the optimizer act on the correct live tensors.
    trainable_params = _get_trainable_params(accelerator.unwrap_model(vae))

    # ── Training loop ───────────────────────────────────────────────────────────

    # Keep inputs in float32 to match model parameters (autocast handles FP16
    # only inside the autocast region during the forward pass).
    vae_dtype = torch.float32
    best_val_loss = float("inf")
    global_step = 0
    history: List[Dict] = []

    progress = tqdm(
        total=args.max_steps,
        desc="Training",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    )

    while global_step < args.max_steps:
        vae.train()
        _freeze_encoder(accelerator.unwrap_model(vae))

        for batch in train_loader:
            image_t1 = batch["image_t1"].to(device=device, dtype=vae_dtype, non_blocking=True)

            with accelerator.accumulate(vae):
                # Encode (frozen) → produce latent
                with torch.no_grad():
                    z = vae.encode(image_t1).latent_dist.sample() * latent_scale

                # Decode (trainable)
                with accelerator.autocast():
                    recon = vae.decode(z / latent_scale).sample  # [-1, 1]

                    loss_l1 = F.l1_loss(recon.float(), image_t1.float())
                    loss_lpips = lpips_fn(recon.float(), image_t1.float()).mean()
                    loss = args.l1_weight * loss_l1 + args.lpips_weight * loss_lpips

                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    accelerator.print(f"[warn] Non-finite loss at step {global_step + 1}; skipping batch.")
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                if global_step % args.log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})

                if global_step % args.eval_every == 0:
                    val_loss, val_l1, val_lpips = evaluate(
                        vae=vae,
                        val_loader=val_loader,
                        device=device,
                        latent_scale=latent_scale,
                        lpips_fn=lpips_fn,
                        l1_weight=args.l1_weight,
                        lpips_weight=args.lpips_weight,
                        eval_batches=args.eval_batches,
                        accelerator=accelerator,
                    )
                    accelerator.print(
                        f"\n[step {global_step}] val_loss={val_loss:.5f}  "
                        f"val_l1={val_l1:.5f}  val_lpips={val_lpips:.5f}"
                    )
                    # Record training history (collected on evaluation steps)
                    try:
                        train_loss = float(loss.item()) if 'loss' in locals() else float('nan')
                    except Exception:
                        train_loss = float('nan')
                    history.append({
                        "step": int(global_step),
                        "train_loss": train_loss,
                        "val_loss": float(val_loss) if np.isfinite(val_loss) else None,
                        "val_l1": float(val_l1) if np.isfinite(val_l1) else None,
                        "val_lpips": float(val_lpips) if np.isfinite(val_lpips) else None,
                        "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
                    })
                    if np.isfinite(val_loss) and val_loss < best_val_loss and accelerator.is_main_process:
                        best_val_loss = val_loss
                        save_checkpoint(
                            accelerator.unwrap_model(vae),
                            output_dir,
                            global_step,
                            {"step": global_step, "best_val_loss": best_val_loss, "is_best": True},
                        )
                        accelerator.print(f"  ↑ New best val_loss={best_val_loss:.5f}")

                if global_step % args.save_every == 0 and accelerator.is_main_process:
                    save_checkpoint(
                        accelerator.unwrap_model(vae),
                        output_dir,
                        global_step,
                        {"step": global_step, "best_val_loss": best_val_loss, "is_best": False},
                    )
                    accelerator.print(f"Saved periodic checkpoint at step {global_step}")

                if global_step >= args.max_steps:
                    break

    progress.close()

    # ── Save final model ────────────────────────────────────────────────────────

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(vae)
        save_full_vae(unwrapped, output_dir)

        with open(output_dir / "final" / "train_summary.json", "w") as f:
            json.dump(
                {
                    "max_steps": args.max_steps,
                    "global_steps": global_step,
                    "best_val_loss": best_val_loss,
                    "seed": args.seed,
                    "vae_model": vae_model,
                    "l1_weight": args.l1_weight,
                    "lpips_weight": args.lpips_weight,
                    "learning_rate": args.learning_rate,
                },
                f,
                indent=2,
            )

        accelerator.print("\nTraining complete.")
        accelerator.print(f"Finetuned VAE saved to: {output_dir / 'final' / 'vae'}")
        # Save train history and config (similar to train_prm)
        try:
            with open(output_dir / "train_history.json", "w") as f:
                json.dump(history, f, indent=2)
        except Exception:
            accelerator.print("[warn] Could not write train_history.json")

        try:
            with open(output_dir / "train_config.json", "w") as f:
                json.dump(vars(args), f, indent=2)
        except Exception:
            accelerator.print("[warn] Could not write train_config.json")


# ── CLI ─────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Phase B: Finetune VAE decoder for Verify2Act")

    parser.add_argument("--dataset-dir", type=str,
                        default="robosuite/data_capture_wm/dataset/nut_assembly")
    parser.add_argument("--output-dir", type=str,
                        default="verify2act/output/decoder")
    parser.add_argument("--pretrained-model", type=str, default="timbrooks/instruct-pix2pix",
                        help="Fallback model name if --vae-model is empty.")
    parser.add_argument("--vae-model", type=str, default="",
                        help="VAE source model; if empty uses --pretrained-model.")
    parser.add_argument("--vae-subfolder", type=str, default="auto",
                        help="VAE subfolder (e.g. 'vae', 'vae_ema', 'root'). 'auto' to resolve.")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Load VAE only from local cache; do not reach HuggingFace Hub.")

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"], default="fp16")

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)

    return parser.parse_args()


if __name__ == "__main__":
    main()
