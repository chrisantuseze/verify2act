import argparse
import json
import os
import re
import warnings

warnings.filterwarnings("ignore", message=".*torch\\.cuda\\.amp\\.GradScaler.*")
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import sys
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import lpips

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed

from verify2act.latent_wm.decoder import FeatureDecoder
from verify2act.latent_wm.train_dynamics import LatentDynamicsDataset, FeatureExtractor


def train_decoder(args):
    set_seed(args.seed)

    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(dataloader_config=dataloader_config, rng_types=[])
    device = accelerator.device

    if accelerator.is_local_main_process:
        print(f"Using device: {device}  |  Num processes: {accelerator.num_processes}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.dataset_type == "calvin":
        from verify2act.data_loader_calvin import build_calvin_datasets
        train_dataset, val_dataset = build_calvin_datasets(
            dataset_dir=args.dataset_dir,
            val_frac=args.val_frac,
            image_size=args.image_size,
            history_len=1,
            seed=args.seed,
            use_cache=False,
        )
    else:
        dataset = LatentDynamicsDataset(
            dataset_dir=args.dataset_dir,
            transitions_file=args.transitions_file,
            history_len=1,
            image_size=args.image_size,
            use_cache=False,
        )
        train_size = int((1.0 - args.val_frac) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

    if accelerator.is_local_main_process:
        print(f"Dataset loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # ── Models ────────────────────────────────────────────────────────────────
    extractor = FeatureExtractor(device, dino_channels=args.dino_channels)

    visualizer = FeatureDecoder(
        dino_channels=args.dino_channels,
        model_channels=256,
    ).to(device)

    # LPIPS for perceptual loss — frozen, no gradients needed
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.requires_grad_(False)
    lpips_fn.eval()

    optimizer = torch.optim.AdamW(visualizer.parameters(), lr=args.lr)

    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume_from:
        if os.path.exists(args.resume_from):
            if accelerator.is_local_main_process:
                print(f"Resuming from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                visualizer.decoder.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"]
                best_val_loss = checkpoint.get("val_loss", float("inf"))
            else:
                visualizer.decoder.load_state_dict(checkpoint)
                match = re.search(r"ep(\d+)\.pt$", args.resume_from)
                if match:
                    start_epoch = int(match.group(1))
            if accelerator.is_local_main_process:
                print(f"Resumed from {args.resume_from} (epoch {start_epoch})")
        else:
            if accelerator.is_local_main_process:
                print(f"Warning: Checkpoint {args.resume_from} not found. Starting from scratch.")

    # ── Accelerate prepare ────────────────────────────────────────────────────
    visualizer, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        visualizer, optimizer, train_dataloader, val_dataloader
    )
    # lpips_fn is frozen; wrap manually so it lives on the right device
    lpips_fn = lpips_fn.to(device)

    # ── Output dir / config ───────────────────────────────────────────────────
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Config saved to {args.output_dir}/config.json")

    # Pre-compute normalisation tensors (they stay on `device` throughout)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.num_epochs):
        visualizer.train()
        train_loss = 0.0

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]",
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            if isinstance(batch, dict):
                target_img = batch["image_t1"].to(device)
            else:
                _, target_img, _, _ = batch
                target_img = target_img.to(device)

            # Extract ground-truth DINO features (no gradients needed)
            with torch.no_grad():
                F_target = extractor.extract_dino(target_img)

            # Decode features → image  (output in [-1, 1])
            unwrapped = accelerator.unwrap_model(visualizer)
            pred_img = unwrapped.decode(F_target)

            # Denormalise ground truth from ImageNet norm → [0,1] → [-1,1]
            target_img_denorm  = (target_img * std + mean).clamp(0, 1)
            target_img_scaled  = target_img_denorm * 2.0 - 1.0

            # Resize if spatial dims differ
            if pred_img.shape[-2:] != target_img_scaled.shape[-2:]:
                pred_img = F.interpolate(
                    pred_img, size=target_img_scaled.shape[-2:], mode="bilinear"
                )

            # Losses
            loss_l1    = F.l1_loss(pred_img, target_img_scaled)
            loss_lpips = lpips_fn(pred_img, target_img_scaled).mean()
            loss       = args.l1_weight * loss_l1 + args.lpips_weight * loss_lpips

            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(visualizer.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({
                "loss":  f"{loss.item():.4f}",
                "l1":    f"{loss_l1.item():.4f}",
                "lpips": f"{loss_lpips.item():.4f}",
            })

        train_loss /= len(train_dataloader)

        # ── Validation Loop ───────────────────────────────────────────────────
        visualizer.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                val_dataloader,
                desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]",
                dynamic_ncols=True,
                disable=not accelerator.is_local_main_process,
            ):
                if isinstance(batch, dict):
                    target_img = batch["image_t1"].to(device)
                else:
                    _, target_img, _, _ = batch
                    target_img = target_img.to(device)

                F_target = extractor.extract_dino(target_img)
                unwrapped = accelerator.unwrap_model(visualizer)
                pred_img  = unwrapped.decode(F_target)

                target_img_denorm = (target_img * std + mean).clamp(0, 1)
                target_img_scaled = target_img_denorm * 2.0 - 1.0

                if pred_img.shape[-2:] != target_img_scaled.shape[-2:]:
                    pred_img = F.interpolate(
                        pred_img, size=target_img_scaled.shape[-2:], mode="bilinear"
                    )

                loss_l1    = F.l1_loss(pred_img, target_img_scaled)
                loss_lpips = lpips_fn(pred_img, target_img_scaled).mean()
                loss       = args.l1_weight * loss_l1 + args.lpips_weight * loss_lpips
                val_loss  += loss.item()

        val_loss = val_loss / max(len(val_dataloader), 1)

        # Reduce val_loss across all processes so every rank has the same value
        val_loss_t = torch.tensor(val_loss, device=device)
        val_loss   = accelerator.reduce(val_loss_t, reduction="mean").item()

        # ── Logging & Checkpointing ───────────────────────────────────────────
        if accelerator.is_main_process:
            print(
                f"Epoch {epoch+1}/{args.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            os.makedirs(args.output_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(visualizer)

            ckpt = {
                "model_state_dict":     unwrapped.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch":                epoch + 1,
                "val_loss":             val_loss,
                "args":                 vars(args),
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, f"{args.output_dir}/latent_decoder_best.pt")
                print(
                    f"  ↳ New best val={val_loss:.4f}, "
                    f"saved to {args.output_dir}/latent_decoder_best.pt"
                )

            if (epoch + 1) % args.checkpoint_freq == 0:
                torch.save(ckpt, f"{args.output_dir}/latent_decoder_ep{epoch+1}.pt")
                print(f"  ↳ Checkpoint saved: latent_decoder_ep{epoch+1}.pt")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Feature Decoder for Latent World Model")
    parser.add_argument("--dataset-dir",      type=str, default="robosuite/data_capture_wm/dataset/nut_assembly_merged")
    parser.add_argument("--dataset-type",     type=str, default="robosuite", choices=["robosuite", "calvin"])
    parser.add_argument("--transitions-file", type=str, default="transitions.jsonl")
    parser.add_argument("--val-frac",         type=float, default=0.1)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output-dir",       type=str,   default="verify2act/output/v2a_wm/decoder")
    parser.add_argument("--image-size",       type=int,   default=224)
    parser.add_argument("--batch-size",       type=int,   default=8)
    parser.add_argument("--num-epochs",       type=int,   default=50)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--l1-weight",        type=float, default=1.0)
    parser.add_argument("--lpips-weight",     type=float, default=0.5)
    parser.add_argument("--checkpoint-freq",  type=int,   default=20)
    parser.add_argument("--resume-from",      type=str,   default=None,
                        help="Path to decoder checkpoint to resume from")
    parser.add_argument("--dino-channels",    type=int,   default=1024)

    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")

    args = parser.parse_args()
    train_decoder(args)
