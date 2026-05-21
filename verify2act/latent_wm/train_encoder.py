"""Pre-train the DeltaEncoder + DeltaDecoder on DINO feature differences.

This is **Stage 1** of the two-stage training pipeline that mirrors RLA-WM's
approach:

    Stage 1 (this script): Train Encoder + Decoder jointly with MSE
        reconstruction loss on ``F_{t+1} - F_t`` samples.  Save the
        trained encoder checkpoint.

    Stage 2 (train_dynamics.py): Load the frozen encoder.  Flow-match in the
        compact latent token space  ``[B, num_latent_tokens, token_dim]``
        instead of the raw DINO feature space  ``[B, 256, 768]``.

Usage
-----
    python -m verify2act.latent_wm.train_encoder \\
        --dataset-dir /path/to/dataset \\
        --output-dir  verify2act/output/delta_encoder \\
        --num-latent-tokens 16 \\
        --token-dim 64 \\
        --num-epochs 30
"""

import json
import os
import argparse
from pathlib import Path
from typing import List
import warnings

warnings.filterwarnings("ignore", message=".*torch\\.cuda\\.amp\\.GradScaler.*")
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed

import sys
from pathlib import Path as _Path
root_dir = _Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from verify2act.latent_wm.delta_encoder import DeltaEncoder, DeltaDecoder
from verify2act.latent_wm.train_dynamics import FeatureExtractor


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)

    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(dataloader_config=dataloader_config, rng_types=[])
    device = accelerator.device

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.dataset_type == "calvin":
        from verify2act.data_loader_calvin import build_calvin_datasets
        train_dataset, val_dataset = build_calvin_datasets(
            dataset_dir=args.dataset_dir,
            val_frac=args.val_frac,
            image_size=args.image_size,
            history_len=args.history_len,
            seed=args.seed,
        )
    else:
        from verify2act.latent_wm.train_dynamics import LatentDynamicsDataset
        dataset = LatentDynamicsDataset(
            dataset_dir=args.dataset_dir,
            transitions_file=args.transitions_file,
            history_len=args.history_len,
            image_size=args.image_size,
        )
        train_size = int((1.0 - args.val_frac) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

    train_dl = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_dl = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # ── Models ───────────────────────────────────────────────────────────────
    extractor = FeatureExtractor(device)

    encoder = DeltaEncoder(
        dino_channels=args.dino_channels,
        model_channels=args.model_channels,
        token_dim=args.token_dim,
        num_latent_tokens=args.num_latent_tokens,
        num_blocks=args.num_enc_blocks,
        num_heads=args.num_heads,
    ).to(device)

    decoder = DeltaDecoder(
        token_dim=args.token_dim,
        model_channels=args.model_channels,
        dino_channels=args.dino_channels,
        num_patches=args.num_patches,
        num_blocks=args.num_dec_blocks,
        num_heads=args.num_heads,
    ).to(device)

    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in decoder.parameters())
    if accelerator.is_local_main_process:
        print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
        print(f"Total trainable: {total_params:,}")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        if accelerator.is_local_main_process:
            print(f"Resumed from {args.resume_from} (epoch {start_epoch})")

    encoder, decoder, optimizer, train_dl, val_dl = accelerator.prepare(
        encoder, decoder, optimizer, train_dl, val_dl
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    writer = None
    if accelerator.is_main_process:
        os.makedirs(f"{args.output_dir}/ckpt", exist_ok=True)
        writer = SummaryWriter(log_dir=f"{args.output_dir}/tb_logs")
        with open(f"{args.output_dir}/config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Config saved to {args.output_dir}/config.json")

    for epoch in range(start_epoch, args.num_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        encoder.train()
        decoder.train()
        train_loss = 0.0

        pbar = tqdm(
            train_dl,
            desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]",
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            # Unpack batch — support both CALVIN dict and RoboSuite tuple
            if isinstance(batch, dict):
                history_imgs = batch["history_imgs"].to(device)
                target_img   = batch["image_t1"].to(device)
            else:
                history_imgs, target_img, _, _ = batch
                history_imgs = history_imgs.to(device)
                target_img   = target_img.to(device)

            # Feature extraction (no grad — extractor is frozen)
            with torch.no_grad():
                F_history = extractor.extract_dino(history_imgs)  # (B, H, P, C)
                F_t  = F_history[:, -1, :, :]                      # (B, P, C)
                F_t1 = extractor.extract_dino(target_img)          # (B, P, C)
                residual_target = F_t1 - F_t                        # (B, P, C)

            # Encode → decode
            unwrapped_enc = accelerator.unwrap_model(encoder)
            unwrapped_dec = accelerator.unwrap_model(decoder)

            latent = unwrapped_enc(residual_target)          # (B, N, token_dim)
            recon  = unwrapped_dec(latent)                   # (B, P, dino_ch)

            loss = F.mse_loss(recon, residual_target.detach())

            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()), 1.0
                )
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"mse": f"{loss.item():.5f}"})

        train_loss /= len(train_dl)

        # ── Validation ────────────────────────────────────────────────────
        encoder.eval()
        decoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                val_dl,
                desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]",
                dynamic_ncols=True,
                disable=not accelerator.is_local_main_process,
            ):
                if isinstance(batch, dict):
                    history_imgs = batch["history_imgs"].to(device)
                    target_img   = batch["image_t1"].to(device)
                else:
                    history_imgs, target_img, _, _ = batch
                    history_imgs = history_imgs.to(device)
                    target_img   = target_img.to(device)

                F_history = extractor.extract_dino(history_imgs)
                F_t  = F_history[:, -1, :, :]
                F_t1 = extractor.extract_dino(target_img)
                residual_target = F_t1 - F_t

                unwrapped_enc = accelerator.unwrap_model(encoder)
                unwrapped_dec = accelerator.unwrap_model(decoder)

                latent = unwrapped_enc(residual_target)
                recon  = unwrapped_dec(latent)
                val_loss += F.mse_loss(recon, residual_target).item()

        val_loss /= max(len(val_dl), 1)

        # ── Logging & checkpointing ────────────────────────────────────────
        if accelerator.is_main_process:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val",   val_loss,   epoch)
            print(
                f"Epoch {epoch+1}/{args.num_epochs}  "
                f"train={train_loss:.5f}  val={val_loss:.5f}"
            )

            unwrapped_enc = accelerator.unwrap_model(encoder)
            unwrapped_dec = accelerator.unwrap_model(decoder)

            ckpt = {
                "encoder":   unwrapped_enc.state_dict(),
                "decoder":   unwrapped_dec.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch":     epoch + 1,
                "args":      vars(args),
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, f"{args.output_dir}/ckpt/delta_encoder_best.pt")
                # Also save encoder-only for easy loading in train_dynamics.py
                torch.save(
                    unwrapped_enc.state_dict(),
                    f"{args.output_dir}/ckpt/encoder_only_best.pt",
                )
                print(f"  ↳ New best val={val_loss:.5f}, checkpoint saved.")

            if (epoch + 1) % args.checkpoint_freq == 0:
                torch.save(ckpt, f"{args.output_dir}/ckpt/delta_encoder_ep{epoch+1}.pt")

    if accelerator.is_main_process and writer:
        writer.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-train DeltaEncoder + DeltaDecoder for verify2act"
    )
    # Dataset
    p.add_argument("--dataset-dir",      type=str, required=True)
    p.add_argument("--dataset-type",     type=str, default="robosuite",
                   choices=["robosuite", "calvin"])
    p.add_argument("--transitions-file", type=str, default="transitions.jsonl")
    p.add_argument("--val-frac",         type=float, default=0.1)
    p.add_argument("--image-size",       type=int,   default=224)
    p.add_argument("--history-len",      type=int,   default=3,
                   help="Only used to load the dataset; encoder sees only F_t and F_{t+1}.")
    # Encoder / decoder architecture
    p.add_argument("--dino-channels",      type=int, default=768)
    p.add_argument("--model-channels",     type=int, default=512)
    p.add_argument("--token-dim",          type=int, default=64)
    p.add_argument("--num-latent-tokens",  type=int, default=16)
    p.add_argument("--num-patches",        type=int, default=256)
    p.add_argument("--num-enc-blocks",     type=int, default=4)
    p.add_argument("--num-dec-blocks",     type=int, default=4)
    p.add_argument("--num-heads",          type=int, default=8)
    # Training
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--num-epochs",       type=int,   default=30)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--checkpoint-freq",  type=int,   default=5)
    p.add_argument("--resume-from",      type=str,   default=None)
    p.add_argument("--output-dir",       type=str,
                   default="verify2act/output/delta_encoder")
    return p.parse_args()


if __name__ == "__main__":
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    train(parse_args())
