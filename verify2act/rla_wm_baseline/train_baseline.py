import json
import os
import argparse
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore", message=".*torch\.cuda\.amp\.GradScaler.*")
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from transformers import CLIPTextModel, CLIPTokenizer

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from verify2act.latent_wm.train_dynamics import LatentDynamicsDataset, FeatureExtractor
from verify2act.rla_wm_baseline.dynamics import BaselineRLAWM
from verify2act.latent_wm.delta_encoder import DeltaEncoder

# Latent normalization constant — matches RLA-WM's latent_scalar_normalization=10.0.
LATENT_SCALE = 10.0

# ─── TRAINING LOOP ───────────────────────────────────────────────────────

def train(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Disable RNG sync to avoid mt19937 state errors in distributed training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(dataloader_config=dataloader_config, rng_types=[], kwargs_handlers=[ddp_kwargs])
    
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    sparsity_weight = args.sparsity_weight
    dataset_dir = args.dataset_dir
    device = accelerator.device
    
    if accelerator.is_local_main_process:
        print(f"Using device: {device} (Accelerate distributed)")
        print(f"Sparsity Weight: {sparsity_weight}")
    
    # ── Automatic Cache Checking & Generation ─────────────────────────────────
    if not args.no_cache:
        if accelerator.is_local_main_process:
            from verify2act.critic.cache_utils import ensure_cache_complete, ensure_calvin_cache_complete
            
            if args.cache_dir is None:
                cache_dir = str(Path(args.output_dir).parent / "dino_features")
            else:
                cache_dir = args.cache_dir
                
            if args.dataset_type == "calvin":
                ensure_calvin_cache_complete(args.dataset_dir, cache_dir=cache_dir, device=str(device))
            else:
                ensure_cache_complete(
                    args.dataset_dir,
                    transitions_file=args.transitions_file,
                    cache_dir=cache_dir,
                    history_len=args.history_len,
                    device=str(device)
                )
        accelerator.wait_for_everyone()

    # 1. Dataset & DataLoader
    if args.dataset_type == "calvin":
        from verify2act.data_loader_calvin import build_calvin_datasets
        train_dataset, val_dataset = build_calvin_datasets(
            dataset_dir=args.dataset_dir,
            val_frac=args.val_frac,
            image_size=args.image_size,
            history_len=args.history_len,
            seed=args.seed,
            use_cache=not args.no_cache,
            cached_dino_dir=cache_dir if not args.no_cache else None
        )
        if accelerator.is_local_main_process:
            print(f"CALVIN Dataset loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        dataset = LatentDynamicsDataset(
            dataset_dir=args.dataset_dir, 
            transitions_file=args.transitions_file, 
            history_len=args.history_len, 
            image_size=args.image_size,
            use_cache=not args.no_cache,
            cached_dino_dir=cache_dir if not args.no_cache else None
        )
        
        # Split into train/val
        train_size = int((1.0 - args.val_frac) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        if accelerator.is_local_main_process:
            print(f"RoboSuite Dataset loaded with {len(dataset)} samples. Train: {train_size}, Val: {val_size}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 2. Models
    extractor = FeatureExtractor(device, dino_channels=args.dino_channels)

    # ── Frozen DeltaEncoder (same checkpoint as V2A-WM Stage 1) ─────────────────────
    # The baseline uses the same latent token space so the comparison is fair:
    # the only difference is in the conditioning stage, not the flow space.
    delta_encoder = DeltaEncoder(
        dino_channels=args.dino_channels,
        model_channels=args.enc_model_channels,
        token_dim=args.token_dim,
        num_latent_tokens=args.num_latent_tokens,
        num_blocks=args.num_enc_blocks,
        num_heads=args.enc_num_heads,
    ).to(device)

    if args.encoder_ckpt:
        enc_state = torch.load(args.encoder_ckpt, map_location=device)
        if isinstance(enc_state, dict) and "encoder" in enc_state:
            enc_state = enc_state["encoder"]
        delta_encoder.load_state_dict(enc_state)
        if accelerator.is_local_main_process:
            print(f"[DeltaEncoder] Loaded from {args.encoder_ckpt}")
    else:
        if accelerator.is_local_main_process:
            print("[DeltaEncoder] No checkpoint — using random weights.")

    delta_encoder.eval()
    for p in delta_encoder.parameters():
        p.requires_grad = False

    # ── Baseline flow model ────────────────────────────────────────────────
    model = BaselineRLAWM(
        dino_channels=args.dino_channels,
        clip_channels=512,
        history_len=args.history_len,
        num_patches=256,
        token_dim=args.token_dim,
        num_latent_tokens=args.num_latent_tokens,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    start_epoch = 0
    if args.resume_from:
        if os.path.exists(args.resume_from):
            if accelerator.is_local_main_process:
                print(f"Resuming from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"]
            else:
                model.load_state_dict(checkpoint)
                import re
                match = re.search(r"ep(\d+)\.pt$", args.resume_from)
                if match:
                    start_epoch = int(match.group(1))
        else:
            if accelerator.is_local_main_process:
                print(f"Warning: Checkpoint {args.resume_from} not found. Starting from scratch.")
    
    # Accelerate Prepare (delta_encoder is frozen, not wrapped)
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    
    # 3. Training Loop
    best_val_loss = float('inf')
    writer = None
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f'{args.output_dir}/ckpt', exist_ok=True)
        writer = SummaryWriter(log_dir=f'{args.output_dir}/tb_logs')
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_cfm_loss = 0.0
        train_sparse_loss = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", dynamic_ncols=True, disable=not accelerator.is_local_main_process)
        
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                # CALVIN batch (dict)
                history_imgs = batch["history_imgs"].to(device)
                target_img   = batch["image_t1"].to(device)
                action_texts = batch["action_text"]
            else:
                # RoboSuite batch (4-tuple)
                history_imgs, target_img, action_texts, _ = batch
                history_imgs = history_imgs.to(device)
                target_img   = target_img.to(device)
            B = target_img.shape[0]

            # Feature extraction (frozen backbone)
            with torch.no_grad():
                # Check if history_imgs holds pre-computed DINOv2 features [B, H, 256, dino_channels]
                if len(history_imgs.shape) == 4 and history_imgs.shape[-1] == args.dino_channels:
                    F_history = history_imgs
                    F_t1 = target_img
                else:
                    F_history = extractor.extract_dino(history_imgs)
                    F_t1 = extractor.extract_dino(target_img)
                F_t  = F_history[:, -1, :, :]          # Baseline uses only F_t (Markovian)
                A_clip = extractor.extract_clip(action_texts)

                # Encode residual into compact latent tokens
                residual_raw = F_t1 - F_t
                gt_tokens = delta_encoder(residual_raw)  # (B, N, token_dim)

            # Flow matching in compact latent space
            # Normalize into flow space - divide by LATENT_SCALE
            x_0 = gt_tokens / LATENT_SCALE

            # logitNormal timestep sampling — same as V2A-WM and RLA-WM
            t = torch.sigmoid(torch.randn(B, device=device))
            noise = torch.randn_like(x_0)
            t_expand = t.view(B, 1, 1)
            noisy_latent = (1 - t_expand) * noise + t_expand * x_0
            velocity_target = x_0 - noise

            # Baseline forward_cond uses only F_t (last frame) — ignores history
            cond = model.forward_cond(F_history, A_clip)
            velocity_pred = model.forward_flow(cond, noisy_latent, t)

            loss_cfm = F.mse_loss(velocity_pred, velocity_target)

            # Sparsity (applied in raw DINO space, consistent with V2A-WM)
            loss_sparsity = torch.tensor(0.0, device=device)
            if sparsity_weight > 0.0:
                patch_movement = residual_raw.norm(dim=-1)
                static_weight  = (patch_movement < 0.05).float().mean(dim=-1)
                latent_activity = velocity_pred.norm(dim=-1).mean(dim=-1)
                loss_sparsity = (static_weight * latent_activity).mean()

            loss = loss_cfm + sparsity_weight * loss_sparsity
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_cfm_loss += loss_cfm.item()
            if sparsity_weight > 0.0:
                train_sparse_loss += loss_sparsity.item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "cfm": f"{loss_cfm.item():.4f}", 
                "sparse": f"{loss_sparsity.item() * sparsity_weight:.4f}"
            })
            
        num_train_batches = len(train_dataloader)
        train_loss /= num_train_batches
        train_cfm_loss /= num_train_batches
        train_sparse_loss /= num_train_batches
        
        if accelerator.is_main_process:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss_CFM/train', train_cfm_loss, epoch)
            if sparsity_weight > 0.0:
                writer.add_scalar('Loss_Sparse/train', train_sparse_loss, epoch)
        
        # --- Evaluation Loop ---
        model.eval()
        val_loss = 0.0
        val_cfm_loss = 0.0
        val_sparse_loss = 0.0
        
        pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", dynamic_ncols=True, disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar_val):
                if isinstance(batch, dict):
                    history_imgs = batch["history_imgs"].to(device)
                    target_img   = batch["image_t1"].to(device)
                    action_texts = batch["action_text"]
                else:
                    history_imgs, target_img, action_texts, _ = batch
                    history_imgs = history_imgs.to(device)
                    target_img   = target_img.to(device)
                B = target_img.shape[0]

                # Check if history_imgs holds pre-computed DINOv2 features [B, H, 256, dino_channels]
                if len(history_imgs.shape) == 4 and history_imgs.shape[-1] == args.dino_channels:
                    F_history = history_imgs
                    F_t1 = target_img
                else:
                    F_history = extractor.extract_dino(history_imgs)
                    F_t1 = extractor.extract_dino(target_img)
                F_t  = F_history[:, -1, :, :]
                A_clip = extractor.extract_clip(action_texts)

                residual_raw  = F_t1 - F_t
                gt_tokens     = delta_encoder(residual_raw)

                # Normalize into flow space - divide by LATENT_SCALE
                x_0 = gt_tokens / LATENT_SCALE

                t          = torch.sigmoid(torch.randn(B, device=device))
                noise      = torch.randn_like(x_0)
                t_expand   = t.view(B, 1, 1)
                noisy_latent    = (1 - t_expand) * noise + t_expand * x_0
                velocity_target = x_0 - noise

                cond          = model.forward_cond(F_history, A_clip)
                velocity_pred = model.forward_flow(cond, noisy_latent, t)

                loss_cfm = F.mse_loss(velocity_pred, velocity_target)

                loss_sparsity = torch.tensor(0.0, device=device)
                if sparsity_weight > 0.0:
                    patch_movement  = residual_raw.norm(dim=-1)
                    static_weight   = (patch_movement < 0.05).float().mean(dim=-1)
                    latent_activity = velocity_pred.norm(dim=-1).mean(dim=-1)
                    loss_sparsity   = (static_weight * latent_activity).mean()

                loss = loss_cfm + sparsity_weight * loss_sparsity
                
                val_loss += loss.item()
                val_cfm_loss += loss_cfm.item()
                if sparsity_weight > 0.0:
                    val_sparse_loss += loss_sparsity.item()
                
                pbar_val.set_postfix({"loss": f"{loss.item():.4f}"})
                
        num_val_batches = len(val_dataloader)
        if num_val_batches > 0:
            val_loss /= num_val_batches
            val_cfm_loss /= num_val_batches
            val_sparse_loss /= num_val_batches
        
        if accelerator.is_main_process:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Loss_CFM/val', val_cfm_loss, epoch)
            if sparsity_weight > 0.0:
                writer.add_scalar('Loss_Sparse/val', val_sparse_loss, epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            os.makedirs(args.output_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(unwrapped_model.state_dict(), f"{args.output_dir}/ckpt/latent_dynamics_best.pt")
                print(f"Saved best checkpoint to {args.output_dir}/ckpt/latent_dynamics_best.pt with val loss: {val_loss:.4f}")
                
            if (epoch + 1) % args.checkpoint_freq == 0:
                torch.save(unwrapped_model.state_dict(), f"{args.output_dir}/ckpt/latent_dynamics_ep{epoch+1}.pt")
                print(f"Saved checkpoint to {args.output_dir}/ckpt/latent_dynamics_ep{epoch+1}.pt")
            
    if accelerator.is_main_process:
        writer.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Baseline RLA-WM (Markovian, pooled-action self-attention conditioning)"
    )

    # Dataset
    parser.add_argument("--dataset-dir", type=str, default="robosuite/data_capture_wm/dataset/nut_assembly_merged")
    parser.add_argument("--dataset-type", type=str, default="robosuite", choices=["robosuite", "calvin"],
                        help="Type of dataset loader to use")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--transitions-file", type=str, default="transitions.jsonl")
    parser.add_argument("--history-len", type=int, default=3,
                        help="Window size for dataloader compat; baseline only uses last frame.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dino-channels", type=int, default=1024,
                        help="DINO feature dimension (768 for ViT-B, 1024 for ViT-L)")

    # Frozen DeltaEncoder (Stage 1, shared with V2A-WM for fair comparison)
    parser.add_argument("--encoder-ckpt", type=str, default=None,
                        help="Path to pre-trained DeltaEncoder (encoder_only_best.pt).")
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--num-latent-tokens", type=int, default=16)
    parser.add_argument("--enc-model-channels", type=int, default=512)
    parser.add_argument("--num-enc-blocks", type=int, default=4)
    parser.add_argument("--enc-num-heads", type=int, default=8)

    # Training
    parser.add_argument("--output-dir", type=str, default="verify2act/output/rla_wm_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sparsity-weight", type=float, default=0.0)
    parser.add_argument("--checkpoint-freq", type=int, default=2)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument(
        "--no-cache", action="store_true", default=False,
        help="Disable using pre-computed DINOv2 feature cache"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Centralized directory to save DINO cache. If None, uses args.output_dir/dino_features"
    )

    return parser.parse_args()


if __name__ == "__main__":
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    train(parse_args())
