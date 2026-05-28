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

import sys
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import DINO-WM baseline dynamics and the shared verify2act dataset utilities
from verify2act.latent_wm.train_dynamics import LatentDynamicsDataset, FeatureExtractor
from verify2act.dino_wm_baseline.dynamics import BaselineDINOWM

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
    device = accelerator.device
    
    if accelerator.is_local_main_process:
        print(f"Using device: {device} (Accelerate distributed)")
        print(f"Dataset type: {args.dataset_type}")
    
    # ── Automatic Cache Checking & Generation ─────────────────────────────────
    if not args.no_cache:
        if args.cache_dir is None:
            args.cache_dir = str(Path(args.output_dir).parent / "dino_features")
            print(f"Cache directory not specified. Using default: {args.cache_dir}")
        
        if accelerator.is_local_main_process:
            from verify2act.critic.cache_utils import ensure_cache_complete, ensure_calvin_cache_complete
            
            if args.dataset_type == "calvin":
                ensure_calvin_cache_complete(args.dataset_dir, cache_dir=args.cache_dir, device=str(device))
            else:
                ensure_cache_complete(
                    args.dataset_dir,
                    transitions_file=args.transitions_file,
                    cache_dir=args.cache_dir,
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
            cached_dino_dir=args.cache_dir if not args.no_cache else None
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
            cached_dino_dir=args.cache_dir if not args.no_cache else None
        )
        
        # Split into train/val
        train_size = int((1.0 - args.val_frac) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        if accelerator.is_local_main_process:
            print(f"RoboSuite Dataset loaded. Train: {train_size}, Val: {val_size}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 2. Models
    extractor = FeatureExtractor(device, dino_channels=args.dino_channels)

    # Strictly aligned DINO-WM baseline model
    model = BaselineDINOWM(
        dino_channels=args.dino_channels,
        clip_channels=512,
        action_dim=args.action_dim,
        action_emb_dim=args.action_emb_dim,
        proprio_dim=args.proprio_dim,
        proprio_emb_dim=args.proprio_emb_dim,
        history_len=args.history_len,
        num_patches=256,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        concat_dim=args.concat_dim
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
        # Save run config
        import json as _json
        with open(f'{args.output_dir}/config.json', 'w') as _f:
            _json.dump(vars(args), _f, indent=2)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        
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

            # Feature extraction (frozen DINOv2 and CLIP backbones)
            with torch.no_grad():
                if len(history_imgs.shape) == 4 and history_imgs.shape[-1] == args.dino_channels:
                    F_history = history_imgs
                    F_t1 = target_img
                else:
                    F_history = extractor.extract_dino(history_imgs)
                    F_t1 = extractor.extract_dino(target_img)
                
                # Combine history and targets to create sequence: (B, H + 1, 256, C)
                F_seq = torch.cat([F_history, F_t1.unsqueeze(1)], dim=1)
                A_clip = extractor.extract_clip(action_texts)

            # Forward pass: yields prediction, target, and standard MSE loss
            z_pred, z_tgt, loss = model(F_seq, A_clip)
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        num_train_batches = len(train_dataloader)
        train_loss /= num_train_batches
        
        if accelerator.is_main_process:
            writer.add_scalar('Loss/train', train_loss, epoch)
        
        # --- Evaluation Loop ---
        model.eval()
        val_loss = 0.0
        
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

                if len(history_imgs.shape) == 4 and history_imgs.shape[-1] == args.dino_channels:
                    F_history = history_imgs
                    F_t1 = target_img
                else:
                    F_history = extractor.extract_dino(history_imgs)
                    F_t1 = extractor.extract_dino(target_img)
                
                F_seq = torch.cat([F_history, F_t1.unsqueeze(1)], dim=1)
                A_clip = extractor.extract_clip(action_texts)

                z_pred, z_tgt, loss = model(F_seq, A_clip)
                val_loss += loss.item()
                pbar_val.set_postfix({"loss": f"{loss.item():.4f}"})
                
        num_val_batches = len(val_dataloader)
        if num_val_batches > 0:
            val_loss /= num_val_batches
        
        if accelerator.is_main_process:
            writer.add_scalar('Loss/val', val_loss, epoch)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
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
        description="Train strictly aligned DINO-WM baseline (Causal sequence prediction in raw DINOv2 space)"
    )

    # Dataset
    parser.add_argument("--dataset-dir", type=str, default="robosuite/data_capture_wm/dataset/nut_assembly_merged")
    parser.add_argument("--dataset-type", type=str, default="robosuite", choices=["robosuite", "calvin"])
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--transitions-file", type=str, default="transitions.jsonl")
    parser.add_argument("--history-len", type=int, default=3, help="Temporal window context length")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dino-channels", type=int, default=1024, help="1024 for ViT-L/14, 768 for ViT-B/14")

    # DINO-WM Hyperparameters
    parser.add_argument("--action-dim", type=int, default=64)
    parser.add_argument("--action-emb_dim", type=int, default=64)
    parser.add_argument("--proprio-dim", type=int, default=16)
    parser.add_argument("--proprio-emb_dim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp-dim", type=int, default=2048)
    parser.add_argument("--concat-dim", type=int, default=0, choices=[0, 1])

    # Training
    parser.add_argument("--output-dir", type=str, default="verify2act/output/dino_wm_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-freq", type=int, default=2)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true", default=False)
    parser.add_argument("--cache-dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    train(parse_args())
