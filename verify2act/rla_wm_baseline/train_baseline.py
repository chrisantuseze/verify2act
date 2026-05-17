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
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed

from transformers import CLIPTextModel, CLIPTokenizer

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from verify2act.latent_wm.train_dynamics import LatentDynamicsDataset, FeatureExtractor
from verify2act.rla_wm_baseline.dynamics import BaselineRLAWM

# ─── TRAINING LOOP ───────────────────────────────────────────────────────

def train(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Disable RNG sync to avoid mt19937 state errors in distributed training
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(dataloader_config=dataloader_config, rng_types=[])
    
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    sparsity_weight = args.sparsity_weight
    dataset_dir = args.dataset_dir
    device = accelerator.device
    
    if accelerator.is_local_main_process:
        print(f"Using device: {device} (Accelerate distributed)")
        print(f"Sparsity Weight: {sparsity_weight}")
    
    # 1. Dataset & DataLoader
    dataset = LatentDynamicsDataset(
        dataset_dir=args.dataset_dir, 
        transitions_file=args.transitions_file, 
        history_len=args.history_len, 
        image_size=args.image_size
    )
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    if accelerator.is_local_main_process:
        print(f"Dataset loaded with {len(dataset)} samples. Train: {train_size}, Val: {val_size}")
    
    # 2. Models
    extractor = FeatureExtractor(device)
    
    model = BaselineRLAWM(
        dino_channels=768,
        clip_channels=512,
        history_len=args.history_len, # API compat
        num_patches=256,
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
    
    # Accelerate Prepare
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
        
        for batch_idx, (history_imgs, target_img, action_texts) in enumerate(pbar):
            history_imgs = history_imgs.to(device)
            target_img = target_img.to(device)
            B = target_img.shape[0]
            
            F_history = extractor.extract_dino(history_imgs) 
            F_t = F_history[:, -1, :, :]                     
            F_t1 = extractor.extract_dino(target_img)        
            A_clip = extractor.extract_clip(action_texts)    
            
            residual_target = F_t1 - F_t                     
            t = torch.rand(B, device=device)
            noise = torch.randn_like(residual_target)
            t_expand = t.view(B, 1, 1)
            noisy_latent = (1 - t_expand) * noise + t_expand * residual_target
            velocity_target = residual_target - noise
            
            unwrapped_model = accelerator.unwrap_model(model)
            cond = unwrapped_model.forward_cond(F_history, A_clip)
            velocity_pred = unwrapped_model.forward_flow(cond, noisy_latent, t)
            
            loss_cfm = F.mse_loss(velocity_pred, velocity_target)
            
            # Controllable Sparsity Regularization
            loss_sparsity = torch.tensor(0.0, device=device)
            if sparsity_weight > 0.0:
                patch_movement = residual_target.norm(dim=-1)
                static_mask = (patch_movement < 0.05).float().unsqueeze(-1)
                loss_sparsity = (static_mask * velocity_pred.abs()).mean()
            
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
            for batch_idx, (history_imgs, target_img, action_texts) in enumerate(pbar_val):
                history_imgs = history_imgs.to(device)
                target_img = target_img.to(device)
                B = target_img.shape[0]
                
                F_history = extractor.extract_dino(history_imgs)
                F_t = F_history[:, -1, :, :]
                F_t1 = extractor.extract_dino(target_img)
                A_clip = extractor.extract_clip(action_texts)
                
                residual_target = F_t1 - F_t
                t = torch.rand(B, device=device)
                noise = torch.randn_like(residual_target)
                t_expand = t.view(B, 1, 1)
                noisy_latent = (1 - t_expand) * noise + t_expand * residual_target
                velocity_target = residual_target - noise
                
                unwrapped_model = accelerator.unwrap_model(model)
                cond = unwrapped_model.forward_cond(F_history, A_clip)
                velocity_pred = unwrapped_model.forward_flow(cond, noisy_latent, t)
                
                loss_cfm = F.mse_loss(velocity_pred, velocity_target)
                
                loss_sparsity = torch.tensor(0.0, device=device)
                if sparsity_weight > 0.0:
                    patch_movement = residual_target.norm(dim=-1)
                    static_mask = (patch_movement < 0.05).float().unsqueeze(-1)
                    loss_sparsity = (static_mask * velocity_pred.abs()).mean()
                
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
    parser = argparse.ArgumentParser(description="Train Baseline RLA-WM")

    parser.add_argument("--dataset-dir", type=str, default="robosuite/data_capture_wm/dataset/nut_assembly_merged")
    parser.add_argument("--transitions-file", type=str, default="transitions.jsonl")
    parser.add_argument("--output-dir", type=str, default="verify2act/output/rla_wm_baseline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--history-len", type=int, default=3, help="Number of past frames (ignored by baseline, kept for dataloader compat)")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for DINOv2 input")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sparsity-weight", type=float, default=0.0, help="Sparsity regularization weight (default 0.0 for baseline)")
    parser.add_argument("--checkpoint-freq", type=int, default=2, help="Checkpoint frequency (epochs)")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args()


if __name__ == "__main__":
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    train(parse_args())
