import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import lpips

from decoder import FeatureDecoder
from train_dynamics import LatentDynamicsDataset, FeatureExtractor

def train_decoder(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader
    if args.dataset_type == "calvin":
        from verify2act.data_loader_calvin import build_calvin_datasets
        # We only need the target image for decoder training
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
            history_len=1, # We only need the target image for decoder training
            image_size=args.image_size,
            use_cache=False,
        )
        
        # Split into train/val
        train_size = int((1.0 - args.val_frac) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Dataset loaded. Train: {train_size}, Val: {val_size}")

    # 2. Models
    extractor = FeatureExtractor(device, dino_channels=args.dino_channels)
    
    visualizer = FeatureDecoder(
        dino_channels=args.dino_channels,
        model_channels=256
    ).to(device)

    # LPIPS for perceptual loss
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.requires_grad_(False)
    lpips_fn.eval()

    optimizer = torch.optim.AdamW(visualizer.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Resuming from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            # Support both full dict checkpoints and plain state_dicts
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                visualizer.decoder.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"]
            else:
                visualizer.decoder.load_state_dict(checkpoint)
                # Try to extract epoch from filename if it matches format
                import re
                match = re.search(r"ep(\d+)\.pt$", args.resume_from)
                if match:
                    start_epoch = int(match.group(1))
            print(f"Resumed from {args.resume_from} (epoch {start_epoch})")
        else:
            print(f"Warning: Checkpoint {args.resume_from} not found. Starting from scratch.")

    # 3. Training Loop
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {args.output_dir}/config.json")

    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        visualizer.train()
        train_loss = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]", dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                target_img = batch["image_t1"].to(device)
            else:
                _, target_img, _, _ = batch
                target_img = target_img.to(device)
            
            # Extract ground truth DINO features
            with torch.no_grad():
                # Extractor outputs (B, 256, 768)
                F_target = extractor.extract_dino(target_img)

            # Decode features back to images
            pred_img = visualizer.decode(F_target) # Outputs [-1, 1] typically

            # Denormalize ground truth target_img from ImageNet norm to [0, 1]
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            target_img_denorm = (target_img * std + mean).clamp(0, 1)
            
            # Convert [0, 1] to [-1, 1] for loss computation against pred_img
            target_img_scaled = target_img_denorm * 2.0 - 1.0

            # Resize if needed
            if pred_img.shape[-2:] != target_img_scaled.shape[-2:]:
                pred_img = F.interpolate(pred_img, size=target_img_scaled.shape[-2:], mode='bilinear')

            # Compute losses
            loss_l1 = F.l1_loss(pred_img, target_img_scaled)
            loss_lpips = lpips_fn(pred_img, target_img_scaled).mean()
            loss = args.l1_weight * loss_l1 + args.lpips_weight * loss_lpips

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(visualizer.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "l1": f"{loss_l1.item():.4f}", "lpips": f"{loss_lpips.item():.4f}"})
            
        train_loss /= len(train_dataloader)
        
        # --- Evaluation Loop ---
        visualizer.eval()
        val_loss = 0.0
        
        pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]", dynamic_ncols=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar_val):
                if isinstance(batch, dict):
                    target_img = batch["image_t1"].to(device)
                else:
                    _, target_img, _, _ = batch
                    target_img = target_img.to(device)
                
                F_target = extractor.extract_dino(target_img)
                pred_img = visualizer.decode(F_target)
                
                target_img_denorm = (target_img * std + mean).clamp(0, 1)
                target_img_scaled = target_img_denorm * 2.0 - 1.0

                if pred_img.shape[-2:] != target_img_scaled.shape[-2:]:
                    pred_img = F.interpolate(pred_img, size=target_img_scaled.shape[-2:], mode='bilinear')

                loss_l1 = F.l1_loss(pred_img, target_img_scaled)
                loss_lpips = lpips_fn(pred_img, target_img_scaled).mean()
                loss = args.l1_weight * loss_l1 + args.lpips_weight * loss_lpips
                
                val_loss += loss.item()
                pbar_val.set_postfix({"loss": f"{loss.item():.4f}"})
                
        if len(val_dataloader) > 0:
            val_loss /= len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(visualizer.decoder.state_dict(), f"{args.output_dir}/latent_decoder_best.pt")
            print(f"Saved best checkpoint to {args.output_dir}/latent_decoder_best.pt with val loss: {val_loss:.4f}")
            
        # Save regular checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            torch.save(visualizer.decoder.state_dict(), f"{args.output_dir}/latent_decoder_ep{epoch+1}.pt")
            print(f"Saved checkpoint to {args.output_dir}/latent_decoder_ep{epoch+1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Feature Decoder for Latent World Model")
    parser.add_argument("--dataset-dir", type=str, default="robosuite/data_capture_wm/dataset/nut_assembly_merged")
    parser.add_argument("--dataset-type", type=str, default="robosuite", choices=["robosuite", "calvin"])
    parser.add_argument("--transitions-file", type=str, default="transitions.jsonl")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="verify2act/output/v2a_wm/decoder")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=0.5)
    parser.add_argument("--checkpoint-freq", type=int, default=5)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to decoder checkpoint to resume from")
    parser.add_argument("--dino-channels", type=int, default=1024)
    
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    
    args = parser.parse_args()
    train_decoder(args)
