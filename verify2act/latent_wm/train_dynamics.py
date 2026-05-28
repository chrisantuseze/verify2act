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

# Import our dynamics and encoder models
from verify2act.latent_wm.dynamics import LatentDynamicsModel
from verify2act.latent_wm.delta_encoder import DeltaEncoder

# ─── DATASET ─────────────────────────────────────────────────────────────

class LatentDynamicsDataset(Dataset):
    """
    Dataset that returns a history window of images [I_{t-H+1}, ..., I_t],
    the target next image I_{t+1}, and the action text.
    """
    def __init__(
        self,
        dataset_dir: str,
        transitions_file: str = "transitions.jsonl",
        history_len: int = 3,
        image_size: int = 224,
        use_cache: bool = True,
        cached_dino_dir: str = None,
    ):
        self.root = Path(dataset_dir)
        self.history_len = history_len
        self.use_cache = use_cache
        self.cached_dino_dir = Path(cached_dino_dir) if cached_dino_dir is not None else None
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # DINOv2 expected normalization
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Load and group by episode
        episodes = {}
        with open(self.root / transitions_file, "r") as f:
            for line in f:
                row = json.loads(line)
                ep_id = row["episode_id"]
                ts = int(row["timestep"])
                if ep_id not in episodes:
                    episodes[ep_id] = []
                episodes[ep_id].append((ts, row))
                
        # Sort each episode by timestep
        for ep_id in episodes:
            episodes[ep_id].sort(key=lambda x: x[0])
            
        # Create sliding windows
        self.samples = []
        for ep_id, ep_rows in episodes.items():
            rows = [r[1] for r in ep_rows]
            for i in range(len(rows)):
                # Target is I_{t+1} (from the row's image_t1)
                target_relpath = rows[i]["image_t1"]
                action_text = rows[i]["action_text"]
                # if "action_params" in rows[i] and "cartesian_target" in rows[i]["action_params"]:
                #     ct = rows[i]["action_params"]["cartesian_target"]
                #     action_text += f" at loc {ct[0]:.2f} {ct[1]:.2f} {ct[2]:.2f}"
                
                # History is [I_{t-H+1}, ..., I_t]
                # Track which slots are genuine vs. clamped (early-episode padding).
                history_relpaths = []
                history_mask = []
                for j in range(history_len - 1, -1, -1):
                    idx = i - j              # can be negative for early frames
                    is_real = (idx >= 0)
                    history_mask.append(is_real)
                    history_relpaths.append(rows[max(0, idx)]["image_t"])

                # Validate that all referenced images exist on disk before adding
                # this sample. Corrupt/incomplete episodes (e.g. a capture that
                # terminated early) can leave dangling JSONL rows; we skip them
                # gracefully instead of crashing mid-training.
                all_paths = history_relpaths + [target_relpath]
                if not all(( self.root / p).exists() for p in all_paths):
                    continue

                self.samples.append({
                    "history_paths": history_relpaths,
                    "history_mask":  history_mask,   # list of bools, True=real frame
                    "target_path":   target_relpath,
                    "action_text":   action_text
                })

    def __len__(self):
        return len(self.samples)

    def _load_image(self, relpath: str) -> torch.Tensor:
        img = Image.open(self.root / relpath).convert("RGB")
        return self.transform(img)

    def _load_image_or_feature(self, relpath: str) -> Tuple[torch.Tensor, bool]:
        """Loads cached dino feature if use_cache and it exists; otherwise loads raw image."""
        if self.use_cache:
            if self.cached_dino_dir is not None:
                feat_name = relpath.replace("/", "_") + ".pt"
                feat_path = self.cached_dino_dir / feat_name
            else:
                full_path = self.root / relpath
                feat_path = full_path.parent / (full_path.stem + "_dino.pt")
            if feat_path.exists():
                try:
                    feat = torch.load(feat_path, map_location="cpu")
                    return feat.float(), True
                except Exception:
                    pass
        return self._load_image(relpath), False

    def __getitem__(self, idx):
        sample = self.samples[idx]

        history_items = [self._load_image_or_feature(p) for p in sample["history_paths"]]
        target_item, target_is_feat = self._load_image_or_feature(sample["target_path"])

        history_imgs = [item[0] for item in history_items]
        history_are_feats = [item[1] for item in history_items]

        # If there's a mix or they are all raw images, force raw image fallback
        if not (all(history_are_feats) and target_is_feat):
            history_tensor = torch.stack([self._load_image(p) for p in sample["history_paths"]])
            target_tensor = self._load_image(sample["target_path"])
        else:
            history_tensor = torch.stack(history_imgs)
            target_tensor = target_item

        history_mask = torch.tensor(sample["history_mask"], dtype=torch.bool)
        return history_tensor, target_tensor, sample["action_text"], history_mask


# ─── FEATURE EXTRACTOR WRAPPER ──────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """Wraps DINOv2 and CLIP to extract features on the fly during training."""
    def __init__(self, device, dino_channels=1024):
        super().__init__()
        self.device = device
        self.dino_channels = dino_channels
        self.dino = None  # Lazily loaded to conserve VRAM when using cache
            
        print("Loading frozen CLIP text encoder...")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_dino(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (..., 3, H, W)
        Returns: (..., num_patches, 768)
        """
        if self.dino is None:
            model_name = "dinov2_vitl14" if self.dino_channels == 1024 else "dinov2_vitb14"
            print(f"Lazy loading frozen DINOv2 {model_name} backbone ({self.dino_channels}-ch)...")
            self.dino = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True).to(self.device)
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad = False

        orig_shape = imgs.shape
        if len(orig_shape) == 5: # (B, H_len, 3, H, W)
            imgs = imgs.view(-1, *orig_shape[2:])
            
        feats = self.dino.forward_features(imgs)["x_norm_patchtokens"]
        # ViT-L/14 outputs 1024-ch features; ViT-B/14 outputs 768-ch.
        if len(orig_shape) == 5:
            feats = feats.view(orig_shape[0], orig_shape[1], feats.shape[1], feats.shape[2])
        return feats

    @torch.no_grad()
    def extract_clip(self, texts: List[str]) -> torch.Tensor:
        """Returns: (B, seq_len, 512)"""
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        outputs = self.clip(**inputs)
        return outputs.last_hidden_state


# ─── TRAINING LOOP ───────────────────────────────────────────────────────

def train(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Disable RNG sync to avoid mt19937 state errors in distributed training
    # dispatch_batches=False is safer for standard datasets
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(dataloader_config=dataloader_config, rng_types=[], kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    # Hyperparameters
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    sparsity_weight = args.sparsity_weight

    # ── Automatic Cache Checking & Generation ─────────────────────────────────
    if not args.no_cache:
        if args.cache_dir is None:
            # Default to saving in the parent of output_dir (e.g. output/v2a_wm/calvin/dino_features)
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
        from verify2act.data_loader_calvin import CalvinTransitionDataset
        # We use CalvinTransitionDataset for both train and val by splitting it natively
        # Or we can just use the build_calvin_datasets function
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
    # Latent normalization constant — matches RLA-WM's latent_scalar_normalization=10.0.
    # Encoder latents are divided by this before flow interpolation so their scale
    # matches the unit-variance Gaussian noise.  Denormalised back after sampling.
    LATENT_SCALE = 10.0

    extractor = FeatureExtractor(device, dino_channels=args.dino_channels)

    # ── Frozen DeltaEncoder (Stage 1 checkpoint) ──────────────────────────────
    # Compresses F_{t+1} - F_t into compact latent tokens before flow matching,
    # mirroring RLA-WM's two-stage design.
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
        # Support both encoder-only and full encoder+decoder checkpoint formats
        if isinstance(enc_state, dict) and "encoder" in enc_state:
            enc_state = enc_state["encoder"]
        delta_encoder.load_state_dict(enc_state)
        if accelerator.is_local_main_process:
            print(f"[DeltaEncoder] Loaded from {args.encoder_ckpt}")
    else:
        if accelerator.is_local_main_process:
            print("[DeltaEncoder] No checkpoint provided — using random weights.")
            print("               Run train_encoder.py first for best results.")

    # Freeze the encoder — it is not updated during dynamics training
    delta_encoder.eval()
    for p in delta_encoder.parameters():
        p.requires_grad = False

    # ── Flow-matching dynamics model ──────────────────────────────────────────
    # Flow stage operates in the compact latent token space (token_dim);
    # conditioning stage still processes full temporal history (dino_channels).
    model = LatentDynamicsModel(
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
            # Support both full dict checkpoints and plain state_dicts
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"]
            else:
                model.load_state_dict(checkpoint)
                # Try to extract epoch from filename if it matches format
                import re
                match = re.search(r"ep(\d+)\.pt$", args.resume_from)
                if match:
                    start_epoch = int(match.group(1))
        else:
            if accelerator.is_local_main_process:
                print(f"Warning: Checkpoint {args.resume_from} not found. Starting from scratch.")
    
    # Accelerate Prepare (delta_encoder is frozen, no need to wrap in Accelerate)
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    if accelerator.is_local_main_process:
        print(f"Using device: {device} (Accelerate distributed)")
    
    # 3. Training Loop
    best_val_loss = float('inf')
    writer = None
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f'{args.output_dir}/ckpt', exist_ok=True)
        writer = SummaryWriter(log_dir=f'{args.output_dir}/tb_logs')
        # Save run config for reproducibility
        import json as _json
        with open(f'{args.output_dir}/config.json', 'w') as _f:
            _json.dump(vars(args), _f, indent=2)
        print(f"Run config saved to {args.output_dir}/config.json")
    
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
                history_mask = batch["history_mask"].to(device) if args.causal_masking else None
            else:
                # RoboSuite batch (4-tuple)
                history_imgs, target_img, action_texts, history_mask_raw = batch
                history_imgs = history_imgs.to(device)
                target_img   = target_img.to(device)
                history_mask = history_mask_raw.to(device) if args.causal_masking else None
                
            B = target_img.shape[0]

            # --- Feature Extraction (frozen backbone, no grad) ---
            with torch.no_grad():
                # Check if history_imgs holds pre-computed DINOv2 features [B, H, 256, 1024]
                if len(history_imgs.shape) == 4 and history_imgs.shape[-1] == args.dino_channels:
                    F_history = history_imgs
                    F_t1 = target_img
                else:
                    F_history = extractor.extract_dino(history_imgs)  # (B, H, 256, 768)
                    F_t1 = extractor.extract_dino(target_img)         # (B, 256, 768)
                F_t  = F_history[:, -1, :, :]                     # (B, 256, 768)
                # CLIP action tokens for cross-attention grounding
                A_clip = extractor.extract_clip(action_texts)     # (B, seq_len, 512)

                # --- Stage 1: Encode F_{t+1} - F_t into compact latent tokens ---
                # This is the key alignment with RLA-WM: the frozen DeltaEncoder
                # compresses the 256x1024 DINO difference into num_latent_tokens x
                # token_dim compact codes before flow matching.
                residual_raw = F_t1 - F_t                         # (B, 256, 1024)
                gt_tokens = delta_encoder(residual_raw)           # (B, N, token_dim)

            # --- Stage 2: Flow Matching in compact latent token space ---
            # 1. Normalize latents into flow space. (#1)
            # Encoder latents can have std >> 1.  Dividing by LATENT_SCALE brings
            # them close to unit-variance so the linear interpolation
            #   x_t = (1-t)*noise + t*x_0
            # is numerically stable across all t.  Mirrors RLA-WM's
            # latent_scalar_normalization=10.0 in _normalize_latent_tokens().
            x_0 = gt_tokens / LATENT_SCALE

            # 2. Sample flow timesteps via logitNormal distribution.
            # Concentrates training on t≈0.5 (hardest part of trajectory)
            # rather than wasting equal budget on trivial t≈0 / t≈1 endpoints.
            # Same technique used by SD3, Flux, and RLA-WM.
            t = torch.sigmoid(torch.randn(B, device=device))

            # 3. Sample noise and construct noisy latent: x_t = (1-t)*noise + t*x_0
            noise = torch.randn_like(x_0)
            t_expand = t.view(B, 1, 1)
            noisy_latent = (1 - t_expand) * noise + t_expand * x_0

            # 4. Velocity target in normalized latent space: v* = x_0 - noise
            velocity_target = x_0 - noise

            # 5. Forward pass — conditioning uses full history, flow uses token space
            velocity_pred = model(
                F_history, A_clip, noisy_latent, t, history_mask=history_mask
            )  # (B, N, token_dim)

            # --- Losses ---
            # Main CFM MSE loss (in compact latent token space)
            loss_cfm = F.mse_loss(velocity_pred, velocity_target)

            # Sparsity loss: operate in raw DINO space on static patches.
            # We compare movement magnitude of the raw DINO residual.
            patch_movement = residual_raw.norm(dim=-1)             # (B, 256)
            static_mask = (patch_movement < 0.05).float()          # (B, 256)
            # For the sparsity loss we penalise the velocity pred projected back
            # to dino space via a simple mean-pool proxy (lightweight, no decoder).
            # If sparsity_weight == 0, this is a no-op.
            if sparsity_weight > 0:
                # velocity_pred is (B, N, token_dim); we use its L2 norm as a
                # global activity proxy averaged over the latent tokens.
                latent_activity = velocity_pred.norm(dim=-1).mean(dim=-1)  # (B,)
                static_weight   = static_mask.mean(dim=-1)                 # (B,)
                loss_sparsity   = (static_weight * latent_activity).mean()
            else:
                loss_sparsity = torch.tensor(0.0, device=device)

            loss = loss_cfm + sparsity_weight * loss_sparsity
            
            # Optimize
            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_cfm_loss += loss_cfm.item()
            train_sparse_loss += loss_sparsity.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "cfm": f"{loss_cfm.item():.4f}", "sparse": f"{loss_sparsity.item():.4f}"})
            
        num_train_batches = len(train_dataloader)
        train_loss /= num_train_batches
        train_cfm_loss /= num_train_batches
        train_sparse_loss /= num_train_batches
        
        if accelerator.is_main_process:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss_CFM/train', train_cfm_loss, epoch)
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
                    history_mask = batch["history_mask"].to(device) if args.causal_masking else None
                else:
                    history_imgs, target_img, action_texts, history_mask_raw = batch
                    history_imgs = history_imgs.to(device)
                    target_img   = target_img.to(device)
                    history_mask = history_mask_raw.to(device) if args.causal_masking else None

                B = target_img.shape[0]

                # Check if history_imgs holds pre-computed DINOv2 features [B, H, 256, 1024]
                if len(history_imgs.shape) == 4 and history_imgs.shape[-1] == args.dino_channels:
                    F_history = history_imgs
                    F_t1 = target_img
                else:
                    F_history = extractor.extract_dino(history_imgs)
                    F_t1 = extractor.extract_dino(target_img)
                F_t        = F_history[:, -1, :, :]
                A_clip     = extractor.extract_clip(action_texts)

                residual_raw  = F_t1 - F_t
                gt_tokens     = delta_encoder(residual_raw)            # (B, N, token_dim)

                # Normalize into flow space — same as training loop (#1)
                x_0           = gt_tokens / LATENT_SCALE

                t          = torch.sigmoid(torch.randn(B, device=device))
                noise      = torch.randn_like(x_0)
                t_expand   = t.view(B, 1, 1)
                noisy_latent    = (1 - t_expand) * noise + t_expand * x_0
                velocity_target = x_0 - noise

                velocity_pred   = model(
                    F_history, A_clip, noisy_latent, t, history_mask=history_mask
                )

                loss_cfm = F.mse_loss(velocity_pred, velocity_target)
                if sparsity_weight > 0:
                    patch_movement = residual_raw.norm(dim=-1)
                    static_weight  = (patch_movement < 0.05).float().mean(dim=-1)
                    latent_activity = velocity_pred.norm(dim=-1).mean(dim=-1)
                    loss_sparsity  = (static_weight * latent_activity).mean()
                else:
                    loss_sparsity = torch.tensor(0.0, device=device)

                loss = loss_cfm + sparsity_weight * loss_sparsity
                
                val_loss += loss.item()
                val_cfm_loss += loss_cfm.item()
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
            writer.add_scalar('Loss_Sparse/val', val_sparse_loss, epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            os.makedirs(args.output_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(unwrapped_model.state_dict(), f"{args.output_dir}/ckpt/latent_dynamics_best.pt")
                print(f"Saved best checkpoint to {args.output_dir}/ckpt/latent_dynamics_best.pt with val loss: {val_loss:.4f}")
                
            # Save regular checkpoint
            if (epoch + 1) % args.checkpoint_freq == 0:
                torch.save(unwrapped_model.state_dict(), f"{args.output_dir}/ckpt/latent_dynamics_ep{epoch+1}.pt")
                print(f"Saved checkpoint to {args.output_dir}/ckpt/latent_dynamics_ep{epoch+1}.pt")
            
    if accelerator.is_main_process:
        writer.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train Verify2Act latent dynamics (Stage 2: flow matching)")

    # Dataset
    parser.add_argument("--dataset-dir", type=str, default="robosuite/data_capture_wm/dataset/nut_assembly_merged")
    parser.add_argument("--dataset-type", type=str, default="robosuite", choices=["robosuite", "calvin"],
                        help="Type of dataset loader to use")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--transitions-file", type=str, default="transitions.jsonl",
                        help="JSONL filename inside dataset-dir")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for DINOv2 input")
    parser.add_argument("--history-len", type=int, default=3,
                        help="Number of past frames to use as conditioning history")
    parser.add_argument("--dino-channels", type=int, default=1024,
                        help="DINO feature dimension (768 for ViT-B, 1024 for ViT-L)")

    # Frozen DeltaEncoder (Stage 1)
    parser.add_argument("--encoder-ckpt", type=str, default=None,
                        help="Path to pre-trained DeltaEncoder checkpoint (encoder_only_best.pt "
                             "from train_encoder.py). Strongly recommended; omit only for debugging.")
    parser.add_argument("--token-dim", type=int, default=64,
                        help="Compact latent token dimension (must match the encoder checkpoint).")
    parser.add_argument("--num-latent-tokens", type=int, default=16,
                        help="Number of compact latent tokens (must match the encoder checkpoint).")
    parser.add_argument("--enc-model-channels", type=int, default=512,
                        help="DeltaEncoder internal width (must match checkpoint).")
    parser.add_argument("--num-enc-blocks", type=int, default=4,
                        help="DeltaEncoder transformer depth (must match checkpoint).")
    parser.add_argument("--enc-num-heads", type=int, default=8,
                        help="DeltaEncoder attention heads (must match checkpoint).")

    # Training
    parser.add_argument("--output-dir", type=str, default="verify2act/output/v2a_wm/nut_assembly")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sparsity-weight", type=float, default=0.0,
                        help="Sparsity regularization weight (0 = disabled). "
                             "Applied as a global latent-activity penalty on static patches.")
    parser.add_argument("--checkpoint-freq", type=int, default=2, help="Checkpoint frequency (epochs)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to dynamics model checkpoint to resume from")
    parser.add_argument(
        "--causal-masking", action="store_true", default=False,
        help="Use Transformer-native causal attention masking + learnable [START] tokens for "
             "early-episode history padding."
    )
    parser.add_argument(
        "--no-cache", action="store_true", default=False,
        help="Disable using pre-computed DINOv2 feature cache"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Centralized directory to save DINO cache. If None, uses args.output_dir/dino_features"
    )

    args = parser.parse_args()
    if not args.causal_masking:
        args.history_len = 1
    return args


if __name__ == "__main__":
    # Ensure huggingface cache doesn't blow up /tmp if needed
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    train(parse_args())
