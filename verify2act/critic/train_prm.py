#!/usr/bin/env python3
"""Train PRM Beta critic on labeled transitions using frozen SD VAE encoder."""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

# Ensure repo root is on sys.path so the package resolves whether this file is
# run directly (python critic/train_prm.py) or as a module (-m verify2act.critic.train_prm).
# _repo_root = Path(__file__).resolve().parents[2]
# if str(_repo_root) not in sys.path:
#     sys.path.insert(0, str(_repo_root))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # robosuite/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))         # data_capture_wm/

from verify2act.critic.losses import BetaNLLLoss
from verify2act.critic.model import SpatialBetaPRMCritic
from verify2act.utils.data_loader import build_train_val_datasets
from verify2act.utils import VAE_LATENT_SCALE, load_vae_encoder

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(acc - conf)
    return float(ece)


def evaluate(
    model,
    vae,
    loader,
    criterion,
    device,
    weight_dtype: torch.dtype,
    latent_scale: float,
    max_batches: int = 0,
):
    model.eval()
    vae.eval()

    losses = []
    probs = []
    labels_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            # Cast images to VAE dtype to avoid fp32/fp16 mismatch
            image_t1 = batch["image_t1"].to(device=device, dtype=weight_dtype)
            goal_image = batch["goal_image"].to(device=device, dtype=weight_dtype)
            labels = batch["label"].to(device)

            z_t1 = vae.encode(image_t1).latent_dist.sample() * latent_scale
            z_goal = vae.encode(goal_image).latent_dist.sample() * latent_scale

            # Critic always runs in float32
            out = model(z_t1.float(), z_goal.float())
            loss = criterion(out["alpha"], out["beta"], labels)
            losses.append(loss.item())

            probs.append(out["mean_feasibility"].squeeze(1).detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())

    probs = np.concatenate(probs)
    labels_np = np.concatenate(labels_all)
    preds = (probs >= 0.5).astype(np.float32)

    acc = float((preds == labels_np).mean())
    brier = float(np.mean((probs - labels_np) ** 2))
    ece = compute_ece(probs, labels_np)

    # AUROC and AUPRC — require both classes present
    n_pos = labels_np.sum()
    if 0 < n_pos < len(labels_np):
        auroc = float(roc_auc_score(labels_np, probs))
        auprc = float(average_precision_score(labels_np, probs))
    else:
        auroc = float("nan")
        auprc = float("nan")

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": acc,
        "brier": brier,
        "ece": ece,
        "auroc": auroc,
        "auprc": auprc,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_train_val_datasets(
        dataset_dir=args.dataset_dir,
        labels_file=args.labels_file,
        default_goal_image=args.default_goal_image,
        val_frac=args.val_frac,
        seed=args.seed,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers // 2 if args.num_workers > 0 else 0,
        pin_memory=torch.cuda.is_available(),
    )

    weight_dtype = get_dtype(args.mixed_precision)
    vae, resolved_subfolder = load_vae_encoder(
        model_name_or_path=args.vae_model,
        device=device,
        torch_dtype=weight_dtype,
        subfolder=args.vae_subfolder,
        local_files_only=args.local_files_only,
    )
    print(
        f"Loaded VAE encoder from model={args.vae_model} "
        f"(subfolder={resolved_subfolder}, dtype={weight_dtype}, device={device})"
    )

    model = SpatialBetaPRMCritic().to(device)
    criterion = BetaNLLLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    latent_scale = VAE_LATENT_SCALE
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            # Cast images to VAE dtype to avoid fp32/fp16 mismatch
            image_t1 = batch["image_t1"].to(device=device, dtype=weight_dtype)
            goal_image = batch["goal_image"].to(device=device, dtype=weight_dtype)
            labels = batch["label"].to(device)

            with torch.no_grad():
                z_t1 = vae.encode(image_t1).latent_dist.sample() * latent_scale
                z_goal = vae.encode(goal_image).latent_dist.sample() * latent_scale

            # Critic always runs in float32
            out = model(z_t1.float(), z_goal.float())

            sample_weight = torch.where(
                labels > 0.5,
                torch.full_like(labels, args.class_weight_pos),
                torch.full_like(labels, args.class_weight_neg),
            )
            loss = criterion(out["alpha"], out["beta"], labels, sample_weight=sample_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{np.mean(train_losses):.4f}"})

        val_metrics = evaluate(
            model,
            vae,
            val_loader,
            criterion,
            device,
            weight_dtype=weight_dtype,
            latent_scale=latent_scale,
            max_batches=args.max_val_batches,
        )
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, indent=2))

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val,
                    "args": vars(args),
                },
                out_dir / "best_prm_critic.pt",
            )

    with open(out_dir / "train_history.json", "w") as handle:
        json.dump(history, handle, indent=2)

    with open(out_dir / "train_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2)

    print(f"Saved best checkpoint to: {out_dir / 'best_prm_critic.pt'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PRM Beta critic")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--labels-file", type=str, default="labels.jsonl")
    parser.add_argument(
        "--default-goal-image",
        type=str,
        default="",
        help="Fallback goal image path (relative to dataset-dir or absolute). Used when row goal_image is missing.",
    )
    parser.add_argument("--output-dir", type=str, default="verify2act/output/prm", required=True)

    parser.add_argument("--vae-model", type=str, default="timbrooks/instruct-pix2pix")
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
    parser.add_argument("--image-size", type=int, default=512)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)

    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--class-weight-pos", type=float, default=1.0)
    parser.add_argument("--class-weight-neg", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()

if __name__ == "__main__":
    main()
