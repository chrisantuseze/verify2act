#!/usr/bin/env python3
"""Train DINOv2DualHeadCritic with joint InfoNCE loss.

Two-phase training:
  Phase 1 (--freeze-backbone-epochs): backbone frozen, only heads train.
  Phase 2 (remaining epochs): full fine-tuning at --backbone-lr.

Validation metrics:
  auroc_gp  — AUROC for goal proximity
  auroc_tc  — AUROC for temporal consistency

Best checkpoint saved by mean(auroc_gp, auroc_tc).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from accelerate.utils import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verify2act.critic.losses import InfoNCELoss
from verify2act.critic.model import DINOv2DualHeadCritic
from verify2act.data_loader import build_contrastive_datasets, ContrastivePairDataset


# ── Training step ──────────────────────────────────────────────────────────────

def _compute_kl(outputs: Dict, mask0: torch.Tensor, mask1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-head KL regularisation from the full-batch log-variance tensors."""
    kl_gp = (
        DINOv2DualHeadCritic.kl_loss(outputs["lv1_a"][mask0])
        + DINOv2DualHeadCritic.kl_loss(outputs["lv1_p"][mask0])
        + DINOv2DualHeadCritic.kl_loss(outputs["lv1_n"][mask0])
    ) / 3.0
    kl_tc = (
        DINOv2DualHeadCritic.kl_loss(outputs["lv2_a"][mask1])
        + DINOv2DualHeadCritic.kl_loss(outputs["lv2_p"][mask1])
        + DINOv2DualHeadCritic.kl_loss(outputs["lv2_n"][mask1])
    ) / 3.0
    return kl_gp, kl_tc


def _build_lang_inputs(batch: Dict, device: torch.device, mask0: torch.Tensor, lang_goal_weight: float):
    """Parse language-goal fields from batch. Returns (use_lang, has_lang_mask, valid_texts)."""
    if lang_goal_weight <= 0.0:
        return False, None, None
    lang_goals = batch.get("lang_goal", [])
    has_lang = batch.get("has_lang_goal", None)
    if not lang_goals or has_lang is None:
        return False, None, None
    has_lang_mask = has_lang.to(device).bool() & mask0
    if int(has_lang_mask.sum().item()) < 1:
        return False, None, None
    valid_texts = [lang_goals[i] for i, v in enumerate(has_lang_mask.tolist()) if v]
    return True, has_lang_mask, valid_texts


def train_step(
    model: DINOv2DualHeadCritic,
    batch: Dict,
    criterion: InfoNCELoss,
    device: torch.device,
    lambda1: float,
    lambda2: float,
    kl_weight: float,
    lang_goal_weight: float,
    use_inbatch: bool,
    use_cached: bool = False,
) -> Tuple[torch.Tensor, float, float, float, int, int]:
    """Single forward pass computing contrastive + alignment losses (DDP-safe)."""
    anchor   = batch["anchor"].to(device)
    positive = batch["positive"].to(device)
    negative = batch["negative"].to(device)
    modes    = batch["mode"].to(device)

    mask0 = (modes == 0)
    mask1 = (modes == 1)
    n_gp  = int(mask0.sum().item())
    n_tc  = int(mask1.sum().item())

    use_lang, has_lang_mask, valid_texts = _build_lang_inputs(batch, device, mask0, lang_goal_weight)

    all_imgs = torch.cat([anchor, positive, negative], dim=0)
    outputs = model(
        all_imgs, mask0, mask1,
        has_lang_mask=has_lang_mask,
        valid_texts=valid_texts,
        use_cached=use_cached,
        mode="ddp_train_step",
    )

    total_loss = torch.tensor(0.0, device=device)
    loss_gp_val = loss_tc_val = loss_lang_val = 0.0
    kl_gp, kl_tc = _compute_kl(outputs, mask0, mask1)

    if n_gp > 1:
        loss_gp = criterion(outputs["a0"], outputs["p0"], outputs["n0"],
                            use_inbatch_negatives=use_inbatch, symmetric=True)
        total_loss = total_loss + lambda1 * loss_gp + kl_weight * kl_gp
        loss_gp_val = loss_gp.item()

    if use_lang:
        loss_lang = F.mse_loss(outputs["lang_proj"], outputs["visual_proj"])
        total_loss = total_loss + lang_goal_weight * loss_lang
        loss_lang_val = loss_lang.item()

    if n_tc > 1:
        loss_tc = criterion(outputs["a1"], outputs["p1"], outputs["n1"],
                            use_inbatch_negatives=use_inbatch)
        total_loss = total_loss + lambda2 * loss_tc + kl_weight * kl_tc
        loss_tc_val = loss_tc.item()

    return total_loss, loss_gp_val, loss_tc_val, loss_lang_val, n_gp, n_tc


# ── Optimizer / Scheduler ──────────────────────────────────────────────────────

def _make_optimizer_scheduler(args, model, phase: int):
    if phase == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        n_epochs = args.freeze_backbone_epochs
    else:
        opt = AdamW(
            [
                {"params": model.backbone.parameters() if model.backbone is not None else [],
                 "lr": args.backbone_lr},
                {
                    "params": (list(model.head1.parameters())
                               + list(model.head2.parameters())
                               + list(model.clip_goal_proj.parameters())
                               + list(model.log_var_head1.parameters())
                               + list(model.log_var_head2.parameters())),
                    "lr": args.learning_rate,
                },
            ],
            weight_decay=args.weight_decay,
        )
        n_epochs = args.epochs - args.freeze_backbone_epochs

    warmup  = min(args.warmup_epochs, n_epochs) if phase == 1 else 0
    eta_min = (args.backbone_lr if phase == 2 else args.learning_rate) * 0.01

    if warmup > 0:
        cosine = CosineAnnealingLR(opt, T_max=max(1, n_epochs - warmup), eta_min=eta_min)
        sched = SequentialLR(opt, [LinearLR(opt, 0.01, 1.0, warmup), cosine], milestones=[warmup])
    else:
        sched = CosineAnnealingLR(opt, T_max=max(1, n_epochs), eta_min=eta_min)

    return opt, sched


# ── Tensorboard tracker ────────────────────────────────────────────────────────

def _init_tracker(use_tensorboard: bool, output_dir: Path, config: dict):
    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = output_dir / "tb_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        writer.add_text("hparams", json.dumps(config, indent=2), global_step=0)

        class _TB:
            def log(self, metrics: dict, step: int):
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step=step)
                writer.flush()
            def close(self):
                writer.close()
        return _TB()

    class _Noop:
        def log(self, metrics: dict, step: int): pass
        def close(self): pass
    return _Noop()


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def _epoch_from_path(path: Path) -> int:
    match = re.search(r"ep(\d+)\.pt$", str(path))
    return int(match.group(1)) if match else 0


def _load_checkpoint(args, model, device) -> Tuple[int, float, "AdamW | None", "object | None"]:
    """Load a checkpoint and return (start_epoch, best_val, opt_state, sched_state)."""
    resume_path = Path(args.resume_from)
    if not resume_path.exists():
        return 0, -1.0, None, None

    print(f"Resuming from checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = {k: v for k, v in ckpt["model_state_dict"].items()
                      if not k.startswith("_clip_model.")}
        model.load_state_dict(state_dict)
        start_epoch = ckpt.get("epoch", _epoch_from_path(resume_path))
        best_val = (ckpt.get("val_auroc_gp", 0.0) + ckpt.get("val_auroc_tc", 0.0)) / 2.0
        opt_state  = ckpt.get("optimizer_state_dict")
        sched_state = ckpt.get("scheduler_state_dict")
    else:
        state_dict = {k: v for k, v in ckpt.items() if not k.startswith("_clip_model.")}
        model.load_state_dict(state_dict)
        start_epoch = _epoch_from_path(resume_path)
        best_val = -1.0
        opt_state = sched_state = None

    return start_epoch, best_val, opt_state, sched_state


def _load_history(out_dir: Path, start_epoch: int) -> List[dict]:
    history_path = out_dir / "train_history.json"
    if not history_path.exists():
        return []
    try:
        with open(history_path) as f:
            history = json.load(f)
        return [r for r in history if r.get("epoch", 0) <= start_epoch]
    except Exception:
        return []


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(
    model: DINOv2DualHeadCritic,
    val_ds: ContrastivePairDataset,
    device: torch.device,
    accelerator: Accelerator = None,
    n_samples: int = 500,
    uncertainty_mc_samples: int = 10,
    use_cached: bool = False,
    num_workers: int = 4,
) -> Dict[str, float]:
    """Compute AUROC and mean uncertainty for goal proximity and temporal consistency."""
    raw_model = accelerator.unwrap_model(model) if accelerator is not None else model
    raw_model.eval()

    gp_pos, gp_neg, gp_unc = [], [], []
    tc_pos, tc_neg, tc_unc = [], [], []
    collected = {"gp": 0, "tc": 0}
    target = n_samples // 2

    val_loader = DataLoader(
        val_ds, batch_size=32, num_workers=num_workers,
        pin_memory=device.type == "cuda", shuffle=False,
    )

    for batch in val_loader:
        if collected["gp"] >= target and collected["tc"] >= target:
            break
        modes = batch["mode"]
        for i in range(len(modes)):
            mode = int(modes[i].item())
            a = batch["anchor"][i].unsqueeze(0).to(device)
            p = batch["positive"][i].unsqueeze(0).to(device)
            n = batch["negative"][i].unsqueeze(0).to(device)
            encode = raw_model.encode_features if use_cached else raw_model.encode
            ea, ep, en = encode(a), encode(p), encode(n)

            if mode == 0 and collected["gp"] < target:
                gp_pos.append(raw_model.goal_sim(ea, ep).item())
                gp_neg.append(raw_model.goal_sim(ea, en).item())
                _, std = raw_model.goal_sim_with_uncertainty(ea, ep, n_samples=uncertainty_mc_samples)
                gp_unc.append(std.item())
                collected["gp"] += 1
            elif mode == 1 and collected["tc"] < target:
                tc_pos.append(raw_model.temporal_sim(ea, ep).item())
                tc_neg.append(raw_model.temporal_sim(ea, en).item())
                _, std = raw_model.temporal_sim_with_uncertainty(ea, ep, n_samples=uncertainty_mc_samples)
                tc_unc.append(std.item())
                collected["tc"] += 1

    def _auroc(pos: List[float], neg: List[float]) -> float:
        pos = [v for v in pos if v == v]
        neg = [v for v in neg if v == v]
        if not pos or not neg:
            return float("nan")
        return float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg))

    auroc_gp = _auroc(gp_pos, gp_neg)
    auroc_tc = _auroc(tc_pos, tc_neg)

    def _mean(lst): return float(np.mean(lst)) if lst else float("nan")

    return {
        "auroc_gp":   auroc_gp,
        "auroc_tc":   auroc_tc,
        "mean_val":   (auroc_gp + auroc_tc) / 2.0
                      if not any(np.isnan(x) for x in [auroc_gp, auroc_tc]) else float("nan"),
        "gp_pos_sim": _mean(gp_pos),
        "gp_neg_sim": _mean(gp_neg),
        "tc_pos_sim": _mean(tc_pos),
        "tc_neg_sim": _mean(tc_neg),
        "mean_unc_gp": _mean(gp_unc),
        "mean_unc_tc": _mean(tc_unc),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.freeze_backbone_epochs == 0:
        args.freeze_backbone_epochs = args.epochs

    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    device  = accelerator.device
    out_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Cache completeness check
    cached_dino_dir = args.cached_dino_dir
    if cached_dino_dir is not None and accelerator.is_local_main_process:
        from verify2act.critic.cache_utils import ensure_cache_complete, ensure_calvin_cache_complete
        print(f"Checking DINOv2 feature cache: {cached_dino_dir}")
        if args.dataset_type == "calvin":
            ensure_calvin_cache_complete(
                args.dataset_dir, cache_dir=cached_dino_dir,
                device=str(device), dino_channels=args.dino_channels,
            )
        else:
            ensure_cache_complete(
                args.dataset_dir, transitions_file=args.transitions_file,
                cache_dir=cached_dino_dir, history_len=1,
                device=str(device), dino_channels=args.dino_channels,
            )
    accelerator.wait_for_everyone()

    # Datasets
    if args.dataset_type == "calvin":
        from verify2act.data_loader_calvin import build_calvin_contrastive_datasets
        train_ds, val_ds = build_calvin_contrastive_datasets(
            dataset_dir=args.dataset_dir, val_frac=args.val_frac, seed=args.seed,
            image_size=args.image_size, mode0_prob=args.mode0_prob,
            cached_dino_dir=cached_dino_dir,
        )
    else:
        train_ds, val_ds = build_contrastive_datasets(
            dataset_dir=args.dataset_dir, transitions_file=args.transitions_file,
            val_frac=args.val_frac, seed=args.seed, image_size=args.image_size,
            mode0_prob=args.mode0_prob, cached_dino_dir=cached_dino_dir,
        )
        accelerator.print(
            f"  train: {len(train_ds._positive_anchors)} pos anchors, "
            f"{len(train_ds._negative_anchors)} neg anchors, {len(train_ds._tc_rows)} TC rows"
        )
    accelerator.print(
        f"Dataset: {len(train_ds)} train / {len(val_ds)} val | "
        f"mode0_prob={args.mode0_prob}  image_size={args.image_size}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # Model
    model = DINOv2DualHeadCritic(
        pretrained=True, load_backbone=True, dino_channels=args.dino_channels,
    ).to(device)
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model: {n_total:,} total | {n_trainable:,} trainable params")

    criterion = InfoNCELoss(temperature=args.temperature)
    optimizer, scheduler = _make_optimizer_scheduler(args, model, phase=1)

    best_val   = -1.0
    start_epoch = 0
    history    = []

    # Resume
    if args.resume_from:
        start_epoch, best_val, opt_state, sched_state = _load_checkpoint(args, model, device)
        history = _load_history(out_dir, start_epoch)

        if start_epoch > args.freeze_backbone_epochs:
            accelerator.print("Resuming in Phase 2: unfreezing backbone")
            model.unfreeze_backbone()
            optimizer, scheduler = _make_optimizer_scheduler(args, model, phase=2)
        else:
            optimizer, scheduler = _make_optimizer_scheduler(args, model, phase=1)

        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)
        else:
            phase_epoch = (start_epoch - args.freeze_backbone_epochs
                           if start_epoch > args.freeze_backbone_epochs else start_epoch)
            for _ in range(phase_epoch):
                scheduler.step()

    tracker = _init_tracker(args.tensorboard, out_dir, vars(args).copy())

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop
    for epoch in range(start_epoch + 1, args.epochs + 1):

        # Phase transition
        if epoch == args.freeze_backbone_epochs + 1 and start_epoch <= args.freeze_backbone_epochs:
            accelerator.print(f"\n[Epoch {epoch}] Unfreezing backbone (lr={args.backbone_lr:.1e})")
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.unfreeze_backbone()
            optimizer, scheduler = _make_optimizer_scheduler(args, unwrapped, phase=2)
            model, optimizer, scheduler = accelerator.prepare(unwrapped, optimizer, scheduler)

        # KL weight warmup
        warmup_epochs = max(1, int(args.epochs * 0.3))
        epoch_kl_weight = args.kl_weight * min(1.0, epoch / warmup_epochs)

        model.train()
        losses_gp, losses_tc, totals = [], [], []

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}",
            dynamic_ncols=True, disable=not accelerator.is_local_main_process,
        )
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.mixed_precision != "no")):
                loss, lgp, ltc, llang, n0, n1 = train_step(
                    model, batch, criterion, device,
                    args.lambda1, args.lambda2,
                    kl_weight=epoch_kl_weight,
                    lang_goal_weight=args.lang_goal_weight,
                    use_inbatch=True,
                    use_cached=(cached_dino_dir is not None),
                )

            if not loss.requires_grad or loss.item() == 0.0:
                continue
            if torch.isnan(loss) or torch.isinf(loss):
                accelerator.print(f"  ⚠️  Skipping batch {batch_idx}: invalid loss {loss.item():.4f}")
                continue

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            totals.append(loss.item())
            if lgp > 0: losses_gp.append(lgp)
            if ltc > 0: losses_tc.append(ltc)

            pbar.set_postfix({
                "loss": f"{np.mean(totals):.4f}",
                "gp":   f"{np.mean(losses_gp) if losses_gp else 0:.3f}",
                "tc":   f"{np.mean(losses_tc) if losses_tc else 0:.3f}",
                "lang": f"{llang:.3f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.1e}",
            })

        scheduler.step()

        # Validation
        val_metrics = evaluate(
            model, val_ds, device, accelerator=accelerator,
            n_samples=args.val_samples, use_cached=(cached_dino_dir is not None),
            num_workers=args.num_workers,
        )
        train_loss = float(np.mean(totals)) if totals else float("nan")
        record = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_gp":   float(np.mean(losses_gp)) if losses_gp else float("nan"),
            "train_tc":   float(np.mean(losses_tc)) if losses_tc else float("nan"),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        if accelerator.is_local_main_process:
            print(json.dumps(record, indent=2))

        tracker.log(
            {
                "train/loss":    train_loss,
                "train/loss_gp": record["train_gp"],
                "train/loss_tc": record["train_tc"],
                "val/auroc_gp":  val_metrics["auroc_gp"],
                "val/auroc_tc":  val_metrics["auroc_tc"],
                "val/mean":      val_metrics["mean_val"],
                "val/unc_gp":    val_metrics["mean_unc_gp"],
                "val/unc_tc":    val_metrics["mean_unc_tc"],
                "lr":            optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        # Checkpoint
        mean_val = val_metrics["mean_val"]
        unwrapped = accelerator.unwrap_model(model)
        ckpt = {
            "model_state_dict":     unwrapped.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch":       epoch,
            "val_auroc_gp": val_metrics["auroc_gp"],
            "val_auroc_tc": val_metrics["auroc_tc"],
            "args":        vars(args),
        }
        if accelerator.is_main_process:
            torch.save(ckpt, out_dir / "latest_contrastive_critic.pt")
            if not np.isnan(mean_val) and mean_val > best_val:
                best_val = mean_val
                torch.save(ckpt, out_dir / "best_contrastive_critic.pt")
                accelerator.print(f"  ✓ New best: mean_auroc={best_val:.4f}")

    # Finalise
    if accelerator.is_main_process:
        with open(out_dir / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(out_dir / "train_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    tracker.close()
    accelerator.print(f"\nBest: {out_dir / 'best_contrastive_critic.pt'}  (mean_auroc={best_val:.4f})")


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DINOv2DualHeadCritic (contrastive)")

    # Data
    p.add_argument("--dataset-dir",      type=str, required=True)
    p.add_argument("--transitions-file", type=str, default="transitions.jsonl")
    p.add_argument("--image-size",       type=int, default=224)
    p.add_argument("--dino-channels",    type=int, default=1024, choices=[768, 1024],
                   help="Backbone output channels (768=ViT-B, 1024=ViT-L)")
    p.add_argument("--mode0-prob",       type=float, default=0.5,
                   help="Fraction of batch that are mode-0 (goal proximity) pairs")
    p.add_argument("--val-frac",         type=float, default=0.1)
    p.add_argument("--val-samples",      type=int,   default=500)
    p.add_argument("--dataset-type",     type=str,   default="robosuite",
                   choices=["robosuite", "calvin"])
    p.add_argument("--cached-dino-dir",  type=str,   default=None,
                   help="Path to pre-extracted DINOv2 feature cache")

    # Training
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--prefetch-factor", type=int,  default=2)
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--learning-rate",  type=float, default=1e-3)
    p.add_argument("--backbone-lr",    type=float, default=1e-5)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--max-grad-norm",  type=float, default=1.0)
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="Cap batches per epoch (0 = all; for debugging)")

    # Phase schedule
    p.add_argument("--freeze-backbone-epochs", type=int, default=0,
                   help="Epochs with backbone frozen (head warm-up)")
    p.add_argument("--warmup-epochs",          type=int, default=2)

    # InfoNCE / losses
    p.add_argument("--temperature",      type=float, default=0.07)
    p.add_argument("--lambda1",          type=float, default=1.0)
    p.add_argument("--lambda2",          type=float, default=1.0)
    p.add_argument("--kl-weight",        type=float, default=1e-3,
                   help="KL regulariser weight for log-variance heads")
    p.add_argument("--lang-goal-weight", type=float, default=0.5,
                   help="CLIP↔DINOv2 alignment loss weight (0 to disable)")

    # Output / misc
    p.add_argument("--output-dir",      type=str, default="verify2act/output/contrastive")
    p.add_argument("--resume-from",     type=str, default=None)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--tensorboard",     action="store_true", default=False,
                   help="Enable TensorBoard logging")

    return p.parse_args()


if __name__ == "__main__":
    main()
