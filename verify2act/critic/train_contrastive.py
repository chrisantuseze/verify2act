#!/usr/bin/env python3
"""Train DINOv2DualHeadCritic with joint InfoNCE loss.

Two-phase training:
  Phase 1 (--freeze-backbone-epochs): backbone frozen, only heads train at --learning-rate.
  Phase 2 (remaining epochs): full backbone fine-tuning at --backbone-lr (lower LR).

Validation metrics:
  auroc_gp  — AUROC for goal proximity: late vs. early frame cosine similarity to goal
  auroc_tc  — AUROC for temporal consistency: consecutive vs. cross-episode frame similarity

Best checkpoint saved by mean(auroc_gp, auroc_tc).
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root

from verify2act.critic.losses import InfoNCELoss
from verify2act.critic.model import DINOv2DualHeadCritic
from verify2act.data_loader import build_contrastive_datasets, ContrastivePairDataset


# ── CLIP language-goal alignment loss ─────────────────────────────────────────

def _compute_lang_gp_loss(
    model: DINOv2DualHeadCritic,
    batch: Dict,
    device: torch.device,
    amp_enabled: bool,
) -> torch.Tensor:
    """Align clip_goal_proj(CLIP(text)) with head1(DINOv2(goal_frame)).

    For each GP-mode sample in the batch, the positive frame is the goal
    frame (close-to-goal), so we train clip_goal_proj to produce an
    embedding close to head1(goal_frame) when given the task's text label.

    Requires batch to contain a ``lang_goal`` list of strings and a binary
    ``has_lang_goal`` mask indicating which samples have a valid text goal.
    Falls back to a zero loss if the dataset does not include text goals.
    """
    lang_goals: list = batch.get("lang_goal", [])
    has_lang = batch.get("has_lang_goal", None)
    modes = batch["mode"].to(device)
    mask0 = (modes == 0)   # goal proximity pairs

    if not lang_goals or has_lang is None:
        return torch.tensor(0.0, device=device)

    has_lang = has_lang.to(device).bool() & mask0
    if has_lang.sum() < 1:
        return torch.tensor(0.0, device=device)

    positive = batch["positive"].to(device)   # [B, 3, H, W] goal frames

    # Encode positive (goal) frames through DINOv2 + head1
    with torch.amp.autocast('cuda', enabled=amp_enabled):
        emb_goal = model.encode(positive[has_lang])   # ProbEmbedding
        visual_proj = model.project(emb_goal.mu, head=1)   # [n, HEAD_DIM]

    # Encode language goals through CLIP + clip_goal_proj
    valid_texts = [lang_goals[i] for i, v in enumerate(has_lang.tolist()) if v]
    lang_proj = torch.stack(
        [model.encode_text_goal(t) for t in valid_texts], dim=0
    ).squeeze(1)   # [n, HEAD_DIM]

    # Symmetric MSE alignment loss between the two projections
    loss = F.mse_loss(lang_proj, visual_proj.detach())
    return loss


# ── Utilities ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_tracker(tracker: str, output_dir: Path, config: dict):
    if tracker == "tensorboard":
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

    if tracker == "wandb":
        import wandb
        wandb.init(project=config.pop("wandb_project", "verify2act-contrastive"), config=config)

        class _WB:
            def log(self, metrics: dict, step: int):
                wandb.log(metrics, step=step)
            def close(self):
                wandb.finish()
        return _WB()

    class _Noop:
        def log(self, metrics: dict, step: int): pass
        def close(self): pass
    return _Noop()


# ── Training step ──────────────────────────────────────────────────────────────

def _train_step(
    model: DINOv2DualHeadCritic,
    batch: Dict,
    criterion: InfoNCELoss,
    device: torch.device,
    lambda1: float,
    lambda2: float,
    kl_weight: float,
    use_inbatch: bool,
    amp_enabled: bool,
) -> Tuple[torch.Tensor, float, float, int, int]:
    """Forward + loss for one mixed batch. Returns (loss, loss_gp, loss_tc, n_gp, n_tc).

    For probabilistic embeddings, each triplet is projected from a *sampled*
    embedding z = μ + ε·exp(0.5·log_var) via the reparameterization trick.
    A KL regulariser pulls each head's log-variance toward 0 (σ→ 1).
    """
    anchor   = batch["anchor"].to(device)    # [B, 3, H, W]
    positive = batch["positive"].to(device)
    negative = batch["negative"].to(device)
    modes    = batch["mode"].to(device)       # [B]  0 or 1

    mask0 = (modes == 0)   # goal proximity
    mask1 = (modes == 1)   # temporal consistency

    # Encode all images in one fused batch for efficiency
    all_imgs = torch.cat([anchor, positive, negative], dim=0)   # [3B, 3, H, W]
    with torch.amp.autocast('cuda', enabled=amp_enabled):
        all_pe = model.encode(all_imgs)   # ProbEmbedding: .mu/.log_var1/.log_var2 each [3B, 768]

    B = anchor.size(0)
    # Split mean embeddings
    mu_anchor   = all_pe.mu[:B]
    mu_positive = all_pe.mu[B:2*B]
    mu_negative = all_pe.mu[2*B:]
    # Split log-variances per head
    lv1_anchor   = all_pe.log_var1[:B]
    lv1_positive = all_pe.log_var1[B:2*B]
    lv1_negative = all_pe.log_var1[2*B:]
    lv2_anchor   = all_pe.log_var2[:B]
    lv2_positive = all_pe.log_var2[B:2*B]
    lv2_negative = all_pe.log_var2[2*B:]

    total_loss = torch.tensor(0.0, device=device)
    loss_gp_val = 0.0
    loss_tc_val = 0.0
    n_gp = mask0.sum().item()
    n_tc = mask1.sum().item()

    # ── Head 1: goal proximity ────────────────────────────────────────────────
    if n_gp > 1:   # InfoNCE needs > 1 sample for in-batch negatives
        # Sample embeddings via reparameterization trick
        z_a0 = model.sample_embed(mu_anchor[mask0],   lv1_anchor[mask0])
        z_p0 = model.sample_embed(mu_positive[mask0], lv1_positive[mask0])
        z_n0 = model.sample_embed(mu_negative[mask0], lv1_negative[mask0])
        a0 = model.project(z_a0, head=1)
        p0 = model.project(z_p0, head=1)
        n0 = model.project(z_n0, head=1)
        loss_gp = criterion(a0, p0, n0, use_inbatch_negatives=use_inbatch)
        # KL regulariser: pull log_var1 toward 0 for GP samples
        kl_gp = (
            model.kl_loss(lv1_anchor[mask0])
            + model.kl_loss(lv1_positive[mask0])
            + model.kl_loss(lv1_negative[mask0])
        ) / 3.0
        total_loss = total_loss + lambda1 * loss_gp + kl_weight * kl_gp
        loss_gp_val = loss_gp.item()

    # ── Head 2: temporal consistency ──────────────────────────────────────────
    if n_tc > 1:
        z_a1 = model.sample_embed(mu_anchor[mask1],   lv2_anchor[mask1])
        z_p1 = model.sample_embed(mu_positive[mask1], lv2_positive[mask1])
        z_n1 = model.sample_embed(mu_negative[mask1], lv2_negative[mask1])
        a1 = model.project(z_a1, head=2)
        p1 = model.project(z_p1, head=2)
        n1 = model.project(z_n1, head=2)
        loss_tc = criterion(a1, p1, n1, use_inbatch_negatives=use_inbatch)
        kl_tc = (
            model.kl_loss(lv2_anchor[mask1])
            + model.kl_loss(lv2_positive[mask1])
            + model.kl_loss(lv2_negative[mask1])
        ) / 3.0
        total_loss = total_loss + lambda2 * loss_tc + kl_weight * kl_tc
        loss_tc_val = loss_tc.item()

    return total_loss, loss_gp_val, loss_tc_val, int(n_gp), int(n_tc)


# ── Main training step wrapper (adds lang-GP loss) ────────────────────────────

def train_step_with_lang(
    model: DINOv2DualHeadCritic,
    batch: Dict,
    criterion: InfoNCELoss,
    device: torch.device,
    lambda1: float,
    lambda2: float,
    kl_weight: float,
    lang_goal_weight: float,
    use_inbatch: bool,
    amp_enabled: bool,
) -> Tuple[torch.Tensor, float, float, float, int, int]:
    """Wrapper around _train_step that adds the optional CLIP-GP alignment loss.

    Returns (total_loss, loss_gp, loss_tc, loss_lang_gp, n_gp, n_tc).
    """
    loss, lgp, ltc, n0, n1 = _train_step(
        model, batch, criterion, device,
        lambda1, lambda2, kl_weight, use_inbatch, amp_enabled,
    )
    loss_lang = torch.tensor(0.0, device=device)
    if lang_goal_weight > 0.0:
        loss_lang = _compute_lang_gp_loss(model, batch, device, amp_enabled)
        loss = loss + lang_goal_weight * loss_lang
    return loss, lgp, ltc, loss_lang.item(), n0, n1


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: DINOv2DualHeadCritic,
    val_ds: ContrastivePairDataset,
    device: torch.device,
    n_samples: int = 500,
    uncertainty_mc_samples: int = 10,
) -> Dict[str, float]:
    """Compute AUROC and uncertainty metrics for goal proximity and temporal consistency.

    Samples n_samples triplets from val_ds and measures:
      auroc_gp : cos_sim(head1(late), head1(goal)) > cos_sim(head1(early), head1(goal))
      auroc_tc : cos_sim(head2(I_t), head2(I_{t+1})) > cos_sim(head2(I_t), head2(I_cross))

    Deterministic similarities use mu only (no MC sampling) for fast AUROC.
    Uncertainty metrics (mean_unc_gp, mean_unc_tc) use uncertainty_mc_samples
    MC draws from the probabilistic embedding.
    """
    model.eval()

    gp_pos_sims: List[float] = []
    gp_neg_sims: List[float] = []
    tc_pos_sims: List[float] = []
    tc_neg_sims: List[float] = []
    gp_uncertainties: List[float] = []
    tc_uncertainties: List[float] = []

    # Force evaluation over a fixed random sample of each mode
    mode0_indices = [i for i in range(min(n_samples * 4, len(val_ds)))]
    collected = {"gp": 0, "tc": 0}
    target_per_mode = n_samples // 2

    for idx in mode0_indices:
        if collected["gp"] >= target_per_mode and collected["tc"] >= target_per_mode:
            break
        item = val_ds[idx]
        mode = item["mode"].item()

        anchor   = item["anchor"].unsqueeze(0).to(device)
        positive = item["positive"].unsqueeze(0).to(device)
        negative = item["negative"].unsqueeze(0).to(device)

        emb_a = model.encode(anchor)
        emb_p = model.encode(positive)
        emb_n = model.encode(negative)

        if mode == 0 and collected["gp"] < target_per_mode:
            # Deterministic similarity for AUROC
            pos_sim = model.goal_sim(emb_a, emb_p).item()
            neg_sim = model.goal_sim(emb_a, emb_n).item()
            gp_pos_sims.append(pos_sim)
            gp_neg_sims.append(neg_sim)
            # MC uncertainty: std of positive similarity
            _, std_pos = model.goal_sim_with_uncertainty(emb_a, emb_p, n_samples=uncertainty_mc_samples)
            gp_uncertainties.append(std_pos.item())
            collected["gp"] += 1
        elif mode == 1 and collected["tc"] < target_per_mode:
            pos_sim = model.temporal_sim(emb_a, emb_p).item()
            neg_sim = model.temporal_sim(emb_a, emb_n).item()
            tc_pos_sims.append(pos_sim)
            tc_neg_sims.append(neg_sim)
            _, std_pos = model.temporal_sim_with_uncertainty(emb_a, emb_p, n_samples=uncertainty_mc_samples)
            tc_uncertainties.append(std_pos.item())
            collected["tc"] += 1

    def _auroc(pos: List[float], neg: List[float]) -> float:
        if not pos or not neg:
            return float("nan")
        pos_clean = [v for v in pos if not (v != v)]   # filter NaN (NaN != NaN)
        neg_clean = [v for v in neg if not (v != v)]
        if not pos_clean or not neg_clean:
            return float("nan")
        scores = pos_clean + neg_clean
        labels = [1] * len(pos_clean) + [0] * len(neg_clean)
        return float(roc_auc_score(labels, scores))

    auroc_gp = _auroc(gp_pos_sims, gp_neg_sims)
    auroc_tc = _auroc(tc_pos_sims, tc_neg_sims)

    mean_pos_gp = float(np.mean(gp_pos_sims)) if gp_pos_sims else float("nan")
    mean_neg_gp = float(np.mean(gp_neg_sims)) if gp_neg_sims else float("nan")
    mean_pos_tc = float(np.mean(tc_pos_sims)) if tc_pos_sims else float("nan")
    mean_neg_tc = float(np.mean(tc_neg_sims)) if tc_neg_sims else float("nan")
    mean_unc_gp = float(np.mean(gp_uncertainties)) if gp_uncertainties else float("nan")
    mean_unc_tc = float(np.mean(tc_uncertainties)) if tc_uncertainties else float("nan")

    return {
        "auroc_gp": auroc_gp,
        "auroc_tc": auroc_tc,
        "mean_val": (auroc_gp + auroc_tc) / 2.0 if not any(
            np.isnan(x) for x in [auroc_gp, auroc_tc]) else float("nan"),
        "gp_pos_sim": mean_pos_gp,
        "gp_neg_sim": mean_neg_gp,
        "tc_pos_sim": mean_pos_tc,
        "tc_neg_sim": mean_neg_tc,
        "mean_unc_gp": mean_unc_gp,   # mean predictive std for Head 1
        "mean_unc_tc": mean_unc_tc,   # mean predictive std for Head 2
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds, val_ds = build_contrastive_datasets(
        dataset_dir=args.dataset_dir,
        transitions_file=args.transitions_file,
        val_frac=args.val_frac,
        seed=args.seed,
        image_size=args.image_size,
        mode0_prob=args.mode0_prob,
    )
    print(
        f"Dataset: {len(train_ds)} train pairs / {len(val_ds)} val pairs | "
        f"mode0_prob={args.mode0_prob}  image_size={args.image_size}"
    )
    print(
        f"  train: {len(train_ds._positive_anchors)} positive anchors, "
        f"{len(train_ds._negative_anchors)} negative anchors, "
        f"{len(train_ds._tc_rows)} TC rows"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,     # required for stable InfoNCE in-batch negatives
    )
    val_ds_for_eval = val_ds   # keep reference for evaluate()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DINOv2DualHeadCritic(pretrained=True).to(device)
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_total:,} total params | {n_trainable:,} trainable "
          f"(projection + log-var heads; backbone frozen in phase 1)")

    criterion = InfoNCELoss(temperature=args.temperature)

    # ── Optimizer factory ──────────────────────────────────────────────────────
    def _make_optimizer_scheduler(phase: int):
        if phase == 1:
            params = [p for p in model.parameters() if p.requires_grad]
            opt = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
            n_epochs = args.freeze_backbone_epochs
        else:
            opt = AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": args.backbone_lr},
                    {
                        "params": list(model.head1.parameters()) +
                                  list(model.head2.parameters()) +
                                  list(model.clip_goal_proj.parameters()) +
                                  list(model.log_var_head1.parameters()) +
                                  list(model.log_var_head2.parameters()),
                        "lr": args.learning_rate,
                    },
                ],
                weight_decay=args.weight_decay,
            )
            n_epochs = args.epochs - args.freeze_backbone_epochs

        warmup = min(args.warmup_epochs, n_epochs) if phase == 1 else 0
        eta_min = (args.backbone_lr if phase == 2 else args.learning_rate) * 0.01

        if warmup > 0:
            cosine = CosineAnnealingLR(opt, T_max=max(1, n_epochs - warmup), eta_min=eta_min)
            sched = SequentialLR(
                opt,
                schedulers=[LinearLR(opt, 0.01, 1.0, warmup), cosine],
                milestones=[warmup],
            )
        else:
            sched = CosineAnnealingLR(opt, T_max=max(1, n_epochs), eta_min=eta_min)

        return opt, sched

    optimizer, scheduler = _make_optimizer_scheduler(phase=1)

    tracker = _init_tracker(args.tracker, out_dir, vars(args).copy())
    scaler  = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda" and args.mixed_precision != "no"))
    amp_enabled = device.type == "cuda" and args.mixed_precision != "no"

    best_val = -1.0
    history  = []

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):

        # Phase transition
        if epoch == args.freeze_backbone_epochs + 1:
            print(f"\n[Epoch {epoch}] Unfreezing backbone (backbone_lr={args.backbone_lr:.1e})")
            model.unfreeze_backbone()
            optimizer, scheduler = _make_optimizer_scheduler(phase=2)

        model.train()
        losses_gp, losses_tc, totals = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda" and args.mixed_precision != "no")):
                loss, lgp, ltc, llang, n0, n1 = train_step_with_lang(
                    model, batch, criterion, device,
                    args.lambda1, args.lambda2,
                    kl_weight=args.kl_weight,
                    lang_goal_weight=args.lang_goal_weight,
                    use_inbatch=True, amp_enabled=amp_enabled,
                )

            if loss.requires_grad is False or loss.item() == 0.0:
                continue   # batch had <2 samples of either mode

            # Guard against NaN/inf loss corrupting all model weights
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠️  Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            totals.append(loss.item())
            if lgp > 0:
                losses_gp.append(lgp)
            if ltc > 0:
                losses_tc.append(ltc)

            pbar.set_postfix({
                "loss":  f"{np.mean(totals):.4f}",
                "gp":    f"{np.mean(losses_gp) if losses_gp else 0:.3f}",
                "tc":    f"{np.mean(losses_tc) if losses_tc else 0:.3f}",
                "lang":  f"{llang:.3f}",
                "lr":    f"{optimizer.param_groups[0]['lr']:.1e}",
            })

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = evaluate(
            model, val_ds_for_eval, device,
            n_samples=args.val_samples,
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
        print(json.dumps(record, indent=2))

        tracker.log(
            {
                "train/loss":   train_loss,
                "train/loss_gp": record["train_gp"],
                "train/loss_tc": record["train_tc"],
                "val/auroc_gp": val_metrics["auroc_gp"],
                "val/auroc_tc": val_metrics["auroc_tc"],
                "val/mean":     val_metrics["mean_val"],
                "val/unc_gp":   val_metrics["mean_unc_gp"],
                "val/unc_tc":   val_metrics["mean_unc_tc"],
                "lr":           optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        mean_val = val_metrics["mean_val"]
        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch":            epoch,
            "val_auroc_gp":     val_metrics["auroc_gp"],
            "val_auroc_tc":     val_metrics["auroc_tc"],
            "args":             vars(args),
        }
        # Always save latest
        torch.save(ckpt, out_dir / "latest_contrastive_critic.pt")
        # Save best
        if not np.isnan(mean_val) and mean_val > best_val:
            best_val = mean_val
            torch.save(ckpt, out_dir / "best_contrastive_critic.pt")
            print(f"  ✓ New best: mean_auroc={best_val:.4f}")

    # ── Finalise ──────────────────────────────────────────────────────────────
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    tracker.close()
    print(f"\nBest checkpoint: {out_dir / 'best_contrastive_critic.pt'}  "
          f"(mean_auroc={best_val:.4f})")


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DINOv2DualHeadCritic (contrastive)")

    # Data
    p.add_argument("--dataset-dir",      type=str, required=True)
    p.add_argument("--transitions-file", type=str, default="transitions.jsonl")
    p.add_argument("--image-size",       type=int, default=224,
                   help="Resize target for DINOv2 (must be divisible by 14; 224 recommended)")
    p.add_argument("--mode0-prob",       type=float, default=0.5,
                   help="Fraction of batch items that are mode-0 (goal proximity) pairs")
    p.add_argument("--val-frac",         type=float, default=0.1)
    p.add_argument("--val-samples",      type=int, default=500,
                   help="Number of samples for validation AUROC estimation")

    # Training
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--learning-rate", type=float, default=1e-3,
                   help="Head LR throughout; also backbone LR during phase 1 (no effect, frozen)")
    p.add_argument("--backbone-lr",   type=float, default=1e-5,
                   help="Backbone LR during phase 2 (full fine-tuning)")
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="Cap batches per epoch (0 = all). Useful for debugging.")

    # Phase schedule
    p.add_argument("--freeze-backbone-epochs", type=int, default=5,
                   help="Epochs with backbone frozen (head warm-up). Phase 2 unfreezes.")
    p.add_argument("--warmup-epochs", type=int, default=2,
                   help="Linear LR warmup epochs during phase 1.")

    # InfoNCE
    p.add_argument("--temperature", type=float, default=0.07,
                   help="InfoNCE temperature τ (0.07 = MoCo/SimCLR default)")
    p.add_argument("--lambda1", type=float, default=1.0,
                   help="Weight for goal proximity InfoNCE loss (Head 1)")
    p.add_argument("--lambda2", type=float, default=1.0,
                   help="Weight for temporal consistency InfoNCE loss (Head 2)")
    p.add_argument("--kl-weight", type=float, default=1e-3,
                   help="Weight for KL regulariser on log-variance heads. "
                        "Pulls σ toward 1 without pulling μ toward 0. "
                        "Start with 1e-3; increase to 1e-2 if uncertainty collapses.")
    p.add_argument("--lang-goal-weight", type=float, default=0.5,
                   help="Weight for CLIP↔DINOv2 language-goal alignment loss on the GP head. "
                        "Set to 0 to disable (pure visual-visual GP training, original behaviour). "
                        "Requires batch items to contain 'lang_goal' and 'has_lang_goal' fields.")

    # Output
    p.add_argument("--output-dir", type=str, default="verify2act/output/contrastive")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str, default="cuda", choices=["cuda", "cuda:1", "cpu"])
    p.add_argument("--mixed-precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--tracker", type=str, default="tensorboard",
                   choices=["tensorboard", "wandb", "none"])
    p.add_argument("--wandb-project", type=str, default="verify2act-contrastive")

    return p.parse_args()


if __name__ == "__main__":
    main()
