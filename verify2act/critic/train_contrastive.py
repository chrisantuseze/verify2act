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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from accelerate.utils import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root

from verify2act.critic.losses import InfoNCELoss
from verify2act.critic.model import DINOv2DualHeadCritic
from verify2act.data_loader import build_contrastive_datasets, ContrastivePairDataset


# ── Unified Training Step (DDP and Multi-GPU safe) ───────────────────────────

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
    use_cached: bool = False,
) -> Tuple[torch.Tensor, float, float, float, int, int]:
    """Single forward pass for all heads to compute contrastive and alignment losses.

    To support PyTorch DistributedDataParallel (DDP) safely, this function delegates
    the entire forward pass to model(..., mode="ddp_train_step"), ensuring each parameter
    group undergoes exactly one DDP forward pass per training step.
    """
    anchor   = batch["anchor"].to(device)    # [B, 3, H, W]
    positive = batch["positive"].to(device)
    negative = batch["negative"].to(device)
    modes    = batch["mode"].to(device)       # [B]  0 or 1

    mask0 = (modes == 0)   # goal proximity
    mask1 = (modes == 1)   # temporal consistency

    # Check for valid text goals for Head 1 alignment loss
    lang_goals = batch.get("lang_goal", [])
    has_lang = batch.get("has_lang_goal", None)
    
    use_lang_loss = False
    has_lang_mask = None
    valid_texts = None
    if lang_goal_weight > 0.0 and lang_goals and has_lang is not None:
        has_lang_mask = has_lang.to(device).bool() & mask0
        n_lang = int(has_lang_mask.sum().item())
        if n_lang >= 1:
            use_lang_loss = True
            valid_texts = [lang_goals[i] for i, v in enumerate(has_lang_mask.tolist()) if v]

    # Combine images for batched visual processing
    all_imgs = torch.cat([anchor, positive, negative], dim=0)   # [3B, 3, H, W] or [3B, 256, 768]

    # Single unified DDP forward pass
    outputs = model(
        all_imgs,
        mask0,
        mask1,
        has_lang_mask=has_lang_mask,
        valid_texts=valid_texts,
        use_cached=use_cached,
        mode="ddp_train_step",
    )

    total_loss = torch.tensor(0.0, device=device)
    loss_gp_val = 0.0
    loss_tc_val = 0.0
    loss_lang_val = 0.0
    n_gp = int(mask0.sum().item())
    n_tc = int(mask1.sum().item())

    # ── Head 1: Goal Proximity (GP) Loss ──────────────────────────────────────
    if n_gp > 1:
        a0 = outputs["a0"]
        p0 = outputs["p0"]
        n0 = outputs["n0"]
        loss_gp = criterion(a0, p0, n0, use_inbatch_negatives=use_inbatch, symmetric=True)
        
        lv1_anchor = outputs["lv1_anchor"]
        lv1_positive = outputs["lv1_positive"]
        lv1_negative = outputs["lv1_negative"]
        
        kl_gp = (
            DINOv2DualHeadCritic.kl_loss(lv1_anchor[mask0])
            + DINOv2DualHeadCritic.kl_loss(lv1_positive[mask0])
            + DINOv2DualHeadCritic.kl_loss(lv1_negative[mask0])
        ) / 3.0
        total_loss = total_loss + lambda1 * loss_gp + kl_weight * kl_gp
        loss_gp_val = loss_gp.item()

    # ── Language Alignment Loss ──────────────────────────────────────────────
    if use_lang_loss:
        visual_proj = outputs["visual_proj"]
        lang_proj = outputs["lang_proj"]
        loss_lang = F.mse_loss(lang_proj, visual_proj.detach())
        total_loss = total_loss + lang_goal_weight * loss_lang
        loss_lang_val = loss_lang.item()

    # ── Head 2: Temporal Consistency (TC) Loss ────────────────────────────────
    if n_tc > 1:
        a1 = outputs["a1"]
        p1 = outputs["p1"]
        n1 = outputs["n1"]
        loss_tc = criterion(a1, p1, n1, use_inbatch_negatives=use_inbatch)
        
        lv2_anchor = outputs["lv2_anchor"]
        lv2_positive = outputs["lv2_positive"]
        lv2_negative = outputs["lv2_negative"]
        
        kl_tc = (
            DINOv2DualHeadCritic.kl_loss(lv2_anchor[mask1])
            + DINOv2DualHeadCritic.kl_loss(lv2_positive[mask1])
            + DINOv2DualHeadCritic.kl_loss(lv2_negative[mask1])
        ) / 3.0
        total_loss = total_loss + lambda2 * loss_tc + kl_weight * kl_tc
        loss_tc_val = loss_tc.item()

    return total_loss, loss_gp_val, loss_tc_val, loss_lang_val, n_gp, n_tc


# ── Utilities ──────────────────────────────────────────────────────────────────

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


 # ── Optimizer factory ──────────────────────────────────────────────────────
def _make_optimizer_scheduler(args, model, phase: int):
    if phase == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        n_epochs = args.freeze_backbone_epochs
    else:
        opt = AdamW(
            [
                {"params": model.backbone.parameters() if model.backbone is not None else [], "lr": args.backbone_lr},
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
    
# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
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
    """Compute AUROC and uncertainty metrics for goal proximity and temporal consistency.

    Samples n_samples triplets from val_ds and measures:
      auroc_gp : cos_sim(head1(late), head1(goal)) > cos_sim(head1(early), head1(goal))
      auroc_tc : cos_sim(head2(I_t), head2(I_{t+1})) > cos_sim(head2(I_t), head2(I_cross))

    Deterministic similarities use mu only (no MC sampling) for fast AUROC.
    Uncertainty metrics (mean_unc_gp, mean_unc_tc) use uncertainty_mc_samples
    MC draws from the probabilistic embedding.
    """
    raw_model = accelerator.unwrap_model(model) if accelerator is not None else model
    raw_model.eval()

    gp_pos_sims: List[float] = []
    gp_neg_sims: List[float] = []
    tc_pos_sims: List[float] = []
    tc_neg_sims: List[float] = []
    gp_uncertainties: List[float] = []
    tc_uncertainties: List[float] = []

    # Force evaluation over a fixed random sample of each mode
    collected = {"gp": 0, "tc": 0}
    target_per_mode = n_samples // 2

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )

    for batch in val_loader:
        if collected["gp"] >= target_per_mode and collected["tc"] >= target_per_mode:
            break

        modes = batch["mode"]
        for i in range(len(modes)):
            mode = int(modes[i].item())
            if mode == 0 and collected["gp"] < target_per_mode:
                anchor   = batch["anchor"][i].unsqueeze(0).to(device)
                positive = batch["positive"][i].unsqueeze(0).to(device)
                negative = batch["negative"][i].unsqueeze(0).to(device)

                if use_cached:
                    emb_a = raw_model.encode_features(anchor)
                    emb_p = raw_model.encode_features(positive)
                    emb_n = raw_model.encode_features(negative)
                else:
                    emb_a = raw_model.encode(anchor)
                    emb_p = raw_model.encode(positive)
                    emb_n = raw_model.encode(negative)

                # Deterministic similarity for AUROC
                pos_sim = raw_model.goal_sim(emb_a, emb_p).item()
                neg_sim = raw_model.goal_sim(emb_a, emb_n).item()
                gp_pos_sims.append(pos_sim)
                gp_neg_sims.append(neg_sim)
                # MC uncertainty: std of positive similarity
                _, std_pos = raw_model.goal_sim_with_uncertainty(emb_a, emb_p, n_samples=uncertainty_mc_samples)
                gp_uncertainties.append(std_pos.item())
                collected["gp"] += 1
            elif mode == 1 and collected["tc"] < target_per_mode:
                anchor   = batch["anchor"][i].unsqueeze(0).to(device)
                positive = batch["positive"][i].unsqueeze(0).to(device)
                negative = batch["negative"][i].unsqueeze(0).to(device)

                if use_cached:
                    emb_a = raw_model.encode_features(anchor)
                    emb_p = raw_model.encode_features(positive)
                    emb_n = raw_model.encode_features(negative)
                else:
                    emb_a = raw_model.encode(anchor)
                    emb_p = raw_model.encode(positive)
                    emb_n = raw_model.encode(negative)

                pos_sim = raw_model.temporal_sim(emb_a, emb_p).item()
                neg_sim = raw_model.temporal_sim(emb_a, emb_n).item()
                tc_pos_sims.append(pos_sim)
                tc_neg_sims.append(neg_sim)
                _, std_pos = raw_model.temporal_sim_with_uncertainty(emb_a, emb_p, n_samples=uncertainty_mc_samples)
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

    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    out_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Cache Setup ───────────────────────────────────────────────────────────
    cached_dino_dir = args.cached_dino_dir
    if cached_dino_dir is not None:
        if accelerator.is_local_main_process:
            from verify2act.critic.cache_utils import ensure_cache_complete, ensure_calvin_cache_complete
            
            print(f"Checking DINOv2 feature cache completeness in: {cached_dino_dir}")
            if args.dataset_type == "calvin":
                ensure_calvin_cache_complete(
                    args.dataset_dir,
                    cache_dir=cached_dino_dir,
                    device=str(device),
                    dino_channels=args.dino_channels
                )
            else:
                ensure_cache_complete(
                    args.dataset_dir,
                    transitions_file=args.transitions_file,
                    cache_dir=cached_dino_dir,
                    history_len=1,  # Contrastive critic only needs single-frame DINO features
                    device=str(device),
                    dino_channels=args.dino_channels
                )
        accelerator.wait_for_everyone()

    # ── Datasets ──────────────────────────────────────────────────────────────
    if args.dataset_type == "calvin":
        from verify2act.data_loader_calvin import build_calvin_contrastive_datasets

        train_ds, val_ds = build_calvin_contrastive_datasets(
            dataset_dir=args.dataset_dir,
            val_frac=args.val_frac,
            seed=args.seed,
            image_size=args.image_size,
            mode0_prob=args.mode0_prob,
            cached_dino_dir=cached_dino_dir,
        )
        accelerator.print(
            f"Dataset: {len(train_ds)} train pairs / {len(val_ds)} val pairs | "
            f"mode0_prob={args.mode0_prob}  image_size={args.image_size}"
        )
    else:
        train_ds, val_ds = build_contrastive_datasets(
            dataset_dir=args.dataset_dir,
            transitions_file=args.transitions_file,
            val_frac=args.val_frac,
            seed=args.seed,
            image_size=args.image_size,
            mode0_prob=args.mode0_prob,
            cached_dino_dir=cached_dino_dir,
        )
        accelerator.print(
            f"Dataset: {len(train_ds)} train pairs / {len(val_ds)} val pairs | "
            f"mode0_prob={args.mode0_prob}  image_size={args.image_size}"
        )
        accelerator.print(
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
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_ds_for_eval = val_ds   # keep reference for evaluate()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DINOv2DualHeadCritic(
        pretrained=True,
        load_backbone=True,
        dino_channels=args.dino_channels,
    ).to(device)
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model: {n_total:,} total params | {n_trainable:,} trainable "
          f"(projection + log-var heads; backbone frozen initially)")

    criterion = InfoNCELoss(temperature=args.temperature)
    optimizer, scheduler = _make_optimizer_scheduler(args, model, phase=1)

    best_val = -1.0
    start_epoch = 0
    history = []

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            # Support both full dict checkpoints and plain state_dicts
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # model.load_state_dict(checkpoint["model_state_dict"], strict=False)

                state_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if not k.startswith("_clip_model.")}
                model.load_state_dict(state_dict)
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"]
                else:
                    import re
                    match = re.search(r"ep(\d+)\.pt$", str(resume_path))
                    if match:
                        start_epoch = int(match.group(1))

                if "val_auroc_gp" in checkpoint and "val_auroc_tc" in checkpoint:
                    best_val = (checkpoint["val_auroc_gp"] + checkpoint["val_auroc_tc"]) / 2.0

                if start_epoch > args.freeze_backbone_epochs:
                    print(f"Resuming in Phase 2: Unfreezing backbone before restoring optimizer")
                    model.unfreeze_backbone()
                    optimizer, scheduler = _make_optimizer_scheduler(args, model, phase=2)
                else:
                    optimizer, scheduler = _make_optimizer_scheduler(args, model, phase=1)

                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                # model.load_state_dict(checkpoint, strict=False)

                state_dict = {k: v for k, v in checkpoint.items() if not k.startswith("_clip_model.")}
                model.load_state_dict(state_dict)
                # Try to extract epoch from filename if it matches format
                import re
                match = re.search(r"ep(\d+)\.pt$", str(resume_path))
                if match:
                    start_epoch = int(match.group(1))

                if start_epoch > args.freeze_backbone_epochs:
                    print(f"Resuming in Phase 2: Unfreezing backbone")
                    model.unfreeze_backbone()
                    optimizer, scheduler = _make_optimizer_scheduler(args, model, phase=2)
                    for _ in range(start_epoch - args.freeze_backbone_epochs):
                        scheduler.step()
                else:
                    optimizer, scheduler = _make_optimizer_scheduler(args, model, phase=1)
                    for _ in range(start_epoch):
                        scheduler.step()
            
            # Try to load existing history if resuming
            history_path = out_dir / "train_history.json"
            if history_path.exists():
                try:
                    with open(history_path, "r") as f:
                        history = json.load(f)
                    # Keep only records up to start_epoch
                    history = [r for r in history if r.get("epoch", 0) <= start_epoch]
                except Exception:
                    pass
        else:
            accelerator.print(f"Warning: Checkpoint {resume_path} not found. Starting from scratch.")

    if accelerator.is_main_process:
        tracker = _init_tracker(args.tracker, out_dir, vars(args).copy())
    else:
        tracker = _init_tracker("none", out_dir, vars(args).copy())

    # Prepare model, optimizer, train_loader, scheduler under Accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    amp_enabled = device.type == "cuda" and args.mixed_precision != "no"

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, args.epochs + 1):

        # Phase transition
        if epoch == args.freeze_backbone_epochs + 1 and start_epoch <= args.freeze_backbone_epochs:
            accelerator.print(f"\n[Epoch {epoch}] Unfreezing backbone (backbone_lr={args.backbone_lr:.1e})")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.unfreeze_backbone()
            optimizer, scheduler = _make_optimizer_scheduler(args, unwrapped_model, phase=2)
            model, optimizer, scheduler = accelerator.prepare(unwrapped_model, optimizer, scheduler)

        # Cosine KL weight annealing/warmup
        warmup_epochs = max(1, int(args.epochs * 0.3))
        if epoch <= warmup_epochs:
            epoch_kl_weight = args.kl_weight * (epoch / warmup_epochs)
        else:
            epoch_kl_weight = args.kl_weight

        model.train()
        losses_gp, losses_tc, totals = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True, disable=not accelerator.is_local_main_process)
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda" and args.mixed_precision != "no")):
                loss, lgp, ltc, llang, n0, n1 = train_step_with_lang(
                    model, batch, criterion, device,
                    args.lambda1, args.lambda2,
                    kl_weight=epoch_kl_weight,
                    lang_goal_weight=args.lang_goal_weight,
                    use_inbatch=True, amp_enabled=amp_enabled,
                    use_cached=(args.cached_dino_dir is not None),
                )

            if loss.requires_grad is False or loss.item() == 0.0:
                continue   # batch had <2 samples of either mode

            # Guard against NaN/inf loss corrupting all model weights
            if torch.isnan(loss) or torch.isinf(loss):
                accelerator.print(f"  ⚠️  Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
                continue

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

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
            accelerator=accelerator,
            n_samples=args.val_samples,
            use_cached=(args.cached_dino_dir is not None),
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
        unwrapped_model = accelerator.unwrap_model(model)
        ckpt = {
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch":            epoch,
            "val_auroc_gp":     val_metrics["auroc_gp"],
            "val_auroc_tc":     val_metrics["auroc_tc"],
            "args":             vars(args),
        }
        if accelerator.is_main_process:
            # Always save latest
            torch.save(ckpt, out_dir / "latest_contrastive_critic.pt")
            # Save best
            if not np.isnan(mean_val) and mean_val > best_val:
                best_val = mean_val
                torch.save(ckpt, out_dir / "best_contrastive_critic.pt")
                accelerator.print(f"  ✓ New best: mean_auroc={best_val:.4f}")

    # ── Finalise ──────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        with open(out_dir / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(out_dir / "train_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    tracker.close()
    accelerator.print(f"\nBest checkpoint: {out_dir / 'best_contrastive_critic.pt'}  "
          f"(mean_auroc={best_val:.4f})")


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DINOv2DualHeadCritic (contrastive)")

    # Data
    p.add_argument("--dataset-dir",      type=str, required=True)
    p.add_argument("--transitions-file", type=str, default="transitions.jsonl")
    p.add_argument("--image-size",       type=int, default=224,
                   help="Resize target for DINOv2 (must be divisible by 14; 224 recommended)")
    p.add_argument("--dino-channels",    type=int, default=1024, choices=[768, 1024],
                   help="Backbone output channel dimension (768 for ViT-B, 1024 for ViT-L)")
    p.add_argument("--mode0-prob",       type=float, default=0.5,
                   help="Fraction of batch items that are mode-0 (goal proximity) pairs")
    p.add_argument("--val-frac",         type=float, default=0.1)
    p.add_argument("--val-samples",      type=int, default=500,
                   help="Number of samples for validation AUROC estimation")
    p.add_argument("--dataset-type", type=str, default="robosuite", choices=["robosuite", "calvin"],
                        help="Type of dataset loader to use")

    # Training
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="Number of batches loaded in advance by each worker")
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
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to critic model checkpoint to resume from")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str, default="cuda", choices=["cuda", "cuda:1", "cpu"])
    p.add_argument("--mixed-precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--tracker", type=str, default="tensorboard",
                   choices=["tensorboard", "wandb", "none"])
    p.add_argument("--wandb-project", type=str, default="verify2act-contrastive")

    p.add_argument("--cached-dino-dir", type=str, default=None,
               help="Path to pre-extracted DINOv2 features cache directory")

    return p.parse_args()


if __name__ == "__main__":
    main()
