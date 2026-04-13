#!/usr/bin/env python3
"""Post-training threshold calibration for DINO dual-head critic.

Calibrates:
- theta_p (goal proximity): final-frame reflect threshold
- theta_c (temporal consistency): per-step requery threshold

Method:
1) Build val split with build_contrastive_datasets(...)
2) Sample mode-0 triplets for proximity scores
3) Sample mode-1 triplets for temporal consistency scores
4) Compute ROC curves and recommend thresholds via Youden's J statistic

Outputs a JSON report with score summaries and threshold recommendations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verify2act.critic.model import DINOv2DualHeadCritic
from verify2act.data_loader import ContrastivePairDataset, build_contrastive_datasets


@torch.no_grad()
def _collect_scores(
    model: DINOv2DualHeadCritic,
    val_ds: ContrastivePairDataset,
    device: torch.device,
    n_mode0: int,
    n_mode1: int,
) -> Dict[str, List[float]]:
    """Collect score distributions from val triplets.

    Returns
    -------
    dict with keys:
      gp_pos, gp_neg, tc_pos, tc_neg
    """
    model.eval()

    gp_pos: List[float] = []
    gp_neg: List[float] = []
    tc_pos: List[float] = []
    tc_neg: List[float] = []

    # Mode 0: goal proximity
    for _ in tqdm(range(n_mode0), desc="Sampling mode-0 (goal proximity)"):
        item = val_ds._sample_mode0()
        anchor = item["anchor"].unsqueeze(0).to(device)
        positive = item["positive"].unsqueeze(0).to(device)
        negative = item["negative"].unsqueeze(0).to(device)

        e_a = model.encode(anchor)
        e_p = model.encode(positive)
        e_n = model.encode(negative)

        gp_pos.append(float(model.goal_sim(e_a, e_p).item()))
        gp_neg.append(float(model.goal_sim(e_a, e_n).item()))

    # Mode 1: temporal consistency
    for _ in tqdm(range(n_mode1), desc="Sampling mode-1 (temporal consistency)"):
        item = val_ds._sample_mode1()
        anchor = item["anchor"].unsqueeze(0).to(device)
        positive = item["positive"].unsqueeze(0).to(device)
        negative = item["negative"].unsqueeze(0).to(device)

        e_a = model.encode(anchor)
        e_p = model.encode(positive)
        e_n = model.encode(negative)

        tc_pos.append(float(model.temporal_sim(e_a, e_p).item()))
        tc_neg.append(float(model.temporal_sim(e_a, e_n).item()))

    return {
        "gp_pos": gp_pos,
        "gp_neg": gp_neg,
        "tc_pos": tc_pos,
        "tc_neg": tc_neg,
    }


def _summarize(pos: List[float], neg: List[float]) -> Dict[str, float]:
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    scores = np.array(pos + neg, dtype=np.float32)

    auroc = float(roc_auc_score(labels, scores))
    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    thr_youden = float(thr[best_idx])

    return {
        "auroc": auroc,
        "mean_pos": float(np.mean(pos)),
        "mean_neg": float(np.mean(neg)),
        "sep": float(np.mean(pos) - np.mean(neg)),
        "threshold_youden": thr_youden,
        "p10_pos": float(np.percentile(pos, 10)),
        "p90_neg": float(np.percentile(neg, 90)),
    }


def _load_model(ckpt_path: str, device: torch.device) -> DINOv2DualHeadCritic:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model = DINOv2DualHeadCritic(pretrained=False).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate theta_p and theta_c after training")

    p.add_argument("--critic-ckpt", type=str, required=True)
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--transitions-file", type=str, default="transitions_subskill.jsonl")
    p.add_argument("--labels-file", type=str, default="labels.jsonl")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--mode0-prob", type=float, default=0.5)

    p.add_argument("--n-mode0", type=int, default=1200, help="Val mode-0 samples")
    p.add_argument("--n-mode1", type=int, default=1200, help="Val mode-1 samples")

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--output-path",
        type=str,
        default="verify2act/output/contrastive/threshold_calibration.json",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    print("Loading validation split...")
    _, val_ds = build_contrastive_datasets(
        dataset_dir=args.dataset_dir,
        transitions_file=args.transitions_file,
        labels_file=args.labels_file,
        val_frac=args.val_frac,
        seed=args.seed,
        image_size=args.image_size,
        mode0_prob=args.mode0_prob,
    )

    print("Loading critic checkpoint...")
    model = _load_model(args.critic_ckpt, device)

    print("Collecting validation score distributions...")
    scores = _collect_scores(model, val_ds, device, args.n_mode0, args.n_mode1)

    gp = _summarize(scores["gp_pos"], scores["gp_neg"])
    tc = _summarize(scores["tc_pos"], scores["tc_neg"])

    results = {
        "checkpoint": args.critic_ckpt,
        "dataset_dir": args.dataset_dir,
        "n_mode0": args.n_mode0,
        "n_mode1": args.n_mode1,
        "goal_proximity": gp,
        "temporal_consistency": tc,
        "recommended": {
            "theta_p": gp["threshold_youden"],
            "theta_c": tc["threshold_youden"],
        },
    }

    print("\n" + "=" * 66)
    print("  POST-TRAIN THRESHOLD CALIBRATION")
    print("=" * 66)
    print(
        f"  Goal proximity    AUROC={gp['auroc']:.4f}  "
        f"mean_pos={gp['mean_pos']:.4f}  mean_neg={gp['mean_neg']:.4f}"
    )
    print(
        f"    -> theta_p (Youden) = {gp['threshold_youden']:.4f}  "
        f"[p10_pos={gp['p10_pos']:.4f}, p90_neg={gp['p90_neg']:.4f}]"
    )
    print(
        f"  Temporal consistency AUROC={tc['auroc']:.4f}  "
        f"mean_pos={tc['mean_pos']:.4f}  mean_neg={tc['mean_neg']:.4f}"
    )
    print(
        f"    -> theta_c (Youden) = {tc['threshold_youden']:.4f}  "
        f"[p10_pos={tc['p10_pos']:.4f}, p90_neg={tc['p90_neg']:.4f}]"
    )
    print("=" * 66)

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved calibration report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
