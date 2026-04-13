#!/usr/bin/env python3
"""Zero-shot DINOv2 diagnostic — validates patch-mean vs CLS-token AUROC.

Step 1 from CRITIC_REDESIGN_HANDOFF.md:
  "Before any fine-tuning, run a quick zero-shot test.  Load frozen
   dinov2_vitb14, encode all late-frames + goal images from successful
   episodes, compute cosine similarities, and check AUROC."

Expected outcome (from DINO-WM Table 2):
  AUROC(patch_mean) >> AUROC(CLS)
  for goal-proximity discrimination between late and early frames.

Usage
-----
  python verify2act/critic/diagnose_dino.py \
      --dataset-dir robosuite/data_capture_wm/dataset/nut_assembly_merged \
      --n-samples 400

Output
------
  Console table + optional JSON report at --output-path.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # project root

# ── Image loading ──────────────────────────────────────────────────────────────

def _load_image(path: str, size: int = 224) -> torch.Tensor:
    """Load an image file to a [1, 3, size, size] float32 tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)   # [3, H, W]
    return (tensor * 2.0 - 1.0).unsqueeze(0)           # [1, 3, H, W]


# ── Normalisation (ImageNet) ───────────────────────────────────────────────────

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def _imagenet_normalise(t: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] tensor to ImageNet-normalised tensor."""
    t01 = (t + 1.0) / 2.0   # → [0, 1]
    return (t01 - _MEAN.to(t.device)) / _STD.to(t.device)


# ── Dataset loading ────────────────────────────────────────────────────────────

def _load_dataset(
    dataset_dir: str,
    transitions_file: str = "transitions_subskill.jsonl",
    labels_file: str = "labels.jsonl",
) -> Tuple[Dict, Dict]:
    """Return (transitions_by_ep, success_set) dicts."""
    ds_path = Path(dataset_dir)

    rows_by_ep: Dict[str, List[dict]] = {}
    with open(ds_path / transitions_file) as f:
        for line in f:
            row = json.loads(line)
            ep = row["episode_id"]
            rows_by_ep.setdefault(ep, []).append(row)

    success_set = set()
    with open(ds_path / labels_file) as f:
        labels_by_ep: Dict[str, List[int]] = {}
        for line in f:
            row = json.loads(line)
            ep = row["episode_id"]
            labels_by_ep.setdefault(ep, []).append(row["label_reachable"])

    for ep, labels in labels_by_ep.items():
        n = len(labels)
        cutoff = max(1, int(n * 0.8))
        if any(labels[cutoff:]):
            success_set.add(ep)

    return rows_by_ep, success_set


# ── Encoding with both strategies ─────────────────────────────────────────────

@torch.no_grad()
def _encode_batch(
    backbone: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (cls_tokens, patch_means) for a batch of images.

    images : [B, 3, 224, 224]  in [-1, 1]
    returns: cls   [B, 768]
             patch [B, 768]
    """
    x = _imagenet_normalise(images.to(device))

    # DINOv2 forward_features returns a dict with 'x_norm_clstoken'
    # and 'x_norm_patchtokens'
    out = backbone.forward_features(x)
    cls    = out["x_norm_clstoken"]           # [B, 768]
    patches = out["x_norm_patchtokens"]        # [B, 256, 768]
    patch_mean = patches.mean(dim=1)           # [B, 768]
    return cls, patch_mean


# ── AUROC computation ──────────────────────────────────────────────────────────

def _auroc(pos_sims: List[float], neg_sims: List[float]) -> float:
    scores = pos_sims + neg_sims
    labels = [1] * len(pos_sims) + [0] * len(neg_sims)
    return float(roc_auc_score(labels, scores))


# ── Main diagnostic ────────────────────────────────────────────────────────────

def run_diagnostic(args: argparse.Namespace) -> Dict:
    device = torch.device(args.device)

    print("Loading DINOv2-B/14 from torch.hub (frozen)...")
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14",
        pretrained=True,
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    print(f"Loading dataset from {args.dataset_dir}...")
    rows_by_ep, success_set = _load_dataset(
        args.dataset_dir, args.transitions_file, args.labels_file
    )
    n_success = len(success_set)
    n_total   = len(rows_by_ep)
    print(f"  {n_success} successful / {n_total} total episodes")

    # Build positive pairs: (late_frame, goal_image) from successful episodes
    # Build negative pairs: (early_frame, goal_image) from any episode
    positives: List[Tuple[str, str]] = []
    negatives: List[Tuple[str, str]] = []

    rng = np.random.default_rng(args.seed)

    for ep_id, rows in rows_by_ep.items():
        rows_sorted = sorted(rows, key=lambda r: r["timestep"])
        n = len(rows_sorted)
        if n < 4:
            continue
        cutoff = max(1, int(n * 0.8))

        goal_path = rows_sorted[0]["goal_image"]

        if ep_id in success_set and rows_sorted[cutoff:]:
            late_row = rows_sorted[cutoff + rng.integers(0, n - cutoff)]
            positives.append((late_row["image_t1"], goal_path))

        early_row = rows_sorted[rng.integers(0, max(1, cutoff // 2))]
        negatives.append((early_row["image_t1"], goal_path))

    rng.shuffle(positives)
    rng.shuffle(negatives)
    n_pos = min(args.n_samples, len(positives))
    n_neg = min(args.n_samples, len(negatives))
    positives = positives[:n_pos]
    negatives = negatives[:n_neg]
    print(f"  {n_pos} positive pairs, {n_neg} negative pairs")

    def _collect_sims(pairs: List[Tuple[str, str]]) -> Tuple[List[float], List[float]]:
        cls_sims, patch_sims = [], []
        base = Path(args.dataset_dir)
        n_skipped = 0

        for img_path, goal_path in tqdm(pairs, leave=False):
            img_p  = (base / img_path).resolve()
            goal_p = (base / goal_path).resolve()

            if not img_p.is_file() or not goal_p.is_file():
                n_skipped += 1
                continue

            img  = _load_image(str(img_p),  args.image_size)
            goal = _load_image(str(goal_p), args.image_size)

            cls_i,  patch_i  = _encode_batch(backbone, img,  device)
            cls_g,  patch_g  = _encode_batch(backbone, goal, device)

            cls_sims.append(F.cosine_similarity(cls_i,  cls_g,  dim=-1).item())
            patch_sims.append(F.cosine_similarity(patch_i, patch_g, dim=-1).item())

        if n_skipped:
            print(f"  (skipped {n_skipped} pairs with missing or invalid image paths)")
        return cls_sims, patch_sims

    print("Encoding positive pairs...")
    cls_pos, patch_pos = _collect_sims(positives)
    print("Encoding negative pairs...")
    cls_neg, patch_neg = _collect_sims(negatives)

    auroc_cls   = _auroc(cls_pos,   cls_neg)
    auroc_patch = _auroc(patch_pos, patch_neg)

    sep_cls   = np.mean(cls_pos)   - np.mean(cls_neg)
    sep_patch = np.mean(patch_pos) - np.mean(patch_neg)

    results = {
        "n_positive_pairs":   n_pos,
        "n_negative_pairs":   n_neg,
        "cls_token": {
            "auroc":     auroc_cls,
            "mean_pos_sim": float(np.mean(cls_pos)),
            "mean_neg_sim": float(np.mean(cls_neg)),
            "separation": float(sep_cls),
        },
        "patch_mean": {
            "auroc":     auroc_patch,
            "mean_pos_sim": float(np.mean(patch_pos)),
            "mean_neg_sim": float(np.mean(patch_neg)),
            "separation": float(sep_patch),
        },
        "recommendation": "patch_mean" if auroc_patch >= auroc_cls else "cls_token",
    }

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ZERO-SHOT DINOv2 DIAGNOSTIC — goal proximity AUROC")
    print("=" * 60)
    print(f"  {'Strategy':<15}  {'AUROC':>6}  {'pos_sim':>8}  {'neg_sim':>8}  {'sep':>8}")
    print(f"  {'-'*55}")
    print(f"  {'CLS token':<15}  {auroc_cls:>6.4f}  "
          f"{np.mean(cls_pos):>8.4f}  {np.mean(cls_neg):>8.4f}  {sep_cls:>8.4f}")
    print(f"  {'Patch mean':<15}  {auroc_patch:>6.4f}  "
          f"{np.mean(patch_pos):>8.4f}  {np.mean(patch_neg):>8.4f}  {sep_patch:>8.4f}")
    print("=" * 60)
    winner = results["recommendation"]
    print(f"  Recommendation: use {winner.upper()} (AUROC={results[winner]['auroc']:.4f})")
    print("=" * 60 + "\n")

    if auroc_patch > 0.6:
        print("✓ patch_mean has discriminative zero-shot signal. Proceed with InfoNCE fine-tuning.")
    elif auroc_patch > 0.5:
        print("~ patch_mean has weak signal. Fine-tuning should still help — proceed.")
    else:
        print("✗ Neither strategy shows useful zero-shot signal. Check data quality before training.")

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport written to {out}")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot DINOv2 goal-proximity AUROC: patch_mean vs CLS"
    )
    p.add_argument("--dataset-dir",      type=str, required=True)
    p.add_argument("--transitions-file", type=str, default="transitions_subskill.jsonl")
    p.add_argument("--labels-file",      type=str, default="labels.jsonl")
    p.add_argument("--image-size",       type=int, default=224)
    p.add_argument("--n-samples",        type=int, default=400,
                   help="Number of positive (and negative) pairs to sample")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--device",           type=str, default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--output-path",      type=str, default=None,
                   help="Optional path to write JSON results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnostic(args)
