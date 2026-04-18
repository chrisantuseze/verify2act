#!/usr/bin/env python3
"""Validate latent embedding shift between rendered and generated states.

This script compares matched pairs of:
  - rendered next-state image (ground truth simulator render)
  - generated next-state image (world model imagination)

Both images are encoded through the same frozen VAE encoder, and the script
reports distance statistics plus linear domain separability on pooled latents.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_pairs(
    dataset_dir: Path,
    generated_manifest: Path,
    real_field: str,
    generated_field: str,
    transitions_file: str = "transitions_subskill.jsonl",
) -> List[Tuple[str, int, Path, Path]]:
    transitions_path = dataset_dir / transitions_file
    if not transitions_path.exists():
        raise FileNotFoundError(f"Missing transitions file: {transitions_path}")
    if not generated_manifest.exists():
        raise FileNotFoundError(f"Missing generated manifest: {generated_manifest}")

    real_rows = _read_jsonl(transitions_path)
    gen_rows = _read_jsonl(generated_manifest)

    real_map: Dict[Tuple[str, int], Dict] = {}
    for row in real_rows:
        key = (row["episode_id"], int(row["timestep"]))
        real_map[key] = row

    gen_map: Dict[Tuple[str, int], Dict] = {}
    for row in gen_rows:
        if "episode_id" not in row or "timestep" not in row:
            continue
        key = (row["episode_id"], int(row["timestep"]))
        gen_map[key] = row

    pairs: List[Tuple[str, int, Path, Path]] = []
    for key, real_row in real_map.items():
        if key not in gen_map:
            continue
        gen_row = gen_map[key]

        real_rel = real_row.get(real_field)
        gen_rel = gen_row.get(generated_field)
        if not real_rel or not gen_rel:
            continue

        real_path = dataset_dir / real_rel
        gen_path = dataset_dir / gen_rel
        if not real_path.exists() or not gen_path.exists():
            continue

        pairs.append((key[0], key[1], real_path, gen_path))

    if not pairs:
        raise RuntimeError(
            "No matched pairs found. "
            f"Check real_field='{real_field}', generated_field='{generated_field}', and manifest alignment."
        )

    return pairs


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), resample=Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    generated_manifest = Path(args.generated_manifest)
    if not generated_manifest.is_absolute():
        generated_manifest = dataset_dir / generated_manifest

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    dtype = get_dtype(args.mixed_precision)
    vae, resolved_subfolder = load_vae_encoder(
        model_name_or_path=args.vae_model,
        device=device,
        torch_dtype=dtype,
        subfolder=args.vae_subfolder,
        local_files_only=args.local_files_only,
    )

    pairs = _build_pairs(
        dataset_dir=dataset_dir,
        generated_manifest=generated_manifest,
        real_field=args.real_field,
        generated_field=args.generated_field,
        transitions_file=args.transitions_file,
    )
    if args.max_pairs > 0 and len(pairs) > args.max_pairs:
        random.shuffle(pairs)
        pairs = pairs[: args.max_pairs]

    real_pooled: List[np.ndarray] = []
    gen_pooled: List[np.ndarray] = []
    pair_records: List[Dict] = []

    latent_scale = VAE_LATENT_SCALE

    for start in tqdm(range(0, len(pairs), args.batch_size), desc="Encoding pairs"):
        chunk = pairs[start : start + args.batch_size]

        real_batch = torch.stack([_load_image_tensor(item[2], args.image_size) for item in chunk], dim=0)
        gen_batch = torch.stack([_load_image_tensor(item[3], args.image_size) for item in chunk], dim=0)

        real_batch = real_batch.to(device=device, dtype=dtype)
        gen_batch = gen_batch.to(device=device, dtype=dtype)

        with torch.no_grad():
            z_real = vae.encode(real_batch).latent_dist.mean * latent_scale
            z_gen = vae.encode(gen_batch).latent_dist.mean * latent_scale

            pooled_real = z_real.mean(dim=(-2, -1)).float()
            pooled_gen = z_gen.mean(dim=(-2, -1)).float()

            l2 = torch.norm(pooled_real - pooled_gen, p=2, dim=1)
            cosine = F.cosine_similarity(pooled_real, pooled_gen, dim=1)

        pooled_real_np = pooled_real.cpu().numpy()
        pooled_gen_np = pooled_gen.cpu().numpy()
        l2_np = l2.cpu().numpy()
        cosine_np = cosine.cpu().numpy()

        real_pooled.extend([row for row in pooled_real_np])
        gen_pooled.extend([row for row in pooled_gen_np])

        for i, (episode_id, timestep, real_path, gen_path) in enumerate(chunk):
            pair_records.append(
                {
                    "episode_id": episode_id,
                    "timestep": timestep,
                    "real_path": str(real_path.relative_to(dataset_dir)),
                    "generated_path": str(gen_path.relative_to(dataset_dir)),
                    "l2_pooled": float(l2_np[i]),
                    "cosine_similarity": float(cosine_np[i]),
                }
            )

    real_pooled_np = np.stack(real_pooled, axis=0)
    gen_pooled_np = np.stack(gen_pooled, axis=0)

    l2_all = np.linalg.norm(real_pooled_np - gen_pooled_np, axis=1)
    cos_all = np.sum(real_pooled_np * gen_pooled_np, axis=1) / (
        np.linalg.norm(real_pooled_np, axis=1) * np.linalg.norm(gen_pooled_np, axis=1) + 1e-8
    )

    X = np.concatenate([real_pooled_np, gen_pooled_np], axis=0)
    y = np.concatenate(
        [np.zeros(len(real_pooled_np), dtype=np.int64), np.ones(len(gen_pooled_np), dtype=np.int64)],
        axis=0,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    clf = LogisticRegression(max_iter=2000, random_state=args.seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    domain_acc = float((y_pred == y_test).mean())
    domain_auc = float(roc_auc_score(y_test, y_prob))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_csv_path = out_dir / "pair_metrics.csv"
    with open(pair_csv_path, "w") as f:
        f.write("episode_id,timestep,real_path,generated_path,l2_pooled,cosine_similarity\n")
        for row in pair_records:
            f.write(
                f"{row['episode_id']},{row['timestep']},{row['real_path']},{row['generated_path']},"
                f"{row['l2_pooled']:.8f},{row['cosine_similarity']:.8f}\n"
            )

    summary = {
        "num_pairs": len(pair_records),
        "vae_model": args.vae_model,
        "vae_subfolder_resolved": resolved_subfolder,
        "real_field": args.real_field,
        "generated_field": args.generated_field,
        "l2_pooled": _summarize(l2_all),
        "cosine_similarity": _summarize(cos_all),
        "domain_classifier": {
            "type": "logistic_regression",
            "test_size": args.test_size,
            "accuracy": domain_acc,
            "auroc": domain_auc,
        },
        "outputs": {
            "pair_metrics_csv": str(pair_csv_path),
        },
    }

    summary_path = out_dir / "embedding_shift_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved pair metrics CSV: {pair_csv_path}")
    print(f"Saved summary JSON:     {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate embedding shift (rendered vs generated)")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument(
        "--transitions-file",
        type=str,
        default="transitions.jsonl",
        help="JSONL filename inside dataset-dir for real rendered transitions.",
    )
    parser.add_argument(
        "--generated-manifest",
        type=str,
        default="transitions.jsonl",
        help="JSONL with episode_id,timestep and generated image path field.",
    )
    parser.add_argument(
        "--real-field",
        type=str,
        default="image_t1",
        help="Field in dataset transitions.jsonl for rendered next-state image.",
    )
    parser.add_argument(
        "--generated-field",
        type=str,
        default="image_t1_hat",
        help="Field in generated-manifest for generated next-state image.",
    )

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
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means use all pairs")
    parser.add_argument("--test-size", type=float, default=0.3)

    parser.add_argument("--output-dir", type=str, default="verify2act/output/embedding_shift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()


if __name__ == "__main__":
    main()
