import json
import torch
import gc
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DINO_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
IMG_SIZE = 224

def _resolve_goal_image(trans: dict, dataset_root: Path) -> str:
    goal = (trans.get("goal_image") or "").strip()
    if goal:
        return goal
    fallback = Path("episodes") / trans["episode_id"] / "goal.png"
    if (dataset_root / fallback).exists():
        return str(fallback)
    return ""

def ensure_cache_complete(
    dataset_dir: str,
    transitions_file: str = "transitions.jsonl",
    cache_dir: str = "dino_features",
    device: str = "cuda",
    batch_size: int = 64,
    co_locate: bool = True,
    history_len: int = 3,
    dino_channels: int = 1024,
) -> None:
    """Checks the feature cache directory, and generates missing features if necessary."""
    root = Path(dataset_dir)
    cache_root = Path(cache_dir)
    if not co_locate:
        cache_root.mkdir(parents=True, exist_ok=True)

    trans_file_path = root / transitions_file
    if not trans_file_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {trans_file_path}")

    # 1. Collect and validate transitions by sliding window (mimics LatentDynamicsDataset)
    episodes = {}
    with open(trans_file_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep_id = row.get("episode_id")
            ts = row.get("timestep")
            if ep_id is not None and ts is not None:
                ts = int(ts)
                if ep_id not in episodes:
                    episodes[ep_id] = []
                episodes[ep_id].append((ts, row))

    # Sort each episode by timestep
    for ep_id in episodes:
        episodes[ep_id].sort(key=lambda x: x[0])

    needed_paths = set()
    for ep_id, ep_rows in episodes.items():
        rows = [r[1] for r in ep_rows]
        for i in range(len(rows)):
            target_relpath = rows[i].get("image_t1", "")
            
            history_relpaths = []
            for j in range(history_len - 1, -1, -1):
                idx = i - j
                history_relpaths.append(rows[max(0, idx)].get("image_t", ""))

            all_paths = history_relpaths + [target_relpath]
            # Check if all these images exist on disk
            if all(p and (root / p).exists() for p in all_paths):
                # Valid sequence! We need these features
                for p in all_paths:
                    needed_paths.add(p)
                # Also resolve goal image if any
                goal = _resolve_goal_image(rows[i], root)
                if goal and (root / goal).exists():
                    needed_paths.add(goal)

    # 2. Check which features are missing
    missing_paths = []
    for p in sorted(needed_paths):
        if co_locate:
            full_path = root / p
            feat_path = full_path.parent / (full_path.stem + "_dino.pt")
        else:
            feat_name = p.replace("/", "_") + ".pt"
            feat_path = cache_root / feat_name
            
        if not feat_path.exists():
            missing_paths.append(p)

    if not missing_paths:
        print(f"✓ DINOv2 feature cache is complete ({len(needed_paths)} features validated).")
        return

    print(f"Detected {len(missing_paths)} missing features out of {len(needed_paths)} unique active images.")
    print("Initializing DINOv2 backbone to precompute features...")

    # 3. Load DINOv2 backbone
    model_name = "dinov2_vitl14" if dino_channels == 1024 else "dinov2_vitb14"
    dino = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    dino = dino.eval().to(device)

    print(f"Device: {device}")

    xform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        DINO_NORM,
    ])

    # 4. Generate missing features in batches
    for i in tqdm(range(0, len(missing_paths), batch_size), desc="Precomputing DINOv2 features", dynamic_ncols=True):
        chunk = missing_paths[i:i+batch_size]
        imgs_list = []
        valid_chunk = []
        for p in chunk:
            img_path = root / p
            try:
                img = Image.open(img_path).convert("RGB")
                imgs_list.append(xform(img))
                valid_chunk.append(p)
            except Exception as e:
                print(f"\n⚠️ Error loading image {img_path}: {e}. Skipping.")
                continue

        if not imgs_list:
            continue

        imgs_tensor = torch.stack(imgs_list).to(device)

        with torch.no_grad():
            feats = dino.forward_features(imgs_tensor)["x_norm_patchtokens"]  # [B, 256, 1024]

        # Save to disk as fp16 (saves 50% disk space)
        for p, feat in zip(valid_chunk, feats):
            if co_locate:
                full_path = root / p
                feat_path = full_path.parent / (full_path.stem + "_dino.pt")
            else:
                feat_name = p.replace("/", "_") + ".pt"
                feat_path = cache_root / feat_name
            torch.save(feat.half().cpu(), feat_path)

    # 5. Clean up VRAM/RAM completely
    print("Feature generation complete. Cleaning up DINOv2 from VRAM...")
    del dino
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM successfully reclaimed.")

def ensure_calvin_cache_complete(
    dataset_dir: str,
    cache_dir: str = "dino_features",
    device: str = "cuda",
    batch_size: int = 64,
    co_locate: bool = True,
    dino_channels: int = 1024,
) -> None:
    """Checks the feature cache directory for CALVIN, and generates missing features if necessary."""
    root = Path(dataset_dir)
    cache_root = Path(cache_dir)
    if not co_locate:
        cache_root.mkdir(parents=True, exist_ok=True)

    # 1. Collect all episode npz files
    npz_files = list(root.glob("episode_*.npz"))
    
    # 2. Check which features are missing
    missing_files = []
    for f in sorted(npz_files):
        if co_locate:
            feat_path = f.parent / (f.stem + "_dino.pt")
        else:
            feat_name = f.name.replace(".npz", ".pt")
            feat_path = cache_root / feat_name
            
        if not feat_path.exists():
            missing_files.append(f)

    if not missing_files:
        print(f"✓ DINOv2 feature cache for CALVIN is complete ({len(npz_files)} features verified).")
        return

    print(f"Detected {len(missing_files)} missing features out of {len(npz_files)} CALVIN episode files.")
    print("Initializing DINOv2 backbone to precompute features...")

    # 3. Load DINOv2 backbone
    model_name = "dinov2_vitl14" if dino_channels == 1024 else "dinov2_vitb14"
    dino = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    dino = dino.eval().to(device)

    xform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        DINO_NORM,
    ])

    # 4. Generate missing features in batches
    for i in tqdm(range(0, len(missing_files), batch_size), desc="Precomputing CALVIN DINOv2 features", dynamic_ncols=True):
        chunk = missing_files[i:i+batch_size]
        imgs_list = []
        valid_chunk = []
        for npz_path in chunk:
            try:
                data = np.load(npz_path, allow_pickle=True)
                img_np = data["rgb_static"]
                img = Image.fromarray(img_np).convert("RGB")
                imgs_list.append(xform(img))
                valid_chunk.append(npz_path)
            except Exception as e:
                print(f"\n⚠️ Error loading image from {npz_path}: {e}. Skipping.")
                continue

        if not imgs_list:
            continue

        imgs_tensor = torch.stack(imgs_list).to(device)

        with torch.no_grad():
            feats = dino.forward_features(imgs_tensor)["x_norm_patchtokens"]  # [B, 256, 1024]

        for npz_path, feat in zip(valid_chunk, feats):
            if co_locate:
                feat_path = npz_path.parent / (npz_path.stem + "_dino.pt")
            else:
                feat_name = npz_path.name.replace(".npz", ".pt")
                feat_path = cache_root / feat_name
            torch.save(feat.half().cpu(), feat_path)

    print("Feature generation complete. Cleaning up DINOv2 from VRAM...")
    del dino
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM successfully reclaimed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Precompute DINOv2 feature cache")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset-type", type=str, default="robosuite", choices=["robosuite", "calvin"], help="Type of dataset")
    parser.add_argument("--transitions-file", type=str, default="transitions.jsonl", help="JSONL transitions file (RoboSuite only)")
    parser.add_argument("--cache-dir", type=str, default="dino_features", help="Fallback cache directory if not co-locating")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--no-co-locate", action="store_true", default=False, help="Disable co-locating .pt files next to raw images")
    parser.add_argument("--history-len", type=int, default=3, help="History window length for validation check (RoboSuite only)")
    parser.add_argument("--dino-channels", type=int, default=1024, help="DINO feature dimension (768 for ViT-B, 1024 for ViT-L)")
    
    args = parser.parse_args()
    
    if args.dataset_type == "calvin":
        ensure_calvin_cache_complete(
            dataset_dir=args.dataset_dir,
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=args.batch_size,
            co_locate=not args.no_co_locate,
            dino_channels=args.dino_channels
        )
    else:
        ensure_cache_complete(
            dataset_dir=args.dataset_dir,
            transitions_file=args.transitions_file,
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=args.batch_size,
            co_locate=not args.no_co_locate,
            history_len=args.history_len,
            dino_channels=args.dino_channels
        )

