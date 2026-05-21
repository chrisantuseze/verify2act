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
) -> None:
    """Checks the feature cache directory, and generates missing features if necessary."""
    root = Path(dataset_dir)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    trans_file_path = root / transitions_file
    if not trans_file_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {trans_file_path}")

    # 1. Collect unique image paths from transitions
    paths = set()
    with open(trans_file_path) as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_t = item.get("image_t", "")
            image_t1 = item.get("image_t1", "")
            if image_t:
                paths.add(image_t)
            if image_t1:
                paths.add(image_t1)
            goal = _resolve_goal_image(item, root)
            if goal:
                paths.add(goal)

    # 2. Check which features are missing
    missing_paths = []
    missing_image_count = 0
    for p in sorted(paths):
        if not (root / p).exists():
            missing_image_count += 1
            continue
        feat_name = p.replace("/", "_") + ".pt"
        feat_path = cache_root / feat_name
        if not feat_path.exists():
            missing_paths.append(p)

    if missing_image_count > 0:
        print(f"⚠️  Warning: {missing_image_count} referenced images do not exist on disk and will be skipped.")

    if not missing_paths:
        print(f"✓ DINOv2 feature cache is complete ({len(paths)} features cached in '{cache_dir}').")
        return

    print(f"Detected {len(missing_paths)} missing features out of {len(paths)} unique images.")
    print("Initializing DINOv2 backbone to precompute features...")

    # 3. Load DINOv2 backbone
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    dino = dino.eval().to(device)

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
            if not img_path.exists():
                continue
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
            feats = dino.forward_features(imgs_tensor)["x_norm_patchtokens"]  # [B, 256, 768]

        # Save to disk as fp16 (saves 50% disk space)
        for p, feat in zip(valid_chunk, feats):
            feat_name = p.replace("/", "_") + ".pt"
            torch.save(feat.half().cpu(), cache_root / feat_name)

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
) -> None:
    """Checks the feature cache directory for CALVIN, and generates missing features if necessary."""
    root = Path(dataset_dir)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    # 1. Collect all episode npz files
    npz_files = list(root.glob("episode_*.npz"))
    
    # 2. Check which features are missing
    missing_files = []
    for f in sorted(npz_files):
        feat_name = f.name.replace(".npz", ".pt")
        feat_path = cache_root / feat_name
        if not feat_path.exists():
            missing_files.append(f)

    if not missing_files:
        print(f"✓ DINOv2 feature cache for CALVIN is complete ({len(npz_files)} features cached in '{cache_dir}').")
        return

    print(f"Detected {len(missing_files)} missing features out of {len(npz_files)} CALVIN episode files.")
    print("Initializing DINOv2 backbone to precompute features...")

    # 3. Load DINOv2 backbone
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
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
            feats = dino.forward_features(imgs_tensor)["x_norm_patchtokens"]  # [B, 256, 768]

        for npz_path, feat in zip(valid_chunk, feats):
            feat_name = npz_path.name.replace(".npz", ".pt")
            torch.save(feat.half().cpu(), cache_root / feat_name)

    print("Feature generation complete. Cleaning up DINOv2 from VRAM...")
    del dino
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM successfully reclaimed.")
