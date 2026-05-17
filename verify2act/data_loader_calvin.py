"""Dataset loader for CALVIN benchmark, parsing continuous trajectories into sparse transitions."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CalvinTransitionDataset(Dataset):
    """Transitions from the CALVIN benchmark for Latent World Model training.

    Parses `auto_lang_ann.npy` to extract start and end indices of sub-goals.
    Each sample returns:
      image_t:  [3, H, W] in [-1, 1] (state at the start of the instruction)
      image_t1: [3, H, W] in [-1, 1] (state at the end of the instruction)
      action_text: str (the language instruction)
    """

    def __init__(
        self,
        dataset_dir: str,
        image_size: int = 224,
        history_len: int = 3,
        split: str = "train",
        val_frac: float = 0.1,
        seed: int = 42,
    ):
        self.root = Path(dataset_dir)
        self.history_len = history_len
        
        # Load language annotations
        lang_ann_path = self.root / "lang_annotations" / "auto_lang_ann.npy"
        if not lang_ann_path.exists():
            lang_ann_path = self.root / "auto_lang_ann.npy"
        
        if not lang_ann_path.exists():
            raise FileNotFoundError(f"Could not find auto_lang_ann.npy in {self.root}")

        annotations = np.load(lang_ann_path, allow_pickle=True).item()
        indices = annotations["info"]["indx"]
        texts = annotations["language"]["ann"]

        # Each element is (start_idx, end_idx, text)
        self.transitions = list(zip(indices, texts))
        
        rng = np.random.RandomState(seed)
        rng.shuffle(self.transitions)
        
        n_val = max(1, int(len(self.transitions) * val_frac))
        if split == "val":
            self.transitions = self.transitions[:n_val]
        else:
            self.transitions = self.transitions[n_val:]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # DINOv2 expected normalization
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        (start_idx, end_idx), action_text = self.transitions[idx]

        # Load history of images: [I_{start_idx - H + 1}, ..., I_{start_idx}]
        # Simultaneously track which slots are real vs. padded (clamped).
        history_imgs = []
        history_mask = []
        for j in range(self.history_len - 1, -1, -1):
            frame_idx = start_idx - j
            is_real = frame_idx >= 0
            history_mask.append(is_real)
            history_imgs.append(self._load_image(max(0, frame_idx)))

        history_tensor = torch.stack(history_imgs)                      # [H, 3, H, W]
        history_mask   = torch.tensor(history_mask, dtype=torch.bool)  # [H] True=valid
        image_t1 = self._load_image(end_idx)

        return {
            "history_imgs": history_tensor,
            "history_mask": history_mask,
            "image_t1":     image_t1,
            "action_text":  action_text,
        }

    def _load_image(self, episode_idx: int) -> torch.Tensor:
        ep_path = self.root / f"episode_{episode_idx:07d}.npz"
        if not ep_path.exists():
            # sometimes the dataset might be split or missing parts
            raise FileNotFoundError(f"Missing episode file: {ep_path}")
            
        data = np.load(ep_path, allow_pickle=True)
        img_np = data["rgb_static"]
        
        # Convert to PIL Image
        img = Image.fromarray(img_np).convert("RGB")
        return self.transform(img)

def build_calvin_datasets(
    dataset_dir: str,
    val_frac: float = 0.1,
    seed: int = 42,
    image_size: int = 224,
    history_len: int = 3,
) -> Tuple[CalvinTransitionDataset, CalvinTransitionDataset]:
    """Build train and val datasets for CALVIN."""
    train_ds = CalvinTransitionDataset(
        dataset_dir=dataset_dir,
        image_size=image_size,
        history_len=history_len,
        split="train",
        val_frac=val_frac,
        seed=seed,
    )
    val_ds = CalvinTransitionDataset(
        dataset_dir=dataset_dir,
        image_size=image_size,
        history_len=history_len,
        split="val",
        val_frac=val_frac,
        seed=seed,
    )
    return train_ds, val_ds
