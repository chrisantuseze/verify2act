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
            return torch.zeros((3, 224, 224))
            
        try:
            data = np.load(ep_path, allow_pickle=True)
            img_np = data["rgb_static"]
            if isinstance(img_np, bytes):
                return torch.zeros((3, 224, 224))
            
            # Convert to PIL Image
            img = Image.fromarray(img_np).convert("RGB")
            return self.transform(img)
        except Exception:
            return torch.zeros((3, 224, 224))

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

class CalvinContrastivePairDataset(Dataset):
    """Triplet dataset for DINOv2 dual-head contrastive training on CALVIN.
    
    Mode 0 (goal proximity):
      anchor   = late frame (near end_idx)
      positive = goal image (end_idx)
      negative = early frame (near start_idx)

    Mode 1 (temporal consistency):
      anchor   = I_t
      positive = I_{t+1} (same instruction)
      negative = I_{t+1} from a different instruction/episode
    """
    def __init__(
        self,
        dataset_dir: str,
        transitions: List[Tuple[Tuple[int, int], str]],
        all_transitions: List[Tuple[Tuple[int, int], str]],
        image_size: int = 224,
        mode0_prob: float = 0.5,
        seed: int = 42,
        cached_dino_dir: str = None,
    ):
        self.root = Path(dataset_dir)
        self.mode0_prob = mode0_prob
        self.rng = np.random.RandomState(seed)
        self.cached_dino_dir = Path(cached_dino_dir) if cached_dino_dir is not None else None

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        self.transitions = transitions
        self.all_transitions = all_transitions

    def __len__(self):
        return len(self.transitions) * 2

    def __getitem__(self, idx):
        if self.rng.random() < self.mode0_prob:
            return self._sample_mode0()
        return self._sample_mode1()

    def _sample_mode0(self):
        idx = self.rng.randint(0, len(self.transitions))
        (start_idx, end_idx), action_text = self.transitions[idx]
        
        n_frames = end_idx - start_idx + 1
        late_start = max(start_idx, start_idx + int(n_frames * 0.80))
        early_end = max(start_idx, start_idx + int(n_frames * 0.20))
        
        anchor_idx = self.rng.randint(late_start, end_idx + 1) if late_start <= end_idx else end_idx
        neg_idx = self.rng.randint(start_idx, early_end + 1) if start_idx <= early_end else start_idx
        
        return {
            "anchor": self._load_image(anchor_idx),
            "positive": self._load_image(end_idx),
            "negative": self._load_image(neg_idx),
            "mode": torch.tensor(0, dtype=torch.long),
            "lang_goal": action_text,
            "has_lang_goal": torch.tensor(True, dtype=torch.bool),
        }

    def _sample_mode1(self):
        idx = self.rng.randint(0, len(self.transitions))
        (start_idx, end_idx), action_text = self.transitions[idx]
        
        if start_idx == end_idx:
            t = start_idx
            t1 = start_idx
        else:
            t = self.rng.randint(start_idx, end_idx)
            t1 = t + 1
            
        neg_idx = idx
        for _ in range(20):
            cand_idx = self.rng.randint(0, len(self.all_transitions))
            if cand_idx != idx:
                neg_idx = cand_idx
                break
                
        (neg_start, neg_end), _ = self.all_transitions[neg_idx]
        neg_t1 = self.rng.randint(neg_start, neg_end + 1) if neg_start <= neg_end else neg_start
        
        return {
            "anchor": self._load_image(t),
            "positive": self._load_image(t1),
            "negative": self._load_image(neg_t1),
            "mode": torch.tensor(1, dtype=torch.long),
            "lang_goal": action_text,
            "has_lang_goal": torch.tensor(True, dtype=torch.bool),
        }

    def _load_image(self, episode_idx: int) -> torch.Tensor:
        if self.cached_dino_dir is not None:
            feat_name = f"episode_{episode_idx:07d}.pt"
            feat_path = self.cached_dino_dir / feat_name
            if feat_path.exists():
                return torch.load(feat_path, map_location="cpu").float()
            return torch.zeros((256, 768))
                
        ep_path = self.root / f"episode_{episode_idx:07d}.npz"
        if not ep_path.exists():
            return torch.zeros((3, 224, 224))
            
        try:
            data = np.load(ep_path, allow_pickle=True)
            img_np = data["rgb_static"]
            if isinstance(img_np, bytes):
                return torch.zeros((3, 224, 224))
            
            img = Image.fromarray(img_np).convert("RGB")
            return self.transform(img)
        except Exception:
            return torch.zeros((3, 224, 224))

def build_calvin_contrastive_datasets(
    dataset_dir: str,
    val_frac: float = 0.1,
    seed: int = 42,
    image_size: int = 224,
    mode0_prob: float = 0.5,
    cached_dino_dir: str = None,
) -> Tuple[CalvinContrastivePairDataset, CalvinContrastivePairDataset]:
    root = Path(dataset_dir)
    lang_ann_path = root / "lang_annotations" / "auto_lang_ann.npy"
    if not lang_ann_path.exists():
        lang_ann_path = root / "auto_lang_ann.npy"
        
    if not lang_ann_path.exists():
        raise FileNotFoundError(f"Could not find auto_lang_ann.npy in {root}")

    annotations = np.load(lang_ann_path, allow_pickle=True).item()
    indices = annotations["info"]["indx"]
    texts = annotations["language"]["ann"]

    transitions = list(zip(indices, texts))
    
    rng = np.random.RandomState(seed)
    rng.shuffle(transitions)
    
    n_val = max(1, int(len(transitions) * val_frac))
    val_transitions = transitions[:n_val]
    train_transitions = transitions[n_val:]
    
    train_ds = CalvinContrastivePairDataset(
        dataset_dir=dataset_dir,
        transitions=train_transitions,
        all_transitions=transitions,
        image_size=image_size,
        mode0_prob=mode0_prob,
        seed=seed,
        cached_dino_dir=cached_dino_dir,
    )
    val_ds = CalvinContrastivePairDataset(
        dataset_dir=dataset_dir,
        transitions=val_transitions,
        all_transitions=transitions,
        image_size=image_size,
        mode0_prob=mode0_prob,
        seed=seed + 1,
        cached_dino_dir=cached_dino_dir,
    )
    return train_ds, val_ds

