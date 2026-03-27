"""
PyTorch Dataset for World Model / Critic Training

Reads transitions.jsonl and (optionally) labels.jsonl.
Returns (image_t, image_t1, goal_image, action_text, label_reachable) tuples.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class WMTransitionDataset(Dataset):
    """
    Loads transitions for world-model training.

    Each sample returns:
        image_t    : [3, H, W] float tensor normalised to [-1, 1]
        image_t1   : [3, H, W]
        action_text: str
    """

    def __init__(
        self,
        dataset_dir: str,
        image_size: int = 512,
        split: str = "train",
        val_frac: float = 0.1,
        seed: int = 42,
        transitions_file: str = "transitions.jsonl",
    ):
        self.root = Path(dataset_dir)
        self.image_size = image_size

        # Load transitions
        jsonl = self.root / transitions_file
        with open(jsonl) as f:
            self.rows = [json.loads(line) for line in f]

        # Deterministic train/val split by episode
        episodes = sorted(set(r["episode_id"] for r in self.rows))
        rng = np.random.RandomState(seed)
        rng.shuffle(episodes)
        n_val = max(1, int(len(episodes) * val_frac))
        val_eps = set(episodes[:n_val])
        if split == "val":
            self.rows = [r for r in self.rows if r["episode_id"] in val_eps]
        else:
            self.rows = [r for r in self.rows if r["episode_id"] not in val_eps]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Dict:
        row = self.rows[idx]
        img_t = self._load_image(row["image_t"])
        img_t1 = self._load_image(row["image_t1"])
        return {
            "image_t": img_t,
            "image_t1": img_t1,
            "action_text": row["action_text"],
        }

    def _load_image(self, relpath: str) -> torch.Tensor:
        path = self.root / relpath
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class CriticDataset(Dataset):
    """
    Loads transitions + labels for critic training.

    Each sample returns:
        image_t1   : [3, H, W] float [-1, 1]
        goal_image : [3, H, W] float [-1, 1]
        label      : int (0 or 1)

    Optionally returns pre-computed latent embeddings instead of images
    (set latents_dir to a folder with .npy files).
    """

    def __init__(
        self,
        dataset_dir: str,
        labels_file: str = "labels.jsonl",
        image_size: int = 512,
        split: str = "train",
        val_frac: float = 0.1,
        seed: int = 42,
        latents_dir: Optional[str] = None,
    ):
        self.root = Path(dataset_dir)
        self.latents_dir = Path(latents_dir) if latents_dir else None
        self.image_size = image_size

        # Load transitions
        with open(self.root / "transitions.jsonl") as f:
            transitions = {
                (r["episode_id"], r["timestep"]): r
                for line in f
                for r in [json.loads(line)]
            }

        # Load labels and merge
        self.rows = []
        with open(self.root / labels_file) as f:
            for line in f:
                lbl = json.loads(line)
                key = (lbl["episode_id"], lbl["timestep"])
                if key in transitions:
                    tr = transitions[key]
                    tr["label_reachable"] = lbl["label_reachable"]
                    # Skip unlabeled or missing goal
                    if tr["label_reachable"] in (0, 1) and tr.get("goal_image"):
                        self.rows.append(tr)

        # Split
        episodes = sorted(set(r["episode_id"] for r in self.rows))
        rng = np.random.RandomState(seed)
        rng.shuffle(episodes)
        n_val = max(1, int(len(episodes) * val_frac))
        val_eps = set(episodes[:n_val])
        if split == "val":
            self.rows = [r for r in self.rows if r["episode_id"] in val_eps]
        else:
            self.rows = [r for r in self.rows if r["episode_id"] not in val_eps]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Dict:
        row = self.rows[idx]
        label = row["label_reachable"]

        if self.latents_dir is not None:
            z_t1 = np.load(
                self.latents_dir
                / f"{row['episode_id']}_{row['timestep']+1}_z.npy"
            )
            z_goal = np.load(
                self.latents_dir
                / f"{row['episode_id']}_goal_z.npy"
            )
            return {
                "z_t1": torch.from_numpy(z_t1),
                "z_goal": torch.from_numpy(z_goal),
                "label": label,
            }

        img_t1 = self._load_image(row["image_t1"])
        goal_img = self._load_image(row["goal_image"])
        return {
            "image_t1": img_t1,
            "goal_image": goal_img,
            "label": label,
        }

    def _load_image(self, relpath: str) -> torch.Tensor:
        path = self.root / relpath
        img = Image.open(path).convert("RGB")
        return self.transform(img)


# ── PRM Critic Dataset ─────────────────────────────────────────────────────────

@dataclass
class CriticRow:
    episode_id: str
    timestep: int
    image_t1: str
    goal_image: str
    label_reachable: int


class PRMCriticDataset(Dataset):
    """
    Dataset for PRM Beta critic training.

    Produces (image_t1, goal_image, label) triples from labeled robosuite
    transitions.  Images are normalised to [-1, 1] for direct input to the
    frozen SD VAE encoder.
    """

    def __init__(self, dataset_dir: str, rows: List[CriticRow], image_size: int = 512):
        self.root = Path(dataset_dir)
        self.rows = rows
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        return {
            "image_t1": self._load_image(row.image_t1),
            "goal_image": self._load_image(row.goal_image),
            "label": torch.tensor(float(row.label_reachable), dtype=torch.float32),
            "episode_id": row.episode_id,
            "timestep": row.timestep,
        }

    def _load_image(self, rel_path: str) -> torch.Tensor:
        img = Image.open(self.root / rel_path).convert("RGB")
        return self.transform(img)


def _resolve_goal_image(trans: Dict, dataset_root: Path, default_goal_image: str = "") -> str:
    goal = (trans.get("goal_image") or "").strip()
    if goal:
        return goal
    episode_goal = Path("episodes") / trans["episode_id"] / "goal.png"
    if (dataset_root / episode_goal).exists():
        return str(episode_goal)
    return default_goal_image


def _load_critic_rows(
    dataset_dir: str,
    labels_file: str,
    default_goal_image: str = "",
) -> List[CriticRow]:
    root = Path(dataset_dir)
    transitions_path = root / "transitions.jsonl"
    labels_path = root / labels_file

    if not transitions_path.exists():
        raise FileNotFoundError(f"Missing transitions file: {transitions_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    transitions: Dict[Tuple[str, int], Dict] = {}
    with open(transitions_path) as f:
        for line in f:
            item = json.loads(line)
            transitions[(item["episode_id"], int(item["timestep"]))] = item

    rows: List[CriticRow] = []
    with open(labels_path) as f:
        for line in f:
            lbl = json.loads(line)
            key = (lbl["episode_id"], int(lbl["timestep"]))
            if key not in transitions:
                continue
            trans = transitions[key]
            label = int(lbl.get("label_reachable", -1))
            goal = _resolve_goal_image(trans, root, default_goal_image)
            if label not in (0, 1) or not goal:
                continue
            rows.append(CriticRow(
                episode_id=trans["episode_id"],
                timestep=int(trans["timestep"]),
                image_t1=trans["image_t1"],
                goal_image=goal,
                label_reachable=label,
            ))

    if not rows:
        raise RuntimeError("No valid labeled rows found for critic training.")
    return rows


def build_train_val_datasets(
    dataset_dir: str,
    labels_file: str = "labels.jsonl",
    default_goal_image: str = "",
    val_frac: float = 0.1,
    seed: int = 42,
    image_size: int = 512,
) -> Tuple[PRMCriticDataset, PRMCriticDataset]:
    """Episode-level train/val split for PRM critic training."""
    rows = _load_critic_rows(dataset_dir, labels_file, default_goal_image)
    episodes = sorted({r.episode_id for r in rows})

    rng = np.random.RandomState(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * val_frac))
    val_eps = set(episodes[:n_val])

    train_rows = [r for r in rows if r.episode_id not in val_eps]
    val_rows   = [r for r in rows if r.episode_id in val_eps]

    return (
        PRMCriticDataset(dataset_dir, train_rows, image_size),
        PRMCriticDataset(dataset_dir, val_rows,   image_size),
    )
