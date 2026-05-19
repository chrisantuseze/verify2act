"""Datasets for world-model and DINO contrastive critic training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class WMTransitionDataset(Dataset):
    """Transitions for world-model training.

    Each sample returns:
      image_t:  [3, H, W] in [-1, 1]
      image_t1: [3, H, W] in [-1, 1]
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

        with open(self.root / transitions_file) as f:
            rows = [json.loads(line) for line in f]

        episodes = sorted({r["episode_id"] for r in rows})
        rng = np.random.RandomState(seed)
        rng.shuffle(episodes)
        n_val = max(1, int(len(episodes) * val_frac))
        val_eps = set(episodes[:n_val])

        if split == "val":
            self.rows = [r for r in rows if r["episode_id"] in val_eps]
        else:
            self.rows = [r for r in rows if r["episode_id"] not in val_eps]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.rows)

    def sample_weights(self) -> torch.Tensor:
        """Per-sample weights that balance the three visual sub-distributions:
          - move_to_nut, t=0  (home start, 54% of data)
          - move_to_nut, t>0  (mid-episode start, 22%)
          - move_to_peg       (pre-insertion, 24%)
        """
        buckets = {
            "home":  [i for i, r in enumerate(self.rows)
                      if r.get("policy_stage_t") == "move_to_nut" and r["source_timestep_t"] == 0],
            "mid":   [i for i, r in enumerate(self.rows)
                      if r.get("policy_stage_t") == "move_to_nut" and r["source_timestep_t"] > 0],
            "peg":   [i for i, r in enumerate(self.rows)
                      if r.get("policy_stage_t") != "move_to_nut"],
        }
        n = len(self.rows)
        num_buckets = sum(1 for b in buckets.values() if b)
        weights = torch.zeros(n, dtype=torch.double)
        for indices in buckets.values():
            if indices:
                w = n / (num_buckets * len(indices))
                for i in indices:
                    weights[i] = w
        return weights

    def __getitem__(self, idx):
        row = self.rows[idx]
        action_text = row["action_text"]
        if "action_params" in row and "cartesian_target" in row["action_params"]:
            ct = row["action_params"]["cartesian_target"]
            action_text += f" at loc {ct[0]:.2f} {ct[1]:.2f} {ct[2]:.2f}"
            
        return {
            "image_t": self._load_image(row["image_t"]),
            "image_t1": self._load_image(row["image_t1"]),
            "action_text": action_text,
        }

    def _load_image(self, relpath: str) -> torch.Tensor:
        img = Image.open(self.root / relpath).convert("RGB")
        return self.transform(img)


@dataclass
class ContrastiveRow:
    """Transition row used by the contrastive critic dataset."""

    episode_id: str
    timestep: int
    image_t: str
    image_t1: str
    goal_image: str
    episode_success: bool


class ContrastivePairDataset(Dataset):
    """Triplet dataset for DINOv2 dual-head contrastive training.

    Mode 0 (goal proximity):
      anchor   = late frame from successful episode
      positive = goal image of same episode
      negative = early frame from any episode

    Mode 1 (temporal consistency):
      anchor   = I_t
      positive = I_{t+1} (same row)
      negative = I_{t+1} from different episode
    """

    def __init__(
        self,
        dataset_dir: str,
        rows: List[ContrastiveRow],
        all_rows: List[ContrastiveRow],
        image_size: int = 224,
        mode0_prob: float = 0.5,
        seed: int = 42,
    ):
        self.root = Path(dataset_dir)
        self.mode0_prob = mode0_prob
        self.rng = np.random.RandomState(seed)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        ep_map: Dict[str, List[ContrastiveRow]] = {}
        for row in rows:
            ep_map.setdefault(row.episode_id, []).append(row)
        for ep in ep_map:
            ep_map[ep].sort(key=lambda r: r.timestep)

        self._positive_anchors: List[ContrastiveRow] = []
        self._negative_anchors: List[ContrastiveRow] = []

        for ep_rows in ep_map.values():
            n = len(ep_rows)
            late_start = max(0, int(n * 0.80))
            early_end = max(1, int(n * 0.20))
            late_rows = ep_rows[late_start:]
            early_rows = ep_rows[:early_end]

            ep_success = any(r.episode_success for r in late_rows)
            if ep_success and late_rows:
                self._positive_anchors.extend(late_rows)
            if early_rows:
                self._negative_anchors.extend(early_rows)

        self._tc_rows: List[ContrastiveRow] = list(rows)
        self._cross_rows: List[ContrastiveRow] = list(all_rows)

        if not self._positive_anchors:
            raise RuntimeError(
                "No positive anchors found. Check that episode_success=True exists in late episode rows of the transitions file."
            )
        if not self._negative_anchors:
            raise RuntimeError("No early-frame negatives found.")
        if not self._tc_rows:
            raise RuntimeError("No temporal-consistency rows found.")

    def __len__(self) -> int:
        return max(len(self._positive_anchors), len(self._tc_rows)) * 2

    def __getitem__(self, idx: int):
        if self.rng.random() < self.mode0_prob:
            return self._sample_mode0()
        return self._sample_mode1()

    def _sample_mode0(self):
        anchor_row = self._positive_anchors[self.rng.randint(0, len(self._positive_anchors))]
        neg_row = self._negative_anchors[self.rng.randint(0, len(self._negative_anchors))]
        return {
            "anchor": self._load(anchor_row.image_t1),
            "positive": self._load(anchor_row.goal_image),
            "negative": self._load(neg_row.image_t1),
            "mode": torch.tensor(0, dtype=torch.long),
        }

    def _sample_mode1(self):
        row = self._tc_rows[self.rng.randint(0, len(self._tc_rows))]

        neg_row = row
        for _ in range(20):
            cand = self._cross_rows[self.rng.randint(0, len(self._cross_rows))]
            if cand.episode_id != row.episode_id:
                neg_row = cand
                break

        return {
            "anchor": self._load(row.image_t),
            "positive": self._load(row.image_t1),
            "negative": self._load(neg_row.image_t1),
            "mode": torch.tensor(1, dtype=torch.long),
        }

    def _load(self, rel_path: str) -> torch.Tensor:
        img = Image.open(self.root / rel_path).convert("RGB")
        return self.transform(img)


def _resolve_goal_image(trans: Dict, dataset_root: Path) -> str:
    goal = (trans.get("goal_image") or "").strip()
    if goal and (dataset_root / goal).exists():
        return goal
    fallback = Path("episodes") / trans["episode_id"] / "goal.png"
    if (dataset_root / fallback).exists():
        return str(fallback)

    print(f"Warning: Goal image not found for episode {trans['episode_id']}. Checked '{goal}' and '{fallback}'. Skipping this transition.")
    return ""


def build_contrastive_datasets(
    dataset_dir: str,
    transitions_file: str = "transitions.jsonl",
    val_frac: float = 0.1,
    seed: int = 42,
    image_size: int = 224,
    mode0_prob: float = 0.5,
) -> Tuple[ContrastivePairDataset, ContrastivePairDataset]:
    """Build episode-level train/val datasets for contrastive critic training."""
    root = Path(dataset_dir)

    trans_map: Dict[Tuple[str, int], Dict] = {}
    with open(root / transitions_file) as f:
        for line in f:
            item = json.loads(line)
            trans_map[(item["episode_id"], int(item["timestep"]))] = item

    all_rows: List[ContrastiveRow] = []
    for (ep, ts), trans in trans_map.items():
        image_t = trans.get("image_t", "")
        image_t1 = trans.get("image_t1", "")
        if not image_t or not image_t1:
            continue
        goal = _resolve_goal_image(trans, root)
        if not goal:
            continue

        all_rows.append(
            ContrastiveRow(
                episode_id=ep,
                timestep=ts,
                image_t=image_t,
                image_t1=image_t1,
                goal_image=goal,
                episode_success=bool(trans.get("episode_success", False)),
            )
        )

    if not all_rows:
        raise RuntimeError(
            f"No valid rows found in {transitions_file}. Ensure image_t, image_t1, and goal_image exist."
        )

    all_rows.sort(key=lambda r: (r.episode_id, r.timestep))

    episodes = sorted({r.episode_id for r in all_rows})
    rng = np.random.RandomState(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * val_frac))
    val_eps = set(episodes[:n_val])

    train_rows = [r for r in all_rows if r.episode_id not in val_eps]
    val_rows = [r for r in all_rows if r.episode_id in val_eps]

    train_ds = ContrastivePairDataset(
        dataset_dir=dataset_dir,
        rows=train_rows,
        all_rows=all_rows,
        image_size=image_size,
        mode0_prob=mode0_prob,
        seed=seed,
    )
    val_ds = ContrastivePairDataset(
        dataset_dir=dataset_dir,
        rows=val_rows,
        all_rows=all_rows,
        image_size=image_size,
        mode0_prob=mode0_prob,
        seed=seed + 1,
    )
    return train_ds, val_ds
