"""
Verify2Act Critic Training
Training utilities, data loaders, and calibration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm

from .critic_config import CriticConfig
from .critic_model import CriticEnsemble, build_critic


class CriticDataset(Dataset):
    """Dataset for critic training."""
    
    def __init__(
        self,
        data: List[Dict],
        augment: bool = False,
    ):
        """
        Args:
            data: List of dicts with keys:
                - z_t, a_t, z_next: numpy arrays
                - predicate_embed, plan_summary: numpy arrays
                - label_predicate, label_feas, label_nonint: binary labels (0/1)
        """
        self.data = data
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Convert to tensors
        z_t = torch.from_numpy(sample["z_t"]).float()
        a_t = torch.from_numpy(sample["a_t"]).float()
        z_next = torch.from_numpy(sample["z_next"]).float()
        predicate_embed = torch.from_numpy(sample["predicate_embed"]).float()
        plan_summary = torch.from_numpy(sample["plan_summary"]).float()
        
        # Labels
        labels = {}
        if "label_predicate" in sample:
            labels["predicate"] = torch.tensor(sample["label_predicate"], dtype=torch.float32)
        if "label_feas" in sample:
            labels["feas"] = torch.tensor(sample["label_feas"], dtype=torch.float32)
        if "label_nonint" in sample:
            labels["nonint"] = torch.tensor(sample["label_nonint"], dtype=torch.float32)
        
        return {
            "z_t": z_t,
            "a_t": a_t,
            "z_next": z_next,
            "predicate_embed": predicate_embed,
            "plan_summary": plan_summary,
            "labels": labels,
        }


class CriticTrainer:
    """Handles critic model training and calibration."""
    
    def __init__(
        self,
        config: CriticConfig,
        model: Optional[nn.Module] = None,
    ):
        self.config = config
        self.device = config.device
        
        # Build model if not provided
        if model is None:
            self.model = build_critic(config.model, use_ensemble=True)
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Scheduler
        self.scheduler = self._build_scheduler()
        
        # Loss function (BCE)
        self.criterion = nn.BCELoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0
    
    def _build_optimizer(self):
        """Build optimizer."""
        if self.config.training.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=self.config.training.betas,
                eps=self.config.training.eps,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=self.config.training.betas,
                eps=self.config.training.eps,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config.training.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
            )
        elif self.config.training.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.lr_decay_steps,
                gamma=self.config.training.lr_decay_gamma,
            )
        else:
            return None
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-head loss.
        
        Returns:
            (total_loss, loss_dict)
        """
        losses = {}
        total_loss = 0.0
        
        if "p_predicate" in predictions and "predicate" in labels:
            loss_pred = self.criterion(predictions["p_predicate"], labels["predicate"])
            losses["predicate"] = loss_pred.item()
            total_loss += self.config.training.loss_weight_predicate * loss_pred
        
        if "p_feas" in predictions and "feas" in labels:
            loss_feas = self.criterion(predictions["p_feas"], labels["feas"])
            losses["feasibility"] = loss_feas.item()
            total_loss += self.config.training.loss_weight_feasibility * loss_feas
        
        if "p_nonint" in predictions and "nonint" in labels:
            loss_nonint = self.criterion(predictions["p_nonint"], labels["nonint"])
            losses["noninterference"] = loss_nonint.item()
            total_loss += self.config.training.loss_weight_noninterference * loss_nonint
        
        losses["total"] = total_loss.item()
        
        return total_loss, losses
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move to device
            z_t = batch["z_t"].to(self.device)
            a_t = batch["a_t"].to(self.device)
            z_next = batch["z_next"].to(self.device)
            predicate_embed = batch["predicate_embed"].to(self.device)
            plan_summary = batch["plan_summary"].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch["labels"].items()}
            
            # Forward pass through ensemble
            if isinstance(self.model, CriticEnsemble):
                # Train all ensemble members
                total_loss = 0.0
                for model in self.model.models:
                    predictions = model(z_t, a_t, z_next, predicate_embed, plan_summary)
                    loss, _ = self.compute_loss(predictions, labels)
                    total_loss += loss
                total_loss /= len(self.model.models)
            else:
                predictions = self.model(z_t, a_t, z_next, predicate_embed, plan_summary)
                total_loss, _ = self.compute_loss(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(total_loss.item())
        
        return {"loss": np.mean(epoch_losses)}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        epoch_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                z_t = batch["z_t"].to(self.device)
                a_t = batch["a_t"].to(self.device)
                z_next = batch["z_next"].to(self.device)
                predicate_embed = batch["predicate_embed"].to(self.device)
                plan_summary = batch["plan_summary"].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch["labels"].items()}
                
                # Forward pass (use ensemble mean)
                if isinstance(self.model, CriticEnsemble):
                    predictions = self.model(
                        z_t, a_t, z_next, predicate_embed, plan_summary,
                        return_uncertainty=False
                    )
                else:
                    predictions = self.model(z_t, a_t, z_next, predicate_embed, plan_summary)
                
                _, losses = self.compute_loss(predictions, labels)
                epoch_losses.append(losses["total"])
        
        return {"loss": np.mean(epoch_losses)}
    
    def train(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Full training loop.
        
        Args:
            train_data: Training data
            val_data: Validation data
            checkpoint_dir: Directory to save checkpoints
        """
        # Create data loaders
        train_dataset = CriticDataset(train_data)
        val_dataset = CriticDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # Training loop
        for epoch in range(self.config.training.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics["loss"])
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics["loss"])
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Early stopping
            if val_metrics["loss"] < self.best_val_loss - self.config.training.early_stopping_delta:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                
                # Save best model
                if checkpoint_dir is not None:
                    self.save_checkpoint(checkpoint_dir, "best_model.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Periodic checkpoint
            if checkpoint_dir is not None and (epoch + 1) % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    
    def save_checkpoint(self, checkpoint_dir: str, filename: str):
        """Save model checkpoint."""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(checkpoint_dir) / filename
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        print(f"Loaded checkpoint: {checkpoint_path}")


def compute_calibration_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE, MCE).
    
    Args:
        predictions: Predicted probabilities [N]
        labels: Ground truth labels [N]
        n_bins: Number of bins for calibration
    
    Returns:
        Dictionary with ECE and MCE
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_confidence = predictions[mask].mean()
            bin_accuracy = labels[mask].mean()
            bin_weight = mask.sum() / len(predictions)
            
            bin_error = np.abs(bin_confidence - bin_accuracy)
            ece += bin_weight * bin_error
            mce = max(mce, bin_error)
    
    return {"ece": ece, "mce": mce}
