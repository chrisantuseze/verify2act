"""
Verify2Act Critic Model
Multi-headed critic with uncertainty estimation for verifying imagined rollouts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from .critic_config import CriticModelConfig


class SharedEncoder(nn.Module):
    """Shared encoder for processing concatenated inputs."""
    
    def __init__(self, config: CriticModelConfig):
        super().__init__()
        self.config = config
        
        # Calculate input dimension
        input_dim = (
            2 * config.latent_dim +  # z_t, z_{t+1}
            config.action_dim +       # a_t
            config.predicate_embed_dim +  # target predicate
            config.plan_summary_dim   # remaining plan
        )
        
        if config.encoder_type == "mlp":
            layers = []
            in_dim = input_dim
            for hidden_dim in config.encoder_hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    self._get_activation(config.encoder_activation),
                    nn.Dropout(config.encoder_dropout),
                ])
                in_dim = hidden_dim
            self.encoder = nn.Sequential(*layers)
            self.output_dim = config.encoder_hidden_dims[-1]
            
        elif config.encoder_type == "transformer":
            # Simple transformer encoder
            self.input_proj = nn.Linear(input_dim, config.encoder_hidden_dims[0])
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.encoder_hidden_dims[0],
                nhead=8,
                dim_feedforward=config.encoder_hidden_dims[1],
                dropout=config.encoder_dropout,
                activation=config.encoder_activation,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.output_proj = nn.Linear(config.encoder_hidden_dims[0], config.encoder_hidden_dims[-1])
            self.output_dim = config.encoder_hidden_dims[-1]
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        if self.config.encoder_type == "mlp":
            return self.encoder(x)
        else:  # transformer
            x = self.input_proj(x)
            x = x.unsqueeze(1)  # Add sequence dimension
            x = self.encoder(x)
            x = x.squeeze(1)  # Remove sequence dimension
            x = self.output_proj(x)
            return x


class CriticHead(nn.Module):
    """Individual critic head with uncertainty support."""
    
    def __init__(self, input_dim: int, config: CriticModelConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = input_dim
        for hidden_dim in config.head_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                self._get_activation(config.head_activation),
                nn.Dropout(config.head_dropout),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through head."""
        logits = self.network(x)
        return torch.sigmoid(logits)


class CriticModel(nn.Module):
    """
    Multi-headed critic model with uncertainty estimation.
    
    Outputs:
        - p_predicate: Probability of predicate satisfaction
        - p_feas: Probability of action feasibility (optional)
        - p_nonint: Probability of non-interference (optional)
    """
    
    def __init__(self, config: CriticModelConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.encoder = SharedEncoder(config)
        
        # Heads
        self.predicate_head = None
        self.feasibility_head = None
        self.noninterference_head = None
        
        if config.use_predicate_head:
            self.predicate_head = CriticHead(self.encoder.output_dim, config)
        
        if config.use_feasibility_head:
            self.feasibility_head = CriticHead(self.encoder.output_dim, config)
        
        if config.use_noninterference_head:
            self.noninterference_head = CriticHead(self.encoder.output_dim, config)
    
    def forward(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_next: torch.Tensor,
        predicate_embed: torch.Tensor,
        plan_summary: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through critic.
        
        Args:
            z_t: Current latent state [batch, latent_dim]
            a_t: Action [batch, action_dim]
            z_next: Next latent state [batch, latent_dim]
            predicate_embed: Target predicate embedding [batch, predicate_embed_dim]
            plan_summary: Remaining plan summary [batch, plan_summary_dim]
        
        Returns:
            Dictionary with predictions for each active head
        """
        # Concatenate inputs
        x = torch.cat([z_t, a_t, z_next, predicate_embed, plan_summary], dim=-1)
        
        # Shared encoding
        encoded = self.encoder(x)
        
        # Head predictions
        outputs = {}
        
        if self.predicate_head is not None:
            outputs["p_predicate"] = self.predicate_head(encoded).squeeze(-1)
        
        if self.feasibility_head is not None:
            outputs["p_feas"] = self.feasibility_head(encoded).squeeze(-1)
        
        if self.noninterference_head is not None:
            outputs["p_nonint"] = self.noninterference_head(encoded).squeeze(-1)
        
        return outputs


class CriticEnsemble(nn.Module):
    """
    Ensemble of critic models for uncertainty estimation.
    """
    
    def __init__(self, config: CriticModelConfig):
        super().__init__()
        self.config = config
        self.ensemble_size = config.ensemble_size
        
        # Create ensemble members
        self.models = nn.ModuleList([
            CriticModel(config) for _ in range(self.ensemble_size)
        ])
    
    def forward(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_next: torch.Tensor,
        predicate_embed: torch.Tensor,
        plan_summary: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            z_t, a_t, z_next, predicate_embed, plan_summary: Input tensors
            return_uncertainty: If True, compute uncertainty metrics
        
        Returns:
            Dictionary with mean predictions and optionally uncertainty metrics
        """
        # Get predictions from all ensemble members
        all_outputs = []
        for model in self.models:
            outputs = model(z_t, a_t, z_next, predicate_embed, plan_summary)
            all_outputs.append(outputs)
        
        # Compute mean predictions
        result = {}
        for key in all_outputs[0].keys():
            preds = torch.stack([out[key] for out in all_outputs], dim=0)
            result[key] = preds.mean(dim=0)
            
            if return_uncertainty:
                # Epistemic variance
                result[f"{key}_var"] = preds.var(dim=0)
                
                # Binary entropy
                mu = result[key]
                eps = 1e-8
                entropy = -(mu * torch.log(mu + eps) + (1 - mu) * torch.log(1 - mu + eps))
                result[f"{key}_entropy"] = entropy
        
        return result
    
    def forward_mc_dropout(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_next: torch.Tensor,
        predicate_embed: torch.Tensor,
        plan_summary: torch.Tensor,
        n_samples: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using MC dropout for uncertainty estimation.
        Uses only the first model with dropout enabled.
        """
        model = self.models[0]
        model.train()  # Enable dropout
        
        # Sample predictions
        all_outputs = []
        for _ in range(n_samples):
            outputs = model(z_t, a_t, z_next, predicate_embed, plan_summary)
            all_outputs.append(outputs)
        
        # Compute statistics
        result = {}
        for key in all_outputs[0].keys():
            preds = torch.stack([out[key] for out in all_outputs], dim=0)
            result[key] = preds.mean(dim=0)
            result[f"{key}_var"] = preds.var(dim=0)
            
            # Binary entropy
            mu = result[key]
            eps = 1e-8
            entropy = -(mu * torch.log(mu + eps) + (1 - mu) * torch.log(1 - mu + eps))
            result[f"{key}_entropy"] = entropy
        
        return result


def build_critic(config: CriticModelConfig, use_ensemble: bool = True) -> nn.Module:
    """
    Build critic model (ensemble or single).
    
    Args:
        config: Model configuration
        use_ensemble: If True, build ensemble; otherwise single model
    
    Returns:
        Critic model
    """
    if use_ensemble:
        return CriticEnsemble(config)
    else:
        return CriticModel(config)
