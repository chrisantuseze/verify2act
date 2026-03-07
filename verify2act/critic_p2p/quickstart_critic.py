#!/usr/bin/env python3
"""
Quick start guide for using the Verify2Act critic.
Run this to see example usage and verify installation.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("VERIFY2ACT CRITIC - QUICK START")
print("="*80)

# Check imports
print("\n1. Checking imports...")
try:
    from critic.critic_config import CriticConfig
    from critic.critic_model import build_critic, CriticEnsemble
    from critic.critic_inference import CriticInference
    from critic.critic_trainer import CriticTrainer
    from critic.critic_data_collector import CriticDataCollector
    from critic.critic_evaluator import CriticEvaluator
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    exit(1)

# Check PyTorch
print("\n2. Checking PyTorch...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Create config
print("\n3. Creating configuration...")
config = CriticConfig()
config.device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {config.device}")
print(f"   Ensemble size: {config.model.ensemble_size}")
print(f"   Active heads:")
print(f"     - Predicate: {config.model.use_predicate_head}")
print(f"     - Feasibility: {config.model.use_feasibility_head}")
print(f"     - Non-interference: {config.model.use_noninterference_head}")

# Build model
print("\n4. Building critic model...")
model = build_critic(config.model, use_ensemble=True)
model.to(config.device)
num_params = sum(p.numel() for p in model.parameters())
print(f"   ✓ Model created")
print(f"   Total parameters: {num_params:,}")

# Test forward pass
print("\n5. Testing forward pass...")
batch_size = 4
z_t = torch.randn(batch_size, config.model.latent_dim).to(config.device)
a_t = torch.randn(batch_size, config.model.action_dim).to(config.device)
z_next = torch.randn(batch_size, config.model.latent_dim).to(config.device)
pred_embed = torch.randn(batch_size, config.model.predicate_embed_dim).to(config.device)
plan_sum = torch.randn(batch_size, config.model.plan_summary_dim).to(config.device)

with torch.no_grad():
    outputs = model(z_t, a_t, z_next, pred_embed, plan_sum, return_uncertainty=True)

print(f"   ✓ Forward pass successful")
print(f"   Outputs:")
for key, val in outputs.items():
    print(f"     {key}: {val.shape}")

# Test inference
print("\n6. Testing inference engine...")
inference = CriticInference(model, config, config.device)

# Simulate a single step evaluation
diag = inference.evaluate_step(
    z_t=z_t[0:1],
    a_t=a_t[0:1],
    z_next=z_next[0:1],
    predicate_embed=pred_embed[0:1],
    plan_summary=plan_sum[0:1],
    target_predicate="ON(cup, table)",
)

print(f"   ✓ Inference successful")
print(f"   Predictions:")
if diag.p_predicate is not None:
    print(f"     p_predicate: {diag.p_predicate:.4f}")
    if diag.predicate_var is not None:
        print(f"     predicate_var: {diag.predicate_var:.4f}")
        print(f"     predicate_entropy: {diag.predicate_entropy:.4f}")
print(f"   Should reflect: {diag.should_reflect}")
print(f"   Failure reason: {diag.failure_reason.value}")

# Test data collector
print("\n7. Testing data collector...")
collector = CriticDataCollector(
    latent_dim=config.model.latent_dim,
    action_dim=config.model.action_dim,
    predicate_embed_dim=config.model.predicate_embed_dim,
    plan_summary_dim=config.model.plan_summary_dim,
)

# Add dummy data
dummy_traj = [
    {
        "z_t": np.random.randn(config.model.latent_dim).astype(np.float32),
        "a_t": np.random.randn(config.model.action_dim).astype(np.float32),
        "z_next": np.random.randn(config.model.latent_dim).astype(np.float32),
    }
    for _ in range(5)
]
dummy_preds = [np.random.randn(config.model.predicate_embed_dim).astype(np.float32) for _ in range(5)]
dummy_sums = [np.random.randn(config.model.plan_summary_dim).astype(np.float32) for _ in range(5)]

collector.add_successful_trajectory(dummy_traj, dummy_preds, dummy_sums)
collector.add_failed_trajectory(dummy_traj, dummy_preds, dummy_sums, failure_step=2)

stats = collector.get_statistics()
print(f"   ✓ Data collector working")
print(f"   Samples collected: {stats['total']}")

print("\n" + "="*80)
print("QUICK START COMPLETE")
print("="*80)
print("\nNext steps:")
print("1. Collect training data using critic_data_collector.py")
print("2. Train the model using train_critic.py")
print("3. Integrate with your planner using verified_planner.py")
print("\nFor detailed usage, see README_CRITIC.md")
print("="*80)
