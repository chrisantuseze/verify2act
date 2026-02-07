# Verify2Act Critic Module

## Overview
The critic module implements uncertainty-aware verification of imagined rollouts from the Points2Plans dynamics model. It provides three types of verification:
1. **Predicate satisfaction**: Does the imagined state satisfy the target predicate?
2. **Action feasibility**: Is the action executable from the current state?
3. **Non-interference**: Does the action interfere with future plan execution?

## Key Features
- Multi-headed architecture with shared encoder
- Ensemble-based epistemic uncertainty estimation
- MC Dropout support as alternative
- Calibrated reflection thresholds
- Detailed trajectory diagnostics
- Targeted reflection prompt generation

## Architecture

```
Input: z_t, a_t, z_{t+1}, predicate_embed, plan_summary
  ↓
Shared Encoder (MLP/Transformer)
  ↓
  ├─→ Predicate Head → p_predicate
  ├─→ Feasibility Head → p_feas
  └─→ Non-Interference Head → p_nonint

Ensemble of 5 models provides uncertainty:
  μ = mean(predictions)
  σ² = var(predictions)
  H = entropy(μ)
```

## Files

- `critic_config.py`: Configuration classes and default thresholds
- `critic_model.py`: Model architecture (encoder + heads + ensemble)
- `critic_inference.py`: Inference utilities and reflection logic
- `critic_trainer.py`: Training loop and calibration
- `CRITIC_IMPLEMENTATION_PLAN.md`: Full implementation plan with insights

## Usage

### Training

```python
from critic_config import CriticConfig
from critic_trainer import CriticTrainer

# Initialize config
config = CriticConfig()
config.model.use_predicate_head = True  # Phase 1

# Create trainer
trainer = CriticTrainer(config)

# Train
trainer.train(train_data, val_data, checkpoint_dir="./checkpoints")
```

### Inference

```python
from critic_config import CriticConfig
from critic_model import build_critic
from critic_inference import CriticInference

# Load model
config = CriticConfig()
model = build_critic(config.model)
model.load_state_dict(torch.load("best_model.pt"))

# Create inference engine
inference = CriticInference(model, config)

# Evaluate single step
diag = inference.evaluate_step(
    z_t=z_t,
    a_t=a_t,
    z_next=z_next,
    predicate_embed=pred_embed,
    plan_summary=plan_summary,
    target_predicate="ON(cup, table)",
)

print(f"Should reflect: {diag.should_reflect}")
print(f"Failure reason: {diag.failure_reason}")
```

### Trajectory Evaluation

```python
# Evaluate entire trajectory
trajectory_data = [...]  # List of step dicts
traj_diag = inference.evaluate_trajectory(trajectory_data)

# Generate reflection prompt if needed
if traj_diag.should_reflect:
    prompt = inference.generate_reflection_prompt(
        primitive_plan=["pickplace(cup, table)", "pickplace(tea, cup)"],
        failure_analysis=inference.aggregate_failure_analysis([traj_diag]),
        trajectory_diagnostics=traj_diag,
    )
    print(prompt)
```

## Reflection Thresholds

Default thresholds (can be adjusted via config):

**Predicate Head:**
- Hard fail: μ < 0.35
- Uncertainty fail: σ > 0.15 or H > 0.55
- Soft fail: 0.35 ≤ μ < 0.55 and σ > 0.10

**Feasibility Head:**
- Hard fail: μ < 0.30
- Uncertainty fail: σ > 0.12 or H > 0.50
- Soft fail: 0.30 ≤ μ < 0.55 and σ > 0.08

**Non-Interference Head:**
- Hard fail: μ < 0.40
- Uncertainty fail: σ > 0.10 or H > 0.45
- Soft fail: 0.40 ≤ μ < 0.60 and σ > 0.07

## Phased Implementation

**Phase 1** (Start Here):
```python
config.model.use_predicate_head = True
config.model.use_feasibility_head = False
config.model.use_noninterference_head = False
```

**Phase 2**:
```python
config.model.use_feasibility_head = True
```

**Phase 3**:
```python
config.model.use_noninterference_head = True
```

## Data Format

Training data should be a list of dicts with:
```python
{
    "z_t": np.ndarray,  # [latent_dim]
    "a_t": np.ndarray,  # [action_dim]
    "z_next": np.ndarray,  # [latent_dim]
    "predicate_embed": np.ndarray,  # [predicate_embed_dim]
    "plan_summary": np.ndarray,  # [plan_summary_dim]
    "label_predicate": int,  # 0 or 1
    "label_feas": int,  # 0 or 1 (optional)
    "label_nonint": int,  # 0 or 1 (optional)
}
```

## Integration with Points2Plans

The critic integrates into the imagination loop:

```python
# During rollout
for step in range(T):
    # Dynamics model predicts next state
    z_next = dynamics_model.predict(z_t, a_t)
    
    # Critic evaluates step
    diag = critic.evaluate_step(z_t, a_t, z_next, ...)
    
    # Store diagnostics
    trajectory_diagnostics.append(diag)
    
    # Continue rollout
    z_t = z_next

# After rollout, check if reflection needed
traj_diag = critic.evaluate_trajectory(trajectory_diagnostics)
if traj_diag.should_reflect:
    # Trigger replanning
    reflection_prompt = critic.generate_reflection_prompt(...)
```

## Calibration

After training, calibrate thresholds on validation set:

```python
# Sweep thresholds
best_thresholds = calibrate_thresholds(
    model=model,
    val_data=val_data,
    targets=config.calibration_targets,
)

# Update config
config.thresholds = best_thresholds
```

Target metrics:
- Predicate: precision ≥ 0.70, recall ≥ 0.80, ECE ≤ 0.05
- Feasibility: precision ≥ 0.75, recall ≥ 0.85
- Non-interference: precision ≥ 0.70, recall ≥ 0.80

## Citation

If you use this critic module, please cite the Verify2Act project and the papers that inspired the design:
- PETS (Chua et al., 2018) - Ensemble uncertainty for MBRL
- MBPO (Janner et al., 2019) - Trust region for model-based planning
- RWM-U (Li et al., 2024) - Uncertainty-aware world models for robotics
