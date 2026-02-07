# Verify2Act

Verify2Act is a framework for uncertainty-aware planning in robotic manipulation that combines LLM-based planning with dynamics models and critic-based verification.

## Structure

```
verify2act/
├── __init__.py                   # Main package
├── RESEARCH_PLAN.md              # Overall research plan
├── README.md                     # This file
│
└── critic/                       # Critic module
    ├── __init__.py               # Module exports
    │
    ├── Core Implementation
    │   ├── critic_config.py      # Configuration classes
    │   ├── critic_model.py       # Model architecture
    │   ├── critic_inference.py   # Inference & reflection
    │   ├── critic_trainer.py     # Training pipeline
    │   ├── critic_data_collector.py  # Data collection
    │   ├── critic_evaluator.py   # Evaluation metrics
    │   └── verified_planner.py   # System integration
    │
    ├── Scripts
    │   ├── train_critic.py       # Training script
    │   └── quickstart_critic.py  # Quick start guide
    │
    └── Documentation
        ├── README_CRITIC.md      # Critic usage guide
        ├── CRITIC_IMPLEMENTATION_PLAN.md  # Implementation plan
        ├── IMPLEMENTATION_SUMMARY.md      # Summary
        ├── MODULE_STRUCTURE.md            # Architecture
        ├── NEXT_STEPS.md                  # Deployment checklist
        ├── IMPLEMENTATION_COMPLETE.txt    # Visual summary
        └── requirements_critic.txt        # Dependencies
```

## Critic Module

The critic module provides uncertainty-aware verification of imagined rollouts from dynamics models. It includes:

- **Multi-headed architecture**: Predicate satisfaction, feasibility, non-interference
- **Ensemble uncertainty**: Epistemic uncertainty estimation with deep ensembles
- **Calibrated thresholds**: Hard/uncertainty/soft failure detection
- **Reflection prompts**: Targeted feedback for LLM replanning

### Quick Start

```python
# Import from the critic module
from verify2act.critic import CriticConfig, build_critic, CriticInference

# Initialize
config = CriticConfig()
model = build_critic(config.model)
inference = CriticInference(model, config)

# Evaluate trajectory
traj_diag = inference.evaluate_trajectory(trajectory_data)

if traj_diag.should_reflect:
    prompt = inference.generate_reflection_prompt(...)
```

### Training

```bash
cd critic
python train_critic.py \
    --data_path ./data/critic_phase1.pkl \
    --use_predicate_head \
    --ensemble_size 5 \
    --num_epochs 100 \
    --checkpoint_dir ./checkpoints
```

### Testing Installation

```bash
cd critic
python quickstart_critic.py
```

## Documentation

- **Critic Module**: See [critic/README_CRITIC.md](critic/README_CRITIC.md)
- **Implementation Plan**: See [critic/CRITIC_IMPLEMENTATION_PLAN.md](critic/CRITIC_IMPLEMENTATION_PLAN.md)
- **Next Steps**: See [critic/NEXT_STEPS.md](critic/NEXT_STEPS.md)
- **Research Plan**: See [RESEARCH_PLAN.md](RESEARCH_PLAN.md)

## Requirements

```bash
# Install critic dependencies
pip install -r critic/requirements_critic.txt
```

## Status

✅ **Critic Module**: Complete implementation, ready for data collection and training
- 8 core modules (~2,700 lines)
- 2 executable scripts
- 7 documentation files
- Full test suite

## Citation

If you use this framework, please cite:
- Verify2Act Project
- PETS (Chua et al., 2018)
- MBPO (Janner et al., 2019)
- RWM-U (Li et al., 2024)