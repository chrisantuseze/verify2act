# Verify2Act Critic - Next Steps Checklist

## ✅ Completed

- [x] Core model architecture (SharedEncoder, CriticHead, CriticEnsemble)
- [x] Uncertainty estimation (ensemble + MC dropout)
- [x] Configuration system with phased rollout support
- [x] Inference engine with reflection logic
- [x] Three-tier failure detection (hard/uncertainty/soft)
- [x] Calibrated thresholds for all three heads
- [x] Training pipeline with early stopping
- [x] Data collection utilities with hard negatives
- [x] Evaluation metrics (accuracy, calibration, ECE/MCE)
- [x] Integration class (VerifiedPlanner)
- [x] Training script with argparse
- [x] Documentation (README, implementation plan, summary)
- [x] Quick start verification script
- [x] Package initialization (__init__.py)
- [x] Requirements file
- [x] Syntax validation

## 🔄 Phase 1: Data Collection & Training (Predicate Head Only)

### 1. Prepare Points2Plans Integration
- [ ] Implement state encoder: `state_dict → z_t` (256D)
  - Extract object poses, gripper state, scene features
  - Use Points2Plans existing encoder
  
- [ ] Implement action encoder: `action_dict → a_t` (64D)
  - Encode primitive type, object ID, target location
  - Use Points2Plans action representation
  
- [ ] Implement predicate embedder: `predicate_string → embedding` (128D)
  - Parse predicates (ON, HOLDING, NEAR, etc.)
  - Use learned embedding or rule-based encoding
  - Align with predicate decoder
  
- [ ] Implement plan summary embedder: `remaining_plan → embedding` (128D)
  - Encode list of future primitives
  - Use bag-of-words or sequential encoding

### 2. Collect Training Data
- [ ] Run successful rollouts in simulation
  - Use existing task demonstrations
  - Collect 1000+ positive samples
  - Save (z_t, a_t, z_next, predicate, label=1)

- [ ] Run failed rollouts
  - Perturb initial state
  - Use incorrect primitives
  - Collect 500+ negative samples
  - Label failure step and type

- [ ] Generate hard negatives
  - Wrong predicate embeddings
  - Noisy state predictions
  - Wrong object selections
  - Target: balance positive/negative 1:1

- [ ] Split dataset: 70% train, 15% val, 15% test

### 3. Train Phase 1 Model
```bash
python train_critic.py \
  --data_path ./data/critic_phase1.pkl \
  --use_predicate_head \
  --ensemble_size 5 \
  --batch_size 256 \
  --lr 3e-4 \
  --num_epochs 100 \
  --checkpoint_dir ./checkpoints/phase1
```

- [ ] Train ensemble of 5 models
- [ ] Monitor validation loss and ECE
- [ ] Apply early stopping
- [ ] Save best checkpoint

### 4. Calibrate Thresholds
- [ ] Sweep μ threshold: [0.2, 0.3, 0.35, 0.4, 0.5]
- [ ] Sweep σ threshold: [0.1, 0.15, 0.2, 0.25]
- [ ] Sweep H threshold: [0.45, 0.5, 0.55, 0.6]
- [ ] Maximize F_1.5 (favoring recall)
- [ ] Target: precision ≥ 0.70, recall ≥ 0.80, ECE ≤ 0.05
- [ ] Update config with calibrated thresholds

### 5. Evaluate Phase 1
- [ ] Test set accuracy, precision, recall, F1
- [ ] Calibration plot (confidence vs accuracy)
- [ ] ECE and MCE metrics
- [ ] Ablation: ensemble vs MC dropout
- [ ] Generate evaluation report

## 🔄 Phase 2: Add Feasibility Head

### 6. Collect Feasibility Data
- [ ] Label existing data with feasibility
  - Check if action is executable
  - Detect collisions, unreachable targets
  
- [ ] Generate infeasible samples
  - Actions to occluded objects
  - Impossible grasps
  - Out-of-reach targets
  
- [ ] Balance dataset

### 7. Train Phase 2 Model
```bash
python train_critic.py \
  --data_path ./data/critic_phase2.pkl \
  --use_predicate_head \
  --use_feasibility_head \
  --checkpoint_dir ./checkpoints/phase2
```

- [ ] Fine-tune or train from scratch
- [ ] Calibrate feasibility thresholds
- [ ] Target: precision ≥ 0.75, recall ≥ 0.85

### 8. Evaluate Phase 2
- [ ] Per-head metrics
- [ ] Combined reflection decisions
- [ ] Compare to Phase 1 baseline

## 🔄 Phase 3: Add Non-Interference Head

### 9. Collect Non-Interference Data
- [ ] Label actions that block future steps
  - Placing objects that occlude others
  - Actions that create instability
  
- [ ] Generate blocking samples
  - Adversarial primitive orderings
  - Suboptimal placements

### 10. Train Phase 3 Model
```bash
python train_critic.py \
  --data_path ./data/critic_phase3.pkl \
  --use_predicate_head \
  --use_feasibility_head \
  --use_noninterference_head \
  --checkpoint_dir ./checkpoints/phase3
```

- [ ] Full three-head training
- [ ] Calibrate all thresholds
- [ ] Target: precision ≥ 0.70, recall ≥ 0.80

### 11. Evaluate Phase 3
- [ ] All three heads separately
- [ ] Combined system performance
- [ ] End-to-end task success rate

## 🔄 Integration & Deployment

### 12. Integrate with Existing System
- [ ] Modify Points2Plans rollout loop
  ```python
  # In dynamics model rollout
  for step in range(T):
      z_next = predict_next_state(z_t, a_t)
      diag = critic.evaluate_step(z_t, a_t, z_next, ...)
      if diag.should_reflect:
          break
  ```

- [ ] Plug into LLM reflection loop
  ```python
  for iteration in range(max_reflections):
      plan = llm.generate_plan(...)
      verified = critic.verify_plan(...)
      if verified:
          break
      reflection = critic.generate_prompt(...)
      llm.add_context(reflection)
  ```

- [ ] Add trajectory diagnostics logging
- [ ] Implement plan selection logic (best of N candidates)

### 13. Test Integration
- [ ] Unit tests for each component
- [ ] Integration test: full planning loop
- [ ] Robustness test: various failure modes
- [ ] Performance test: inference latency

### 14. Run End-to-End Experiments
- [ ] Task suite: stacking, sorting, assembly
- [ ] Baseline: no critic (LLM + dynamics only)
- [ ] Ablations:
  - No uncertainty (single model)
  - No reflection (critic verification only)
  - Different threshold settings
  
- [ ] Metrics:
  - Task success rate
  - Number of reflections per task
  - Planning time
  - Execution success given verified plan

### 15. Analysis & Iteration
- [ ] Failure analysis: when does critic fail?
- [ ] Calibration drift: does performance degrade?
- [ ] Threshold sensitivity analysis
- [ ] Consider conformal prediction for coverage guarantees

## 📊 Expected Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Data Collection | 1-2 weeks | Points2Plans integration |
| Phase 1 Training | 3-5 days | Data collection |
| Phase 1 Evaluation | 2-3 days | Phase 1 training |
| Phase 2 | 1 week | Phase 1 complete |
| Phase 3 | 1 week | Phase 2 complete |
| Integration | 1 week | Phase 3 complete |
| Experiments | 2-3 weeks | Integration complete |
| **Total** | **7-10 weeks** | |

## 🎯 Success Criteria

### Phase 1 (Predicate)
- ✓ Precision ≥ 0.70
- ✓ Recall ≥ 0.80
- ✓ ECE ≤ 0.05

### Phase 2 (+ Feasibility)
- ✓ Feasibility precision ≥ 0.75
- ✓ Feasibility recall ≥ 0.85
- ✓ Combined system improves task success

### Phase 3 (+ Non-Interference)
- ✓ Non-interference precision ≥ 0.70
- ✓ Non-interference recall ≥ 0.80
- ✓ End-to-end task success > baseline

### Integration
- ✓ Inference latency < 50ms per step
- ✓ Task success rate ≥ 80% on test suite
- ✓ Reflection precision ≥ 0.75 (correct failure detection)
- ✓ Reflection recall ≥ 0.85 (catches most failures)

## 📝 Documentation Updates

As you progress:
- [ ] Document final threshold values
- [ ] Record calibration procedure and results
- [ ] Add real examples to README
- [ ] Create troubleshooting guide
- [ ] Write paper section on critic design

## 🐛 Known Limitations & Future Work

### Current Limitations
- Requires manual predicate/plan embedding
- Fixed threshold values (not adaptive)
- No online learning/adaptation
- No multi-task transfer

### Potential Extensions
- [ ] Learned predicate embeddings
- [ ] Conformal prediction for coverage
- [ ] Adaptive thresholds based on task context
- [ ] Long-horizon uncertainty penalty in scoring
- [ ] Set-valued predictions for ambiguous cases
- [ ] Meta-learning across tasks
- [ ] Uncertainty-aware action selection

## 📞 Support

For questions or issues:
1. Check README_CRITIC.md
2. Review IMPLEMENTATION_SUMMARY.md
3. Examine example usage in quickstart_critic.py
4. Refer to CRITIC_IMPLEMENTATION_PLAN.md for design rationale

---

**Status**: Implementation complete, ready for data collection
**Last Updated**: 2026-02-01
