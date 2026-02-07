# Verify2Act Critic: Implementation Plan

## 0) Scope and Objective
Design and implement a critic that verifies imagined rollouts from the Points2Plans dynamics model, flags failures with calibrated uncertainty, and triggers targeted replanning. The critic must (1) detect predicate satisfaction failures, (2) detect infeasible actions, and (3) detect non-interference violations, while providing uncertainty-aware reflection signals.

## 1) Key Insights Incorporated
### From project plan (Verify2Act)
- Multi-headed critic (predicate, feasibility, non-interference), phased rollout.
- Uncertainty-aware reflection (MC dropout or ensembles).
- Trajectory tracking for diagnostics; use most common failure step for targeted replanning.

### From 2504.16680v3 (RWM-U + MOPO-PPO)
- **Ensemble disagreement** provides epistemic uncertainty; **aleatoric** can be captured per-head.
- Uncertainty should be **propagated over long rollouts** and used as a **penalty / gating** signal.
- **Threshold calibration** is critical: too low -> overconfident acceptance; too high -> over-conservative replanning.

### From 2510.22680v1 (AV uncertainty control)
- **Entropy-based gating** is a stable and interpretable control signal.
- Uncertainty should **modulate downstream decisions** (here: reflection aggressiveness).
- Set-valued predictions can improve robustness when the target class is ambiguous (optional extension).

### From web search (classic world-model MBRL)
- PETS (Chua et al., 2018): Ensemble dynamics models + trajectory sampling for uncertainty-aware planning.
- MBPO (Janner et al., 2019): Trust short rollouts and use model-generalization estimates to prevent bias.
- World Models (Ha & Schmidhuber, 2018): Latent imagination as a planning substrate.
- Dreamer (Hafner et al., 2019): Latent rollouts + value gradients for long-horizon reasoning.

These reinforce using latent rollouts, ensemble uncertainty, and conservative gating to avoid model bias.

## 2) Inputs / Outputs
**Inputs (per step)**
- `z_t`, `a_t`, `z_{t+1}` from dynamics model
- `predicates_target_t` from LLM plan
- `remaining_plan_summary` (future predicates / primitives)

**Outputs (heads)**
- `p_predicate`: predicate satisfaction
- `p_feas`: action feasibility (executability)
- `p_nonint`: non-interference (future plan feasibility)

## 3) Model Architecture
- **Shared encoder**: MLP or transformer over concatenated embeddings of `z_t`, `a_t`, `z_{t+1}`, target predicate, remaining plan summary.
- **Head MLPs**: one per output head, sigmoid output.
- **Uncertainty**: deep ensemble (preferred), or MC dropout in each head.

## 4) Data and Supervision
**Positive samples**
- Successful rollouts from Points2Plans
- (z_t, a_t, z_{t+1}, target_predicate, label=1)

**Negative samples**
- Perturbations: wrong object/target, missing prerequisites
- Hard negatives: failed rollouts with divergence step labels
- LLM-generated incorrect plans (imagine + label failure)

**Auxiliary labels**
- Feasibility: executable vs non-executable action
- Non-interference: “future feasible” vs “blocks future”

## 5) Losses
- Binary cross-entropy per head
- Weighted sum: `L = λ_pred L_pred + λ_feas L_feas + λ_nonint L_nonint`
- Start: `λ_pred=1.0`, `λ_feas=0.5`, `λ_nonint=0.5`

## 6) Uncertainty Computation
For each head (ensemble size `B`):
- Mean: `μ = (1/B) Σ p_b`
- Epistemic variance: `σ² = (1/B) Σ (p_b - μ)²`
- Binary entropy: `H = -μ log μ - (1-μ) log(1-μ)`

Log aleatoric uncertainty if the head predicts variance; only **epistemic** drives reflection.

## 7) Reflection Thresholds (Defaults)
### Predicate head
- Hard fail: `μ < 0.35`
- Uncertainty fail: `σ > 0.15` or `H > 0.55`
- Soft fail: `0.35 ≤ μ < 0.55` and `σ > 0.10`

### Feasibility head
- Hard fail: `μ < 0.30`
- Uncertainty fail: `σ > 0.12` or `H > 0.50`
- Soft fail: `0.30 ≤ μ < 0.55` and `σ > 0.08`

### Non-interference head
- Hard fail: `μ < 0.40`
- Uncertainty fail: `σ > 0.10` or `H > 0.45`
- Soft fail: `0.40 ≤ μ < 0.60` and `σ > 0.07`

**Decision rule**: reflect if any head fails. Tag the failure reason.

## 8) Calibration Targets
- Predicate: precision ≥ 0.70, recall ≥ 0.80, ECE ≤ 0.05
- Feasibility: precision ≥ 0.75, recall ≥ 0.85
- Non-interference: precision ≥ 0.70, recall ≥ 0.80

Calibrate thresholds by maximizing `F_β` with `β=1.5` (favor recall). Apply temperature scaling if needed.

## 9) Trajectory Tracking + Failure Analysis
For each imagined rollout:
- Store per-step `μ`, `σ`, `H`, predicted predicates, and target predicate
- Identify **first failing step** and record reason
- Aggregate over candidates to find most common failing step
- Generate targeted reflection prompt (predicate mismatch, infeasible action, or future blocking)

## 10) Phased Implementation
**Phase 1**: predicate head only + uncertainty gating
**Phase 2**: add feasibility head + thresholds
**Phase 3**: add non-interference head + remaining plan summary

## 11) Integration Points
- Plug into the dynamics model rollout loop in Points2Plans
- Return `terminal_score` for action selection
- Return `trajectory_diagnostics` for reflection prompt generation

## 12) Evaluation
- Head-level accuracy + calibration
- Reflection trigger precision/recall
- End-to-end task success vs no-critic baseline
- Ablations: no uncertainty vs uncertainty gating; ensemble vs MC dropout

## 13) Optional Extensions
- Set-valued predicate satisfaction (if decoder ambiguity is high)
- Conformal thresholds on critic outputs for guaranteed coverage
- Long-horizon uncertainty penalty in rollout scoring (like MOPO-PPO)

## 14) Deliverables
- Critic module with 1–3 heads
- Uncertainty gating + reflection policy
- Logging + evaluation scripts
- Calibration report
