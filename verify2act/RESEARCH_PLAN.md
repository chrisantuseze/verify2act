# Verify2Act: Critic-Guided Reflective Imagination for Sequential Planning

## Overview

This document outlines the research direction for **Verify2Act**, a system that enables long-horizon robotic manipulation planning through:
1. **Planning** — LLM generates multi-step plans with symbolic predicates
2. **Verification** — Dynamics model imagines execution; critic evaluates feasibility
3. **Replanning** — Targeted reflection repairs failed subtasks

**Core Thesis:** Uncertainty-aware reflection enables robust long-horizon planning under model error. By explicitly modeling where the dynamics model is unreliable and triggering targeted replanning at those points, we can achieve higher success rates on multi-step manipulation tasks compared to filtering-based approaches.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VERIFY2ACT PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────┐  │
│   │   LLM    │───▶│ Dynamics Model  │───▶│    Critic    │───▶│ Executor │  │
│   │ Planner  │    │ (Points2Plans)  │    │   (NEW)      │    │          │  │
│   └──────────┘    └─────────────────┘    └──────────────┘    └──────────┘  │
│        │                   │                    │                  │        │
│        │                   │                    │                  │        │
│        ▼                   ▼                    ▼                  ▼        │
│   Plan Sequence      Imagined States      Pass/Fail +        Real States   │
│   + Predicates       (z₁, z₂, ..., zₙ)    Uncertainty        (Rollout)     │
│                            │                    │                  │        │
│                            ▼                    │                  │        │
│                    ┌───────────────┐            │                  │        │
│                    │   Predicate   │◀───────────┘                  │        │
│                    │    Decoder    │                               │        │
│                    │ (Points2Plans)│                               │        │
│                    └───────────────┘                               │        │
│                            │                                       │        │
│                            ▼                                       │        │
│                    Symbolic State                                  │        │
│                    ON(cup, table)                                  │        │
│                    IN(tea, cup)                                    │        │
│                                                                    │        │
│   ◀────────────────── REFLECTION LOOP ─────────────────────────▶  │        │
│   If critic flags failure → LLM replans from failed subtask        │        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Building on Points2Plans

Points2Plans provides the following components that Verify2Act will leverage:

| Component | Points2Plans Provides | Verify2Act Extends |
|-----------|----------------------|-------------------|
| **Dynamics Model** | Predicts z_{t+1} from (z_t, action) | Use as imagination engine |
| **Predicate Decoder** | Converts z → symbolic predicates | Use for critic supervision |
| **LLM Planner** | Generates action sequences | Add reflection conditioning |
| **Point Cloud Encoder** | Observation → latent state | Use as-is |

**Key Insight:** Points2Plans already bridges latent states ↔ symbolic predicates. The critic can directly compare:
- Predicted predicates (from imagined z_{t+1}) 
- Target predicates (from LLM plan)

---

## The Critic Model

### Architecture: Multi-Headed Critic

```
                    ┌─────────────────────────────────────┐
                    │           CRITIC MODEL              │
                    ├─────────────────────────────────────┤
                    │                                     │
  Inputs:           │   ┌─────────────────────────────┐   │
  ─────────────────▶│   │      Shared Encoder         │   │
  • z_{t+1}         │   │   (Transformer / MLP)       │   │
  • action a_t      │   └──────────┬──────────────────┘   │
  • target predicate│              │                      │
  • remaining plan  │              ▼                      │
                    │   ┌──────────┴──────────┐           │
                    │   │                     │           │
                    │   ▼                     ▼           │
                    │ ┌───────────┐    ┌───────────┐      │
                    │ │ HEAD 1    │    │ HEAD 2    │      │   ┌───────────┐
                    │ │Feasibility│    │ Predicate │      │   │ HEAD 3    │
                    │ │           │    │Satisfaction│     │   │Non-Interf.│
                    │ └─────┬─────┘    └─────┬─────┘      │   └─────┬─────┘
                    │       │                │            │         │
                    │       ▼                ▼            │         ▼
                    │   P(executable)   P(predicate)      │   P(future ok)
                    │                                     │   [Phase 2]
                    └─────────────────────────────────────┘
```

### Head Descriptions

| Head | Input | Output | Failure Mode Addressed |
|------|-------|--------|----------------------|
| **Feasibility** | z_t, a_t | P(action executable) | Unreachable objects, invalid grasps, collisions |
| **Predicate Satisfaction** | z_{t+1}, target_pred | P(predicate achieved) | Action doesn't achieve intended effect |
| **Non-Interference** | z_{t+1}, remaining_plan | P(future feasible) | Current action blocks future subtasks |

### Implementation Phases

**Phase 1 (Start Here):**
- Single-head critic: Predicate Satisfaction only
- Binary classification: Does imagined state satisfy target predicate?
- Simplest to train, most direct signal

**Phase 2:**
- Add Feasibility head
- Requires negative samples of infeasible actions

**Phase 3:**
- Add Non-Interference head
- Requires planning-aware supervision (hardest)

---

## Training the Critic

### Data Collection Strategy

#### Positive Samples (Successful Plans)
```
Source: Successful rollouts from Points2Plans
Format: (z_t, a_t, z_{t+1}, predicate, label=1)

Example:
  z_t:        latent state with cup in gripper
  a_t:        pickplace(cup, table)
  z_{t+1}:    latent state after placement
  predicate:  ON(cup, table)
  label:      1 (success)
```

#### Negative Samples (Failed Plans)

**Strategy 1: Perturbation-Based**
```python
# Modify successful plans to create failures
def generate_negative(positive_plan):
    modified = positive_plan.copy()
    
    # Option A: Wrong target location
    modified.action = pickplace(cup, wrong_location)
    
    # Option B: Wrong object
    modified.action = pickplace(wrong_object, table)
    
    # Option C: Skip prerequisite step
    modified.plan = remove_prerequisite(plan)
    
    return modified, label=0
```

**Strategy 2: Hard Negatives from Failures**
```python
# Mine from actual failed rollouts
def collect_hard_negatives(rollout_buffer):
    negatives = []
    for rollout in rollout_buffer:
        if not rollout.success:
            # Find the step where failure occurred
            failure_step = find_divergence_point(rollout)
            negatives.append((
                rollout.states[failure_step],
                rollout.actions[failure_step],
                rollout.next_states[failure_step],
                rollout.target_predicates[failure_step],
                label=0
            ))
    return negatives
```

**Strategy 3: LLM-Generated Incorrect Plans**
```python
# Ask LLM to generate plausible but incorrect plans
prompt = """
Given the goal: Make tea
Generate an incorrect plan that would fail.
Keep predicates correct, but make subtask order wrong.
"""
# Use these as negative samples after dynamics imagination
```

### Loss Functions

**Predicate Satisfaction Head:**
```
L_pred = -[y·log(σ(f(z_{t+1}, pred))) + (1-y)·log(1-σ(f(z_{t+1}, pred)))]

where:
  y = 1 if predicate satisfied, 0 otherwise
  f = critic network
  σ = sigmoid
```

**Feasibility Head:**
```
L_feas = -[y·log(σ(g(z_t, a_t))) + (1-y)·log(1-σ(g(z_t, a_t)))]

where:
  y = 1 if action executable, 0 otherwise
```

**Combined (Multi-Task):**
```
L_total = λ_1·L_pred + λ_2·L_feas + λ_3·L_non_interf
```

---

## Uncertainty Quantification

### Why Uncertainty Matters

The critic should know when it doesn't know. High uncertainty → trigger reflection even if prediction is "pass".

### Methods to Consider

| Method | Complexity | Quality | Recommendation |
|--------|-----------|---------|----------------|
| MC Dropout | Low | Medium | **Start here** |
| Deep Ensembles | Medium | High | Phase 2 |
| Evidential DL | Medium | High | Alternative |

### MC Dropout Implementation
```python
class UncertainCritic(nn.Module):
    def __init__(self, ...):
        self.dropout = nn.Dropout(p=0.1)
    
    def forward_with_uncertainty(self, x, n_samples=10):
        self.train()  # Enable dropout
        predictions = []
        for _ in range(n_samples):
            pred = self.forward(x)
            predictions.append(pred)
        
        mean = torch.stack(predictions).mean(dim=0)
        uncertainty = torch.stack(predictions).std(dim=0)
        return mean, uncertainty
```

### Decision Logic
```python
def should_reflect(critic_output, uncertainty, thresholds):
    pred_score, pred_unc = critic_output['predicate'], uncertainty['predicate']
    
    # Reflect if: low confidence OR high uncertainty
    if pred_score < thresholds['score'] or pred_unc > thresholds['uncertainty']:
        return True, "predicate_failure"
    
    return False, None
```

---

## Lookahead with Trajectory Tracking

### The Problem: Terminal-Only Evaluation Loses Diagnostics

When using multi-step lookahead for action selection, evaluating only the terminal state creates a critical gap:

- **Selection works:** We can rank actions by terminal state quality
- **Diagnostics lost:** When ALL samples fail, we don't know:
  - Which step in the sequence caused failure?
  - What type of failure (collision? predicate mismatch? unreachable?)
  - What should we tell the LLM to fix?

### Solution: Track Trajectory + Evaluate Terminal

Track intermediate states during lookahead, but still **select based on terminal feasibility**:

```
┌─────────────────────────────────────────────────────────────────────┐
│              LOOKAHEAD WITH TRAJECTORY TRACKING                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  For each sampled action:                                           │
│                                                                     │
│    trajectory = []                                                  │
│    z = z₀                                                           │
│                                                                     │
│    for step_idx, primitive in enumerate(lookahead_sequence):        │
│      z_next = dynamics(z, action)                                   │
│      pred = decoder(z_next)                                         │
│                                                                     │
│      step_info = {                                                  │
│        'step': step_idx,                                            │
│        'primitive': primitive,                                      │
│        'latent_state': z_next,                                      │
│        'predicted_predicates': pred,                                │
│        'target_predicate': target[step_idx],                        │
│        'step_score': critic(z_next, target[step_idx]),  ← Per-step  │
│        'failure_reasons': detect_failures(...)                      │
│      }                                                              │
│      trajectory.append(step_info)                                   │
│      z = z_next                                                     │
│                                                                     │
│    terminal_score = critic(z, goal_predicates)  ← For selection     │
│                                                                     │
│    return action, terminal_score, trajectory  ← Keep diagnostics    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Analyzing Failures Across Trajectories

When all samples fail, aggregate trajectory data to identify systematic issues:

```python
def _analyze_failures(self, all_trajectories: List[Dict]) -> Dict:
    """Analyze failure patterns across all sampled trajectories."""
    
    failure_analysis = {
        'most_common_failure_step': None,
        'failure_step_counts': defaultdict(int),
        'failure_reasons': defaultdict(int),
        'problematic_predicates': [],
        'best_partial_trajectory': None,
        'best_partial_score': -1.0,
    }
    
    for traj in all_trajectories:
        # Find first failing step in this trajectory
        for step_info in traj['steps']:
            if step_info['step_score'] < threshold:
                failure_analysis['failure_step_counts'][step_info['step']] += 1
                for reason in step_info['failure_reasons']:
                    failure_analysis['failure_reasons'][reason] += 1
                
                # Track which predicates are problematic
                failed_preds = compare_predicates(
                    step_info['predicted_predicates'],
                    step_info['target_predicate']
                )
                failure_analysis['problematic_predicates'].extend(failed_preds)
                break
        
        # Track best partial progress
        partial_score = sum(s['step_score'] for s in traj['steps']) / len(traj['steps'])
        if partial_score > failure_analysis['best_partial_score']:
            failure_analysis['best_partial_score'] = partial_score
            failure_analysis['best_partial_trajectory'] = traj
    
    # Identify most problematic step
    if failure_analysis['failure_step_counts']:
        failure_analysis['most_common_failure_step'] = max(
            failure_analysis['failure_step_counts'], 
            key=failure_analysis['failure_step_counts'].get
        )
    
    return failure_analysis
```

### Generating Targeted Reflection Prompts

Use trajectory analysis to give LLM specific, actionable feedback:

```python
def _generate_reflection_prompt(self, primitive_plan, failure_analysis) -> str:
    """Generate targeted reflection prompt based on trajectory analysis."""
    
    failed_step = failure_analysis['most_common_failure_step']
    failed_primitive = primitive_plan[failed_step] if failed_step is not None else "unknown"
    
    # Get most common failure reasons
    top_reasons = sorted(
        failure_analysis['failure_reasons'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    prompt = f"""
The plan failed during dynamics model verification.

Original Plan: {primitive_plan}

Failure Analysis:
- Most failures occurred at step {failed_step + 1 if failed_step else '?'}: "{failed_primitive}"
- {failure_analysis['failure_step_counts'][failed_step]} out of {self.num_samples} samples failed at this step
- Common failure reasons: {dict(top_reasons)}
- Problematic predicates: {set(failure_analysis['problematic_predicates'][:5])}

Best Partial Progress:
- Best trajectory achieved {failure_analysis['best_partial_score']:.2f} average step score
- Successfully completed steps: {failed_step if failed_step else 0}

Please suggest an alternative approach. Consider:
1. Is there a prerequisite step missing before step {failed_step + 1 if failed_step else '?'}?
2. Should objects be manipulated in a different order?
3. Is the target location for "{failed_primitive}" appropriate?
4. Are there collision or reachability issues to address?
"""
    
    return prompt
```

### Key Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Action Selection** | Terminal state score | Lookahead should evaluate long-term outcomes |
| **When to Track** | Every step during lookahead | Need diagnostics for all intermediate states |
| **When to Analyze** | Only when ALL samples fail | Avoid overhead when feasible action found |
| **What to Store** | Per-step scores + predicates + failure reasons | Rich info for targeted reflection |
| **Reflection Trigger** | All samples below threshold | Exhaustive search before fallback |

### Overhead Considerations

```python
# Lightweight trajectory storage (per sample)
trajectory_entry = {
    'step': int,                    # 4 bytes
    'primitive': str,               # ~50 bytes
    'step_score': float,            # 8 bytes  
    'failure_reasons': List[str],   # ~100 bytes
    # Skip storing full latent states unless debugging
    # 'latent_state': tensor,       # ~4KB per object (SKIP in production)
}

# For K=50 samples, L=3 lookahead: ~50 * 3 * 200 bytes = 30KB total
# Negligible compared to model forward passes
```

---

## Reflection Mechanism

### Structured Feedback to LLM

When reflection is triggered, provide the LLM with:

```python
reflection_prompt = f"""
The plan failed at step {failed_step}.

Original Plan:
{format_plan(original_plan)}

Failure Analysis:
- Failed subtask: {failed_action}
- Target predicate: {target_predicate}
- Predicted predicate: {predicted_predicate}
- Failure reason: {failure_type}  # e.g., "object unreachable", "wrong placement"

Current State (after step {failed_step - 1}):
{current_predicates}

Goal:
{goal_predicates}

Generate a corrected plan starting from step {failed_step}.
Ensure the new plan achieves: {remaining_predicates}
"""
```

### Preventing Reflection Loops

```python
class ReflectionManager:
    def __init__(self, max_reflections=3):
        self.max_reflections = max_reflections
        self.reflection_history = []
    
    def should_continue(self, failed_step, failure_reason):
        # Check if same failure repeated
        key = (failed_step, failure_reason)
        if self.reflection_history.count(key) >= 2:
            return False, "repeated_failure"
        
        if len(self.reflection_history) >= self.max_reflections:
            return False, "max_reflections_reached"
        
        self.reflection_history.append(key)
        return True, None
```

---

## Two Verification Modes

### Mode 1: Imagination-Time Verification (Pre-Execution)

```
┌─────────────────────────────────────────────────────────────────┐
│                    IMAGINATION VERIFICATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each subtask in plan:                                      │
│    1. z_{t+1} = dynamics_model(z_t, action)                     │
│    2. pred_{t+1} = predicate_decoder(z_{t+1})                   │
│    3. score, unc = critic(z_{t+1}, target_pred)                 │
│    4. if score < θ or unc > θ_unc:                              │
│         trigger_reflection(step=t)                              │
│    5. z_t = z_{t+1}  # Chain forward                            │
│                                                                 │
│  If all steps pass → Execute plan                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Mode 2: Execution-Time Verification (During Rollout)

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION VERIFICATION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each subtask during execution:                             │
│    1. Execute action in environment                             │
│    2. z_observed = encoder(observation)                         │
│    3. z_predicted = dynamics_model(z_prev, action)              │
│    4. deviation = compute_deviation(z_observed, z_predicted)    │
│    5. if deviation > θ_dev:                                     │
│         trigger_reflection(step=t, from_execution=True)         │
│                                                                 │
│  Deviation Metrics:                                             │
│    • Euclidean: ||z_obs - z_pred||_2                            │
│    • Predicate: disagreement(pred(z_obs), pred(z_pred))         │
│    • KL: KL(p(z_obs) || p(z_pred)) for distributional encoders  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Experimental Plan

### Ablation Studies

| Experiment | Components | Purpose |
|------------|-----------|---------|
| **Baseline** | LLM only | Lower bound |
| **+ Dynamics** | LLM + imagination (no critic) | Value of verification |
| **+ Critic** | LLM + imagination + critic | Value of learned scoring |
| **+ Uncertainty** | Full system with uncertainty | Value of selective reflection |
| **+ Execution-Time** | Both verification modes | Value of closed-loop correction |

### Metrics

- **Success Rate:** % of tasks completed
- **Reflection Efficiency:** Success rate improvement per reflection
- **Reliability Horizon:** Max steps before dynamics model becomes unreliable
- **Computation Cost:** Planning time, number of reflections

### Task Complexity Progression

1. **2-3 step tasks:** Validate basic pipeline
2. **4-6 step tasks:** Test reflection mechanism
3. **7+ step tasks:** Stress test uncertainty handling

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up Verify2Act codebase structure
- [ ] Integrate Points2Plans dynamics model
- [ ] Implement basic imagination loop
- [ ] Create data collection pipeline

### Phase 2: Single-Head Critic (Weeks 3-4)
- [ ] Implement Predicate Satisfaction critic
- [ ] Collect positive/negative training data
- [ ] Train and validate critic
- [ ] Integrate with imagination loop

### Phase 3: Reflection (Weeks 5-6)
- [ ] Implement structured feedback to LLM
- [ ] Add reflection loop with safeguards
- [ ] Test on 2-3 step tasks
- [ ] Iterate on prompt engineering

### Phase 4: Uncertainty (Weeks 7-8)
- [ ] Add MC Dropout uncertainty
- [ ] Tune thresholds for reflection triggering
- [ ] Compare with non-uncertainty baseline
- [ ] Analyze reliability horizon

### Phase 5: Multi-Head & Execution-Time (Weeks 9-12)
- [ ] Add Feasibility head
- [ ] Implement execution-time verification
- [ ] Add Non-Interference head (if time permits)
- [ ] Full ablation studies

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Dynamics model too inaccurate | Critic receives garbage | Focus on shorter horizons; use execution-time verification |
| Negative samples not diverse | Critic overfits | Mine hard negatives from actual failures |
| Reflection loops | System hangs | Strict loop limits; failure history tracking |
| Uncertainty calibration poor | Wrong reflection decisions | Validate calibration on held-out set |

---

## References

- **Points2Plans:** Base dynamics model and predicate decoder
- **REFINER:** https://arxiv.org/pdf/2304.01904 — Reflection/refinement framework
- **Inner Monologue:** Feedback-driven replanning in robotics
- **SayCan:** Grounding LLMs in robot affordances

---

## Next Steps

**Option A: Sketch out the Critic Model Architecture**
- Detailed PyTorch implementation
- Input/output specifications
- Integration points with Points2Plans

**Option B: Integration with Existing Relational Dynamics Code**
- Review current `relational_dynamics/` structure
- Map Points2Plans components to Verify2Act needs
- Create integration plan

---

## Appendix A: Integration Plan with Points2Plans

### Codebase Analysis Summary

After reviewing the existing Points2Plans codebase, here's what we have:

#### Points2Plans Architecture

```
Points2Plans/
├── relational_dynamics/
│   ├── base_RD.py              # Main orchestration class (RelationalDynamics)
│   ├── model_utils.py          # Data processing utilities
│   ├── model/
│   │   ├── models.py           # Core neural network modules
│   │   │   ├── PointConv       # Point cloud encoder (obs → embedding)
│   │   │   ├── EmbeddingNetTorch    # Dynamics model (z_t, a → z_{t+1})
│   │   │   └── QuickReadoutNet      # Decoder (z → predicates, poses)
│   ├── dataloader/
│   └── config/
│
└── robosuite/planning/
    ├── dynamics_model_planner.py   # Wrapper for closed-loop planning
    ├── llm_task_planner.py         # LLM interface for goals/plans
    ├── state_converter.py          # Robosuite state → model input
    ├── primitive_executor.py       # Execute actions in simulation
    └── collision_checker.py        # Collision detection utilities
```

#### Key Components & Their Roles

| Component | Class/File | What It Does | Verify2Act Integration |
|-----------|-----------|--------------|----------------------|
| **Point Cloud Encoder** | `PointConv` in models.py | Converts per-object point clouds to 128-dim embeddings | Use as-is for observation encoding |
| **Dynamics Model** | `EmbeddingNetTorch` in models.py | Predicts next latent state given current state + action | Core imagination engine |
| **Predicate Decoder** | `QuickReadoutNet` in models.py | Decodes latent state to symbolic predicates + poses | Use for critic supervision |
| **Planning Wrapper** | `DynamicsModelPlanner` | Rejection sampling over action candidates | Extend with critic-based filtering |
| **LLM Interface** | `LLMTaskPlanner` | Generates goals and plans from natural language | Extend with reflection prompts |

---

### Key Data Structures

#### Latent State (`z_t`)
```python
# Shape: [batch_size, num_objects, embedding_dim]
# embedding_dim = node_emb_size * 2 = 256 (default)

# Created by concatenating:
# - PointConv output (128-dim per object)
# - One-hot object ID embedding (128-dim per object)

node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim=-1)
# node_pose.shape = [batch, num_objects, 256]
```

#### Action Embedding
```python
# Discrete component: which object to move
discrete_action = classif_model.one_hot_encoding_embed(object_id)  # [batch, 128]

# Continuous component: (dx, dy) displacement
continuous_action = classif_model.continuous_action_emb(action[:, -3:-1])  # [batch, 128]

# Combined: [discrete; continuous] = [batch, 256]
current_action = torch.cat([discrete_action, continuous_action], dim=-1)
```

#### Decoder Outputs
```python
outs_decoder = classif_model_decoder(latent_state, edge_index)

# Returns dict with:
{
    'pred_sigmoid': binary_predicates,     # [batch, num_pairs, num_predicate_types]
    'predicted_pose': pose_deltas,         # [batch, num_objects, 2] (dx, dy)
    'env_identity': environment_features,  # [batch, num_objects, 3]
    'grasp_identity': feasibility_scores,  # [batch, num_pairs, 2]
}
```

#### Predicate Structure
```python
# Predicates are pairwise relations between objects
# For N objects: N * (N-1) pairs
# Each pair has multiple predicate types (typically 6):
#   - ON(obj_i, obj_j)
#   - ABOVE(obj_i, obj_j)
#   - BELOW(obj_i, obj_j)
#   - LEFT(obj_i, obj_j)
#   - RIGHT(obj_i, obj_j)
#   - INSIDE(obj_i, obj_j)

# Shape: [batch, N*(N-1), 6]
predicates = outs_decoder['pred_sigmoid']
```

---

### Integration Points for Verify2Act

#### 1. Where to Hook the Critic

The critic should be inserted into `DynamicsModelPlanner.plan_next_primitive()`:

```python
# Current flow in dynamics_model_planner.py:
def plan_next_primitive(self, state_dict, goal_predicates, primitive_plan):
    # 1. Sample K action candidates
    candidates = self._sample_action_candidates(...)
    
    # 2. Forward simulate each through dynamics model
    for action in candidates:
        pred_latent = self._simulate_forward(state_dict, action)
        pred_predicates = self._decode_predicates(pred_latent)
        
        # 3. Check if predicates match goals (current: hard threshold)
        if self._predicates_match(pred_predicates, goal_predicates):
            return action
    
    # 4. Return best if none match
    return best_action

# ─────────────────────────────────────────────────────────────────
# PROPOSED: Insert critic after dynamics simulation
# ─────────────────────────────────────────────────────────────────

def plan_next_primitive_with_critic(self, state_dict, goal_predicates, primitive_plan):
    candidates = self._sample_action_candidates(...)
    
    scored_candidates = []
    for action in candidates:
        pred_latent = self._simulate_forward(state_dict, action)
        pred_predicates = self._decode_predicates(pred_latent)
        
        # NEW: Critic evaluation
        critic_score, uncertainty = self.critic.evaluate(
            latent_state=pred_latent,
            target_predicates=self._get_target_predicate(action, primitive_plan),
            remaining_plan=primitive_plan[1:]
        )
        
        scored_candidates.append({
            'action': action,
            'pred_latent': pred_latent,
            'pred_predicates': pred_predicates,
            'critic_score': critic_score,
            'uncertainty': uncertainty
        })
    
    # Filter by critic score and uncertainty
    feasible = [c for c in scored_candidates 
                if c['critic_score'] > self.score_threshold 
                and c['uncertainty'] < self.uncertainty_threshold]
    
    if feasible:
        return feasible[0]['action']
    else:
        # Trigger reflection
        return self._trigger_reflection(scored_candidates, primitive_plan)
```

#### 2. Where to Get Training Data for Critic

**Positive samples:** Extract from successful planning runs
```python
# In DynamicsModelPlanner, after successful execution:
def _log_successful_transition(self, state_dict, action, next_state_dict, achieved_predicate):
    self.positive_buffer.append({
        'z_t': self._get_latent(state_dict),
        'action': action,
        'z_t1': self._get_latent(next_state_dict),
        'target_predicate': achieved_predicate,
        'label': 1  # success
    })
```

**Negative samples:** From failed rollouts or perturbations
```python
# Collect when dynamics prediction diverges from actual outcome:
def _log_failed_transition(self, state_dict, action, predicted_predicate, actual_predicate):
    self.negative_buffer.append({
        'z_t': self._get_latent(state_dict),
        'action': action,
        'z_t1_predicted': predicted_latent,
        'target_predicate': predicted_predicate,
        'actual_predicate': actual_predicate,
        'label': 0  # failure
    })
```

#### 3. Where to Insert Reflection Loop

Extend `LLMTaskPlanner` with reflection capability:

```python
# In llm_task_planner.py, add:

def generate_reflection_prompt(
    self,
    original_plan: List[str],
    failed_step: int,
    failure_info: Dict
) -> str:
    """Generate a structured reflection prompt for the LLM."""
    return f"""
The plan failed at step {failed_step}.

Original Plan:
{self._format_plan(original_plan)}

Failure Analysis:
- Failed subtask: {failure_info['action']}
- Target predicate: {failure_info['target_predicate']}
- Predicted predicate: {failure_info['predicted_predicate']}
- Critic assessment: {failure_info['critic_reason']}

Current State:
{failure_info['current_predicates']}

Goal:
{failure_info['goal_predicates']}

Generate a corrected plan starting from step {failed_step}.
"""

def replan_from_failure(
    self,
    original_plan: List[str],
    failed_step: int,
    failure_info: Dict
) -> List[str]:
    """Request LLM to replan after a failure."""
    prompt = self.generate_reflection_prompt(original_plan, failed_step, failure_info)
    response = self.model.generate(prompt)
    new_plan = self._parse_plan(response)
    return new_plan
```

---

### Proposed Directory Structure for Verify2Act

```
verify2act/
├── verify2act/
│   ├── __init__.py
│   ├── RESEARCH_PLAN.md           # This document
│   │
│   ├── critic/                    # NEW: Critic model module
│   │   ├── __init__.py
│   │   ├── critic_model.py        # Multi-headed critic architecture
│   │   ├── uncertainty.py         # MC Dropout / ensemble uncertainty
│   │   ├── data_collector.py      # Positive/negative sample collection
│   │   └── trainer.py             # Critic training loop
│   │
│   ├── reflection/                # NEW: Reflection/replanning module
│   │   ├── __init__.py
│   │   ├── reflection_manager.py  # Loop control, history tracking
│   │   ├── feedback_generator.py  # Structured feedback for LLM
│   │   └── prompts/               # Reflection prompt templates
│   │       └── reflection_template.txt
│   │
│   ├── verification/              # NEW: Verification module
│   │   ├── __init__.py
│   │   ├── imagination_verifier.py    # Pre-execution verification
│   │   └── execution_verifier.py      # During-rollout verification
│   │
│   └── integration/               # NEW: Points2Plans integration
│       ├── __init__.py
│       ├── dynamics_wrapper.py    # Wrapper for RelationalDynamics
│       └── planner_extension.py   # Extended DynamicsModelPlanner
│
├── Points2Plans/                  # EXISTING (submodule or symlink)
│   └── ... (unchanged)
│
└── robosuite/                     # EXISTING 
    └── planning/                  # Extend existing planners
        ├── dynamics_model_planner.py  # Modify to use critic
        └── llm_task_planner.py        # Add reflection methods
```

---

### Implementation Order

#### Phase 1: Foundation Setup
```
1. Create verify2act/critic/ directory structure
2. Create dynamics_wrapper.py to cleanly interface with Points2Plans
3. Implement data_collector.py for training sample collection
4. Add logging hooks to existing DynamicsModelPlanner
```

#### Phase 2: Critic Model
```
1. Implement critic_model.py (start with single predicate-satisfaction head)
2. Implement trainer.py with BCE loss
3. Collect initial training data from existing planner runs
4. Train and validate basic critic
```

#### Phase 3: Integration
```
1. Implement imagination_verifier.py
2. Extend DynamicsModelPlanner with critic scoring
3. Add uncertainty estimation
4. Test on 2-3 step tasks
```

#### Phase 4: Reflection
```
1. Implement reflection_manager.py
2. Add reflection prompts to LLMTaskPlanner
3. Create feedback_generator.py for structured failure feedback
4. Implement loop control (max reflections, repeated failure detection)
```

---

### API Design

#### Critic Interface
```python
class Critic:
    """Multi-headed critic for action evaluation."""
    
    def evaluate(
        self,
        latent_state: torch.Tensor,      # [batch, num_obj, emb_dim]
        action: torch.Tensor,             # [batch, action_dim]
        target_predicate: torch.Tensor,   # [batch, num_pairs, pred_dim]
        remaining_plan: Optional[List[str]] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate action quality.
        
        Returns:
            scores: {'predicate': 0.85, 'feasibility': 0.92, 'non_interference': 0.78}
            uncertainties: {'predicate': 0.05, 'feasibility': 0.03, 'non_interference': 0.12}
        """
        pass
    
    def should_reflect(
        self,
        scores: Dict[str, float],
        uncertainties: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide if reflection should be triggered.
        
        Returns:
            (should_reflect, reason)
            e.g., (True, "predicate_score_low")
        """
        pass
```

#### Reflection Interface
```python
class ReflectionManager:
    """Manages reflection loops with safety guardrails."""
    
    def __init__(self, max_reflections: int = 3):
        self.max_reflections = max_reflections
        self.history = []
    
    def request_reflection(
        self,
        original_plan: List[str],
        failed_step: int,
        critic_assessment: Dict,
        current_state: Dict
    ) -> Tuple[bool, Optional[List[str]]]:
        """
        Request reflection and return new plan if allowed.
        
        Returns:
            (reflection_allowed, new_plan)
        """
        pass
```

#### Extended Planner Interface
```python
class Verify2ActPlanner(DynamicsModelPlanner):
    """Extended planner with critic-guided verification."""
    
    def __init__(self, critic: Critic, reflection_manager: ReflectionManager, **kwargs):
        super().__init__(**kwargs)
        self.critic = critic
        self.reflection_manager = reflection_manager
    
    def plan_with_verification(
        self,
        state_dict: Dict,
        goal_predicates: np.ndarray,
        primitive_plan: List[str]
    ) -> Tuple[str, np.ndarray, Dict]:
        """
        Plan next action with critic verification.
        
        Returns:
            (action_name, action_params, verification_info)
        """
        pass
```

---

### Key Implementation Details

#### Extracting Latent States for Critic Input

```python
# From base_RD.py, the latent state creation:

def get_latent_state(self, state_dict):
    """Extract latent state from observation for critic input."""
    
    # 1. Get point cloud voxels
    voxel_data = state_dict['batch_voxel_list_single']
    
    # 2. Encode with PointConv
    # Shape: [batch, num_objects, voxel_channels, num_points]
    reshaped = voxel_data.reshape(-1, voxel_data.shape[2], voxel_data.shape[3])
    img_emb = self.emb_model(reshaped)  # [batch*num_obj, 128]
    img_emb = img_emb.reshape(voxel_data.shape[0], voxel_data.shape[1], -1)
    
    # 3. Add one-hot object encoding
    one_hot = state_dict['batch_one_hot_encoding']
    latent_one_hot = self.classif_model.one_hot_encoding_embed(torch.argmax(one_hot, dim=2))
    
    # 4. Concatenate for full latent
    latent_state = torch.cat([img_emb, latent_one_hot], dim=-1)
    # Shape: [batch, num_objects, 256]
    
    return latent_state
```

#### Forward Simulation Through Dynamics

```python
def simulate_forward(self, latent_state, action, skill_type=0):
    """Simulate one step forward through dynamics model."""
    
    # 1. Encode action
    discrete_action = self.classif_model.one_hot_encoding_embed(action['object_id'])
    continuous_action = self.classif_model.continuous_action_emb(action['displacement'])
    action_emb = torch.cat([discrete_action, continuous_action], dim=-1)
    
    # 2. Combine with state
    graph_input = torch.cat([latent_state, action_emb], dim=1)
    
    # 3. Forward through dynamics (skill-specific)
    if skill_type == 0:  # pick-place
        next_latent = self.classif_model.graph_dynamics_0(graph_input)
    else:  # push
        next_latent = self.classif_model.graph_dynamics_1(graph_input)
    
    # 4. Handle delta vs absolute prediction
    if self.args.delta_forward:
        next_latent = next_latent[:, :-2, :] + latent_state
    else:
        next_latent = next_latent[:, :-2, :]
    
    return next_latent
```

#### Decoding Predicates from Latent

```python
def decode_predicates(self, latent_state, edge_index):
    """Decode symbolic predicates from latent state."""
    
    # Forward through decoder
    outputs = self.classif_model_decoder(latent_state, edge_index)
    
    # Extract predicates (sigmoid already applied)
    predicates = outputs['pred_sigmoid']  # [batch, num_pairs, num_pred_types]
    
    # Optionally threshold to binary
    binary_predicates = (predicates > 0.5).float()
    
    return {
        'raw': predicates,           # For critic training (soft labels)
        'binary': binary_predicates  # For verification (hard labels)
    }
```

---

### Testing Strategy

#### Unit Tests
```python
# tests/test_critic.py

def test_critic_forward():
    """Test critic forward pass shapes."""
    critic = Critic(input_dim=256, hidden_dim=128)
    
    batch_size, num_objects, emb_dim = 4, 5, 256
    latent = torch.randn(batch_size, num_objects, emb_dim)
    target_pred = torch.randint(0, 2, (batch_size, 20, 6)).float()
    
    scores, uncertainties = critic.evaluate(latent, target_pred)
    
    assert 'predicate' in scores
    assert 0 <= scores['predicate'] <= 1

def test_reflection_loop_limit():
    """Test that reflection manager respects max iterations."""
    manager = ReflectionManager(max_reflections=3)
    
    for i in range(5):
        allowed, _ = manager.request_reflection(
            original_plan=['a', 'b', 'c'],
            failed_step=1,
            critic_assessment={},
            current_state={}
        )
        if i < 3:
            assert allowed
        else:
            assert not allowed
```

#### Integration Tests
```python
# tests/test_integration.py

def test_planner_with_critic():
    """Test full planning loop with critic."""
    # Load pretrained dynamics model
    planner = Verify2ActPlanner(
        checkpoint_path="Points2Plans/ckpt/checkpoint/cp_1.pth",
        critic=Critic.load("critic_checkpoint.pth")
    )
    
    # Mock state
    state_dict = create_mock_state(num_objects=5)
    goal_predicates = create_mock_goals()
    plan = ["pick(obj_1)", "place(obj_1, shelf)"]
    
    action, params, info = planner.plan_with_verification(
        state_dict, goal_predicates, plan
    )
    
    assert action in ["pick", "place"]
    assert 'critic_score' in info
```

---

*Document created: January 15, 2026*
*Project: Verify2Act*
*Author: Chrisantus Eze*
