# Research Conversation: World Model + Critic for Robot Manipulation Planning

## Context
PhD research discussion on designing a system combining a world/dynamics model, LLM/VLM planner, and critic model for robot manipulation. References two papers:
- **Goal-VLA**: Image-Generative VLMs as Object-Centric World Models
- **ReflectVLM**: Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation

---

## System Overview

The proposed pipeline is:

```
LLM/VLM generates plan → action params → World Model imagines scene → Critic evaluates feasibility → (refine plan)
```

Formally:
```
LLM plan → a_t → ŝ_{t+1} = WorldModel(s_t, a_t) → Critic(ŝ_{t+1}, s_goal)
```

---

## Key Design Decisions & Discussion

### 1. Goal-VLA vs ReflectVLM World Model

**Goal-VLA's "world model"** is really a goal-conditioned image editor — it uses a generative VLM (Gemini Flash) to synthesize a goal image from a language instruction, then derives a 6-DoF object pose via feature matching + point cloud registration. It has no notion of intermediate states or action sequences. Structurally insufficient for training a critic that needs to reason about intermediate states.

**ReflectVLM's dynamics model** is a diffusion-based forward model predicting $I_{t+1}$ given $(I_t, a_t)$. Much better suited for critic training signal over a horizon. Their trajectory relabeling approach for training data generation (Figure 2 in paper) is clean and relatively low-effort to implement.

**Important clarification on ReflectVLM's iterative rollout**: The diffusion model generates one imagined state per *high-level action primitive* (e.g. "pick up purple", "insert purple"), not per low-level timestep. So for a 4-5 step horizon you only generate 4-5 imagined states total. This makes cascading error much less severe than it would be at dense timestep level — each rollout step is a meaningful semantic transition, not a tiny motion increment.

**Decision: Go with a lighter-weight version of ReflectVLM's diffusion dynamics model**, scoped to the task domain (nut assembly in robosuite). Lighter-weight because:
- Only needs faithful dynamics for nut assembly, not general manipulation
- Far less training data needed since visual domain is narrow
- Can finetune fewer layers of the base model

---

### 2. Task & Horizon

- **Task**: Modified nut assembly in robosuite
- **Horizon**: 4-5 steps
- **Example plan**: `grasp nut → move to peg → insert`
- May extend to more complex tasks with longer horizons later

---

### 3. Critic Design

#### 3a. Framing: Process Reward Model (PRM)

**The critic is framed as a Process Reward Model (PRM)** — a concept from LLM reasoning literature (Lightman et al., 2023 "Let's Verify Step by Step") applied to robot manipulation planning. A PRM assigns a reward signal to *intermediate steps* of a plan, not just the final outcome.

This framing is stronger than "binary feasibility classifier" because:
- It connects to a rapidly growing literature on test-time compute and process supervision
- It positions the critic as evaluating *plan quality at each step*, generalizing beyond any single task
- It naturally handles tasks beyond nut assembly since it doesn't assume sequential structure
- It provides a richer signal — a step-level quality score with uncertainty — rather than binary pass/fail

Formally, the critic estimates the probability that an optimal policy reaches the goal from the imagined intermediate state:

$$\text{Critic}(\hat{s}_{t+1}, s_\text{goal}) \rightarrow P(\text{goal reachable} \mid \hat{s}_{t+1}, s_\text{goal})$$

The critic does **not** need next action params as input. The next action params are the parameter generator's problem.

---

#### 3b. Output: Beta Distribution for Calibrated Uncertainty

**The critic outputs a Beta distribution over the feasibility score**, not a single scalar. This is the central design decision that makes the critic robust and informative.

```python
# Shared CNN backbone output
shared_features = CNN(concat_map)              # (B, 128)

# Output two parameters of a Beta distribution
alpha = Softplus(Linear(128 → 64 → 1))        # shape parameter α > 0
beta  = Softplus(Linear(128 → 64 → 1))        # shape parameter β > 0

# Derived quantities
mean_feasibility = alpha / (alpha + beta)      # point estimate ∈ (0, 1)
uncertainty      = (alpha * beta) / (          # variance of Beta distribution
    (alpha + beta)**2 * (alpha + beta + 1)
)
```

**Why Beta distribution:**
- Naturally bounded to [0,1] — appropriate for a probability estimate
- Represents predictive distributional uncertainty over feasibility; confidence is derived from concentration ($\alpha + \beta$)
- Calibrated uncertainty is a genuine methodological contribution: virtually no current manipulation critics/reward models output uncertainty estimates
- Directly handles diffusion model hallucinations — when the world model generates an implausible state, the critic should output high uncertainty, not a confident wrong answer

**Uncertainty interpretation note:** A single Beta-head network primarily captures *predictive/distributional uncertainty*. To claim stronger epistemic uncertainty, add either MC-dropout at inference or a small deep ensemble (3-5 critics) and report ensemble variance.

**Replanning trigger using uncertainty:**

```python
# Only replan when critic is CONFIDENTLY predicting infeasibility
# High uncertainty → gather more information rather than immediately replanning
trigger_replan = (mean_feasibility < feasibility_threshold) AND \
                 (uncertainty < confidence_threshold)
```

This is more robust than a simple threshold on the point estimate — it avoids false replanning triggers caused by world model hallucinations that the critic is uncertain about.

**Training loss — Beta NLL:**

```python
# Negative log-likelihood of Beta distribution at label y ∈ {0, 1}
# y is the ground truth feasibility label from robosuite rollouts
loss = -BetaDistribution(alpha, beta).log_prob(y.clamp(1e-6, 1 - 1e-6))
```

**Stability note with binary labels:** Beta NLL can be numerically brittle when labels are exactly 0/1 and predictions become over-confident early. Keep two safeguards:
- Train with clipped/soft labels (e.g., positives in `[0.95, 0.99]`, negatives in `[0.01, 0.05]`), or at minimum clamp labels as above.
- Always maintain a sigmoid + weighted BCE baseline run. If Beta NLL is unstable on a split, fall back to BCE for that experiment and keep Beta as an ablation.

---

#### 3c. Architecture: Spatial Difference + Lightweight CNN + Beta Head

**Encoder outputs from Instructpix2pix** (relevant context):

The diffusion model has two encoders:
- **VAE encoder**: shape `(B, 4, 64, 64)` — spatially structured feature map where each location `(i,j)` corresponds to an 8×8 pixel patch in the original 512×512 image
- **CLIP text encoder**: shape `(B, 77, 768)` — text token embeddings; not used for the critic since neither input is text

**Critic uses the VAE encoder only:**
- $z_{t+1}$: VAE latent of the imagined next state → `(4, 64, 64)`
- $z_\text{goal}$: VAE latent of the goal image → `(4, 64, 64)`

```python
# ── Input construction ─────────────────────────────────────────────
diff_map   = z_t1 - z_goal                                     # (B, 4,  64, 64)
concat_map = torch.cat([z_t1, z_goal, diff_map], dim=1)        # (B, 12, 64, 64)

# ── Lightweight CNN backbone ───────────────────────────────────────
x = Conv2d(12, 32, 3, padding=1) → ReLU → MaxPool2d(2)         # (B, 32, 32, 32)
x = Conv2d(32, 64, 3, padding=1) → ReLU → MaxPool2d(2)         # (B, 64, 16, 16)
x = Conv2d(64, 128, 3, padding=1) → ReLU → AdaptiveAvgPool2d(1)# (B, 128, 1, 1)
shared_features = flatten(x)                                   # (B, 128)

# ── Beta distribution head ─────────────────────────────────────────
alpha = Softplus(Linear(128, 64) → ReLU → Linear(64, 1))       # (B, 1)
beta  = Softplus(Linear(128, 64) → ReLU → Linear(64, 1))       # (B, 1)

mean_feasibility = alpha / (alpha + beta)                       # (B, 1) point estimate
uncertainty      = beta_variance(alpha, beta)                   # (B, 1) predictive uncertainty proxy
```

**Why spatial CNN over plain MLP:**
- VAE latent is spatially structured — flattening discards geometric relationships meaningful for manipulation
- Difference map `z_t1 - z_goal` at each spatial location encodes *where* in the scene the imagined state deviates from goal
- CNN activations are interpretable and visualizable — useful for debugging and paper figures
- `[z_t1, z_goal, diff_map]` → `(B, 12, 64, 64)` gives access to absolute state AND relative difference simultaneously

**Alternatives considered:**

| Architecture | Why Deferred |
|---|---|
| Flatten + Difference + MLP | 32768-dim input loses spatial structure |
| Cross-attention on spatial tokens | 4096 tokens → quadratic cost; upgrade path if CNN plateaus |

**CNN upgrade path**: downsample VAE latents to `(B, 4, 16, 16)` = 256 tokens via strided convolutions, then apply cross-attention between $z_{t+1}$ tokens (queries) and $z_\text{goal}$ tokens (keys/values). Natural ablation for the paper.

---

#### 3d. Intermediate Step Evaluation with Early Termination

**The critic evaluates at every intermediate step**, not just at the end of the horizon. This is a stronger design than ReflectVLM which only reflects after the full imagined rollout.

```
For each step t in horizon:
    a_t → WorldModel(s_t, a_t) → ŝ_{t+1}
    mean_feasibility, uncertainty = Critic(z_{t+1}, z_goal)

    if (mean_feasibility < θ_f) AND (uncertainty < θ_u):
        trigger reflection — pass (ŝ_t, mean_feasibility, uncertainty, t) to LLM/VLM
        break
    elif (mean_feasibility < θ_f) AND (uncertainty >= θ_u):
        request one additional imagination sample / critic query
        # if still uncertain after K retries, trigger conservative reflection
    else:
        continue to t+1
```

**Advantages over end-of-horizon-only evaluation:**
- Catches infeasible trajectories earlier, saving world model computation on remaining steps
- Gives the LLM/VLM targeted replanning context — failure at step 2 vs step 4 implies different corrections
- Reduces cascading diffusion error since imagination stops at confident failure detection
- Avoids blindly continuing low-feasibility trajectories when critic confidence is low (explicit uncertain branch)
- **Meaningful contribution over ReflectVLM** — they always roll out the full horizon before replanning

---

#### 3e. Reflection Context Passed to LLM/VLM on Failure

```python
reflection_context = {
    "imagined_state":    ŝ_t,               # image — visually informative for VLM replanning
    "mean_feasibility":  mean_feasibility,   # scalar ∈ (0,1)
    "uncertainty":       uncertainty,        # scalar — how confident the critic is
    "failure_step":      t,                  # index into the plan
    "failed_subtask":    plan[t],            # e.g. "move to peg" — for targeted correction
}
```

The LLM/VLM uses this structured context to replan specifically at the failure point rather than regenerating the entire plan from scratch.

---

### 4. Action Parameters

- **In simulation**: Object/target locations (Cartesian space) used as action params
- **In real world**: A learned policy that outputs Cartesian space action parameters directly (not joint space)
- Outputting Cartesian space from the learned policy keeps the critic interface consistent between sim and real world — the critic always receives Cartesian action parameters regardless of source
- Sim-to-real gap becomes a **policy generalization problem**, not a critic interface problem — clean separation of concerns

---

### 5. Critic Input Space Decision

Three options considered:
1. Raw image space — publishable, aligns with visual world modeling trend, but hard to train sensitivity to subtle misalignments
2. Geometric/structured representation — clean in sim, breaks down in real world without pose estimator
3. **Diffusion model's latent embeddings** ← recommended

**Decision: Critic operates in the diffusion model's latent embedding space.**

Reasoning:
- World model is a diffusion model, so its latent space already encodes information relevant to scene transitions
- Critic operating in that same latent space evaluates feasibility in terms of features the world model itself found meaningful — natural alignment
- For nut assembly, the latent space of a finetuned diffusion model should encode peg-nut relative pose implicitly since that's the dominant source of variation
- Transfers naturally when swapping in the diffusion model at inference time

---

### 6. World Model: Diffusion Dynamics Model

**Base model**: Instructpix2pix (pretrained, as used in ReflectVLM)

**Finetuning procedure**:
1. Collect transitions $(I_t, a_t, I_{t+1})$ in robosuite using a mix of expert and suboptimal policies
2. Finetune Instructpix2pix where the "edit instruction" is the action description (e.g. "grasp nut") conditioned on Cartesian action params
3. Finetune the decoder separately for precise reconstruction (important for precise spatial relationships in nut-peg alignment)

**Training data collection** (following ReflectVLM):
- Use sub-optimal policies with varying noise levels to get broader state coverage
- Collect transitions across the full task horizon

---

### 7. Critic Training Using Robosuite Renderer (Key Insight)

The diffusion model is **not needed during critic training**. Use ground-truth simulator states instead:

**Procedure**:
1. Use robosuite's renderer to generate ground truth $(s_t, a_t, s_{t+1})$ transitions by actually simulating the actions
2. Pass $s_{t+1}$ (ground truth next state image) through the **frozen encoder** of the finetuned diffusion model to extract latent embeddings $z_{t+1}$
3. Similarly encode goal state $s_\text{goal}$ to get $z_\text{goal}$
4. Train critic as a classifier on $(z_{t+1}, z_\text{goal})$ with binary feasibility labels generated by checking if the expert policy can reach the goal from $s_{t+1}$ within a step budget

**Advantages**:
- Critic training completely decoupled from diffusion model quality
- Ground truth feasibility labels from simulator
- Critic learns in diffusion model's latent space → transfers naturally at inference time
- No dependency on diffusion model during critic training phase

---

### 8. Goal Representation

**Decision: Goal image encoded through the same frozen diffusion encoder.**

**How ReflectVLM gets the goal image**: They procedurally generate it — their tasks are synthetic assembly puzzles where the goal configuration is fully specified programmatically, so they literally render the goal state in the simulator. The goal image is always available by construction.

**Why goal image over alternatives:**

| Representation | Pros | Cons |
|---|---|---|
| Goal image | Information-rich, unambiguous, works naturally with VLMs, same representation space as imagined states | Requires knowing what goal looks like visually |
| Language goal | Generalizable, easy to specify | Underspecified for precise manipulation — "nut on peg" doesn't specify orientation/height |
| Geometric goal (target pose) | Precise, low-dimensional, directly measurable | Harder to specify in real world, less natural for VLM planners |

**The key architectural reason for goal image**: The critic operates in the diffusion model's latent space. Encoding the goal image through the same frozen encoder gives $z_\text{goal}$ in the same embedding space as $z_{t+1}$. The critic then learns a reachability function in a *consistent representation*. A language goal would require a separate text encoder, comparing embeddings across modalities — a harder learning problem.

**Goal representation and critic input space are coupled decisions** — whatever goal representation you choose must be encodable into the same space the critic operates in.

**In simulation**: Render goal state directly from robosuite (trivial).
**In real world**: Capture a reference image of the correctly assembled state once during setup.

---

### 9. Key Risk to Validate Early

**Distribution shift between real rendered states and diffusion-generated states in latent space.**

At inference time, the critic receives embeddings of *diffusion-generated* imagined states. But it was trained on embeddings of *ground-truth rendered* states. If these are far apart in latent space, the critic will be unreliable.

**Validation approach**: Before committing, compare encoder embeddings of matched real vs. diffusion-generated states directly. Measure the embedding distance distribution.

---

### 10. Research Directions and Development Order

#### Direction A (Primary): PRM Critic with Calibrated Uncertainty

The primary contribution is the critic as a calibrated Process Reward Model with Beta distribution output, intermediate step evaluation, and structured reflection context. This is a complete and publishable contribution on its own.

**Development sequence:**

1. **Validate pipeline with simulator as oracle world model** — use robosuite renderer to generate ground-truth imagined states. Gives a clean upper bound on critic quality before the diffusion model is introduced as a confound.
2. **Train PRM critic (Direction A)** — CNN backbone + Beta distribution head on ground-truth simulator states + frozen VAE encoder. Run BCE baseline in parallel. Validate calibration and uncertainty estimates.
3. **Validate intermediate step evaluation** — confirm early termination + structured reflection context improves replanning over end-of-horizon-only evaluation.
4. **Finetune diffusion dynamics model** on robosuite transitions.
5. **Embedding-shift gate before full swap** — compare latent distributions for rendered $I_{t+1}$ vs diffusion-generated $\hat{I}_{t+1}$; only proceed if similarity/calibration stay within threshold.
6. **Swap in diffusion model** at inference time — measure degradation vs. sim oracle; validate critic uncertainty correctly increases for hallucinated states.

---

#### Direction B (Upgrade): Joint World Model + Critic Co-Training

Direction B adds a gradient pathway from the critic loss back into the world model's UNet, so the world model learns to generate states that are maximally informative for feasibility estimation — not just visually plausible.

**Direction A is a strict prerequisite for Direction B** — by the time Direction A is complete, all components needed for co-training are already in place.

```python
# Direction A: critic loss isolated, world model frozen after finetuning
critic_loss = BetaNLL(alpha, beta, feasibility_label)
critic_loss.backward()   # gradients flow only through critic

# Direction B: critic loss also updates world model UNet
imagined_state = WorldModel(s_t, a_t)           # diffusion UNet — now trainable
z_t1 = VAEEncoder(imagined_state)               # VAE encoder stays frozen
alpha, beta = Critic(z_t1, z_goal)
critic_loss = BetaNLL(alpha, beta, feasibility_label)

total_loss = reconstruction_loss + lambda * critic_loss
total_loss.backward()    # gradients flow through critic AND world model UNet
```

**Why co-training is a stronger contribution:** There is a genuine tension between the world model's two objectives under co-training:
- **Reconstruction objective**: generate visually plausible next states
- **Critic objective**: generate states that are maximally discriminative for feasibility

Co-training forces the world model to resolve this tension — it learns to be not just visually faithful but *feasibility-aware* in what it generates. This is a meaningful and publishable property.

**Practical note**: Direction B requires backpropagating through the diffusion UNet, which is GPU memory intensive. Verify hardware can handle joint training before committing to it as a deliverable. Direction A is a complete contribution if Direction B is computationally prohibitive.

---

#### Ablation Table

The phased development naturally produces a clean 2×2 ablation that tells a complete story:

| Condition | World Model Training | Critic Output | Expected Result |
|---|---|---|---|
| Baseline | Independent | Single scalar sigmoid | Lower bound |
| **+PRM (Direction A)** | Independent | Beta distribution | Primary contribution |
| +Cotrain | Joint w/ critic loss | Single scalar sigmoid | Tests co-training value alone |
| **Full (Direction B)** | Joint w/ critic loss | Beta distribution | Upper bound |

Each row is a publishable incremental result. The table directly answers: (1) does calibrated uncertainty help? (2) does co-training help? (3) do they interact?

---

#### Evaluation Protocol (Required for Section 3 Claims)

For each condition in the ablation table, report:

- **Task metrics**: success rate, replans per episode, average failure step, average horizon used
- **Classification metrics**: AUROC, AUPRC, F1 on infeasible class
- **Calibration metrics**: ECE, Brier score, reliability diagram
- **Uncertainty behavior**: uncertainty histogram for correct vs incorrect predictions; uncertainty on hallucinated vs rendered states

This prevents over-claiming uncertainty quality from point metrics alone.

---

### 11. Implementation Plan (Execution Checklist)

#### Phase 0 — Reproducible Setup (1-2 days)

1. Freeze seeds, train/val split by **episode id**, and output directory schema.
2. Add experiment config templates for: `critic_bce`, `critic_beta`, `oracle_loop_eval`, `wm_swap_eval`.
3. Define canonical metric logger (CSV + JSON summary per run).

**Exit criteria:** Two repeated runs of the same config produce near-identical validation metrics.

#### Phase 1 — Critic Data + Embedding Cache (2-3 days)

1. Build embedding cache job:
    - Input: `image_t1`, `goal_image`, `label_reachable`
    - Encoder: frozen VAE encoder
    - Output: cached `z_t1`, `z_goal`, pooled + spatial tensors
2. Add integrity checks (missing files, NaNs, label coverage, class ratio).
3. Create dataloader variants for pooled input and spatial input.

**Exit criteria:** Cache creation passes on full dataset and class stats are logged.

#### Phase 2 — Critic Baselines (3-4 days)

1. Train **BCE baseline** critic (same CNN backbone, sigmoid head).
2. Train **Beta-head** critic with label smoothing/clipping and numerically stable loss.
3. Add optional MC-dropout or 3-model ensemble for epistemic proxy.
4. Tune thresholds $(\theta_f, \theta_u)$ on validation set.

**Exit criteria:** Beta or BCE reaches target AUROC and acceptable calibration (ECE/Brier) on validation.

#### Phase 3 — Oracle Planner Loop Integration (2-3 days)

1. Integrate critic in oracle world-model loop (simulated next state instead of diffusion output).
2. Implement 3-way control:
    - feasible → continue
    - confidently infeasible → reflect/replan
    - low-feasibility + high-uncertainty → re-imagine/requery up to `K`, then conservative reflect
3. Log per-step decision traces for qualitative analysis.

**Exit criteria:** Early termination improves compute/replanning efficiency without reducing success rate vs end-of-horizon baseline.

#### Phase 4 — Diffusion Swap + Shift Gate (3-5 days)

1. Run latent shift analysis on matched pairs `(I_{t+1}, \hat{I}_{t+1})`.
2. If shift acceptable, swap diffusion model into planner loop.
3. Recompute full metrics and compare against oracle upper bound.

**Exit criteria:** Controlled degradation and uncertainty increases on shifted/hallucinated samples.

#### Phase 5 — Direction B Feasibility Spike (optional, 5-7 days)

1. Prototype joint loss `L = L_recon + \lambda L_critic` with small `\lambda`.
2. Use alternating updates to stabilize training.
3. Run reduced-scale ablation to test if co-training helps before full launch.

**Exit criteria:** Demonstrable gain on at least one of: success rate, calibration, or sample efficiency without destabilizing reconstruction.

---

## Full Pipeline Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                          At Inference                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LLM/VLM generates plan                                         │
│  "grasp nut → move to peg → insert"                             │
│                    │                                             │
│                    ▼                                             │
│  ┌─────────── For each step t ──────────────────────────────┐   │
│  │                                                          │   │
│  │  Parameter Generator                                     │   │
│  │  (sim: object locations / real: learned policy)          │   │
│  │       │                                                  │   │
│  │       ▼                                                  │   │
│  │  Cartesian action params a_t                             │   │
│  │       │                                                  │   │
│  │       ▼                                                  │   │
│  │  Diffusion World Model (finetuned Instructpix2pix)       │   │
│  │       │                                                  │   │
│  │       ▼                                                  │   │
│  │  Imagined state ŝ_{t+1}                                 │   │
│  │       │                                                  │   │
│  │       ▼                                                  │   │
│  │  Frozen VAE Encoder                                      │   │
│  │       │                                                  │   │
│  │       ▼                                                  │   │
│  │  z_{t+1} ──► PRM Critic(z_{t+1}, z_goal)               │   │
│  │                    │                                     │   │
│  │         Beta(α, β) distribution                         │   │
│  │         → mean_feasibility, uncertainty                 │   │
│  │                    │                                     │   │
│  │    ┌───────────────┴───────────────┐                     │   │
│  │    ▼                               ▼                     │   │
│  │  feasible OR uncertain         confidently infeasible    │   │
│  │  continue to t+1               trigger reflection        │   │
│  │                                pass to LLM/VLM:         │   │
│  │                                  ŝ_t, mean_f,           │   │
│  │                                  uncertainty, t,        │   │
│  │                                  plan[t]                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  z_goal = FrozenVAEEncoder(goal image rendered from sim)        │
│  Same encoder for z_{t+1} and z_goal → consistent latent space  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Paper References

- **Goal-VLA**: Chen et al. "Goal-VLA: Image-Generative VLMs as Object-Centric World Models Empowering Zero-shot Robot Manipulation." arXiv:2506.23919v2, 2025.
- **ReflectVLM**: Feng et al. "Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation." 2025.
- **Instructpix2pix**: Brooks et al. 2022 — base model for diffusion dynamics model finetuning
- **Process Reward Models**: Lightman et al. 2023 "Let's Verify Step by Step" — conceptual grounding for PRM framing

---

## Open Questions / Next Steps

- What LLM/VLM specifically for plan generation?
- How complex will the real-world learned policy need to be?
- Feasibility and confidence threshold tuning — how to set $\theta_f$ and $\theta_u$; consider learning them or sweeping on a validation set
- Which layers of Instructpix2pix to finetune (ReflectVLM freezes latent + text encoder, finetunes UNet + decoder)
- How action description text + Cartesian params are jointly conditioned into the diffusion model
- Expert policy step budget for generating feasibility labels during critic training
- Maximum number of replanning attempts before system gives up
- GPU memory feasibility of Direction B (backprop through diffusion UNet)
- Which tasks beyond nut assembly to target for generalization evaluation