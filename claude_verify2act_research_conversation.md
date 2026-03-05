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

**The critic is a feasibility evaluator / goal-conditioned reachability classifier**, not a dense value function or RL reward model.

It asks: *"Given the imagined state after executing this sub-plan, is the overall goal still reachable?"*

This is a binary or probabilistic classifier conditioned on `(imagined state, goal)`.

**The critic does NOT need next action params as input.** It purely evaluates:
$$\text{Critic}(\hat{s}_{t+1}, s_\text{goal}) \rightarrow \text{feasibility} \in [0, 1]$$

The next action params are the parameter generator's problem — it looks at $\hat{s}_{t+1}$ and figures out what to do next.

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

### 8. Key Risk to Validate Early

**Distribution shift between real rendered states and diffusion-generated states in latent space.**

At inference time, the critic receives embeddings of *diffusion-generated* imagined states. But it was trained on embeddings of *ground-truth rendered* states. If these are far apart in latent space, the critic will be unreliable.

**Validation approach**: Before committing, compare encoder embeddings of matched real vs. diffusion-generated states directly. Measure the embedding distance distribution.

---

### 9. Recommended Development Order

1. **Validate pipeline with simulator as oracle world model** — use robosuite renderer to generate "imagined" states (i.e. actually simulate the plan). This gives a clean upper bound on critic quality and validates the overall pipeline before diffusion model becomes a confound.
2. **Train and validate critic** using ground-truth simulator states + frozen diffusion encoder
3. **Finetune diffusion dynamics model** on robosuite transitions
4. **Swap in diffusion model** at inference time and measure degradation vs. sim oracle
5. **Ablations**: critic w/ sim vs. critic w/ diffusion (reviewers will expect this — ReflectVLM provides precedent)

---

## Full Pipeline Summary

```
┌─────────────────────────────────────────────────────────────┐
│                        At Inference                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LLM/VLM                                                    │
│  generates ──► "grasp nut → move to peg → insert"          │
│  plan                                                       │
│     │                                                       │
│     ▼                                                       │
│  Parameter                                                  │
│  Generator ──► Cartesian action params a_t                  │
│     │          (sim: object locations,                      │
│     │           real: learned policy output)                │
│     ▼                                                       │
│  Diffusion                                                  │
│  World Model ──► imagined state ŝ_{t+1}                    │
│  (finetuned        │                                        │
│  Instructpix2pix)  │                                        │
│                    ▼                                        │
│              Frozen Encoder                                 │
│                    │                                        │
│                    ▼                                        │
│              z_{t+1} (latent embedding)                     │
│                    │                                        │
│                    ▼                                        │
│  Critic ◄── (z_{t+1}, z_goal)                              │
│  (feasibility                                               │
│   classifier) ──► feasibility score                        │
│                    │                                        │
│                    ▼                                        │
│              Refine plan if needed                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Paper References

- **Goal-VLA**: Chen et al. "Goal-VLA: Image-Generative VLMs as Object-Centric World Models Empowering Zero-shot Robot Manipulation." arXiv:2506.23919v2, 2025.
- **ReflectVLM**: Feng et al. "Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation." 2025.
- **Instructpix2pix**: Brooks et al. 2022 — base model for diffusion dynamics model finetuning
- **ReflectVLM diffusion model base**: Same Instructpix2pix architecture

---

## Open Questions / Next Steps

- What LLM/VLM specifically for plan generation?
- How complex will the real-world learned policy need to be?
- Extension to longer horizon tasks beyond nut assembly
- Critic evaluation frequency: after each horizon step vs. end of full horizon only
