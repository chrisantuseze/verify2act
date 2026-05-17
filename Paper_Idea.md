# Hybrid VLM‑Driven Feature‑Based World Model

## Motivation & Limitations of Prior Work

### DINO‑WM / Baseline RLA‑WM
- **No high‑level semantic / language reasoning:** Pure latent dynamics models are excellent for physics prediction but cannot inherently interpret natural-language goals or constraints.
- **Search‑space explosion:** They rely on random or CEM sampling of continuous actions, which scales poorly to long‑horizon, multi‑step manipulation tasks.
- **Weak Action Grounding:** Baseline feature-models typically just concatenate text embeddings (like CLIP) to image patches, which fails to precisely ground the language instruction (e.g., "pick up nut") to the specific object in a cluttered scene.
- **Markovian Amnesia & Drift:** Standard autoregressive rollouts often forget occluded objects and suffer from compounding numerical drift in static backgrounds.

### VLM‑Based Planners (e.g., Reflect‑VLM)
- **Pixel‑bottleneck:** Require a generative diffusion model (InstructPix2Pix) to render future RGB images for the VLM to “reflect” on.
- **Compounding hallucination / ghosting:** Each diffusion step introduces visual artifacts, background erasure, and object disappearance, making the critic entirely unreliable over long horizons.
- **Slow iterative inference:** Repeatedly running a multi-step image diffusion model for every node in a search tree is computationally prohibitive.

## Proposed Hybrid Approach & Novelties

This architecture bridges Semantic Reasoning and Latent Physics by utilizing a VLM purely for action proposal, and a highly specialized Flow Matching dynamics model for physical simulation and evaluation.

### 1. Semantic Action Proposal
Keep the VLM in the loop **only for high‑level language grounding**. The VLM receives the current RGB observation and the goal description and proposes a **small set of plausible candidate actions** (e.g., $K=3$). This acts as an intelligent heuristic that prunes the search tree.

### 2. Specialized Latent Dynamics Engine
Instead of generic video-prediction dynamics, we design a custom **Flow Matching model** predicting Residual Latent Actions ($\Delta F$) in DINOv2 space, specialized for robotic manipulation:
- **Cross-Attention Action Grounding:** Instead of concatenation, DINO patch features cross-attend to the CLIP tokens of the proposed text action. This strictly grounds the language instruction to the physical object patches.
- **History Context:** The dynamics model receives a short temporal window (e.g., $t-2, t-1, t$) instead of a single frame, providing short-term memory to handle severe occlusions (like an object disappearing inside a hole or gripper).
- **Multi-Scale Spatial Resolution:** Merging shallow (geometric) and deep (semantic) layers from the DINO backbone to achieve the sub-patch precision required for tight-tolerance assembly tasks.
- **Sparsity Regularization (Drift Prevention):** An L1 penalty explicitly applied to non-manipulated patches during training, mathematically anchoring the static background and preventing compounding errors over long horizons.

### 3. Zero-Shot Latent Critic & Dynamic Rollout
- **Objective Evaluation:** We compute the cosine similarity between the predicted terminal latent state and the latent goal representation using a Dual-Head Contrastive Critic.
- **Dynamic Horizon:** Because the rollout occurs entirely in the latent space and evaluation requires no image rendering, we can afford **Monte‑Carlo Tree Search (MCTS)** or Beam Search. The horizon is dynamic, continuing the rollout until the critic's goal similarity exceeds a confidence threshold.

## Contributions
1. **Decoupling Semantic Reasoning from Physical Simulation:** Demonstrates that a VLM does not need to “reflect” on generated pixels. Semantics are handled by the VLM (Proposal), and physics by the latent model (Evaluation).
2. **VLM‑Guided Latent Search:** Uses the VLM as a high‑level heuristic to solve the search-space explosion problem in continuous latent world models, enabling tractable planning for long‑horizon manipulation.
3. **Specialized Manipulation Dynamics:** Introduces architectural upgrades to residual flow-matching (Cross-Attention grounding, Sparsity loss, History context) that explicitly solve occlusion and drift in robotic assembly tasks.
4. **Unified, Ghost-Free Framework:** Outperforms pure latent models and pure VLM pipelines by perfectly combining common-sense reasoning with hallucination-free, deterministic physical prediction.

## Narrative for the Introduction
> *“Recent feature‑based world models (e.g., DINO‑WM, RLA‑WM) have removed the hallucination problems of pixel‑space diffusion by predicting dynamics in a pre‑trained latent space, yet they lack high‑level semantic reasoning and suffer from inefficient sampling for long‑horizon tasks. Conversely, VLM‑based planners excel at language grounding but are limited by a pixel‑generation bottleneck that introduces ghosting and scales poorly. In this work, we present **[Your System Name]**, a hybrid neuro-symbolic architecture. We decouple reasoning from physics by leveraging a VLM solely for semantic action proposal, while utilizing a specialized, sparsity-regularized latent flow-matching model for deterministic feature rollout and zero-shot evaluation, thereby achieving scalable, accurate, and language‑conditioned robotic planning.”*
