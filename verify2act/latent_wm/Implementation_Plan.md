# Implementation Plan: Hybrid VLM‑Feature World Model

This document provides a comprehensive, step-by-step blueprint for migrating the Reflect-VLM pipeline from pixel-level diffusion to a highly specialized, feature-based latent dynamics model. This implementation is tailored explicitly for precise robotic manipulation tasks.

## Phase 1: Upgraded Latent Dynamics Core (The Physics Engine)

We will build a custom Flow Matching model that significantly improves upon the baseline RLA-WM by addressing occlusions, action grounding, and background drift.

### [NEW] `verify2act/latent_wm/dynamics.py`
This file will contain the `LatentDynamicsModel`, a Transformer-based Flow Matching model predicting DINOv2 residuals ($\Delta F$).

**Key Architectural Details:**
1. **History Context (Handling Occlusion):** 
   - Instead of a Markovian input $F_t$, the model takes a concatenated window of the last $N$ frames (e.g., $N=3$): $[F_{t-2}, F_{t-1}, F_t]$. 
   - This temporal context allows the attention mechanism to "remember" occluded objects during complex insertions.
2. **Cross-Attention Action Grounding:**
   - **Input:** The VLM's string action (e.g., "pick round nut") is tokenized and passed through a frozen `CLIPTextModel`, yielding a sequence of action tokens.
   - **Mechanism:** The DINO patch sequence (acting as Queries) cross-attends to the CLIP action tokens (Keys/Values). This explicitly grounds the language instruction to specific spatial patches.
3. **Multi-Scale Spatial Resolution:**
   - To achieve sub-patch precision, the input feature map will concatenate intermediate transformer layers from the DINOv2 backbone (e.g., merging layer 4 for high-frequency geometry with layer 11 for deep semantics).
4. **Integration (Inference):**
   - Implements `step(F_history, A_clip) -> F_{t+1}`. Uses an ODE solver (e.g., Euler, 5 steps) to integrate the predicted velocity field into the final residual $\Delta F$, returning $F_t + \Delta F$.

## Phase 2: Local Training Pipeline

We will train this highly specialized dynamics model directly on your dataset.

### [NEW] `verify2act/latent_wm/train_dynamics.py`
**Key Training Details:**
1. **Dataset:** Loads from `transitions.jsonl`. Pre-computes and caches DINOv2 multi-scale features and CLIP embeddings for efficiency.
2. **Flow Matching Objective:** Uses standard Conditional Flow Matching (CFM) loss to map from noise to the ground truth feature residual $(F_{t+1} - F_t)$.
3. **Sparsity Regularization (Drift Prevention):**
   - We will calculate a target mask indicating which patches actually changed between $F_t$ and $F_{t+1}$.
   - Add an **L1 Sparsity Loss** term forcing the model to predict exactly `0.0` velocity for patches outside the manipulation zone. This mathematically anchors the static background, preventing compounding drift over long horizons.

## Phase 3: Critic Integration

We will reuse your well-trained contrastive critic to evaluate the latent rollouts.

### [MODIFY] `verify2act/critic/model.py`
**Precise Code Changes:**
1. Add a new method to `DINOv2DualHeadCritic`:
   ```python
   def encode_features(self, patch_tokens: torch.Tensor) -> "ProbEmbedding":
       """Bypasses the backbone and evaluates predicted DINO patches directly."""
       # If patch_tokens is multi-scale, project back to 768 or take the deepest layer
       mu = patch_tokens.mean(dim=1)  # Mean pool over patches
       log_var1 = self.log_var_head1(mu).clamp(-4.0, 4.0)
       log_var2 = self.log_var_head2(mu).clamp(-4.0, 4.0)
       return ProbEmbedding(mu=mu, log_var1=log_var1, log_var2=log_var2)
   ```

## Phase 4: Hybrid Planner & Dynamic Rollout

### [NEW] `verify2act/latent_wm/planner.py`
This module orchestrates the MCTS/Beam Search loop.

**Key Logic (`HybridPlanner.plan()`):**
1. **Initialize:** Extract base features $F_{current}$ and $F_{goal}$ using DINOv2. Maintain a history buffer for $F_{current}$.
2. **VLM Action Proposal:** 
   - Prompt the VLM with the current and goal RGB images.
   - *Prompt Update:* Instruct the VLM to output a JSON list of $K$ (e.g., 3) distinct, plausible candidate actions.
3. **Dynamic Latent Rollout (Beam Search):**
   - Embed the $K$ candidates using CLIP.
   - For each candidate, call `LatentDynamicsModel.step()`.
   - **Dynamic Horizon ($H$):** Continue proposing and unrolling actions until the terminal node's `goal_sim` (via the Critic) exceeds a pre-defined confidence threshold (e.g., > 0.85), or a maximum depth limit (e.g., $H=10$) is reached.
4. **Selection:** Backtrack the search tree and return the optimal action trajectory.

## Phase 5: Visualization Decoder

### [NEW] `verify2act/latent_wm/visualizer.py`
- Imports and wraps `DinoToImageDecoderV1` (from the `rla-wm` codebase).
- Used purely for human interpretability. After `HybridPlanner` selects the best latent trajectory, this module decodes the sequence of $F_{t}$ back to RGB images and saves a GIF/Collage for logging.
