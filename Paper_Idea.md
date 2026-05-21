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

This architecture bridges Semantic Reasoning and Latent Physics by utilizing a VLM purely for action proposal, and a highly specialized two-stage Flow Matching dynamics model for physical simulation and evaluation.

### 1. Semantic Action Proposal
Keep the VLM in the loop **only for high‑level language grounding**. The VLM receives the current RGB observation and the goal description and proposes a **small set of plausible candidate actions** (e.g., $K=3$). This acts as an intelligent heuristic that prunes the search tree.

### 2. Two-Stage Specialized Latent Dynamics Engine
Instead of generic video-prediction dynamics, we design a two-stage system that mirrors and extends the RLA-WM architecture:

**Stage 1 — Bottleneck Encoder Pre-training:**
A `DeltaEncoder` (perceiver-style `SimpleTokenTransformer`) is trained to compress the raw DINO feature difference $F_{t+1} - F_t \in \mathbb{R}^{256 \times 768}$ into a compact latent token set $z \in \mathbb{R}^{16 \times 64}$. A paired `DeltaDecoder` reconstructs $\Delta F$ from $z$ under an MSE loss. After pre-training, the encoder is **frozen** for Stage 2. This stage uses only adjacent frame pairs — no action text, no temporal history.

**Stage 2 — Flow Matching in Compact Latent Space:**
The frozen `DeltaEncoder` produces target codes $z_{gt}$ for each training transition. A conditional flow-matching model learns to transport Gaussian noise to $z_{gt}$ in the low-dimensional $\mathbb{R}^{16 \times 64}$ space, conditioned on:
- **Temporal History Context:** $[F_{t-2}, F_{t-1}, F_t]$ with causal masking and learnable `[START]` tokens — the unique novelty relative to RLA-WM, which conditions only on the single current frame $F_t$.
- **Cross-Attention Action Grounding:** The temporal history DINO tokens cross-attend to the CLIP token sequence of the proposed text action. This provides language grounding at the conditioning stage. RLA-WM by contrast uses a single action vector (task embedding + qpos GRU output) passed as a modulation signal to self-attention — it does **not** cross-attend to a language token sequence.
- **Sparsity Regularization (Drift Prevention):** A global latent-activity penalty on static patches, applied via the raw DINO residual.

**Key difference vs. RLA-WM:**
| | RLA-WM | Ours |
|---|---|---|
| Conditioning input | Single frame $F_t$ | History $[F_{t-2}, F_{t-1}, F_t]$ |
| Action conditioning | Single vector (task ID + qpos GRU) via self-attn modulation | CLIP token sequence via cross-attention |
| Flow space | $\mathbb{R}^{N_{\text{lat}} \times 64}$ | $\mathbb{R}^{16 \times 64}$ (same concept) |
| Timestep sampling | logit-Normal | logit-Normal (adopted) |

### 3. Zero-Shot Latent Critic & Dynamic Rollout
- **Objective Evaluation:** We compute the cosine similarity between the predicted terminal latent state and the latent goal representation using a Dual-Head Contrastive Critic.
- **Dynamic Horizon:** Because the rollout occurs entirely in the latent space and evaluation requires no image rendering, we can afford **Monte‑Carlo Tree Search (MCTS)** or Beam Search. The horizon is dynamic, continuing the rollout until the critic's goal similarity exceeds a confidence threshold.

### 4. Causal Temporal Masking & Learnable Start-of-Sequence Context
Instead of using the standard CNN‑era hack of repeating the first frame (e.g., $[I_0, I_0, I_0]$) to fill the history buffer at early-episode timesteps—which introduces a mathematically flawed zero-momentum prior and misleads the model—we leverage a true Transformer‑native **Causal Masking and Learnable Padding** scheme:
- **Autoregressive Attention Masking:** We track a dynamic historical validity mask (e.g., `[False, False, True]` at $t=0$) and supply it directly to the Transformer self-attention blocks. Padded positions are strictly causal-masked, forcing the network to only attend to physically occurred history steps.
- **Learnable [START] Embeddings:** We substitute empty history indices with a specialized learnable sequence-initiator embedding (synonymous with `<BOS>` tokens in LLMs). This provides an explicit semantic signal to the model that it is operating at the genesis of an episode and should rely purely on the action prompt rather than searching for non-existent past momentum.

#### From Chat
Yes, absolutely! In fact, this exact approach is the gold standard for modern sequential vision models and robotics transformers.

The "repeating the first frame" trick is actually an outdated hack leftover from the era of Convolutional Neural Networks (CNNs). In older reinforcement learning setups (like playing Atari), agents stacked the last 4 frames into a 12-channel image. Because CNNs require a fixed number of input channels and can't use "attention masks," researchers had to copy the first frame 4 times just to make the tensor shape fit.

However, since your Latent World Model uses a Transformer architecture (via ModCrossAttentionBlock), you are not bound by CNN limitations. Using padding and attention masks is standard practice in:

Robotic Transformers (RT-1, RT-2, Decision Transformer): These models treat states, actions, and text as a stream of tokens. At step $t=0$, the context window isn't full yet, so they pad the empty slots with zeros and pass a causal attention mask to the transformer to ignore the padding.
Video Generation Models (VideoGPT, Sora-like architectures): Autoregressive video models explicitly use causal masking across the time dimension. For the first frame prediction, the attention heads are masked to only see the initial frame and the text prompt, never a repeated fake history.
Vision Transformers (ViViT, TimeSformer): Sequence-based vision models almost always use learnable [START] or [CLASS] tokens combined with padding masks when dealing with variable-length clips.
By implementing attention masking, you are essentially upgrading the temporal processing of your world model from a "CNN-style hack" to a proper "Transformer-style sequence modeling" approach.

## Contributions
1. **Decoupling Semantic Reasoning from Physical Simulation:** Demonstrates that a VLM does not need to "reflect" on generated pixels. Semantics are handled by the VLM (Proposal), and physics by the latent model (Evaluation).
2. **VLM‑Guided Latent Search:** Uses the VLM as a high‑level heuristic to solve the search-space explosion problem in continuous latent world models, enabling tractable planning for long‑horizon manipulation.
3. **Two-Stage Specialized Manipulation Dynamics:** Adopts and extends RLA-WM's two-stage design (bottleneck encoder → flow matching). Stage 1 compresses $\Delta F$ into compact latent tokens; Stage 2 flow-matches in that low-dimensional space conditioned on temporal history and cross-attention action grounding — both absent from RLA-WM.
4. **Causal Autoregressive History Alignment:** Replaces the standard but mathematically flawed first-frame duplication hack at episode boundaries with a clean causal attention masking mechanism and a learnable sequence-initiator (`[START]`) token, maintaining true physical momentum priors across the temporal context window.
5. **Unified, Ghost-Free Framework:** Outperforms pure latent models and pure VLM pipelines by perfectly combining common-sense reasoning with hallucination-free, deterministic physical prediction.

## Narrative for the Introduction
> *“Recent feature‑based world models (e.g., DINO‑WM, RLA‑WM) have removed the hallucination problems of pixel‑space diffusion by predicting dynamics in a pre‑trained latent space, yet they lack high‑level semantic reasoning and suffer from inefficient sampling for long‑horizon tasks. Conversely, VLM‑based planners excel at language grounding but are limited by a pixel‑generation bottleneck that introduces ghosting and scales poorly. In this work, we present **[Your System Name]**, a hybrid neuro-symbolic architecture. We decouple reasoning from physics by leveraging a VLM solely for semantic action proposal, while utilizing a specialized, sparsity-regularized latent flow-matching model for deterministic feature rollout and zero-shot evaluation, thereby achieving scalable, accurate, and language‑conditioned robotic planning.”*
