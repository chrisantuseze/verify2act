# Module Components, Inputs/Outputs, and Training Specifics

## System Overview

```
[Image_t]  [action_text]
     │            │
     ▼            ▼
  [VAE Encoder] [Text Encoder]      ← both FROZEN at all times
     │            │
  z_t (latent)  z_text
     │            │
     └──────┬─────┘
            ▼
       [Diffusion UNet]             ← trainable (LoRA, Phase A)
            │
         ẑ_{t+1}
            │
            ▼
       [VAE Decoder]                ← trainable (full finetune, Phase B)
            │
         Î_{t+1}  (imagined frame)
            │
            ▼
       [VAE Encoder]  (same frozen encoder)
            │
        ẑ_{t+1} ──────────────────────┐
                                       │
  [goal_image] → [VAE Encoder] → z_goal│
                                       ▼
                               [Critic MLP]         ← trainable (Phase C)
                                       │
                               feasibility ∈ [0,1]
```

---

## Module 1: VAE Encoder (Frozen)

**Role:** Maps RGB images to the latent space shared by all modules.

| Property | Value |
|---|---|
| Base model | Stable Diffusion 1.5 VAE encoder (KL-regularised) |
| Input | `image` — `[B, 3, 512, 512]` float32, normalised to `[-1, 1]` |
| Output | `z` — `[B, 4, 64, 64]` float32 (latent) |
| Spatial compression | 8× (512 → 64 per spatial dim) |
| Channel dimension | 4 (latent channels) |
| Parameters | ~34M |
| Frozen? | **Always frozen.** Never trained in any phase. |

**Note:** The VAE encoder is the same object shared by the world model pipeline and the critic. This is the key design decision — both the imagined state and the ground-truth state live in the same latent space.

**Usage:**
```python
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
vae.requires_grad_(False)  # freeze

# Encode image to latent (scaling factor 0.18215 is SD convention)
z = vae.encode(image_tensor).latent_dist.sample() * 0.18215   # [B, 4, 64, 64]

# Decode latent to image
image_recon = vae.decode(z / 0.18215).sample            # [B, 3, 512, 512]
```

---

## Module 2: Text Encoder (Frozen)

**Role:** Encodes action text prompts into conditioning embeddings for the UNet cross-attention.

| Property | Value |
|---|---|
| Base model | CLIP ViT-L/14 text encoder (from Stable Diffusion 1.5) |
| Input | `action_text` string, tokenised to `[B, 77]` token ids |
| Output | `text_emb` — `[B, 77, 768]` float32 |
| Max sequence length | 77 tokens |
| Embedding dim | 768 |
| Parameters | ~123M |
| Frozen? | **Always frozen.** |

**Usage:**
```python
from transformers import CLIPTextModel, CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder.requires_grad_(False)

tokens = tokenizer(action_text, return_tensors="pt", padding="max_length",
                   max_length=77, truncation=True).input_ids
text_emb = text_encoder(tokens).last_hidden_state    # [B, 77, 768]
```

---

## Module 3: Diffusion UNet (Phase A — LoRA Finetune)

**Role:** The core denoiser. Learns the dynamics function: given current state latent `z_t` and action text conditioning, predict the noise in a noisy `z_{t+1}`.

| Property | Value |
|---|---|
| Base model | InstructPix2Pix UNet (based on SD 1.5 UNet) |
| Input 1 | `noisy_latent` — `[B, 4, 64, 64]` (noisy version of target `z_{t+1}`) |
| Input 2 | `image_latent` — `[B, 4, 64, 64]` (current frame `z_t`) |
| Concatenated UNet input | `[B, 8, 64, 64]` (channel-wise cat of inputs 1 and 2) |
| Input 3 | `text_emb` — `[B, 77, 768]` (action text, via cross-attention) |
| Input 4 | `timestep` — `[B]` scalar diffusion timestep |
| Output | `noise_pred` — `[B, 4, 64, 64]` predicted noise |
| Full UNet parameter count | ~860M |
| LoRA trainable parameters | ~2–8M (rank r=8, applied to attention layers) |
| Frozen? | Base weights frozen; LoRA adapters **trainable** in Phase A |

**InstructPix2Pix-specific:** The UNet in InstructPix2Pix is modified to accept 8 input channels (4 for the noisy target latent + 4 for the source image latent). This is the mechanism by which the model conditions on `Image_t`.

**LoRA config:**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # attention projections
    lora_dropout=0.05,
    bias="none",
)
unet = get_peft_model(unet, lora_config)
```

---

## Module 4: VAE Decoder (Phase B — Full Finetune)

**Role:** Decodes predicted latents back to pixel space with high fidelity on small critical objects (nuts, pegs). The pretrained SD decoder blurs fine spatial detail at this scale.

| Property | Value |
|---|---|
| Base model | Stable Diffusion 1.5 VAE decoder |
| Input | `z` — `[B, 4, 64, 64]` (predicted latent from UNet, de-scaled) |
| Output | `image` — `[B, 3, 512, 512]` float32, range `[-1, 1]` |
| Parameters | ~49M |
| Frozen? | **No.** Fully finetuned in Phase B (separate from Phase A). |
| Phase A dependency | Phase B can run in parallel with Phase A (uses frozen encoder only) |

**Key point:** The encoder is frozen in Phase B. Only the decoder is updated. This means `z_{t+1}` embeddings used by the critic remain stable — the encoder output doesn't change.

---

## Module 5: Critic Classifier (Phase C — Full Train from Scratch)

**Role:** Feasibility classifier. Given latent embeddings of the imagined next state and the goal, predicts the probability that the goal is still reachable.

$$\text{Critic}(z_{t+1}, z_\text{goal}) \rightarrow p_\text{feasible} \in [0, 1]$$

| Property | Value |
|---|---|
| Input 1 | `z_{t+1}` — `[B, 4, 64, 64]` latent, spatially pooled to `[B, 4]` (mean over spatial dims) |
| Input 2 | `z_goal` — `[B, 4, 64, 64]` latent, spatially pooled to `[B, 4]` |
| Concatenated input | `[B, 8]` |
| Architecture | 3-layer MLP: `8 → 256 → 128 → 1`, ReLU, dropout 0.1 |
| Output | `[B, 1]` logit → sigmoid → feasibility probability |
| Parameters | ~66k (tiny) |
| Label source | `label_reachable` from `compute_labels.py` |
| Loss | Binary cross-entropy |
| Trained on | Ground-truth `Image_{t+1}` embeddings from simulator (NOT diffusion-generated) |

**Spatial pooling:**
```python
# z: [B, 4, 64, 64]
z_pooled = z.mean(dim=(-2, -1))   # [B, 4]
```

**Alternative (richer) input:** If mean-pooling is too lossy, use flattened + projected:
```python
z_flat = z.reshape(B, -1)         # [B, 16384]
z_proj = nn.Linear(16384, 128)    # learnable projection
```
Start with mean-pooled (faster to train); switch to projected if critic calibration is poor.

---

## Training Phase A — UNet LoRA

**Objective:** Train the UNet LoRA adapters to predict the noise at each diffusion timestep, conditioned on `(z_t, action_text)`.

$$\mathcal{L}_A = \mathbb{E}_{z_{t+1}, \epsilon \sim \mathcal{N}(0,I), t}\left[\|\epsilon - \epsilon_\theta(\tilde{z}_{t+1}, z_t, \tau(a), t)\|^2\right]$$

where $\tilde{z}_{t+1} = z_{t+1} + \sigma_t \epsilon$ is the noisy latent, $z_t$ is the encoded source frame, $\tau(a)$ is the text embedding.

| Hyperparameter | Value |
|---|---|
| Base model | `timbrooks/instruct-pix2pix` |
| Resolution | 512 |
| Batch size | 2 (gradient accumulation steps 4 → effective batch 8) |
| Learning rate | 2e-4 (linear warmup 500 steps, then constant) |
| Max steps | 10 000 (adjust based on dataset size) |
| LoRA rank `r` | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Mixed precision | fp16 |
| Gradient checkpointing | Yes |
| xFormers | Yes (if available) |
| Optimizer | AdamW 8-bit (bitsandbytes) |
| Image conditioning scale | 1.5 (InstructPix2Pix) |
| Text conditioning scale | 7.5 (InstructPix2Pix) |

**What is frozen:**
- VAE encoder + decoder (completely frozen)
- Text encoder (completely frozen)
- UNet base weights (frozen; only LoRA adapters trained)

**What is saved:** LoRA adapter weights (`unet_lora.safetensors`) + config. (~10–30 MB)

---

## Training Phase B — Decoder Finetune

**Objective:** Improve VAE decoder precision on the nut assembly visual domain (sharp nut-peg geometry).

$$\mathcal{L}_B = \lambda_1 \|I_{t+1} - \hat{I}_{t+1}\|_1 + \lambda_2 \,\mathcal{L}_{\text{LPIPS}}(I_{t+1}, \hat{I}_{t+1})$$

| Hyperparameter | Value |
|---|---|
| Batch size | 8 |
| Learning rate | 5e-6 |
| Max epochs | 5 |
| L1 weight $\lambda_1$ | 1.0 |
| LPIPS weight $\lambda_2$ | 0.5 |
| Input | Ground-truth `z_{t+1}` from frozen encoder |
| Target | Ground-truth `Image_{t+1}` |
| Optimizer | AdamW |
| Mixed precision | fp16 |

**What is frozen:** VAE encoder (freeze to preserve embedding space for critic).
**What is trained:** VAE decoder weights (full update, no LoRA needed — decoder is small).

**Can run in parallel with Phase A** because it uses only the frozen encoder, not the UNet.

---

## Training Phase C — Critic

**Objective:** Binary classifier for goal reachability in diffusion latent space.

$$\mathcal{L}_C = -\left[y \ln \hat{p} + (1-y) \ln(1-\hat{p})\right]$$

where $y = \texttt{label\_reachable}$ and $\hat{p} = \text{Critic}(z_{t+1}, z_\text{goal})$.

| Hyperparameter | Value |
|---|---|
| Architecture | MLP: 8 → 256 → 128 → 1 (or variant below) |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Max epochs | 30 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Class weights | Inversely proportional to class frequency |
| Input source (training) | Ground-truth `Image_{t+1}` from simulator (NOT diffusion-generated) |
| Val split | 10% held-out episodes |

**What is frozen:** VAE encoder (embeddings are precomputed and cached as `.npy` before training).
**What is trained:** Critic MLP from scratch.

**Precompute embeddings before critic training:**
```python
# Run once, save to disk for fast critic training
z_t1 = vae.encode(image_t1).latent_dist.sample() * 0.18215
z_t1_pooled = z_t1.mean(dim=(-2,-1))    # [B, 4]
np.save("z_t1_pooled.npy", z_t1_pooled.cpu().numpy())
```

---

## Embedding-Shift Validation (Between Phases A and C)

After Phase A UNet training, before deploying the critic at inference, validate:

1. Generate `\hat{I}_{t+1}` from diffusion for N matched (image_t, action_text) pairs.
2. Encode both ground-truth `I_{t+1}` and generated `\hat{I}_{t+1}` with the frozen encoder.
3. Compute:
   - Mean L2 distance: $\mathbb{E}\|z_{t+1} - \hat{z}_{t+1}\|_2$
   - Cosine similarity distribution
   - Domain classifier accuracy (train a linear classifier to separate real vs generated embeddings; lower accuracy = less shift)
4. **Accept threshold:** Mean cosine similarity > 0.85; domain classifier accuracy < 65%.
5. If shift is large: add alignment loss in Phase A (e.g., MMD or cosine loss between z-real and z-generated) or collect more diverse data.

---

## Inference Pipeline (Full)

```python
# Given: current image, action text, goal image

# 1. Encode current frame
z_t = vae.encode(preprocess(image_t)).latent_dist.sample() * 0.18215   # [1,4,64,64]

# 2. Encode goal frame (precomputed or cached)
z_goal = vae.encode(preprocess(goal_image)).latent_dist.sample() * 0.18215

# 3. Encode action text
text_emb = text_encoder(tokenizer(action_text)).last_hidden_state       # [1,77,768]

# 4. Run InstructPix2Pix denoising loop
pipeline = StableDiffusionInstructPix2PixPipeline(...)
image_t1_hat = pipeline(prompt=action_text, image=image_t, ...)  # returns PIL image

# 5. Encode imagined frame
z_t1_hat = vae.encode(preprocess(image_t1_hat)).latent_dist.sample() * 0.18215

# 6. Critic query
z_t1_pooled =  z_t1_hat.mean(dim=(-2,-1))    # [1, 4]
z_goal_pooled = z_goal.mean(dim=(-2,-1))      # [1, 4]
critic_input = torch.cat([z_t1_pooled, z_goal_pooled], dim=-1)    # [1, 8]
feasibility = torch.sigmoid(critic(critic_input)).item()           # scalar
```

---

## Summary Table

| Module | Trainable | Input Shape | Output Shape | Phase |
|---|---|---|---|---|
| VAE Encoder | No (frozen) | `[B,3,512,512]` | `[B,4,64,64]` | Used in all |
| Text Encoder | No (frozen) | `[B,77]` token ids | `[B,77,768]` | Phase A |
| Diffusion UNet | LoRA only | `[B,8,64,64]` + text | `[B,4,64,64]` noise | Phase A |
| VAE Decoder | Yes (full) | `[B,4,64,64]` | `[B,3,512,512]` | Phase B |
| Critic MLP | Yes (full) | `[B,8]` pooled latents | `[B,1]` logit | Phase C |
