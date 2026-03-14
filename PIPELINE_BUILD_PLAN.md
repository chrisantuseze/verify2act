# Verify2Act: Pipeline Build Plan

This document is the authoritative implementation guide for the Verify2Act inference and training
pipeline, derived from the research design conversations. It is intended to be handed directly to an
implementation agent. All architectural decisions, data shapes, training recipes, and interface
contracts are specified here.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Component Inventory](#2-component-inventory)
3. [Inference Pipeline (Full Detail)](#3-inference-pipeline-full-detail)
4. [Reflection Mechanism](#4-reflection-mechanism)
5. [Training Phases](#5-training-phases)
6. [Goal Image Acquisition](#6-goal-image-acquisition)
7. [Build Order and Milestones](#7-build-order-and-milestones)
8. [File/Module Layout](#8-filemodule-layout)
9. [Key Constants and Hyperparameters](#9-key-constants-and-hyperparameters)
10. [Validation Gates](#10-validation-gates)
11. [Out of Scope (Future Work)](#11-out-of-scope-future-work)

---

## 1. System Overview

Verify2Act is a critic-guided reflective planning system for robot manipulation. At each real
timestep the system:

1. Asks a VLM to propose a full action plan for the task horizon (one VLM call).
2. Chains the world model forward through each imagined step, evaluating feasibility with the critic
   at every step.
3. If the critic confidently predicts failure at any step, it triggers early termination and passes
   enriched reflection context back to the VLM which revises the plan.
4. If all steps pass, executes only the **first** action on the real robot (receding horizon
   control), then replans from the new real state.

```
[Real timestep t]

 ┌─────────────────────────────────────────────────────────────────────┐
 │  Stage 1 — Plan Generation (1 VLM call)                            │
 │                                                                     │
 │  Inputs:  current_image_real, goal_image, history                   │
 │  Output:  [a_0, a_1, ..., a_{H-1}]     (H = horizon length)        │
 └────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Stage 2+3 — Imagination + Critic Loop                              │
 │                                                                     │
 │  ŝ_0 = current_image_real                                           │
 │  For k = 0, 1, ..., H-1:                                           │
 │                                                                     │
 │    ŝ_{k+1} = WorldModel(ŝ_k, a_k)                                  │
 │    z_{k+1} = FrozenVAEEncoder(ŝ_{k+1})                             │
 │    (alpha, beta) = Critic([z_{k+1}, z_goal, z_{k+1} - z_goal])     │
 │    mean_f    = alpha / (alpha + beta)                               │
 │    uncert    = (alpha*beta) / ((alpha+beta)^2 * (alpha+beta+1))     │
 │                                                                     │
 │    if mean_f < θ_f  AND  uncert < θ_u:                             │
 │        → TRIGGER REFLECTION  (see Section 4)                       │
 │        → VLM revises plan → restart Stage 2+3 with new plan        │
 │        break                                                        │
 │                                                                     │
 │    elif mean_f < θ_f  AND  uncert >= θ_u:                          │
 │        → re-run WorldModel up to K times (default K=2)             │
 │        → if still uncertain after K retries: trigger reflection     │
 │                                                                     │
 │    else:  continue to k+1                                           │
 │                                                                     │
 │  If all H steps pass:                                               │
 │    Execute a_0 on real robot                                        │
 │    history.append(a_0)                                              │
 │    t += 1; new current_image_real = env.read_pixels()              │
 └─────────────────────────────────────────────────────────────────────┘
```

**Receding horizon rationale:** Only `a_0` is executed even when all H steps pass. This prevents
compounding sim-to-real mismatch across H sequential open-loop actions. The plan is refreshed from
the new real observation at every real timestep.

**History:** A list of action strings already executed on the real robot in this episode,
e.g. `["pick up red nut", "insert red nut"]`. The last 10 entries are included in every VLM prompt.
Updated only when an action is physically executed, never for imagined steps.

---

## 2. Component Inventory

### 2.1 VAE Encoder — always frozen

| Property | Value |
|---|---|
| Base | Stable Diffusion 1.5 VAE encoder (KL-regularised) |
| Input | `[B, 3, 512, 512]` float32, normalised to `[-1, 1]` |
| Output | `[B, 4, 64, 64]` float32 |
| Spatial compression | 8× (512 → 64 per dim) |
| SD scaling factor | `0.18215` (multiply after sampling) |
| Frozen | Always. Never updated in any phase. |

```python
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
vae.requires_grad_(False)

def encode_image(vae, image_tensor):
    # image_tensor: [B, 3, 512, 512], range [-1, 1]
    return vae.encode(image_tensor).latent_dist.sample() * 0.18215  # [B, 4, 64, 64]
```

The same VAE encoder instance is shared by the world model pipeline and the critic. This is the
foundation of the consistent latent space assumption.

---

### 2.2 Text Encoder — always frozen

| Property | Value |
|---|---|
| Base | CLIP ViT-L/14 text encoder (from SD 1.5) |
| Input | tokenised action text `[B, 77]` token ids |
| Output | `[B, 77, 768]` float32 |
| Frozen | Always. |

Used only by the world model UNet (cross-attention conditioning). Not used by the critic.

---

### 2.3 Diffusion UNet (World Model) — LoRA finetuned

| Property | Value |
|---|---|
| Base | `timbrooks/instruct-pix2pix` |
| UNet input | `[B, 8, 64, 64]` — channel-wise concat of noisy target latent (4ch) + source latent z_t (4ch) |
| Text conditioning | `[B, 77, 768]` via cross-attention |
| Output | `[B, 4, 64, 64]` predicted noise |
| Trainable params | LoRA adapters only (~2–8M, rank r=8 on attention projections) |
| Base weights | Frozen |

The UNet is used at inference via the full InstructPix2Pix diffusion pipeline
(`StableDiffusionInstructPix2PixPipeline`). It takes the current image and action text and produces
an imagined next-state image.

**Action text format:** Semantic natural language with spatial qualifiers for disambiguation.
No coordinates. InstructPix2Pix was pretrained on natural-language image editing instructions
(e.g. *"make the sky purple"*) and conditions on descriptions of visual change, not metric
positions. Coordinates are meaningless to it and degrade conditioning quality.

**Spatial qualifier scheme** (`prompt_utils.spatial_qualifier`, `policy_wrappers.NutAssemblyPolicyAdapter`):

When multiple nuts of the same type appear in the scene, a positional label is prepended to
disambiguate. The qualifier is derived from each nut's world-frame (x, y) position relative to all
same-type siblings:

- **x-axis:** robot's left (negative) → robot's right (positive) → label: `left` / `center` / `right`
- **y-axis:** near robot (negative) → far from robot (positive) → label: `front` / `middle` / `back`

The algorithm first checks whether the x-label alone is unique among siblings. If yes, only x is
used (e.g. `"left square nut"`). If two or more nuts share the same x-bucket, both axes are
combined (e.g. `"front-left round nut"`, `"back-right round nut"`). For nuts aligned in a
pure vertical column, only y is used (e.g. `"front round nut"`).

Examples produced by `build_action_prompt`:

```
"pick left square nut"         # 2 square nuts, x is sufficient
"pick front-left round nut"    # 4 round nuts in 2×2 grid
"insert square nut"            # only 1 square nut — no qualifier
"insert round nut"             # peg type implicitly matches nut type — no qualifier needed
```

**Qualifier uniqueness and image context:**
3-label bucketing (left/center/right, front/middle/back) cannot guarantee unique labels when N > 3
nuts share an axis. For example, 4 nuts in a column produce at most 3 distinct y-labels, so two
nuts may share `"front"`. This is acceptable because InstructPix2Pix conditions jointly on
`(image_t, action_text)` — at `pick_start` the robot arm is already positioned near the target
nut, and its shadow/proximity in the source image resolves any remaining label collision. This is
strictly better than ReflectVLM which uses no spatial qualifier at all for identical-shape objects.

**Known gap:** Verify the y-axis front/back direction in your agentview camera after the first
collection run. If "front" and "back" appear inverted in the rendered images, negate the y
coordinate when building `sibling_positions` in `NutAssemblyPolicyAdapter.get_action_info()`.

Coordinates are stored separately in `action_params.cartesian_target` in the transition manifest
and are consumed by the robot controller at execution time. The world model and controller are
fed different representations of the same action:

```
VLM plan: "pick up the front-left round nut"
                │
         ┌──────┴────────────────┐
         │                       │
  Diffusion world model    Robot controller
  "pick front-left          (x, y, z) from
   round nut"               action_params
  → visual conditioning
```

---

### 2.4 VAE Decoder — full finetune

| Property | Value |
|---|---|
| Base | Stable Diffusion 1.5 VAE decoder |
| Input | `[B, 4, 64, 64]` latent (de-scaled: divide by 0.18215 before passing) |
| Output | `[B, 3, 512, 512]` float32 range `[-1, 1]` |
| Trainable | Full weights (no LoRA needed — decoder is small ~49M params) |
| Encoder frozen during Phase B | Yes — encoder weights not touched, preserving latent space |

Finetuned separately from the UNet to improve spatial precision on small objects (nuts, pegs) that
the pretrained SD decoder blurs.

---

### 2.5 Critic — CNN + Beta head, trained from scratch

The critic is a Process Reward Model (PRM): it estimates the probability that the goal is still
reachable from imagined state `ŝ_{k+1}`.

**Input construction:**
```python
diff_map   = z_t1 - z_goal                              # [B, 4, 64, 64]
concat_map = torch.cat([z_t1, z_goal, diff_map], dim=1) # [B, 12, 64, 64]
```

**Architecture:**
```python
class CriticCNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )  # → [B, 32, 32, 32]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )  # → [B, 64, 16, 16]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )  # → [B, 128, 16, 16]   ← Grad-CAM target layer
        self.pool = nn.AdaptiveAvgPool2d(1)             # → [B, 128, 1, 1]

        self.alpha_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus()
        )
        self.beta_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus()
        )

    def forward(self, concat_map):  # [B, 12, 64, 64]
        x = self.conv1(concat_map)
        x = self.conv2(x)
        x = self.conv3(x)           # [B, 128, 16, 16]  — keep reference for Grad-CAM
        x = self.pool(x).flatten(1) # [B, 128]
        alpha = self.alpha_head(x)  # [B, 1]
        beta  = self.beta_head(x)   # [B, 1]
        return alpha, beta
```

**Derived quantities:**
```python
mean_feasibility = alpha / (alpha + beta)        # point estimate ∈ (0, 1)
uncertainty      = (alpha * beta) / (
    (alpha + beta)**2 * (alpha + beta + 1)
)                                                # predictive variance of Beta dist
```

**Why Beta distribution:** Naturally bounded [0,1]; concentration `(alpha + beta)` encodes
calibrated confidence; high uncertainty explicitly distinguishes "world model hallucinated something
implausible" from "confident failure". This is a distinct contribution from binary BCE critics.

---

### 2.6 VLM Planner

The VLM is **GPT-4o** accessed via the OpenAI Chat Completions API. The prompt system follows the
Points2Plans pattern: YAML-driven system prompts, YAML few-shot examples, and a `PromptManager`
class that assembles complete OpenAI Chat Completion message lists.

**Implementation:** `verify2act/pipeline/planner.py` (VLMPlanner class) and
`verify2act/pipeline/prompt_utils.py` (PromptManager, SystemPrompt, ExamplePrompt).

#### Prompt architecture (Points2Plans-style)

```
configs/prompts/
├── planner.yaml             # top-level config: wires system + example YAMLs
├── system/
│   ├── propose.yaml         # system prompt for planning calls
│   └── reflect.yaml         # system prompt for replanning calls
└── examples/
    ├── example_1.yaml       # few-shot: fresh start, 2 nuts
    ├── example_2.yaml       # few-shot: mid-episode with spatial qualifiers
    └── example_3.yaml       # few-shot: reflect behaviour (critic rejects plan)
```

Each system prompt YAML has the structure:
```yaml
behavior: propose          # or "reflect"
role: system
content: |
  I am the task-planning assistant for a robot nut-assembly task.
  ...
  Output format — output ONLY a JSON object:
    {"plan": ["action1", "action2", ...]}
```

Each example YAML has:
```yaml
task: example_1
objects: ["round nut", "square nut"]
history: []
plan: ["pick round nut", "insert round nut", ...]
role: system
name_query: example_user
name_response: example_assistant
```

#### Message assembly

`PromptManager.build_propose_messages()` produces:

| # | role | name | content |
|---|------|------|---------|
| 0 | system | — | System prompt from `propose.yaml` |
| 1 | system | example_user | Few-shot query (text-only) |
| 2 | system | example_assistant | Few-shot response (JSON) |
| … | … | … | (repeat for each example) |
| N | user | — | Multimodal: goal img + current img + history + request |

Images are passed as base64 `image_url` content blocks in the user message.

#### Usage

```python
from verify2act.pipeline.planner import VLMPlanner

planner = VLMPlanner.from_yaml("verify2act/configs/prompts/planner.yaml")

# Stage 1 — propose
plan = planner.propose(current_image_np, goal_image_np, history, obj_labels, horizon=5)
# plan: ["pick round nut", "insert round nut", ...]

# Stage 3 — reflect (after critic failure; ctx from build_reflection_context())
result = planner.reflect(current_image_np, goal_image_np, history, obj_labels, plan, ctx)
revised_plan = result["revised_plan"]
diagnosis    = result["analysis"]
```

The reflect message builder and `build_reflection_context()` are in Section 4.5.

---

## 3. Inference Pipeline (Full Detail)

### 3.1 Preprocessing

```python
def preprocess_image(img_np):
    # img_np: HxWx3 uint8 numpy array from env.read_pixels()
    img = Image.fromarray(img_np).resize((512, 512))
    img_tensor = transforms.ToTensor()(img)    # [3, 512, 512] in [0, 1]
    img_tensor = img_tensor * 2 - 1            # normalise to [-1, 1]
    return img_tensor.unsqueeze(0)             # [1, 3, 512, 512]
```

### 3.2 Step-by-step pseudocode

```python
def run_episode(env, vae, diffusion_pipeline, critic, planner, goal_image_np, horizon=5, max_steps=50,
                theta_f=0.4, theta_u=0.15, max_retries=2, max_replans=3):
    # planner: VLMPlanner instance
    history = []
    z_goal  = encode_image(vae, preprocess_image(goal_image_np))     # [1, 4, 64, 64]
    obj_labels = env.get_obj_labels()

    for t in range(max_steps):
        current_image_np = env.read_pixels()

        # ── Stage 1: generate full plan (1 VLM call) ──────────────────────────
        propose_prompt = get_propose_prompt(history, obj_labels, horizon)
        plan = vlm.act(current_image_np, goal_image_np, propose_prompt)
        # plan: list of H action strings, e.g. ["pick up red nut", "insert red nut", ...]

        # ── Stage 2+3: imagination + critic loop ──────────────────────────────
        action_executed = None
        for attempt in range(max_replans):
            imagined_img  = current_image_np
            all_scores    = []
            failed        = False

            for k, action_text in enumerate(plan):
                # World model forward pass
                imagined_img_next = diffusion_pipeline(
                    prompt=action_text,
                    image=Image.fromarray(imagined_img),
                    image_guidance_scale=1.5,
                    guidance_scale=7.5,
                    num_inference_steps=20,
                ).images[0]
                imagined_img_next_np = np.array(imagined_img_next)

                # Encode imagined state
                z_t1 = encode_image(vae, preprocess_image(imagined_img_next_np))  # [1, 4, 64, 64]

                # Critic evaluation
                diff_map   = z_t1 - z_goal
                concat_map = torch.cat([z_t1, z_goal, diff_map], dim=1)  # [1, 12, 64, 64]
                alpha, beta = critic(concat_map)
                mean_f  = (alpha / (alpha + beta)).item()
                uncert  = ((alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))).item()
                all_scores.append((mean_f, uncert))

                if mean_f < theta_f and uncert >= theta_u:
                    # Uncertain failure — retry world model
                    retry_scores = []
                    for _ in range(max_retries):
                        # re-run diffusion with same inputs (stochastic)
                        ...  # same as above
                        retry_scores.append((mean_f_r, uncert_r))
                    # pick the most confident evaluation among retries
                    best = min(retry_scores, key=lambda x: x[1])  # lowest uncertainty
                    mean_f, uncert = best
                    all_scores[-1] = (mean_f, uncert)
                    if mean_f >= theta_f:
                        imagined_img = imagined_img_next_np
                        continue  # retry resolved; proceed

                if mean_f < theta_f and uncert < theta_u:
                    # Confident failure — build reflection context and break
                    reflection_ctx = build_reflection_context(
                        imagined_state=imagined_img_next_np,
                        z_t1=z_t1, z_goal=z_goal, diff_map=diff_map,
                        critic=critic, concat_map=concat_map,
                        all_scores=all_scores, failed_step=k,
                        full_plan=plan
                    )
                    failed = True
                    break

                imagined_img = imagined_img_next_np  # chain forward

            if not failed:
                # All steps passed — execute first action
                action_executed = plan[0]
                env.act_txt(action_executed)
                history.append(action_executed)
                break
            else:
                # Reflect and get revised plan
                result = planner.reflect(
                    current_image_np, goal_image_np, history, obj_labels, plan, reflection_ctx
                )
                plan = result["revised_plan"]
                # plan is now the revised plan; loop back for re-evaluation

        if env.is_success():
            break
```

### 3.3 World model chaining

After each imagined step, `imagined_img` is updated to `imagined_img_next_np`. The next world model
call takes this as the source image. This means the world model sees imagined states, not real
states, for steps k > 0. This introduces compounding diffusion error, which is the primary reason
the critic must be evaluated at every step (early stop before error accumulates too much) rather
than only at the end of the horizon.

---

## 4. Reflection Mechanism

When the critic triggers a confident failure at step k, the system constructs enriched reflection
context at three layers of increasing detail. All three layers are passed to the VLM in the reflect
prompt.

### 4.1 Layer 1 — Trajectory trend analysis

```python
def classify_failure_pattern(all_scores):
    """
    all_scores: list of (mean_f, uncert) for steps 0..k
    Returns a human-readable failure pattern string.
    """
    scores = [s for s, _ in all_scores]
    k = len(scores) - 1

    if all(s < 0.4 for s in scores):
        return f"the initial planned action is fundamentally misaligned with the goal " \
               f"(feasibility was low from step 0)"
    
    delta = scores[k] - scores[k-1] if k > 0 else 0
    if delta < -0.3:
        kind = "sudden"
    else:
        kind = "gradual"
    
    return f"the plan was progressing until step {k}, where a {kind} failure occurred " \
           f"(scores: {[f'{s:.2f}' for s in scores]})"
```

### 4.2 Layer 2 — Spatial attribution from diff map

```python
def get_worst_region(diff_map):
    """
    diff_map: [1, 4, 64, 64] tensor
    Returns the 3x3 grid region name with the highest goal mismatch.
    The 64x64 latent grid is 8x downsampled — each of the 9 regions covers ~21x21 latent cells,
    corresponding to ~170x170 pixels in the original 512x512 image.
    """
    pixel_diff = diff_map.norm(dim=1).squeeze(0)  # [64, 64]
    
    region_labels = [
        ["top-left",    "top-center",    "top-right"   ],
        ["middle-left", "center",        "middle-right"],
        ["bottom-left", "bottom-center", "bottom-right"]
    ]
    
    H, W = pixel_diff.shape
    grid_scores = {}
    for row in range(3):
        for col in range(3):
            r0, r1 = row * H // 3, (row + 1) * H // 3
            c0, c1 = col * W // 3, (col + 1) * W // 3
            label = region_labels[row][col]
            grid_scores[label] = pixel_diff[r0:r1, c0:c1].mean().item()
    
    worst = max(grid_scores, key=grid_scores.get)
    return worst, grid_scores
```

### 4.3 Layer 3 — Grad-CAM overlay (visual grounding)

Grad-CAM is computed from the `mean_feasibility` scalar back to `critic.conv3` (the last conv layer
before the global pool), producing a `[16, 16]` spatial heatmap that is upsampled to 512×512 and
overlaid on the imagined state image.

```python
def compute_gradcam(critic, concat_map):
    """
    concat_map: [1, 12, 64, 64], already on the correct device
    Returns: heatmap as [512, 512] numpy array in [0, 1], and PIL overlay image
    """
    critic.eval()
    activations, gradients = {}, {}

    def save_activation(m, inp, out):
        activations["feat"] = out.detach()

    def save_gradient(m, grad_in, grad_out):
        gradients["feat"] = grad_out[0].detach()

    hook_a = critic.conv3[0].register_forward_hook(save_activation)   # Conv2d inside conv3
    hook_g = critic.conv3[0].register_full_backward_hook(
        lambda m, gi, go: gradients.update({"feat": go[0].detach()})
    )

    concat_map_req = concat_map.requires_grad_(True)
    alpha, beta = critic(concat_map_req)
    mean_f = alpha / (alpha + beta)
    mean_f.backward()

    hook_a.remove()
    hook_g.remove()

    weights = gradients["feat"].mean(dim=(-2, -1), keepdim=True)  # [1, 128, 1, 1]
    cam = (weights * activations["feat"]).sum(dim=1, keepdim=True).relu()  # [1, 1, 16, 16]
    cam = F.interpolate(cam, size=(512, 512), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalise to [0, 1]

    return cam


def make_gradcam_overlay(imagined_img_np, cam):
    """
    imagined_img_np: [512, 512, 3] uint8
    cam: [512, 512] float32 in [0, 1]
    Returns: PIL Image with heatmap overlaid
    """
    import matplotlib.cm as cm
    heatmap = cm.hot(cam)[:, :, :3]                       # [512, 512, 3] float
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    overlay = (0.5 * imagined_img_np + 0.5 * heatmap_uint8).astype(np.uint8)
    return Image.fromarray(overlay)
```

### 4.4 Full reflection context

```python
def build_reflection_context(imagined_state, z_t1, z_goal, diff_map, critic, concat_map,
                              all_scores, failed_step, full_plan):
    mean_f = all_scores[failed_step][0]
    uncert = all_scores[failed_step][1]
    failure_pattern = classify_failure_pattern(all_scores)
    worst_region, grid_scores  = get_worst_region(diff_map)
    cam = compute_gradcam(critic, concat_map)
    gradcam_overlay = make_gradcam_overlay(imagined_state, cam)

    return {
        "imagined_state":    imagined_state,           # numpy [512,512,3] — shown as <image> to VLM
        "gradcam_overlay":   gradcam_overlay,          # PIL Image — shown as <image> to VLM
        "mean_feasibility":  mean_f,
        "uncertainty":       uncert,
        "all_scores":        all_scores,               # [(mean_f, uncert), ...]
        "failure_pattern":   failure_pattern,
        "worst_region":      worst_region,
        "failed_step":       failed_step,
        "failed_action":     full_plan[failed_step],
        "full_plan":         full_plan,
    }
```

### 4.5 Reflect prompt

**Implementation:** `verify2act/pipeline/prompt_utils.py` → `PromptManager.build_reflect_messages()`
and `verify2act/pipeline/reflection.py` → `build_reflection_context()`.

The system prompt is loaded from `configs/prompts/system/reflect.yaml`. Few-shot reflect examples
(e.g. `example_3.yaml`) show a critic rejection + revised plan, enabling in-context learning.

#### Reflect user message structure

`build_reflect_messages()` produces a multimodal user message with 6 numbered sections:

| Section | Content |
|---------|---------- |
| 1. Task images | Goal image + current real state image (base64 `image_url` blocks) |
| 2. Execution history | Last 10 executed actions (text) |
| 3. Original proposed plan | Indexed action list (text) |
| 4. Critic diagnosis | Failed step, feasibility score, score trajectory, failure pattern, worst region (text) |
| 5. World model output | Imagined scene image + Grad-CAM attention overlay image |
| 6. Replanning instruction | Available objects + JSON output format spec |

#### Usage

```python
from verify2act.pipeline.reflection import build_reflection_context
from verify2act.pipeline.planner import VLMPlanner

# After critic flags a failure at step k:
ctx = build_reflection_context(
    imagined_state=imagined_img_np, z_t1=z_t1, z_goal=z_goal,
    diff_map=diff_map, critic=critic, concat_map=concat_map,
    all_scores=all_scores, failed_step=k, full_plan=plan,
)

result = planner.reflect(
    current_image_np, goal_image_np,
    history, obj_labels, plan, ctx,
)
revised_plan = result["revised_plan"]
diagnosis    = result["analysis"]   # logged to episode trace
```

---

## 5. Training Phases

### Phase A — UNet LoRA Finetune (World Model)

**Objective:** Train LoRA adapters to predict noise at each diffusion timestep conditioned on
`(z_t, action_text)`.

$$\mathcal{L}_A = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(\tilde{z}_{t+1}, z_t, \tau(a), t)\|^2\right]$$

**Data:** Transitions `(Image_t, action_text, Image_{t+1})` collected from robosuite using a mix of
expert and suboptimal policies. Action text is generated by `prompt_utils.build_action_prompt` which
produces semantic-only labels with spatial qualifiers for disambiguation (see Section 2.3). Each
transition spans one complete high-level primitive (e.g. the full `"pick front-left round nut"`
primitive from `pick_start` to `pick_end`), not a low-level timestep.

**LoRA config:**
```python
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.05,
    bias="none",
)
unet = get_peft_model(unet, lora_config)
```

| Hyperparameter | Value |
|---|---|
| Base | `timbrooks/instruct-pix2pix` |
| Resolution | 512 |
| Batch size | 2 (grad accum 4 → effective 8) |
| Learning rate | 2e-4 |
| LR schedule | Linear warmup 500 steps, then constant |
| Max steps | 10 000 (scale based on dataset size) |
| Mixed precision | fp16 |
| Optimizer | AdamW 8-bit |
| Image guidance scale | 1.5 |
| Text guidance scale | 7.5 |

**Saved:** LoRA adapter weights `unet_lora.safetensors` + config. (~10–30 MB)

---

### Phase B — VAE Decoder Finetune

**Objective:** Improve pixel-space precision on nut/peg geometry.

$$\mathcal{L}_B = \|I_{t+1} - \hat{I}_{t+1}\|_1 + 0.5 \cdot \mathcal{L}_\text{LPIPS}(I_{t+1}, \hat{I}_{t+1})$$

**Important:** The encoder and `quant_conv` are frozen throughout Phase B. Only the decoder and
`post_quant_conv` are updated. This preserves the latent space that the critic will be trained on.

**Can run in parallel with Phase A** since it only uses the frozen encoder.

| Hyperparameter | Value |
|---|---|
| Batch size | 8 |
| Learning rate | 5e-6 |
| Epochs | 5 |
| Optimizer | AdamW |
| Mixed precision | fp16 |

---

### Phase C — Critic Training

**Objective:** Train the CNN + Beta head critic from scratch as a feasibility classifier.

**Critical design decision:** The critic is trained on ground-truth rendered states from the
robosuite simulator, **not** on diffusion-generated states. This completely decouples critic
training from world model quality.

**Data collection procedure:**
1. Render ground-truth `(Image_t, action_text, Image_{t+1})` transitions in robosuite.
2. For each `Image_{t+1}`, determine the feasibility label: run the expert policy from that state;
   label `reachable=1` if the task succeeds within a fixed step budget, else `reachable=0`.
   `compute_labels.py` in `robosuite/data_capture_wm/` is the starting point for this.
3. Encode each `Image_{t+1}` and `goal_image` through the **frozen VAE encoder** to get `z_t1` and
   `z_goal`. Precompute and cache these as `.npy` files before training (one pass over all data).
4. Train critic on the cached embeddings.

**Loss — Beta NLL with label smoothing:**
```python
from torch.distributions import Beta

def beta_nll_loss(alpha, beta, y):
    # y: ground truth label, smoothed to [0.05, 0.95] to avoid log(0) issues
    y_smooth = y.clamp(0.05, 0.95)
    return -Beta(alpha, beta).log_prob(y_smooth).mean()
```

Always run a BCE baseline in parallel (same CNN backbone, sigmoid head) as a sanity check and lower
bound comparison.

| Hyperparameter | Value |
|---|---|
| Architecture | CriticCNN (Section 2.5) |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Epochs | 30 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Class weights | Inversely proportional to class frequency |
| Val split | 10% held-out by episode id (not by transition) |
| Label smoothing | positives→0.95, negatives→0.05 |

**Precompute embeddings once before training:**
```python
for img_t1, goal_img, label in dataset:
    z_t1   = encode_image(vae, preprocess_image(img_t1))    # [1, 4, 64, 64]
    z_goal = encode_image(vae, preprocess_image(goal_img))  # [1, 4, 64, 64]
    np.save(f"cache/z_t1_{idx}.npy",   z_t1.cpu().numpy())
    np.save(f"cache/z_goal_{idx}.npy", z_goal.cpu().numpy())
    # label cached separately
```

---

## 6. Goal Image Acquisition

The goal image is obtained directly from the robosuite environment at reset time:

```python
env.reset(seed=reset_seed)
goal_image_np = env.goal_images.get(camera_name, None)
# goal_image_np: [H, W, 3] uint8 numpy array
assert goal_image_np is not None
```

This is identical to how ReflectVLM obtains it (confirmed from their `run.py`). The environment
renders the goal configuration programmatically at reset. No separate goal-image pipeline is needed.

For real-robot deployment (future work): capture a reference image of the correctly assembled state
once during setup and use it as `goal_image_np`.

---

## 7. Build Order and Milestones

Build proceeds in strict sequence — each phase validates correctness before the next introduces a
new variable.

### Milestone 0 — Reproducible Setup ✅ COMPLETE

- Fix global seeds, train/val split by **episode id** (not transition id, to prevent leakage),
  output directory schema.
- Define canonical metric logger outputting CSV + JSON summary per run.
- Config templates for: `critic_bce`, `critic_beta`, `oracle_loop_eval`, `wm_swap_eval`.

Exit: Two repeated runs of the same config produce near-identical validation metrics.

---

### Milestone 1 — Critic Data + Embedding Cache ✅ COMPLETE

- Implement feasibility labelling via robosuite expert rollout (`compute_labels.py`).
- Encode all `(Image_{t+1}, goal_image)` pairs through frozen VAE encoder; cache to disk.
- Add integrity checks: missing files, NaNs, label coverage, class balance ratio.

Exit: Cache generation succeeds on full dataset; class stats logged.

---

### Milestone 2 — Critic Baselines ~~(skipped — baselines not required)~~

- ~~Train BCE baseline (same CNN backbone, sigmoid output head). Validate AUROC.~~
- Train Beta-head critic with label smoothing + numerically stable Beta NLL.
- Tune thresholds `(theta_f, theta_u)` on held-out validation set.
- Report: AUROC, AUPRC, F1 on infeasible class, ECE, Brier score, reliability diagram.

Exit: Beta critic reaches acceptable AUROC and ECE on validation.

---

### Milestone 3 — Oracle Planner Loop

Swap the world model out entirely. Use the robosuite simulator as an oracle: step the environment
for each imagined action to get ground-truth next states. This is Phase D=1 validation with a
perfect world model — it gives the upper bound on what the critic can achieve.

- Implement the full Stage 2+3 loop (Section 3.2) with oracle simulator.
- Wire up early termination and replanning trigger.
- Use a scripted or GPT-4o VLM for the reflect call (fast to prototype without finetuning).
- Log per-step decision traces, replanning frequency, success rate.

Exit: Early termination improves task success rate and/or reduces unnecessary world model calls
compared to an end-of-horizon-only evaluation baseline.

---

### Milestone 4 — World Model Training (Phases A + B) ✅ Data collection infrastructure complete — training pending

- ✅ Implement data collection script for `(Image_t, action_text, Image_{t+1})` transitions
  (`batch_collect.py`, `episode_recorder.py`, `prompt_utils.py`, `policy_wrappers.py`).
- ⏳ Run Phase A (UNet LoRA finetune).
- ⏳ Run Phase B (decoder finetune).
- ⏳ Run embedding-shift validation (Section 10) before using diffusion model with critic.

Exit: Mean cosine similarity between rendered `z_{t+1}` and diffusion-generated `ẑ_{t+1}` > 0.85;
domain classifier accuracy < 65%.

---

### Milestone 5 — Full Pipeline Integration

- Swap the oracle simulator in Stage 2+3 for the finetuned diffusion pipeline.
- Measure degradation vs. oracle upper bound.
- Validate that critic uncertainty correctly increases for diffusion-hallucinated states.

Exit: Controlled degradation vs oracle; no catastrophic failure modes; uncertainty metric tracks
hallucination quality.

---

## 8. File/Module Layout

Suggested layout under `verify2act/`:

```
verify2act/
├── configs/
│   └── prompts/
│       ├── planner.yaml          # top-level config wiring system + example YAMLs
│       ├── system/
│       │   ├── propose.yaml      # system prompt: planning behaviour
│       │   └── reflect.yaml      # system prompt: replanning behaviour
│       └── examples/
│           ├── example_1.yaml    # few-shot: fresh start, 2 nuts
│           ├── example_2.yaml    # few-shot: mid-episode, spatial qualifiers
│           └── example_3.yaml    # few-shot: reflect (critic rejects plan)
├── pipeline/
│   ├── __init__.py
│   ├── inference.py          # run_episode() and full Stage 1+2+3 loop
│   ├── planner.py            # VLMPlanner class (GPT-4o wrapper)
│   ├── prompt_utils.py       # PromptManager, SystemPrompt, ExamplePrompt
│   ├── world_model.py        # diffusion pipeline wrapper + oracle sim wrapper
│   └── reflection.py         # build_reflection_context(), compute_gradcam(),
│                             #   make_gradcam_overlay(), classify_failure_pattern(),
│                             #   get_worst_region()
├── critic/
│   ├── model.py              # CriticCNN definition
│   ├── train.py              # Phase C training loop
│   ├── loss.py               # beta_nll_loss(), bce_loss()
│   └── embed_cache.py        # precompute + cache z_t1, z_goal to disk
├── world_model/
│   ├── train_unet_lora.py    # Phase A training script
│   └── train_decoder.py      # Phase B training script (existing train_decoder.py)
├── data/
│   ├── collect_transitions.py  # robosuite data collection for world model training
│   └── compute_labels.py       # feasibility label generation for critic training
└── eval/
    ├── run_oracle_loop.py    # Milestone 3: oracle simulator loop evaluation
    └── run_full_pipeline.py  # Milestone 5: full pipeline evaluation
```

---

## 9. Key Constants and Hyperparameters

| Symbol | Default | Description |
|---|---|---|
| `H` | 5 | Planning horizon (number of steps per VLM call) |
| `theta_f` | 0.4 | Feasibility threshold below which failure may be triggered |
| `theta_u` | 0.15 | Uncertainty threshold: below this = critic is confident |
| `K` (max_retries) | 2 | Max world-model re-samples on uncertain failure |
| `max_replans` | 3 | Max reflect-replan cycles per real timestep before giving up |
| VAE scale | 0.18215 | SD convention scaling on VAE encoder output |
| SD resolution | 512 | Input/output image size for all modules |
| Latent size | `[4, 64, 64]` | VAE encoder output spatial dimensions |
| CNN Grad-CAM layer | `conv3` | Last conv before global pool — target for Grad-CAM |
| Label smoothing range | [0.05, 0.95] | Clamp range for Beta NLL training labels |

Thresholds `theta_f` and `theta_u` should be tuned on the validation set after Milestone 2;
defaults above are starting points only.

---

## 10. Validation Gates

### Embedding-shift validation (between Phase A and Milestone 5)

Before using the diffusion model with the critic in the live loop:

```python
# For N matched pairs (Image_t, action_text) → renders ground truth Image_{t+1}
# and generates diffusion Î_{t+1}

z_real = encode_image(vae, preprocess_image(Image_t1_real))
z_gen  = encode_image(vae, preprocess_image(Image_t1_diffusion))

# Metric 1: mean cosine similarity
cos_sim = F.cosine_similarity(
    z_real.flatten(1), z_gen.flatten(1), dim=1
).mean()

# Metric 2: domain classifier accuracy (linear probe)
# Train a logistic regression on [z_real, z_gen] labels to separate distributions.
# Lower accuracy = more similar distributions = less shift.

# Accept if:
assert cos_sim > 0.85, "Embedding shift too large; add alignment loss in Phase A"
assert domain_cls_acc < 0.65, "Embedding shift too large"
```

If shift is too large: add an MMD or cosine alignment loss between `z_real` and `z_gen` as an
auxiliary objective in Phase A, or collect more diverse transition data.

### Critic calibration validation (Milestone 2)

Report reliability diagram (expected calibration error, ECE) in addition to AUROC. A well-
calibrated Beta critic should have ECE < 0.05 on the validation set. If ECE is poor but AUROC is
good, the point estimate is discriminative but confidence values are unreliable — Grad-CAM and
uncertainty-gated replanning will not behave as designed.

---

## 11. Out of Scope (Future Work)

The following are identified contributions reserved for later:

### VLM Finetuning

The VLM can be used with a strong API model (GPT-4o, Gemini) zero-shot for prototyping. If
finetuning is pursued:

- **Stage 1 (base policy):** Finetune LLaVA-1.5-13b on expert demonstrations. Start from
  `yunhaif/ReflectVLM-llava-v1.5-13b-base` if the task vocabulary is similar.
- **Stage 2 (reflection policy):** Post-train on reflection pairs collected via DAgger-style
  rollouts. Training example: `(goal_image, current_image, imagined_state_at_k, gradcam_overlay,
  enriched_text_context) → corrected_plan`. This requires Stage 1 to be complete and a working
  world model + critic to generate supervision signal.

The Grad-CAM overlay input to the VLM specifically requires Stage 2 finetuning — no pretrained VLM
understands red-hot heatmaps on robot scenes out of the box.

### Direction B — Joint World Model + Critic Co-Training

After the full pipeline is working (Milestone 5), backpropagate the critic loss into the UNet LoRA
adapters:

```python
total_loss = L_reconstruction + lambda_critic * L_critic
total_loss.backward()  # gradients flow through critic AND world model UNet
```

This makes the world model feasibility-aware: it learns to generate states that are maximally
discriminative for the critic, not just visually plausible. Requires GPU memory audit before
committing — backprop through a diffusion UNet is expensive.
