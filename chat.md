User: 
Claude finished the response
You said: The image generation quality is poor.

train_wm.py
py


data_loader.py
271 lines

py

The image generation quality is poor.  what am I missing? `

This is what a transition looks like:

{"episode_id": "ep_17658", "timestep": 724, "image_t": "episodes/ep_17658/frame_00724.png", "image_t1": "episodes/ep_17658/frame_00917.png", "goal_image": "episodes/ep_17655/goal.png", "goal_image_source": "fallback", "action_text": "insert left round nut", "action_params": {"skill": "insert", "object": "left round nut", "cartesian_target": [0.07854, -0.09973, 0.97724]}, "state_t": "episodes/ep_17658/state_00724.npz", "state_t1": "episodes/ep_17658/state_00917.npz", "policy_type": "expert", "policy_stage_t": "move_to_peg", "policy_stage_t1": "reset_orientation", "event_tag_t": "pick_end|insert_start", "event_tag_t1": "insert_end", "source_timestep_t": 724, "source_timestep_t1": 916, "episode_success": false, "label_reachable": -1}
Show more
11:01 AM
Claude responded: There are several issues, ranging from a hard bug to data-level problems.

Diagnosed critical data and training configuration issues
Diagnosed critical data and training configuration issues
There are several issues, ranging from a hard bug to data-level problems. Here's a precise breakdown ranked by severity:

1. z_t is not scaled (hard bug, highest impact)
In training (and eval):

python
z_t  = vae.encode(image_t).latent_dist.mode()         # ← NOT scaled
z_t1 = vae.encode(image_t1).latent_dist.sample() * latent_scale  # ← scaled
model_input = torch.cat([noisy_z_t1, z_t], dim=1)
IP2P concatenates 8 channels: [noisy_z_t1 | z_t]. The noisy target is properly scaled by the VAE scaling_factor (~0.18215 for SD), but the conditioning latent z_t lives in raw encoder space — roughly 5–6× larger in magnitude. The UNet is now seeing wildly inconsistent channel statistics across its input, which severely corrupts the conditioning signal. Fix:

python
z_t = vae.encode(image_t).latent_dist.mode() * latent_scale
2. Massive temporal gaps in your transitions (data-level, critical)
Your example: frame_00724 → frame_00917 — Δt = 193 frames. A world model trained on ε-prediction over a diffusion process is a single-step predictor. Asking it to bridge 193 frames of robot manipulation is intractable — the state distribution at t+193 given only t and a text action has enormous variance, so the model optimally converges to predicting a blurry/mean image (high perceptual entropy, low MSE but poor quality).

You need to either:

Filter transitions to a small Δt window (e.g., Δt ≤ 5 for dense prediction, or the subskill boundary if you're doing keyframe prediction)
Make the Δt explicit as a conditioning signal so the model can modulate uncertainty
3. lora_alpha < lora_rank — effectively halved update scale
lora_rank=64, lora_alpha=32  →  effective scale = alpha/rank = 0.5
LoRA applies updates as (alpha/rank) * BA. At 0.5×, you're systematically underscaling all learned residuals. Especially at rank 64, this means your adapter has capacity but can't use it. Set lora_alpha=64 (or at minimum equal to rank). The comment #16 next to the default suggests these were copied from a rank-16 config without updating alpha.

4. EMA decay of 0.9999 is too aggressive for 10k steps
At decay=0.9999, the EMA shadow weights require ~10,000 steps just to integrate half the training signal. With only 10k max_steps and effective batch size of 8, the EMA model barely moves. Use 0.999 or 0.9995 for runs this short, otherwise your best checkpoint is checkpointing the pretrained weights with a thin perturbation.

5. Most training data is episode_success: false
Your example has episode_success: false. If this is representative of your dataset, the world model is predominantly learning transition dynamics of failed manipulation episodes. This isn't a correctness bug per se, but the model will learn to predict trajectories toward failure states, which degrades quality for downstream planning/verification tasks like Unveiler. Filter to successful episode transitions, or at least stratify sampling.

Summary table
Issue	Location	Severity
z_t unscaled in 8-ch input	train_wm.py:478, evaluate:181	🔴 Bug — corrupts conditioning
Δt = 193 frames	transitions.jsonl	🔴 Data — makes task intractable
lora_alpha=32 < rank=64	train_wm.py:756	🟠 Suboptimal — halved LoRA capacity
ema_decay=0.9999 too high	train_wm.py:779	🟠 Suboptimal for 10k steps
Failed-episode data dominance	dataset	🟡 Task-dependent
Fix #1 and #2 first — the quality improvement should be substantial.

You said: So this is what a GPT-codex said regarding point 1:
generated_collage_v8.png
So this is what a GPT-codex said regarding point 1:

# z_t is the conditioning image latent: must NOT be scaled,
                # matching IP2P's inference pipeline (prepare_image_latents)
                # which passes raw VAE latents without scaling_factor.
                # Also use mode() (deterministic) to match inference.
point 2: The image contains the input (left), the groundtruth (middle) and the predicted (right). As you can see, the prediction isn't blurred, it just deviates from the actual groundtruth. Also, the previous work (ReflectVLM) predicted actions like pick, grasp, place, etc. and not subskills.

point 3: This is the actual training cmd args:
accelerate launch --num_processes 3 \
  verify2act/world_model/train_wm.py \
  --dataset-dir robosuite/data_capture_wm/dataset/nut_assembly_merged \
  --transitions-file transitions.jsonl \
  --output-dir verify2act/output/wm_v8 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lora-target-ff \
  --lora-target-conv \
  --change-mask-weight 5.0 \
  --change-mask-threshold 15.0 \
  --train-batch-size 3 \
  --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 \
  --max-steps 20000 \
  --eval-every 2000 \
  --eval-batches 20 \
  --snr-gamma 5.0 \
  --use-ema \
  --ema-decay 0.9999 \
  --val-frac 0.2 \
  --enable-gradient-checkpointing \
  --mixed-precision fp16 \
  --enable-xformers \
  --tracker none

point 5: That isn't a representative of my training data

Show more
11:11 AM
Claude responded: Fair corrections — let me revise accordingly.

Recalibrated analysis after acknowledging prior errors
Recalibrated analysis after acknowledging prior errors
Fair corrections — let me revise accordingly.

On point 1: You're right. IP2P's prepare_image_latents intentionally passes unscaled VAE latents for the conditioning channels. The asymmetry is by design — the noisy target lives in scaled latent space, the conditioning image lives in raw encoder space. That point is retracted.

On point 3: With --lora-rank 32 --lora-alpha 32 you have scale = 1.0, which is correct. The default args in the file are misleading but irrelevant to your actual run.

On point 5: Acknowledged.

Now, looking at the actual collage — the failure mode is clearer than I initially characterized:

The generated image is sharp and structurally coherent, but it's essentially predicting minimal change from the input. The robot arm barely moves, and the table objects stay near their input positions, whereas ground truth shows significant arm displacement and object rearrangement. This is the input-copying problem, not a blurriness problem.

What's actually causing this
1. The large Δt is still the root cause — but not via blurriness

At Δt=193, the mapping (frame_724, "insert left round nut") → frame_917 is extremely many-to-one from the model's perspective, because the arm trajectory, grasp dynamics, and object contact physics between those 193 frames are unobserved. The model can't recover a deterministic target. Under MSE-on-noise training, the optimal solution under high-variance targets is to push the prediction toward the mean of the conditional distribution — which, for IP2P with strong image conditioning, collapses to the input image. You get a sharp-but-static prediction rather than a blurry one because the sharpness comes from copying z_t, not from learning the transition.

2. Image conditioning dominance is reinforcing this

The unscaled z_t channels have ~5–6× higher magnitude than the noise-corrupted z_t1 channels at mid-to-high diffusion timesteps (since raw VAE encoder std ≈ 1/scaling_factor ≈ 5.5). The UNet is effectively seeing the conditioning image louder than the noisy target at many timesteps. Combined with the high-variance target from large Δt, gradient descent settles on "reconstruct the conditioning input" as the path of least resistance. Your --change-mask-weight 5.0 partially counteracts this, but the mask needs a coherent target to focus on — which it doesn't have at Δt=193.

3. Image conditioning dropout probability may be insufficient

Your current implementation:

python
# random_p ∈ [0,1], p = 0.05
image_mask = ((random_p >= p) & (random_p < 3.0 * p))  # prob ≈ 10%
10% image dropout may not be enough to force the model to actually learn the text-conditioned transition without leaning on z_t. The original IP2P paper uses ~15% total with overlapping regions. More aggressively zeroing z_t (e.g., 15–20%) during training discourages the copy shortcut.

4. 20k steps at effective batch size ~18 (3 × 2 × 3 GPUs) is borderline

For a 512×512 IP2P fine-tune on robotics transitions with LoRA targeting QKV + FF + conv, 20k steps ≈ 360k image pairs seen. This is likely enough to fit, but not enough to generalize well if the target distribution is high-variance. Extending to 40–50k steps or increasing batch size would help, but only after fixing the Δt issue — more steps on bad data amplifies the copy-input bias.

What to actually fix
The highest-leverage intervention is filtering or bucketing by Δt. Even if you're predicting subskill boundaries rather than single-step transitions, you should identify what your actual Δt distribution looks like:

python
# Quick diagnostic
import json, numpy as np
with open("transitions.jsonl") as f:
    rows = [json.loads(l) for l in f]
dts = [abs(r["source_timestep_t1"] - r["source_timestep_t"]) for r in rows]
print(np.percentile(dts, [25, 50, 75, 95, 99]))
If the distribution is skewed toward large gaps, either cap at a max Δt threshold or train separate models per subskill boundary type (which aligns with your policy_stage_t → policy_stage_t1 field — e.g., move_to_peg → reset_orientation is a well-defined, bounded transition).

The second fix is bumping --conditioning-dropout-prob to 0.10–0.15 to weaken the copy-input shortcut during training.

New Chat from another model given your (Claude) responses:

Don't make any code changes, let's reason through the recommendations first.

GitHub Copilot: Let me get the terminal output from that diagnostic you already ran: 

Checked terminal output

Ran terminal command:  python3 - <<'PY'
import json, numpy as np
with open("robosuite/data_capture_wm/dataset/nut_assembly_merged/transitions.jsonl") as f:
    rows=[json.loads(l) for l in f]
dts=[abs(r["source_timestep_t1"]-r["source_timestep_t"]) for r in rows]
print("Δt percentiles [25,50,75,95,99]:", np.percentile(dts,[25,50,75,95,99]))
print(f"Total transitions: {len(rows)}")
print(f"Δt min/max: {min(dts)}/{max(dts)}")

# Also break down by policy_stage transition type
from collections import Counter
stage_pairs = Counter()
for r in rows:
    k = f"{r.get('policy_stage_t','?')} → {r.get('policy_stage_t1','?')}"
    stage_pairs[k] += 1
print("\nStage-pair distribution:")
for k, v in sorted(stage_pairs.items(), key=lambda x: -x[1]):
    print(f"  {v:5d}  {k}")
PY


The Δt data is the key diagnostic. Here's what it reveals and how to reason through each recommendation:

**The data confirms this is a keyframe prediction task, not a transition prediction task**

The minimum Δt is 145 frames, 50th percentile is 216. There are no short-horizon transitions at all — every sample spans an entire subskill execution. And there are only two stage-pair types:

- `move_to_nut → move_to_peg`: 43,817 samples (76%)
- `move_to_peg → reset_orientation`: 13,765 samples (24%)

This reframes the problem: the model isn't being asked "what happens one step forward?" — it's being asked "given the start of a subskill, what does the robot's end state look like?" That's a harder, more structured prediction task, but it's also more tractable than it appears, because the expert policy is deterministic.

**Why the model copies the input (precise mechanism)**

With Δt ~210 frames, the target frame is a genuinely distant future state. However, the target is actually *low-variance conditioned on object positions in `image_t`* — because the expert policy always completes the subskill. The problem is the model isn't extracting the spatial structure from `z_t` well enough to use it for prediction. Under MSE-on-noise training, the path of least resistance is: copy `z_t` (which has high magnitude since it's unscaled) for static background regions, and produce low-confidence noise for the arm region. The change-mask-weight is supposed to counteract this, but the "changed" region is the *entire arm trajectory endpoint* — the mask identifies it correctly, but the model still can't predict where the arm ends up.

**What's actually missing: spatial grounding**

The text conditioning ("insert left round nut") is the same for all ~44k `move_to_nut → move_to_peg` transitions. The distinguishing information about *where* the arm ends up is entirely encoded in `image_t` — specifically, the object positions, which determine the Cartesian target. But your transitions already have `action_params.cartesian_target` ([x, y, z]). This is unused.

Prepending the Cartesian target to the text prompt is likely the highest-leverage data-side fix: `"move to (0.079, -0.100, 0.977): insert left round nut"` gives the UNet a direct spatial grounding signal through text cross-attention, rather than expecting it to infer geometry from pixels alone.

**The other two levers**

The inference guidance scales likely matter more than anything in the training config. IP2P has two CFG axes: `text_guidance_scale` and `image_guidance_scale`. If you're running inference at `image_guidance_scale > 1.5–2.0`, the model will strongly copy the conditioning image regardless of what it learned. This is worth checking before any retraining.

The conditioning dropout at 5% for both text and image means 90% of training batches see both signals simultaneously — the model never has to rely on either one independently. Bumping to 10–15% image dropout specifically would force the model to learn text-conditioned pathways more aggressively, reducing the copy-input shortcut.

**Recommended priority order**

1. Check inference `image_guidance_scale` — cheap, immediate diagnostic
2. Add cartesian target to text prompts (data pipeline change) + retrain
3. Increase image conditioning dropout to 0.10–0.15 (one-line change in training args)

The stage-pair split is also worth considering: training two separate LoRA adapters (one per stage-pair) would give each model a more homogeneous distribution and remove the need to disambiguate subskill type from the image alone.

User: I don't know if adding cartesian target works, since I don't have access to it during inference.

GitHub Copilot: Good point — that eliminates cartesian target as a conditioning signal. Let me check what's actually available in your inference pipeline before suggesting anything else.  

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline)

Read [](file:///home/scratch1/cheze/verify2act/verify2act/world_model)

Read [](file:///home/scratch1/cheze/verify2act/verify2act/world_model/demo_wm.py#1-1), lines 1 to 80

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline/world_model.py#1-1), lines 1 to 100

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline/world_model.py#100-100), lines 100 to 200

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline/world_model.py#200-200), lines 200 to 350

Right, that closes off the cartesian target idea cleanly. Let me look at your inference call to understand the full picture. 

Read [](file:///home/scratch1/cheze/verify2act/verify2act/world_model/sweep_guidance.py#1-1), lines 1 to 60

Now I have a complete picture. Here's the honest assessment:

**The core constraint is structural, not a tuning problem.**

Your inference interface is `imagine(image_t, action_text)`. With 44k transitions all using identical text "insert left round nut", the *only* information the model has to predict where the arm ends up is the pixel content of `image_t`. The text provides zero spatial grounding — it just disambiguates "which subskill" (though even that isn't unique since both stage pairs use the same text). This means you're asking the model to:

1. Visually localize the nut from `image_t`
2. Reason about kinematics to predict the arm endpoint
3. Render a full 512×512 image of that endpoint state

That's a hard visual perception + dynamics problem, and IP2P wasn't designed for it.

**What's actually actionable without changing the inference interface:**

**1. Add stage label to the action_text prompt (zero inference cost)**  
You know at inference time which stage you're executing (you have to, to call the policy). The text "move_to_nut→move_to_peg: insert left round nut" vs "move_to_peg→reset_orientation: insert left round nut" is real information the model currently doesn't have — it has to infer stage type from the image alone. This is a data pipeline change + retrain, but adds no inference-time requirements.

**2. `image_guidance_scale` at inference is your strongest free lever**  
You're using 2.8 by default. Higher IGS amplifies the "copy input" tendency because it strengthens the image conditioning signal relative to the noisy target. For a task where the arm moves substantially, try 1.2–1.8. This doesn't require retraining. Your sweep_guidance.py already supports this — have you run it on your v8 checkpoint?

**3. The deeper question: what does the downstream critic need?**  
If the critic/verifier only needs to answer "did the arm approximately reach the target?" and not pixel-perfect reconstruction, then even an imperfect prediction may be useful. The question is whether the model's copy-input failure mode produces images that systematically mislead the critic into a false positive (arm didn't move, critic says success). If so, the quality problem is critical for the pipeline; if the critic is robust to it, it's less urgent.

The practical path forward is (2) first (costs nothing, run today), then (1) as a retraining experiment.

User: I previously considered subskill as you may know (see a trace of it), but the reason I chose to veer off that path is the fact that it was too granular an unnatural to instruct the robot on the entire subskills. Plus at the time, I really didn't see how it improved the model. what do you think? Give me your honest evaluation/recommendation.

GitHub Copilot: Your reasoning to abandon subskills was correct, but let me be precise about *why* it was correct, because the reason matters for what you do next.

**Why subskills were right to drop**

The inference interface is `imagine(image_t, action_text)`. For subskill conditioning to work at inference, you'd need to know whether you're currently in "approach", "carry", etc. — which requires either a stage classifier on the live observation or access to policy internals. Neither is available in the `DiffusionWorldModel` path. Chaining 4–5 subskill predictions per skill would also compound errors. And the planner reasons at the skill level ("insert left round nut"), not the subskill level — the interface would be mismatched with how the planner generates action_text.

**But here's the honest problem subskills would have actually solved**

The real reason subskills might have improved generation quality isn't granularity — it's *text discriminability*. Right now, 44k training samples share essentially the same action_text. The text conditioning carries ~0 bits of information distinguishing one `move_to_nut → move_to_peg` sample from another. "approach left round nut" vs "carry left round nut" are genuinely different visual transformations with different action texts — the model could actually use that signal. You dropped it for the right reason (inference interface), but the underlying information problem it would have solved is real and unsolved.

**The honest diagnosis**

Your world model is effectively doing *text-unconditional* image-conditioned generation, because the text is degenerate across the training distribution. The copy-input bias isn't a training failure — it's the Bayes-optimal response to a near-zero-information text signal combined with high-variance targets.

**What's actually available at inference that's informative?**

Looking at your `NutAssemblyPolicyAdapter`: during pipeline execution with the oracle setup, `policy.stage` IS accessible. The `OracleWorldModel` already uses `execute_action(action_text)` which implies the policy is running. If your `DiffusionWorldModel` inference path runs alongside the heuristic policy (even just for stage determination), the current stage label is obtainable.

The recommendation: enrich action_text with the **stage transition label** (`policy_stage_t` → `policy_stage_t1`) during both training *and* inference, where the stage is read from the running policy. This satisfies the inference constraint (stage is known when the policy is executing), makes text conditioning discriminative between the two distinct visual transformations in your data, and doesn't require any architectural change — just a text formatting change.

The format would be something like: `"[move_to_nut→move_to_peg] insert left round nut"`. At training time this comes from the JSONL field already populated. At inference time it comes from `adapter.policy.stage` before and after the action executes.

If the inference path truly has no access to policy stage — e.g., you want to run the diffusion world model standalone without a running policy — then you're stuck, and the quality ceiling is fundamentally limited by the information problem.

User: won't all samples for that skill basically end up having the same "[move_to_nut→move_to_peg]" prefix? Thats the same path for everything, isn't it?

GitHub Copilot: Yes, you're exactly right. Within each bucket:
- All 43,817 `move_to_nut→move_to_peg` samples → same prefix
- All 13,765 `move_to_peg→reset_orientation` samples → same prefix

The stage label distinguishes the two buckets from each other, but adds zero bits within each bucket. And since 76% of your data is a single bucket, it barely moves the needle.

**The honest conclusion**

Your action_text is already about as informative as it can be given what's available at inference. The spatial qualifier ("left round nut", "right round nut") is already there. Stage label would help marginally. But none of it solves the core problem: the distinguishing information about where the arm ends up in any given transition is *purely visual* — it lives in the pixel positions of the nut and peg in `image_t`, not in any text signal you can construct.

This means the model fundamentally has to learn to decode spatial structure from `image_t` and use it for prediction. IP2P with LoRA on attention layers *can* in principle do this through cross-frame spatial attention — but it needs enough capacity and enough signal to learn to do so, rather than taking the copy-input shortcut.

**Where does this leave you?**

The copy-input bias isn't going away through text enrichment. The real levers are:

1. **Guidance scale at inference** — still the cheapest thing to test. Lower `image_guidance_scale` (1.0–1.5) reduces the copy-input pull and forces the model to commit to what it learned. Have you swept this on v8?

2. **More aggressive image conditioning dropout during training** — currently 10%, pushing to 20–25% forces the model to learn image-independent pathways, which paradoxically makes it use the image *better* at inference rather than just copying it.

3. **Accept the quality ceiling and focus on the critic** — if the downstream goal is verification ("did the action succeed?"), the world model doesn't need to be pixel-accurate. It needs to produce images that the critic can correctly classify. A blurry or shifted prediction can still carry enough structural signal for a good critic. The question is whether your copy-input failure produces false positives *specifically* for the critic, which is a different question than perceptual quality.

User: Lets thoroughly consider other options for enriching the text both during training and inference.

Granted during inference we send the current image + goal to the VLM and it returns a list of actions to us. Is there a way to use these list and add more spatial attributes to it so we can enrich the textual signal to the wm? 

What do you think?

GitHub Copilot: I’ll inspect how your VLM action list is produced and consumed so I can propose text-enrichment options that are actually available at both training and inference time.

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline/inference.py#1-1), lines 1 to 260

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline/decompose.py)

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline/prompt_utils.py#1-1), lines 1 to 320

I found the planner path: it outputs nut ordering, and your world model text currently comes from decomposition templates. Next I’ll inspect planner prompts and the world-model call site to propose enrichment that fits your actual runtime.

Searched for regex `expand_nut_plan|world_model.imagine|sub_skill_prompt|execute_nut_assembly|plan` (`**/verify2act/pipeline/inference.py`), 20 results

Searched for files matching `**/verify2act/configs/prompts/**/*.yaml`, 6 matches

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline/planner.py)

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline/inference.py#260-260), lines 260 to 520

Read [](file:///home/scratch1/cheze/verify2act/verify2act/configs/prompts/system/propose.yaml)

Read [](file:///home/scratch1/cheze/verify2act/verify2act/configs/prompts/planner.yaml)

Searched for regex `class PromptManager|build_propose_messages|build_reflect_messages|Available nuts|obj_labels|history_str` (`**/verify2act/pipeline/prompt_utils.py`), 20 results

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline/prompt_utils.py#320-320), lines 320 to 525

Read [](file:///home/scratch1/cheze/verify2act/robosuite/data_capture_wm/prompt_utils.py)

I confirmed prompts already carry left/right qualifiers from object labels; I’ll quickly verify what labels are available at runtime and then lay out concrete enrichment schemes you can use without cartesian targets.

Searched for regex `def get_obj_labels|spatial_qualifier|current_nut|round nut|front-left` (`**/verify2act/pipeline/env_wrapper.py`), 15 results

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline/env_wrapper.py#1-1), lines 1 to 260

Read [](file:///home/scratch1/cheze/verify2act/verify2act/pipeline/env_wrapper.py#260-260), lines 260 to 493

Yes, and this is a strong direction.

Short answer: enriching WM text from the VLM plan can help, but only if the enrichment is available both at train and inference and is actually discriminative per sample. You have at least 3 good options.

Most important finding first: there is currently a train-inference text mismatch that likely hurts a lot.

- Training samples include labels like “insert left round nut” (your JSONL example).
- In inference, available labels are currently coarse (“round nut” / “square nut”) in env_wrapper.py.
- So the planner cannot output spatially specific nut names, because the “Available nuts” list is already collapsed.
- Also, your current loop is temporarily hardcoded to one plan entry in inference.py, which bypasses planner diversity entirely.

My honest recommendation is to fix this first before any fancy prompt engineering.

What you can do to enrich text without cartesian targets

1. Recover object identity tokens from vision, then use them end-to-end
- Goal: make action text identify a specific nut instance, not just type.
- Example style: “insert nut_A_round” instead of “insert round nut”.
- How:
  - At each episode start, assign stable IDs to visible nuts from image (or from env metadata in sim): nut_A, nut_B.
  - Planner outputs IDs, not generic type names.
  - WM prompt uses the same IDs at train and inference.
- Why this helps:
  - It adds discriminative information that is available at inference.
  - Unlike “move_to_nut→move_to_peg”, it separates samples within the same skill bucket.

2. Add relational language derived from current+goal images (no coordinates)
- Goal: give WM text explicit spatial intent that is inferable from images.
- Example style:
  - “insert left round nut toward right peg”
  - “insert round nut that is farther from its peg first”
- How:
  - Build a lightweight “prompt refiner” stage:
    - Inputs: current image, goal image, chosen nut label from plan.
    - Output: compact relational phrase from a controlled vocabulary.
  - Append this phrase to WM prompt.
- Critical requirement:
  - Generate the same relational tags offline for training rows too, using the same refiner pipeline.
- Why:
  - You avoid unavailable robot-state variables while still adding geometry-relevant text.

3. Add plan-context tokens (sequence position + remaining set)
- Goal: give WM temporal context that can disambiguate expected appearance.
- Example style:
  - “insert left round nut | step 1 of 2 | remaining: right round nut”
- How:
  - At inference: you already have full plan list and history.
  - At training: reconstruct synthetic plan context from episode timeline fields.
- Why:
  - It is easy and fully inference-available.
  - Helps WM distinguish early-vs-late episode visual priors.

What not to prioritize
- Stage-prefix alone (like move_to_nut→move_to_peg) is weak in your data because most rows share the same transition bucket.
- Generic wording changes (synonyms, longer prose) usually add little signal.

Practical phased plan

Phase A: no model retrain, quick diagnostics
1. Un-hardcode planner output in inference.py.
2. Make available nut labels instance-specific (not only “round nut”) in env_wrapper.py.
3. Run a guidance sweep to check how much copy-bias is pure inference-time CFG.

Phase B: smallest retrain with high upside
1. Add stable nut ID token to action text in dataset creation and inference prompts.
2. Keep vocabulary compact and templated to reduce language noise.
3. Retrain and compare critic-facing metrics, not only visual fidelity.

Phase C: stronger enrichment
1. Add relational phrase from current+goal image via a deterministic tagger or a small VLM refiner.
2. Generate identical tags for train and inference.

Bottom line
- Yes, you can use the VLM action list to enrich WM text.
- The best path is not “more words”, but “more discriminative, inference-available tokens”.
- Instance identity + relational tags are the highest-value additions.
- Your current coarse label collapse in env_wrapper.py is likely a major bottleneck and worth fixing immediately.

User: Here are my evaluations:

point 1: I don't think this adds anything actually since we already have something similar as in "insert left round nut" for instance.

point 2: We don't want to manually determine the assemblly order. We want the VLM to handle that. 

point 3: This looks interesting. Claude, what do you think about this proposal made by GPT-5.3-Codex?

GitHub Copilot: My honest evaluation: it's syntactically appealing but semantically weak for this task.

**Why it doesn't actually help the world model**

The WM needs to predict what the scene looks like after the arm completes "insert left round nut." That outcome depends entirely on the spatial configuration visible in `image_t` — where the nut is, where the peg is, and where the arm ends up. Neither "step 1 of 2" nor "remaining: right round nut" has any causal relationship to that visual outcome. The right round nut's existence elsewhere in the scene is already *visible* in `image_t`; the WM doesn't need it stated in text.

What you'd be doing is adding tokens that are unique per sample but uncorrelated with the prediction target. The model might latch onto them as spurious features, which is worse than ignoring them.

**The one narrow case where it could help**

If episode length (number of nuts) correlates with visual scene complexity at a dataset level, then "step 1 of N" might weakly encode that. But that's already in `image_t` as visible clutter, and the effect would be trivially small.

**The deeper problem this proposal doesn't solve**

The copy-input bias isn't about text diversity — it's about the model finding a gradient shortcut. Adding more text tokens doesn't close that shortcut; lowering `image_guidance_scale` at inference or increasing image conditioning dropout during training directly addresses the mechanism. Those are the two levers actually worth pulling before any data-side changes.

Have you run sweep_guidance.py on v8 yet? That's the cheapest diagnostic and the answer will tell you whether this is a training failure or an inference-time CFG issue.

User: Yes, I have ran it. 2.5 was the best for the input image, but for another input image, it was awful, although better than 1.5.

GitHub Copilot: That result is actually very informative — and it shifts the diagnosis significantly.

**What the sweep result tells you**

Per-sample inconsistency at a fixed guidance scale means the failure isn't a hyperparameter problem. If it were purely a guidance scale issue, all samples would improve or degrade together as you sweep. The fact that 2.5 works on one input but is awful on another at the same scale means the model has learned a *partial, non-generalizing* mapping. For some input configurations it "clicks" and produces a reasonable prediction; for others it never learned anything useful.

This is a coverage/generalization failure in the learned weights, not a CFG calibration problem.

**Why the partial learning happens**

Your ~57k samples have the same text but enormous visual diversity — different nut counts, positions, episode states. The LoRA cross-attention modifications learn to associate text tokens with visual features, but the task fundamentally requires dense spatial correspondence from input pixels to output pixels. That's not what IP2P's cross-attention was designed for. What the model actually needs to do — "track where the nut is, predict where the arm will deposit it" — is a visual correspondence problem, not a language-conditioned editing problem.

The samples where it works are likely configurations that appeared frequently enough in training for the LoRA to partially fit. The samples where it fails are visual configurations that are underrepresented or require spatial reasoning the architecture cannot easily express through text cross-attention.

**The honest architectural conclusion**

IP2P with LoRA is a reasonable first attempt for this problem, but it has a structural mismatch: the mechanism through which IP2P conditions on instructions (cross-attention on text tokens) is weak for a task where all the discriminative information is purely visual. You're asking text cross-attention to do spatial reasoning that the image conditioning channels should be doing — but LoRA on attention layers doesn't have a direct path to strengthen image-channel-to-output correspondence.

**What actually moves the needle from here**

Two genuine options:

1. **More data with higher coverage** — not a small increase. If the model partially generalizes on current data, you need enough episodes to densely cover the visual input space (nut positions, arm states). If you can generate more episodes, this is worth trying. Doubling from 57k to ~120k+ samples would meaningfully improve coverage.

2. **Reconsider the role of the world model** — your critic takes a predicted image and compares it to the goal. If the world model is sample-inconsistent, the critic will see garbage on hard inputs and can't recover. It may be worth asking whether a simpler *state delta predictor* (predict object position deltas, not full images) would be a more tractable intermediate representation — or whether the oracle WM is just the right path and the diffusion WM effort is better spent elsewhere in the pipeline.