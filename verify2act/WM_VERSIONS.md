# World Model Version History

Tracks key differences between training runs. Parameters not listed for a version are unchanged from the previous version.

---

## v2
- **transitions_file**: `transitions_subskill.jsonl` (subskill-level transitions)
- **max_steps**: 15000
- **lora_target_ff**: false
- **snr_gamma**: not set
- **gradient_checkpointing**: false
- **xformers**: false

## v3
- **max_steps**: 20000 *(+5k)*
- **lora_target_ff**: true *(added FF LoRA targets)*
- **snr_gamma**: 5.0 *(added Min-SNR-γ weighting)*

## v4
- **conditioning_dropout_prob**: 0.05 *(explicitly set; noise_offset: 0.05 added)*
- No other changes from v3

## v5
- No `train_config.json` saved — config unknown

## v6
- **transitions_file**: `transitions.jsonl` *(switched from subskill to skill-level transitions)*
- **lora_rank**: 64 *(doubled)*; **lora_alpha**: 32 *(not updated → scale=0.5, later identified as bug)*
- **lora_target_conv**: false
- **train_batch_size**: 2
- **use_ema**: true; **ema_decay**: 0.9999

## v7
- **lora_rank**: 32 *(back to 32)*; **lora_alpha**: 32 *(scale=1.0, fixed)*
- **lora_target_conv**: false
- **train_batch_size**: 8
- **tracker**: tensorboard
- No `change_mask_weight` / `change_mask_threshold`

## v8
- **lora_target_conv**: true *(added ResNet conv LoRA targets)*
- **train_batch_size**: 3 *(reduced for 3-GPU run with --num_processes 3)*
- **change_mask_weight**: 5.0; **change_mask_threshold**: 15.0 *(added change-mask loss)*
- **tracker**: none; **xformers**: true
- **eval_every**: 2000; **eval_batches**: 20

## v9
- **lora_target_conv**: false *(removed — ResNet block conv LoRA dropped)*
- **change_mask_weight**: 2.0 *(reduced from 5.0)*
- **conditioning_dropout_prob**: 0.15 *(increased from 0.05 — stronger image dropout to reduce copy-input shortcut)*
- **best_val_loss**: 0.01914 @ step 12000; oscillated/did not improve after that
- **result**: no visible improvement over v8. Increasing dropout (+) was cancelled out by reducing change_mask_weight (−), which weakened the spatial gradient signal on changed regions.

## v10
- **sampler**: WeightedRandomSampler balancing 3 visual sub-distributions:
  - `move_to_nut, t=0` (home start, 54%) → weight 0.61×
  - `move_to_nut, t>0` (mid-episode, 22%) → weight 1.55×
  - `move_to_peg` (pre-insertion, 24%) → weight 1.40×
- **conv_in unfrozen**: full fine-tuning of the 8→320 input projection (23,040 params), at 0.1× LR (separate param group). This layer controls how `[noisy_z_t1 | z_t]` channels are mixed before any LoRA can act — absent from all prior versions.
- **rationale (sampler)**: Experiment 2 confirmed per-sample inconsistency caused by 71.6% of `move_to_nut` transitions starting from t=0 (home position), causing the model to fit home-start configurations well but fail on mid-episode arm states.
- **rationale (conv_in)**: LoRA targets `conv1/conv2` inside ResNet blocks handle intra-block feature transformation but cannot fix the upstream channel-mixing problem at the UNet's input boundary.
- **best_val_loss**: 0.02304 @ step 12000 — *worse than v9*. The conv_in lr=0.1× (5e-6) was still too aggressive, destabilising the run before LoRA adapters could compensate.
- **result**: no improvement. Ghost-object hallucination (3rd square nut) persists across all versions to this point.

---

## v11
- **change_mask_weight**: 5.0 *(restored from 2.0 — reverts the v9 regression)*
- **conditioning_dropout_prob**: 0.15 *(kept from v9)*
- **change_mask_threshold**: 10.0 *(lowered from 15.0 — flags more pixels as changed, giving more spatial gradient coverage)*
- **max_steps**: 30000 *(extended — both v9/v10 plateaued at step 12000/20000, indicating underfitting beyond that point under the current loss balance)*
- **conv_in**: frozen again *(removed from v10 — isolates the conv_in variable; re-evaluate separately)*
- **sampler**: default random *(removed weighted sampler — isolates that variable too)*
- **negative_mask_weight**: 1.0 *(disabled — baseline without directional masking, for ablation against v12)*
- **result**: Image quality completely degraded. The square nuts on the table were entirely removed. Due to a bug in `train_wm.py` at the time, this run only applied the spatial upweight to pixels that *became brighter* (revealing the background) and didn't apply any upweight to pixels that *became darker* (the objects at their new locations). This perfectly explains why it deleted the ghost objects but failed to render the new objects.

## v12
- **negative_mask_weight**: 8.0 *(NEW — directional disappeared-region loss weight)*
- **change_mask_weight**: 5.0
- **change_mask_threshold**: 10.0
- **conditioning_dropout_prob**: 0.15
- **max_steps**: 30000
- **result**: Extremely chaotic hallucination (produced ~6 random nuts on the table). The flawed directional mask code assigned 8x weight to pixels that became darker, breaking the geometry. The directional mask approach based on color sign was conceptually flawed and removed.

---

## v13
- **lora_target_conv**: true *(restored from v8 to give the UNet capacity to actively erase ghost objects in the convolutional skip connections)*
- **change_mask_weight**: 2.5 *(moderate weight: high enough to penalize ghost objects, low enough to avoid catastrophic interference that erases the background)*
- **change_mask**: perceptual luminance difference *(switched from per-channel mean difference for more stable region detection)*
- **conditioning_dropout_prob**: 0.10 *(reduced from 0.15 to keep more batches fully supervised, while still allowing dual CFG)*
- **rationale**: Resolves the catch-22 identified in v8-v11. High mask weights (5.0) erased the background by starving static pixels of gradient. Low mask weights (2.0) with frozen convs safely passed the background but failed to erase the ghost object. By unfreezing the convs and using a moderate weight, we give the model the capacity to rewrite the feature maps without destroying its ability to copy.
- **training command**:
  ```bash
  accelerate launch --num_processes 3 \
    verify2act/world_model/train_wm.py \
    --dataset-dir robosuite/data_capture_wm/dataset/nut_assembly_merged \
    --output-dir verify2act/output/wm_v13 \
    --lora-rank 32 --lora-alpha 32 \
    --lora-target-ff \
    --lora-target-conv \
    --conditioning-dropout-prob 0.10 \
    --change-mask-weight 2.5 \
    --change-mask-threshold 10.0 \
    --train-batch-size 3 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --max-steps 30000 \
    --eval-every 2000 --eval-batches 20 \
    --snr-gamma 5.0 \
    --use-ema --ema-decay 0.9999 \
    --val-frac 0.2 \
    --enable-gradient-checkpointing \
    --mixed-precision fp16 \
    --enable-xformers \
    --tracker tensorboard
  ```

## v14
- **lora_target_conv**: false *(Frozen convolutions to perfectly preserve the background, exactly like v9)*
- **change_mask_weight**: 2.0 *(Low mask weight to avoid catastrophic interference, exactly like v9)*
- **conditioning_dropout_prob**: 0.10
- **action_text augmentation**: Appends 3D `cartesian_target` to the text prompt during dataset loading (`"pick right square nut at loc -0.01 0.14 0.84"`).
- **rationale**: InstructPix2Pix fundamentally fails to solve spatial ambiguity through loss masking (proven by v8 and v13 erasing the background). v14 returns to the stable v9 configuration and relies exclusively on Data-Side Instance Grounding (prompt augmentation) to give the text encoder the spatial uniqueness required to erase the ghost object.
- **training command**:
  ```bash
  accelerate launch --num_processes 3 \
    verify2act/world_model/train_wm.py \
    --dataset-dir robosuite/data_capture_wm/dataset/nut_assembly_merged \
    --output-dir verify2act/output/wm_v14 \
    --lora-rank 32 --lora-alpha 32 \
    --lora-target-ff \
    --conditioning-dropout-prob 0.10 \
    --change-mask-weight 2.0 \
    --change-mask-threshold 10.0 \
    --train-batch-size 3 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --max-steps 30000 \
    --eval-every 2000 --eval-batches 20 \
    --snr-gamma 5.0 \
    --use-ema --ema-decay 0.9999 \
    --val-frac 0.2 \
    --enable-gradient-checkpointing \
    --mixed-precision fp16 \
    --enable-xformers \
    --tracker tensorboard
  ```
