#!/bin/bash

# ==============================================================================
# SETUP & HPCC ENVIRONMENT COMMANDS
# ==============================================================================
# conda activate verify2act
# cd robosuite/data_capture_wm
#
# HPCC Slurm Helpers:
# ssh echris@pete.hpc.okstate.edu
# squeue -u echris
# scancel <jobid>
# sbatch verify2act.sbatch

# ==============================================================================
# CRITIC TRAINING
# ==============================================================================

# RoboSuite Critic Training
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=bf16 \
  verify2act/critic/train_contrastive.py \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --output-dir verify2act/output/contrastive/nut_assembly \
  --dataset-type robosuite \
  --epochs 25 \
  --batch-size 20 \
  --learning-rate 1e-4 \
  --lambda1 0.5 \
  --lambda2 0.7 \
  --kl-weight 5e-4 \
  --resume-from verify2act/output/contrastive/nut_assembly/best_contrastive_critic.pt

# Calvin Critic Training
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=bf16 \
  verify2act/critic/train_contrastive.py \
  --dataset-dir calvin/dataset/task_ABCD_D_filtered/training \
  --output-dir verify2act/output/contrastive/calvin \
  --dataset-type calvin \
  --epochs 25 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --lambda1 0.5 \
  --lambda2 0.7 \
  --kl-weight 5e-4 \
  --cached-dino-dir "verify2act/output/v2a_wm/calvin/dino_features" \
  --resume-from verify2act/output/contrastive/calvin/best_contrastive_critic.pt

# ==============================================================================
# LATENT WM TRAINING (v2a_wm - Flow Matching)
# ==============================================================================

# -------- NUT ASSEMBLY (RoboSuite) --------

# Stage 1: Train DeltaEncoder (Bottleneck)
# Extended to 200 epochs — encoder val loss was still descending at ep100.
# Resume from best checkpoint so training continues from ep100.
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/latent_wm/train_encoder.py \
  --dataset-type robosuite \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --output-dir verify2act/output/v2a_wm/nut_assembly/encoder \
  --token-dim 128 --num-latent-tokens 32 \
  --num-epochs 300 --batch-size 128 --lr 1e-4 \
  --resume-from verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/delta_encoder_best.pt

# Stage 2 (aux): Train Decoder (DINO features -> image)
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/latent_wm/train_decoder.py \
  --dataset-type robosuite \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --output-dir verify2act/output/v2a_wm/nut_assembly/decoder \
  --num-epochs 100 --batch-size 128 \
  --resume-from verify2act/output/v2a_wm/nut_assembly/decoder/latent_decoder_best.pt

# Stage 2: Flow Matching 
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/latent_wm/train_dynamics.py \
  --dataset-type robosuite \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --output-dir verify2act/output/v2a_wm/nut_assembly/wm_non_cross_attn \
  --encoder-ckpt verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/encoder_only_best.pt \
  --token-dim 128 --num-latent-tokens 32 \
  --num-epochs 100 --batch-size 32 --lr 1e-4 --checkpoint-freq 10 \
  --causal-masking \
  --action-conditioning adaln \
  --resume-from verify2act/output/v2a_wm/nut_assembly/wm_causal/ckpt/latent_dynamics_ep60.pt

# -------- CALVIN --------

# Filter Calvin dataset utility
python3 verify2act/utils/filter_calvin_dataset.py \
  --input-dir calvin/dataset/task_ABC_D/training \
  --output-dir calvin/dataset/task_ABC_D_filtered/training \
  --history-len 5

# Stage 1: Train DeltaEncoder (Bottleneck)
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/latent_wm/train_encoder.py \
  --dataset-type calvin \
  --dataset-dir calvin/dataset/task_ABCD_D_filtered/training \
  --output-dir verify2act/output/v2a_wm/calvin/encoder_wider \
  --token-dim 128 --num-latent-tokens 32 \
  --num-epochs 300 --batch-size 32 --lr 1e-4 \
  --resume-from verify2act/output/v2a_wm/calvin/encoder/ckpt/delta_encoder_best.pt

# Stage 2 (aux): Train Decoder (DINO features -> image)
python verify2act/latent_wm/train_decoder.py \
  --dataset-type calvin \
  --dataset-dir calvin/dataset/task_ABCD_D_filtered/training \
  --output-dir verify2act/output/v2a_wm/calvin/decoder \
  --num-epochs 100 --batch-size 32 \
  --resume-from verify2act/output/v2a_wm/calvin/decoder/latent_decoder_best.pt

# Stage 2: Flow Matching
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/latent_wm/train_dynamics.py \
  --dataset-type calvin \
  --dataset-dir calvin/dataset/task_ABCD_D_filtered/training \
  --output-dir verify2act/output/v2a_wm/calvin/wm_non_cross_attn \
  --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder_wider/ckpt/encoder_only_best.pt \
  --token-dim 128 --num-latent-tokens 32 \
  --num-epochs 100 --batch-size 32 --lr 1e-4 --checkpoint-freq 10 \
  --causal-masking \
  --action-conditioning adaln \
  --resume-from verify2act/output/v2a_wm/calvin/wm/ckpt/latent_dynamics_best.pt

CUDA_VISIBLE_DEVICES=1 python # for training on gpu 1

# ==============================================================================
# ABLATION: --action-conditioning  (cross_attn vs adaln)
# ==============================================================================
# Two orthogonal axes control the conditioning design:
#
#   --action-conditioning cross_attn  (default) — full CLIP token sequence via
#     cross-attention.  This is V2A-WM's core novelty.
#
#   --action-conditioning adaln       (ablation)  — CLIP tokens are mean-pooled
#     to a single vector and injected as an AdaLN modulation signal. Mirrors the
#     original RLA-WM action-grounding design.
#
# Combined with --causal-masking you get a clean 2×2 ablation table:
#
#   | conditioning \ history  | single-frame  | causal history |
#   |-------------------------|---------------|----------------|
#   | adaln                   | ≈ Baseline    | adaln + causal |
#   | cross_attn (default)    | cross + Markov| V2A-WM (full)  |

# Ablation A: AdaLN action grounding + causal history (isolates cross-attn contribution)
# accelerate launch --num_processes=3 ... verify2act/latent_wm/train_dynamics.py \
#   ... \
#   --action-conditioning adaln \
#   --causal-masking

# Ablation B: AdaLN + no causal masking (≈ BaselineRLAWM, run inside train_dynamics.py)
# accelerate launch --num_processes=3 ... verify2act/latent_wm/train_dynamics.py \
#   ... \
#   --action-conditioning adaln
#   # (omit --causal-masking → forces history_len=1 internally)

# Full V2A-WM (cross_attn is the default, no flag needed):
# accelerate launch --num_processes=3 ... verify2act/latent_wm/train_dynamics.py \
#   ... \
#   --causal-masking
#   # --action-conditioning cross_attn is implied

# ==============================================================================
# VISUALIZATION
# ==============================================================================


# RoboSuite Visualization
python verify2act/latent_wm/visualize_wm.py \
  --dataset-type robosuite \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --wm-ckpt verify2act/output/v2a_wm/nut_assembly/wm_causal_v2/ckpt/latent_dynamics_best_weights.pt \
  --encoder-ckpt verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/delta_encoder_best.pt \
  --decoder-ckpt verify2act/output/v2a_wm/nut_assembly/decoder/latent_decoder_best.pt \
  --history-len 3 \
  --num-samples 10 \
  --token-dim 128 --num-latent-tokens 32 

python verify2act/latent_wm/visualize_wm.py \
  --dataset-type calvin \
  --dataset-dir calvin/dataset/task_ABCD_D_filtered/training \
  --wm-ckpt verify2act/output/v2a_wm/calvin/wm_wider/ckpt/latent_dynamics_best_weights.pt \
  --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder_wider/ckpt/delta_encoder_best.pt \
  --decoder-ckpt verify2act/output/v2a_wm/calvin/decoder/latent_decoder_best.pt \
  --history-len 3 \
  --num-samples 10 \
  --token-dim 128 --num-latent-tokens 32 

# ==============================================================================
# INFERENCE PIPELINE — Production Evaluation Commands (25 episodes each)
# ==============================================================================

# (1) v2a_wm  — proposed method (causal world model + encoder + critic)
xvfb-run -a python verify2act/pipeline/inference.py \
  --critic-ckpt verify2act/output/contrastive/nut_assembly/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/v2a_wm/nut_assembly/wm_causal/ckpt/latent_dynamics_best_weights.pt \
  --encoder-ckpt verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/delta_encoder_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/nut_assembly/decoder \
  --num-round 4 \
  --num-square 3 \
  --guarantee-overlap \
  --randomize-nut-counts \
  --num-episodes 100 \
  --base-seed 42 \
  --device cuda \
  --dtype fp16 \
  --wm-mode v2a_wm \
  --theta-c 0.5 \
  --theta-p 0.05 \
  --history-len 3 \
  --action-conditioning cross_attn

# (2) rla_wm  — ablation: narrower (original-size) world model + encoder, same critic
xvfb-run -a python verify2act/pipeline/inference.py \
  --critic-ckpt verify2act/output/contrastive/nut_assembly/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/rla_wm/nut_assembly/wm/ckpt/latent_dynamics_best.pt \
  --encoder-ckpt verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/delta_encoder_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/nut_assembly/decoder \
  --num-round 4 \
  --num-square 3 \
  --guarantee-overlap \
  --randomize-nut-counts \
  --num-episodes 100 \
  --base-seed 42 \
  --device cuda \
  --dtype fp16 \
  --wm-mode rla_wm \
  --theta-c 0.5 \
  --theta-p 0.2

# (3) dino_wm  — ablation: latent WM without the encoder (raw DINO features)
xvfb-run -a python verify2act/pipeline/inference.py \
  --critic-ckpt verify2act/output/contrastive/nut_assembly/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/dino_wm/nut_assembly/wm/ckpt/latent_dynamics_best.pt \
  --num-round 4 \
  --num-square 3 \
  --guarantee-overlap \
  --randomize-nut-counts \
  --num-episodes 100 \
  --base-seed 42 \
  --device cuda \
  --dtype fp16 \
  --wm-mode dino_wm \
  --theta-c 0.5 \
  --theta-p 0.2 \
  --history-len 3

# (4) diffusion  — baseline: pixel-space diffusion world model (ReflectVLM-style)
xvfb-run -a python verify2act/pipeline/inference.py \
  --wm-model timbrooks/instruct-pix2pix \
  --wm-adapter-dir verify2act/output/diffusion_wm/nut_assembly/wm/best/unet_lora \
  --wm-decoder-dir verify2act/output/diffusion_wm/nut_assembly/decoder/checkpoint-5000 \
  --num-round 4 \
  --num-square 3 \
  --guarantee-overlap \
  --randomize-nut-counts \
  --num-episodes 100 \
  --base-seed 42 \
  --device cuda \
  --dtype fp16 \
  --wm-mode diffusion \
  --history-len 1

# (5) vlm_only  — baseline: no world model, VLM plans and verifies directly
xvfb-run -a python verify2act/pipeline/inference.py \
  --num-round 4 \
  --num-square 3 \
  --guarantee-overlap \
  --randomize-nut-counts \
  --num-episodes 100 \
  --base-seed 42 \
  --device cuda \
  --dtype fp16 \
  --wm-mode vlm_only \
  --history-len 1

# (6) wm_non_cross_attn  — ablation: adaln action conditioning + causal history
xvfb-run -a python verify2act/pipeline/inference.py \
  --critic-ckpt verify2act/output/contrastive/nut_assembly/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/v2a_wm/nut_assembly/wm_non_cross_attn/ckpt/latent_dynamics_best_weights.pt \
  --encoder-ckpt verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/delta_encoder_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/nut_assembly/decoder \
  --num-round 4 \
  --num-square 3 \
  --guarantee-overlap \
  --randomize-nut-counts \
  --num-episodes 100 \
  --base-seed 42 \
  --device cuda \
  --dtype fp16 \
  --wm-mode v2a_wm \
  --theta-c 0.5 \
  --theta-p 0.05 \
  --history-len 3 \
  --action-conditioning adaln

# (7) wm_non_causal  — ablation: cross_attn action conditioning + single-frame history (history_len=1)
xvfb-run -a python verify2act/pipeline/inference.py \
  --critic-ckpt verify2act/output/contrastive/nut_assembly/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/v2a_wm/nut_assembly/wm_non_causal/ckpt/latent_dynamics_best_weights.pt \
  --encoder-ckpt verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/delta_encoder_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/nut_assembly/decoder \
  --num-round 4 \
  --num-square 3 \
  --guarantee-overlap \
  --randomize-nut-counts \
  --num-episodes 100 \
  --base-seed 42 \
  --device cuda \
  --dtype fp16 \
  --wm-mode v2a_wm \
  --theta-c 0.5 \
  --theta-p 0.05 \
  --history-len 1 \
  --action-conditioning cross_attn

# ─────────────────────────────────────────────────────────────────────────────
# CALVIN Inference — Production Evaluation Commands (100 sequences each)
# ─────────────────────────────────────────────────────────────────────────────
# TIP: find your GCP project ID with: gcloud config get-value project
#      or set it permanently: gcloud config set project <YOUR_PROJECT_ID>
#
# Shared critic hyperparameters (tuned on 5-seq pilot):
#   --theta-c 0.5        Head 2 temporal consistency gate (TC range 0.35-0.84)
#   --theta-p 0.2        Head 1 goal proximity gate (GP range -0.17 to 0.56)
#   --critic-unc-threshold 0.08   MC uncertainty gate (observed std 0.030-0.065)
# ─────────────────────────────────────────────────────────────────────────────

# (1) v2a_wm  — proposed method (wider world model + encoder + critic)
python3 verify2act/pipeline/inference_calvin.py \
  --critic-ckpt verify2act/output/contrastive/calvin/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/v2a_wm/calvin/wm_wider/ckpt/latent_dynamics_best_weights.pt \
  --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder_wider/ckpt/delta_encoder_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/calvin/decoder \
  --train-folder calvin/models/hulc_baseline \
  --dataset-path calvin/dataset/task_ABCD_D_filtered \
  --low-level-policy diffusion \
  --low-level-policy-ckpt calvin/models/diffusion_baseline \
  --device cuda \
  --wm-mode v2a_wm \
  --gcp-project verify2act \
  --theta-c 0.5 \
  --theta-p 0.05 \
  --critic-unc-threshold 0.08 \
  --num-sequences 100 \
  --history-len 3 \
  --action-conditioning cross_attn

# (2) rla_wm  — ablation: narrower (original-size) world model + encoder, same critic
python3 verify2act/pipeline/inference_calvin.py \
  --critic-ckpt verify2act/output/contrastive/calvin/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/rla_wm/calvin/wm/ckpt/latent_dynamics_best.pt \
  --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder_wider/ckpt/delta_encoder_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/calvin/decoder \
  --train-folder calvin/models/hulc_baseline \
  --dataset-path calvin/dataset/task_ABCD_D_filtered \
  --low-level-policy diffusion \
  --low-level-policy-ckpt calvin/models/diffusion_baseline \
  --device cuda \
  --wm-mode rla_wm \
  --gcp-project verify2act \
  --theta-c 0.5 \
  --theta-p 0.2 \
  --critic-unc-threshold 0.08 \
  --num-sequences 100

# (3) dino_wm  — ablation: latent WM without the encoder (raw DINO features)
python3 verify2act/pipeline/inference_calvin.py \
  --critic-ckpt verify2act/output/contrastive/calvin/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/v2a_wm/calvin/wm_wider/ckpt/latent_dynamics_best_weights.pt \
  --train-folder calvin/models/hulc_baseline \
  --dataset-path calvin/dataset/task_ABCD_D_filtered \
  --low-level-policy diffusion \
  --low-level-policy-ckpt calvin/models/diffusion_baseline \
  --device cuda \
  --wm-mode dino_wm \
  --gcp-project verify2act \
  --theta-c 0.5 \
  --theta-p 0.2 \
  --critic-unc-threshold 0.08 \
  --num-sequences 100

# (4) diffusion  — baseline: pixel-space diffusion world model (ReflectVLM-style)
python3 verify2act/pipeline/inference_calvin.py \
  --wm-model timbrooks/instruct-pix2pix \
  --wm-adapter-dir verify2act/output/diffusion_wm/calvin/wm/best/unet_lora \
  --wm-decoder-dir verify2act/output/diffusion_wm/calvin/decoder/checkpoint-5000 \
  --train-folder calvin/models/hulc_baseline \
  --dataset-path calvin/dataset/task_ABCD_D_filtered \
  --low-level-policy diffusion \
  --low-level-policy-ckpt calvin/models/diffusion_baseline \
  --device cuda \
  --wm-mode diffusion \
  --gcp-project verify2act \
  --num-sequences 100

# (5) vlm_only  — baseline: no world model, VLM plans and verifies directly
python3 verify2act/pipeline/inference_calvin.py \
  --train-folder calvin/models/hulc_baseline \
  --dataset-path calvin/dataset/task_ABCD_D_filtered \
  --low-level-policy diffusion \
  --low-level-policy-ckpt calvin/models/diffusion_baseline \
  --device cuda \
  --wm-mode vlm_only \
  --gcp-project verify2act \
  --num-sequences 100

# (6) wm_non_cross_attn  — ablation: adaln action conditioning + causal history
python3 verify2act/pipeline/inference_calvin.py \
  --critic-ckpt verify2act/output/contrastive/calvin/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/v2a_wm/calvin/wm_non_cross_attn/ckpt/latent_dynamics_best_weights.pt \
  --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder_wider/ckpt/delta_encoder_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/calvin/decoder \
  --train-folder calvin/models/hulc_baseline \
  --dataset-path calvin/dataset/task_ABCD_D_filtered \
  --low-level-policy diffusion \
  --low-level-policy-ckpt calvin/models/diffusion_baseline \
  --device cuda \
  --wm-mode v2a_wm \
  --gcp-project verify2act \
  --theta-c 0.5 \
  --theta-p 0.05 \
  --critic-unc-threshold 0.08 \
  --num-sequences 100 \
  --history-len 3 \
  --action-conditioning adaln

# (7) wm_non_causal  — ablation: cross_attn action conditioning + single-frame history (history_len=1)
python3 verify2act/pipeline/inference_calvin.py \
  --critic-ckpt verify2act/output/contrastive/calvin/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/v2a_wm/calvin/wm_non_causal/ckpt/latent_dynamics_best_weights.pt \
  --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder_wider/ckpt/delta_encoder_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/calvin/decoder \
  --train-folder calvin/models/hulc_baseline \
  --dataset-path calvin/dataset/task_ABCD_D_filtered \
  --low-level-policy diffusion \
  --low-level-policy-ckpt calvin/models/diffusion_baseline \
  --device cuda \
  --wm-mode v2a_wm \
  --gcp-project verify2act \
  --theta-c 0.5 \
  --theta-p 0.05 \
  --critic-unc-threshold 0.08 \
  --num-sequences 100 \
  --history-len 1 \
  --action-conditioning cross_attn

