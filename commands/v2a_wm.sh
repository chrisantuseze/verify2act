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
  --epochs 20 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --lambda1 0.4 \
  --lambda2 0.8 \
  --kl-weight 5e-4 \
  --resume-from verify2act/output/contrastive/nut_assembly/best_contrastive_critic.pt

# Calvin Critic Training
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=bf16 \
  verify2act/critic/train_contrastive.py \
  --dataset-dir calvin/dataset/task_ABC_D_filtered/training \
  --output-dir verify2act/output/contrastive/calvin \
  --dataset-type calvin \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --lambda1 0.4 \
  --lambda2 0.8 \
  --kl-weight 5e-4 \
  --resume-from verify2act/output/contrastive/calvin/best_contrastive_critic.pt

# ==============================================================================
# LATENT WM TRAINING (v2a_wm - Flow Matching)
# ==============================================================================

# -------- NUT ASSEMBLY (RoboSuite) --------

# Stage 1: Train DeltaEncoder (Bottleneck)
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/latent_wm/train_encoder.py \
  --dataset-type robosuite \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --output-dir verify2act/output/v2a_wm/nut_assembly/encoder \
  --num-epochs 50 --batch-size 64 --lr 1e-4 \
  --resume-from verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/delta_encoder_best.pt

# Stage 2 (aux): Train Decoder (DINO features -> image)
python verify2act/latent_wm/train_decoder.py \
  --dataset-type robosuite \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --output-dir verify2act/output/v2a_wm/nut_assembly/decoder \
  --batch-size 32 \
  --num-epochs 25 \
  --resume-from verify2act/output/v2a_wm/nut_assembly/decoder/latent_decoder_best.pt

# Stage 2: Flow Matching
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/latent_wm/train_dynamics.py \
  --dataset-type robosuite \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --output-dir verify2act/output/v2a_wm/nut_assembly/wm \
  --encoder-ckpt verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/encoder_only_best.pt \
  --num-epochs 50 --batch-size 16 --lr 1e-4 --checkpoint-freq 5 \
  --causal-masking \
  --resume-from verify2act/output/v2a_wm/nut_assembly/wm/ckpt/latent_dynamics_best.pt

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
  --dataset-dir calvin/dataset/task_ABC_D_filtered/training \
  --output-dir verify2act/output/v2a_wm/calvin/encoder \
  --num-epochs 50 --batch-size 16 --lr 1e-4 \
  --resume-from verify2act/output/v2a_wm/calvin/encoder/ckpt/delta_encoder_best.pt

# Stage 2 (aux): Train Decoder (DINO features -> image)
python verify2act/latent_wm/train_decoder.py \
  --dataset-type calvin \
  --dataset-dir calvin/dataset/task_ABC_D_filtered/training \
  --output-dir verify2act/output/v2a_wm/calvin/decoder \
  --batch-size 32 \
  --num-epochs 50 \
  --resume-from verify2act/output/v2a_wm/calvin/decoder/latent_decoder_best.pt

# Stage 2: Flow Matching
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/latent_wm/train_dynamics.py \
  --dataset-type calvin \
  --dataset-dir calvin/dataset/task_ABC_D_filtered/training \
  --output-dir verify2act/output/v2a_wm/calvin/wm \
  --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder/ckpt/encoder_only_best.pt \
  --num-epochs 50 --batch-size 16 --lr 1e-4 --checkpoint-freq 5 \
  --causal-masking \
  --resume-from verify2act/output/v2a_wm/calvin/wm/ckpt/latent_dynamics_best.pt

CUDA_VISIBLE_DEVICES=1 python # for training on gpu 1
# ==============================================================================
# VISUALIZATION
# ==============================================================================

# RoboSuite Visualization
python verify2act/latent_wm/visualize_wm.py \
  --dataset-type robosuite \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --wm-ckpt verify2act/output/v2a_wm/nut_assembly/ckpt/latent_dynamics_best.pt \
  --encoder-ckpt verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/delta_encoder_best.pt \
  --decoder-ckpt verify2act/output/v2a_wm/nut_assembly/decoder/latent_decoder_best.pt \
  --num-samples 5

python verify2act/latent_wm/visualize_wm.py \
  --dataset-type calvin \
  --dataset-dir calvin/dataset/task_ABC_D_filtered/training \
  --wm-ckpt verify2act/output/v2a_wm/calvin/wm_causal/ckpt/latent_dynamics_best.pt \
  --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder/ckpt/delta_encoder_best.pt \
  --decoder-ckpt verify2act/output/v2a_wm/calvin/decoder/latent_decoder_best.pt \
  --num-samples 5

# ==============================================================================
# INFERENCE PIPELINE
# ==============================================================================

# RoboSuite Inference (using v2a_wm)
xvfb-run -a python verify2act/pipeline/inference.py \
  --critic-ckpt verify2act/output/contrastive/nut_assembly/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/v2a_wm/nut_assembly/wm/ckpt/latent_dynamics_best.pt \
  --encoder-ckpt verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/encoder_only_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/nut_assembly/decoder \
  --num-round 2 \
  --num-square 1 \
  --initial-stacking-prob 0.0 \
  --device cuda \
  --dtype fp16 \
  --wm-mode v2a_wm \
  --output-dir verify2act/output/inference_run/v2a_wm/nut_assembly

# Calvin Inference (using v2a_wm)
xvfb-run -a python3 verify2act/pipeline/inference_calvin.py \
  --critic-ckpt verify2act/output/contrastive/calvin/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/v2a_wm/calvin/wm/ckpt/latent_dynamics_best.pt \
  --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder/ckpt/encoder_only_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/calvin/decoder \
  --train-folder calvin/models/hulc_baseline \
  --dataset-path calvin/dataset/task_ABC_D_filtered \
  --device cuda
