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
python3 verify2act/critic/train_contrastive.py \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --output-dir verify2act/output/contrastive/nut_assembly \
  --dataset-type robosuite \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --lambda1 0.4 \
  --lambda2 0.8 \
  --kl-weight 5e-4 \
  --device cuda

# Calvin Critic Training
python3 verify2act/critic/train_contrastive.py \
  --dataset-dir calvin/dataset/task_ABC_D_filtered/training \
  --output-dir verify2act/output/contrastive/calvin \
  --dataset-type calvin \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --lambda1 0.4 \
  --lambda2 0.8 \
  --kl-weight 5e-4 \
  --device cuda

# ==============================================================================
# DINO-WM BASELINE TRAINING (Causal raw DINOv2 feature space sequence model)
# ==============================================================================

# -------- NUT ASSEMBLY (RoboSuite) --------
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/dino_wm_baseline/train_baseline.py \
  --dataset-type robosuite \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --output-dir verify2act/output/dino_wm/nut_assembly/wm \
  --num-epochs 100 \
  --batch-size 16 \
  --lr 1e-4 \
  --checkpoint-freq 5

# -------- CALVIN --------
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/dino_wm_baseline/train_baseline.py \
  --dataset-type calvin \
  --dataset-dir calvin/dataset/task_ABC_D_filtered/training \
  --output-dir verify2act/output/dino_wm/calvin/wm \
  --num-epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --checkpoint-freq 5

# ==============================================================================
# INFERENCE & EVALUATION PIPELINES
# ==============================================================================

# -------- NUT ASSEMBLY (RoboSuite) --------
# RoboSuite Inference (using dino_wm)
xvfb-run -a python verify2act/pipeline/inference.py \
  --critic-ckpt verify2act/output/contrastive/nut_assembly/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/dino_wm/nut_assembly/wm/ckpt/latent_dynamics_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/nut_assembly/decoder \
  --num-round 2 \
  --num-square 1 \
  --initial-stacking-prob 0.0 \
  --device cuda \
  --dtype fp16 \
  --wm-mode dino_wm \
  --output-dir verify2act/output/inference_run/dino_wm/nut_assembly

# -------- CALVIN --------
# Calvin Standalone Inference (using dino_wm)
xvfb-run -a python verify2act/pipeline/inference.py \
  --critic-ckpt verify2act/output/contrastive/calvin/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/dino_wm/calvin/wm/ckpt/latent_dynamics_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/calvin/decoder \
  --num-round 2 \
  --num-square 1 \
  --initial-stacking-prob 0.0 \
  --device cuda \
  --dtype fp16 \
  --wm-mode dino_wm \
  --output-dir verify2act/output/inference_run/dino_wm/calvin

# Calvin Benchmark evaluate_policy Evaluation (using dino_wm)
python3 verify2act/pipeline/inference_calvin.py \
  --critic-ckpt verify2act/output/contrastive/calvin/best_contrastive_critic.pt \
  --latent-wm-ckpt verify2act/output/dino_wm/calvin/wm/ckpt/latent_dynamics_best.pt \
  --wm-decoder-dir verify2act/output/v2a_wm/calvin/decoder \
  --train-folder calvin/models/hulc_baseline \
  --dataset-path calvin/dataset/task_ABC_D_filtered \
  --history-len 3 \
  --device cuda \
  --wm-mode dino_wm \
  --debug
