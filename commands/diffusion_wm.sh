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
# WM TRAINING (InstructPix2Pix diffusion)
# ==============================================================================

# -------- NUT ASSEMBLY (RoboSuite) --------
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/diffusion_wm/train_wm.py \
  --dataset-type robosuite \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --output-dir verify2act/output/diffusion_wm/nut_assembly/wm \
  --max-steps 16000 \
  --eval-every 1000 \
  --train-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-5 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lora-target-ff \
  --snr-gamma 5.0 \
  --conditioning-dropout-prob 0.15 \
  --change-mask-weight 2.0 \
  --change-mask-threshold 15.0 \
  --noise-offset 0.05 \
  --val-frac 0.1 \
  --mixed-precision fp16 \
  --enable-gradient-checkpointing \
  --device cuda \
  --resume-from verify2act/output/diffusion_wm/nut_assembly/wm/checkpoint-12000

# -------- CALVIN --------
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/diffusion_wm/train_wm.py \
  --dataset-type calvin \
  --dataset-dir calvin/dataset/task_ABCD_D_filtered/training \
  --output-dir verify2act/output/diffusion_wm/calvin/wm \
  --max-steps 16000 \
  --eval-every 1000 \
  --train-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-5 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lora-target-ff \
  --snr-gamma 5.0 \
  --conditioning-dropout-prob 0.15 \
  --change-mask-weight 2.0 \
  --change-mask-threshold 15.0 \
  --noise-offset 0.05 \
  --val-frac 0.1 \
  --mixed-precision fp16 \
  --enable-gradient-checkpointing \
  --device cuda \
  --resume-from verify2act/output/diffusion_wm/calvin/wm/checkpoint-12000

# ==============================================================================
# VAE DECODER TRAINING
# ==============================================================================

# -------- NUT ASSEMBLY (RoboSuite) --------
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/diffusion_wm/train_decoder.py \
  --dataset-type robosuite \
  --dataset-dir robosuite/data_capture/dataset/nut_assembly_merged \
  --output-dir verify2act/output/diffusion_wm/nut_assembly/decoder \
  --max-steps 5000 \
  --eval-every 500 \
  --batch-size 8 \
  --mixed-precision fp16 \
  --device cuda

# -------- CALVIN --------
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/diffusion_wm/train_decoder.py \
  --dataset-type calvin \
  --dataset-dir calvin/dataset/task_ABCD_D_filtered/training \
  --output-dir verify2act/output/diffusion_wm/calvin/decoder \
  --max-steps 5000 \
  --eval-every 500 \
  --batch-size 8 \
  --mixed-precision fp16 \
  --device cuda

# ==============================================================================
# INFERENCE PIPELINE
# ==============================================================================

# RoboSuite Inference (using diffusion)
xvfb-run -a python verify2act/pipeline/inference.py \
  --critic-ckpt verify2act/output/contrastive/nut_assembly/best_contrastive_critic.pt \
  --wm-adapter-dir verify2act/output/diffusion_wm/nut_assembly/wm/best/unet_lora \
  --wm-decoder-dir verify2act/output/diffusion_wm/nut_assembly/decoder \
  --num-round 4 \
  --num-square 3 \
  --guarantee-overlap \
  --randomize-nut-counts \
  --num-episodes 25 \
  --base-seed 42 \
  --device cuda \
  --dtype fp16 \
  --wm-mode diffusion \
  --vae-model runwayml/stable-diffusion-v1-5

# Calvin Inference (using diffusion)
xvfb-run -a python verify2act/pipeline/inference.py \
  --critic-ckpt verify2act/output/contrastive/calvin/best_contrastive_critic.pt \
  --wm-adapter-dir verify2act/output/diffusion_wm/calvin/wm/best/unet_lora \
  --wm-decoder-dir verify2act/output/diffusion_wm/calvin/decoder \
  --num-round 2 \
  --num-square 1 \
  --guarantee-overlap \
  --device cuda \
  --dtype fp16 \
  --wm-mode diffusion \
  --vae-model runwayml/stable-diffusion-v1-5
