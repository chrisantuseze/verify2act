#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mail-user=chrisantus.eze@okstate.edu
#SBATCH --mail-type=end
#SBATCH --gres=gpu:1
module load apptainer/1.2.2 cuda/12.1
export APPTAINER_BIND=$PWD

apptainer run --nv apptainerfile.sif python3 main_act.py \
--task_name sim_object_unveiler \
--ckpt_dir act/ckpt \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 5 \
--hidden_dim 512 \
--batch_size 4 \
--dim_feedforward 3200 \
--num_epochs 100000 \
--lr 1e-4 \
--num_patches 10 \
--seed 0