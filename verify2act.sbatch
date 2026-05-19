#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mail-user=chrisantus.eze@okstate.edu
#SBATCH --mail-type=end
#SBATCH --gres=gpu:1 
module load apptainer/1.2.2 cuda/12.1
export APPTAINER_BIND=$PWD

# # Run 10 data collection processes in parallel (you can add more seeds to the list)
# for seed in 63 64 65 66 67; do
#     apptainer exec --nv apptainerfile.sif xvfb-run -a python3 robosuite/data_capture_wm/batch_collect.py \
#             --env ClutteredNutAssembly --policy-mode expert \
#             --transition-mode keyframe \
#             --output-dir robosuite/data_capture_wm/dataset/nut_assembly \
#             --num-round 1 --num-square 2 --initial-stacking-prob 0.0 \
#             --nut-type-mode random --num-episodes 5000 \
#             --image-size 512 --seed $seed &
# done

# # Wait for all background processes to finish before ending the slurm job
# wait

# apptainer exec --nv verify2act.sif python3 verify2act/latent_wm/train_dynamics.py \
#   --dataset-type calvin \
#   --dataset-dir calvin/dataset/task_ABC_D/training \
#   --output-dir verify2act/output/v2a_wm/calvin/task_ABC_D \
#   --num-epochs 100 --batch-size 32 --lr 1e-4 --checkpoint-freq 5 \
#   --causal-masking \
#   --resume-from verify2act/output/v2a_wm/calvin/task_ABC_D/ckpt/latent_dynamics_best.pt

apptainer exec --nv verify2act.sif python3 verify2act/utils/filter_calvin_dataset.py \
  --input-dir calvin/dataset/task_ABC_D/training \
  --output-dir calvin/dataset/task_ABC_D_filtered/training \
  --history-len 5

# apptainer exec --nv verify2act.sif python3 unzip.py