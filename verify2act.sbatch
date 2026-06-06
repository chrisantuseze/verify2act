#!/bin/bash
# #SBATCH -p bullet
#SBATCH -p batch
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mail-user=chrisantus.eze@okstate.edu
#SBATCH --mail-type=end
# #SBATCH --gres=gpu:1
# module load apptainer/1.2.2 cuda/12.1
module load apptainer/1.2.2
export APPTAINER_BIND=$PWD

# Run 10 data collection processes in parallel (you can add more seeds to the list)
# for seed in 76 77 78 79 80; do
#     apptainer exec verify2act.sif xvfb-run -a python3 robosuite/data_capture/batch_collect.py \
#             --env ClutteredNutAssembly --policy-mode noisy --noise-sigma 0.05 \
#             --output-dir robosuite/data_capture/dataset/nut_assembly \
#             --num-round 3 --num-square 2 \
#             --nut-type-mode random --num-episodes 5000 \
#             --image-size 512 --seed $seed &
# done

# # Wait for all background processes to finish before ending the slurm job
# wait

# apptainer exec verify2act.sif bash -c "cd calvin/dataset && bash download_data.sh ABCD"

# apptainer exec verify2act.sif python3 -c "import os, zipfile; p = 'calvin/dataset/task_ABCD_D.zip' if os.path.exists('calvin/dataset/task_ABCD_D.zip') else 'task_ABCD_D.zip'; print(f'Extracting {p}...'); zipfile.ZipFile(p).extractall('calvin/dataset')"

# apptainer exec verify2act.sif python3 verify2act/utils/filter_calvin_dataset.py \
#   --input-dir calvin/dataset/task_ABCD_D/training \
#   --output-dir calvin/dataset/task_ABCD_D_filtered/training \
#   --history-len 10 &

# apptainer exec verify2act.sif python3 verify2act/utils/filter_calvin_dataset.py \
#   --input-dir calvin/dataset/task_ABCD_D/validation \
#   --output-dir calvin/dataset/task_ABCD_D_filtered/validation \
#   --history-len 10 &

# wait

apptainer exec verify2act.sif python3 -c "import os, shutil; p = 'calvin/dataset/task_ABCD_D_filtered'; print(f'Zipping {p}...') if os.path.exists(p) else print(f'Directory {p} not found.'); shutil.make_archive(p, 'zip', root_dir='calvin/dataset', base_dir='task_ABCD_D_filtered') if os.path.exists(p) else None"