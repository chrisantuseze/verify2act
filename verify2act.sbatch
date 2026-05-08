#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mail-user=chrisantus.eze@okstate.edu
#SBATCH --mail-type=end
module load apptainer/1.2.2 cuda/12.1
export APPTAINER_BIND=$PWD

# Run 10 data collection processes in parallel (you can add more seeds to the list)
for seed in 63 64 65 66 67 68 69 70 71 72 ; do
    apptainer exec --nv apptainerfile.sif xvfb-run -a python3 robosuite/data_capture_wm/batch_collect.py \
            --env ClutteredNutAssembly --policy-mode expert \
            --transition-mode keyframe \
            --output-dir robosuite/data_capture_wm/dataset/nut_assembly \
            --num-round 1 --num-square 2 --initial-stacking-prob 0.0 \
            --nut-type-mode random --num-episodes 5000 \
            --image-size 512 --seed $seed &
done

# Wait for all background processes to finish before ending the slurm job
wait