#!/bin/bash

export MUJOCO_GL=egl 
export PYOPENGL_PLATFORM=egl

# Validate argument
if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 {sim|diffusion}"
    exit 1
fi

if [[ "$1" == "sim" || "$1" == "diffusion" ]]; then
    echo "Running ReflectVLM with $1 as dynamics model"
else
    echo "Error: Argument must be 'sim' or 'diffusion'"
    exit 1
fi

dynamics_model=$1
exp_name="eval_reflect_vlm_${dynamics_model}"
model_path="yunhaif/ReflectVLM-llava-v1.5-13b-post-trained"

if [ $dynamics_model == "diffusion" ]; then
    diffusion_args="--diffuser_pretrained_model=yunhaif/ReflectVLM-diffusion"
else
    diffusion_args=""
fi

python run.py \
    --seed=1000000 \
    --n_trajs=100 \
    --save_dir="logs/$exp_name" \
    --start_traj_id=0 \
    --start_board_id=1000000 \
    --logging.online=True \
    --logging.group=$exp_name \
    --logging.prefix='eval' \
    --agent_type="llava" \
    --revise_action=True \
    --imagine_future_steps=5 \
    $diffusion_args \
    --level='hard' \
    --oracle_prob=0 \
    --model_path=$model_path \
    --record=True
