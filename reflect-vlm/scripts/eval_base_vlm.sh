#!/bin/bash

export MUJOCO_GL=egl 
export PYOPENGL_PLATFORM=egl

exp_name="eval_base_vlm"
model_path='yunhaif/ReflectVLM-llava-v1.5-13b-base'

python run.py \
    --seed=1000000 \
    --reset_seed_start=0 \
    --n_trajs=100 \
    --save_dir="logs/$exp_name" \
    --start_traj_id=0 \
    --start_board_id=1000000 \
    --logging.online=True \
    --logging.group=$exp_name \
    --logging.prefix='eval' \
    --agent_type="llava" \
    --level='hard' \
    --oracle_prob=0 \
    --model_path=$model_path \
    --record=True
