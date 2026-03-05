#!/bin/bash

export MUJOCO_GL=egl 
export PYOPENGL_PLATFORM=egl

exp_name="eval_expert"

python run.py \
    --seed=1000000 \
    --reset_seed_start=0 \
    --n_trajs=100 \
    --save_dir="logs/$exp_name" \
    --start_traj_id=0 \
    --start_board_id=1000000 \
    --logging.online=True \
    --logging.group=$exp_name \
    --logging.prefix='prerelease' \
    --agent_type="expert" \
    --level='hard' \
    --oracle_prob=0 \
    --record=True
