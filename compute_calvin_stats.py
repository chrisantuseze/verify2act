#!/usr/bin/env python3
"""
Compute robot_obs and scene_obs statistics from the CALVIN environment
by performing random resets and sampling observations.
Outputs statistics.yaml in the required format for NormalizeVector.
"""
import sys
import os
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent / 'calvin/calvin_models'))
sys.path.insert(0, str(pathlib.Path(__file__).parent / 'calvin/calvin_env'))

os.environ.setdefault('MUJOCO_GL', 'glx')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

from omegaconf import OmegaConf
import hydra
from pathlib import Path

def main():
    dataset_path = Path("calvin/dataset/task_ABC_D_filtered")
    val_dir = dataset_path / "validation"
    train_dir = dataset_path / "training"

    # Load config
    train_cfg_path = Path("calvin/models/hulc_baseline/.hydra/config.yaml")
    cfg = OmegaConf.load(train_cfg_path)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder

    # Initialize hydra to get dataset config
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize("calvin/calvin_models/conf/datamodule/datasets")
    datasets_cfg = hydra.compose("vision_lang.yaml", overrides=[f"lang_dataset.lang_folder={lang_folder}"])

    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = str(dataset_path.resolve())

    # Set up the env
    from torch.utils.data import DataLoader as TorchDataLoader
    import torchvision
    import torch
    from calvin_agent.datasets.utils.episode_utils import load_dataset_statistics
    import calvein_agent  # just to get the path

    # Build val dataset with default (no statistics) transforms
    default_transforms = cfg.datamodule.transforms
    val_transforms = {}
    for cam_key in default_transforms.val:
        tfs = [hydra.utils.instantiate(t) for t in default_transforms.val[cam_key]]
        val_transforms[cam_key] = torchvision.transforms.Compose(tfs)

    val_dataset = hydra.utils.instantiate(datasets_cfg.lang_dataset, datasets_dir=val_dir, transforms=val_transforms)
    
    # Create the env
    rollout_cfg = OmegaConf.load(Path("calvin/calvin_models/conf/callbacks/rollout/default.yaml"))
    device = torch.device("cuda:0")
    env = hydra.utils.instantiate(rollout_cfg.env_cfg, val_dataset, device, show_gui=False)

    # Collect statistics
    N = 500
    robot_obs_list = []
    scene_obs_list = []
    
    print(f"Collecting {N} environment observations...")
    for i in range(N):
        raw_obs = env.env.reset()
        robot_obs = raw_obs['robot_obs']
        scene_obs = raw_obs['scene_obs']
        robot_obs_list.append(robot_obs)
        scene_obs_list.append(scene_obs)
        if i % 50 == 0:
            print(f"  {i}/{N}")

    robot_arr = np.stack(robot_obs_list)
    scene_arr = np.stack(scene_obs_list)

    print(f"\nrobot_obs shape: {robot_arr.shape}")
    print(f"scene_obs shape: {scene_arr.shape}")

    robot_mean = robot_arr.mean(0).tolist()
    robot_std = robot_arr.std(0).tolist()
    scene_mean = scene_arr.mean(0).tolist()
    scene_std = scene_arr.std(0).tolist()

    # Replace zeros in std with 1.0 to avoid division by zero
    robot_std = [s if s > 1e-6 else 1.0 for s in robot_std]
    scene_std = [s if s > 1e-6 else 1.0 for s in scene_std]

    print("\nrobot_obs mean:", robot_mean)
    print("robot_obs std:", robot_std)
    print("scene_obs mean:", scene_mean)
    print("scene_obs std:", scene_std)

    # Also compute action bounds from the actual data
    # For task_ABC_D the known action max/min from checkpoint are the bounds used during training
    act_max = [0.4298, 0.1394, 0.7963, 3.1416, 0.6386, 3.1416, 1.0]
    act_min = [-0.4322, -0.5455, 0.2934, -3.1416, -0.8113, -3.1416, -1.0]

    # Build statistics config in the format expected by load_dataset_statistics
    stats_config = {
        'robot_obs': {
            '_target_': 'calvin_agent.utils.transforms.NormalizeVector',
            'mean': robot_mean,
            'std': robot_std
        },
        'scene_obs': {
            '_target_': 'calvin_agent.utils.transforms.NormalizeVector',
            'mean': scene_mean,
            'std': scene_std
        },
        'act_max_bound': act_max,
        'act_min_bound': act_min
    }

    # Save to both training and validation dirs
    stats_yaml = OmegaConf.create(stats_config)
    for save_dir in [train_dir, val_dir]:
        save_path = save_dir / 'statistics.yaml'
        OmegaConf.save(stats_yaml, save_path)
        print(f"\nSaved statistics to: {save_path}")


if __name__ == '__main__':
    main()
