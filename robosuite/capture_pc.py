import os
import sys

import mujoco
import robosuite

# Set rendering backend before any robosuite imports
if sys.platform == 'darwin':  # macOS
    os.environ['MUJOCO_GL'] = 'osmesa'
else:  # Linux
    os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import robosuite as suite
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
from robosuite.environments.base import make


# Convert depth to point cloud
def depth_to_pointcloud(depth, intrinsics):
    height, width = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert to 3D coordinates
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into point cloud
    points = np.stack([x, y, z], axis=-1)
    return points.reshape(-1, 3)

# pointcloud = depth_to_pointcloud(depth, intrinsics)


def create_env():
    return make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["frontview", "agentview"],
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
    )

if __name__ == "__main__":
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Robosuite version: {robosuite.__version__}")

    env = create_env()
    obs = env.reset()
    print("Environment created successfully!")

    # Get depth image (normalized between 0-1)
    depth = obs['frontview_depth']

    # Get camera intrinsics
    camera_height, camera_width = depth.shape
    fovy = 45  # Default robosuite FOV
    intrinsics = get_camera_intrinsic_matrix(
        camera_height, camera_width, fovy
    )