import argparse
import os
import platform

import numpy as np

# macOS: force osmesa for true headless rendering (no NSWindow/GLFW issues)
if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.point_cloud import get_pointcloud
from robosuite.environments.base import make

def create_environment(env_name: str = "Stack4", args=None):
    """
    Create and configure the robosuite stacking environment.
    
    Args:
        env_name: Name of the environment ("Stack", "Stack3", or "Stack4")
        
    Returns:
        Configured environment instance
    """
    controller_config = load_composite_controller_config(controller="BASIC")
    
    env = make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
        camera_names=[args.camera],
        camera_heights=[args.height],
        camera_widths=[args.width],
        camera_depths=[True],
    )
    
    return env


def main():
    parser = argparse.ArgumentParser(description="Capture a point cloud from a robosuite camera")
    parser.add_argument("--env", type=str, default="Lift", help="Environment name")
    parser.add_argument("--camera", type=str, default="agentview", help="Camera name")
    parser.add_argument("--height", type=int, default=240, help="Image height")
    parser.add_argument("--width", type=int, default=240, help="Image width")
    parser.add_argument("--rgb", action="store_true", help="Attach per-point RGB")
    parser.add_argument("--seg", action="store_true", help="Attach per-point body id")
    parser.add_argument("--out", type=str, default="pointcloud.npz", help="Output .npz path")
    args = parser.parse_args()

    # env = make(
    #     args.env,
    #     robots=["Panda"],
    #     controller_configs=load_composite_controller_config(controller="BASIC"),
    #     has_renderer=False,
    #     has_offscreen_renderer=True,
    #     ignore_done=True,
    #     use_object_obs=True,
    #     use_camera_obs=False,
    #     reward_shaping=True,
    #     control_freq=20,
    #     camera_names=[args.camera],
    #     camera_heights=[args.height],
    #     camera_widths=[args.width],
    #     camera_depths=[True],
    # )

    env = create_environment(env_name=args.env, args=args)

    env.reset()

    pc = get_pointcloud(
        env.sim,
        camera_name=args.camera,
        camera_height=args.height,
        camera_width=args.width,
        return_world=True,
        with_rgb=args.rgb,
        with_segmentation=args.seg,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    save_dict = {
        "points_cam": pc.get("points_cam", np.zeros((0, 3), dtype=np.float32)),
        "pixels": pc.get("pixels", np.zeros((0, 2), dtype=np.int32)),
    }
    if "points_world" in pc:
        save_dict["points_world"] = pc["points_world"]
    if "rgb" in pc:
        save_dict["rgb"] = pc["rgb"]
    if "body_ids" in pc:
        save_dict["body_ids"] = pc["body_ids"]

    np.savez_compressed(args.out, **save_dict)
    print(f"Saved point cloud with {save_dict['points_cam'].shape[0]} points to {args.out}")

    env.close()


if __name__ == "__main__":
    main()
