import mujoco
import robosuite
import numpy as np
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
from robosuite.environments.base import make

# Set matplotlib backend to non-GUI for saving images
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

from robosuite.utils.pointcloud_generator import PointCloudGenerator

# Convert depth to point cloud
def depth_to_pointcloud(depth, intrinsics):
    """
    Convert depth image to 3D point cloud.
    
    Args:
        depth: Depth image (H, W) with values in meters
        intrinsics: Camera intrinsic matrix (3, 3)
    
    Returns:
        Point cloud array (N, 3) in camera coordinates
    """
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


def capture_camera_data(env, camera_name="frontview", camera_height=256, camera_width=256):
    """
    Manually capture RGB and depth images from a camera using MuJoCo's rendering.
    
    Args:
        env: Robosuite environment
        camera_name: Name of the camera to capture from
        camera_height: Height of the captured image
        camera_width: Width of the captured image
    
    Returns:
        rgb: RGB image (H, W, 3)
        depth: Depth image (H, W) in meters
    """
    sim = env.sim
    
    # Get the underlying MuJoCo model and data (robosuite wraps them)
    mj_model = sim.model._model  # Access underlying mujoco.MjModel
    mj_data = sim.data._data      # Access underlying mujoco.MjData
    
    # Get camera ID
    camera_id = sim.model.camera_name2id(camera_name)
    
    # Create MuJoCo renderer with the unwrapped model
    renderer = mujoco.Renderer(mj_model, camera_height, camera_width)
    
    # Update renderer with current simulation state
    renderer.update_scene(mj_data, camera=camera_id)
    
    # Render RGB
    rgb = renderer.render()
    
    # Render depth
    renderer.enable_depth_rendering()
    depth = renderer.render()
    
    # MuJoCo depth is in the format we need (distance from camera)
    # Values are already in meters
    
    renderer.close()
    
    return rgb, depth


def visualize_pointcloud(pointcloud, rgb=None, title="Point Cloud Visualization", filename="pointcloud_3d.png"):
    """
    Visualize point cloud using matplotlib and save to file.
    
    Args:
        pointcloud: Point cloud array (N, 3)
        rgb: Optional RGB colors for points (N, 3) with values in [0, 255]
        title: Title for the plot
        filename: Output filename for the visualization
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points if too many (for performance)
    max_points = 50000
    if len(pointcloud) > max_points:
        indices = np.random.choice(len(pointcloud), max_points, replace=False)
        pc_sample = pointcloud[indices]
        rgb_sample = rgb[indices] if rgb is not None else None
    else:
        pc_sample = pointcloud
        rgb_sample = rgb
    
    # Filter out invalid points (e.g., very far or at origin)
    valid_mask = (pc_sample[:, 2] > 0.01) & (pc_sample[:, 2] < 10.0)
    pc_sample = pc_sample[valid_mask]
    if rgb_sample is not None:
        rgb_sample = rgb_sample[valid_mask]
    
    # Plot point cloud
    if rgb_sample is not None:
        # Normalize RGB to [0, 1] if needed
        if rgb_sample.max() > 1.0:
            rgb_sample = rgb_sample / 255.0
        ax.scatter(pc_sample[:, 0], pc_sample[:, 1], pc_sample[:, 2], 
                  c=rgb_sample, s=1, alpha=0.6)
    else:
        # Color by depth (Z coordinate)
        ax.scatter(pc_sample[:, 0], pc_sample[:, 1], pc_sample[:, 2], 
                  c=pc_sample[:, 2], cmap='viridis', s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (depth)')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([
        pc_sample[:, 0].max() - pc_sample[:, 0].min(),
        pc_sample[:, 1].max() - pc_sample[:, 1].min(),
        pc_sample[:, 2].max() - pc_sample[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (pc_sample[:, 0].max() + pc_sample[:, 0].min()) * 0.5
    mid_y = (pc_sample[:, 1].max() + pc_sample[:, 1].min()) * 0.5
    mid_z = (pc_sample[:, 2].max() + pc_sample[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


def save_pointcloud_image(rgb, depth, pointcloud, filename_prefix="output"):
    """
    Save RGB, depth, and point cloud visualization to files.
    
    Args:
        rgb: RGB image (H, W, 3)
        depth: Depth image (H, W)
        pointcloud: Point cloud array (N, 3)
        filename_prefix: Prefix for output files
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show RGB
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Show depth
    depth_vis = axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title('Depth Image')
    axes[1].axis('off')
    plt.colorbar(depth_vis, ax=axes[1], label='Depth (m)')
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_rgb_depth.png', dpi=150, bbox_inches='tight')
    print(f"Saved {filename_prefix}_rgb_depth.png")
    plt.close()

def create_env():
    """Create environment with visible renderer for macOS compatibility."""
    return make(
        "Stack",
        robots="Panda",
        has_renderer=True,  # Use visible renderer (works on macOS)
        has_offscreen_renderer=False,  # Don't need offscreen
        use_camera_obs=False,  # We'll capture manually
        use_object_obs=True,  # Get object states
        control_freq=20,
    )


def main():
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Robosuite version: {robosuite.__version__}")

    # Create environment
    env = create_env()
    obs = env.reset()
    print("Environment created successfully!")
    
    # Camera parameters
    camera_name = "frontview"
    camera_height = 256
    camera_width = 256
    
    # Capture RGB and depth images manually
    print(f"\nCapturing from camera: {camera_name}")
    rgb, depth = capture_camera_data(env, camera_name, camera_height, camera_width)
    
    print(f"RGB shape: {rgb.shape}")
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}] meters")
    
    # Get camera intrinsics using robosuite's function
    intrinsics = get_camera_intrinsic_matrix(env.sim, camera_name, camera_height, camera_width)
    print(f"Intrinsics shape: {intrinsics.shape}")
    print(f"Intrinsics:\n{intrinsics}")
    
    # Convert to point cloud
    pointcloud = depth_to_pointcloud(depth, intrinsics)
    print(f"\nPoint cloud shape: {pointcloud.shape}")
    print(f"Point cloud range:")
    print(f"  X: [{pointcloud[:, 0].min():.3f}, {pointcloud[:, 0].max():.3f}]")
    print(f"  Y: [{pointcloud[:, 1].min():.3f}, {pointcloud[:, 1].max():.3f}]")
    print(f"  Z: [{pointcloud[:, 2].min():.3f}, {pointcloud[:, 2].max():.3f}]")
    
    # Save RGB and depth visualization
    save_pointcloud_image(rgb, depth, pointcloud, filename_prefix="frontview")
    
    # Visualize point cloud with RGB colors
    # Reshape RGB to match point cloud
    rgb_flat = rgb.reshape(-1, 3)
    print("\nSaving point cloud visualization...")
    visualize_pointcloud(pointcloud, rgb=rgb_flat, 
                        title=f"Point Cloud from {camera_name}",
                        filename="frontview_pointcloud_3d.png")
    
    print("\nAll visualizations saved!")
    print("  - frontview_rgb_depth.png (RGB and depth side-by-side)")
    print("  - frontview_pointcloud_3d.png (3D point cloud visualization)")


    # Initialize
    pcd_gen = PointCloudGenerator(
        voxel_size=0.005,  # 5mm voxels
        # bounds=workspace_bounds  # Optional filtering
    )

    # Generate from multiple views
    pcd = pcd_gen.generate(env, ["frontview", "agentview", "birdview"])

    # Use the point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # pcd_gen.visualize_with_matplotlib(pcd, subsample=5)

    pcd_gen.save_point_cloud_file(pcd)
    
    print(f"\nGenerated point cloud with {len(points)} points from multiple views")

    
    # Optional: Render the viewer for a moment to see the scene
    env.render()
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()