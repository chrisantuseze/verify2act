"""
Point Cloud Generation for Robosuite Environments

This module provides clean utilities for generating point clouds from
multiple camera views in Robosuite simulation environments.
"""
import mujoco
import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple, Dict


class PointCloudGenerator:
    """Handles point cloud generation from Robosuite camera observations."""
    
    def __init__(self, voxel_size: float = 0.005, bounds: Optional[np.ndarray] = None):
        """
        Initialize the point cloud generator.
        
        Args:
            voxel_size: Size of voxels for downsampling (in meters)
            bounds: Optional workspace bounds as (3, 2) array [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        """
        self.voxel_size = voxel_size
        self.bounds = bounds
    
    def generate(self, env, camera_names: List[str]) -> o3d.geometry.PointCloud:
        """
        Generate a fused point cloud from multiple camera views.
        
        Args:
            env: Robosuite environment instance
            camera_names: List of camera names to use (e.g., ['frontview', 'agentview'])
            
        Returns:
            Open3D PointCloud object with fused observations
        """
        points_list = []
        colors_list = []
        
        for cam_name in camera_names:
            # Get observations from camera
            obs = self._get_camera_observation(env, cam_name)
            
            if obs is None:
                print(f"Warning: Could not get observation from camera '{cam_name}'")
                continue
            
            color, depth, intrinsics, extrinsics = obs
            
            # Backproject depth to 3D points in camera frame
            points_cam = self._backproject_depth(depth, intrinsics)
            
            # Transform to world frame
            points_world = self._transform_points(points_cam, extrinsics)
            
            # Flatten color to match points shape
            colors = color.reshape(-1, 3)
            points_world = points_world.reshape(-1, 3)
            
            # Filter by workspace bounds if provided
            if self.bounds is not None:
                mask = self._within_bounds(points_world, self.bounds)
                points_world = points_world[mask]
                colors = colors[mask]
            
            # Filter invalid depth readings (zero or NaN)
            valid_mask = np.isfinite(points_world).all(axis=1)
            points_world = points_world[valid_mask]
            colors = colors[valid_mask]
            
            points_list.append(points_world)
            colors_list.append(colors)
        
        if not points_list:
            raise ValueError("No valid point clouds generated from any camera")
        
        # Merge all views
        points = np.concatenate(points_list, axis=0)
        colors = np.concatenate(colors_list, axis=0)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        # Downsample
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.voxel_size)
        
        return pcd
    
    def _capture_camera_data(self, env, camera_name="frontview", camera_height=256, camera_width=256):
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
    
    def _get_camera_observation(
        self, 
        env, 
        camera_name: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get camera observation including color, depth, intrinsics, and extrinsics.
        
        Args:
            env: Robosuite environment
            camera_name: Name of the camera
            
        Returns:
            Tuple of (color, depth, intrinsics, extrinsics) or None if camera not found
        """
        # Get camera observations from environment
        # obs_dict = env.sim.render(
        #     camera_name=camera_name,
        #     width=env.camera_widths[0] if hasattr(env, 'camera_widths') else 256,
        #     height=env.camera_heights[0] if hasattr(env, 'camera_heights') else 256,
        #     depth=True
        # )

        color, depth = self._capture_camera_data(env, camera_name)
        
        # color = obs_dict[:, :, :3]  # RGB
        # depth = obs_dict[:, :, 3]    # Depth channel
        
        # Get camera parameters
        camera_id = env.sim.model.camera_name2id(camera_name)
        intrinsics = self._get_camera_intrinsics(env, camera_name)
        extrinsics = self._get_camera_extrinsics(env, camera_id)
        
        return color, depth, intrinsics, extrinsics
    
    def _get_camera_intrinsics(self, env, camera_name: str) -> np.ndarray:
        """
        Compute camera intrinsic matrix.
        
        Args:
            env: Robosuite environment
            camera_name: Name of the camera
            
        Returns:
            3x3 intrinsic matrix
        """
        # Get camera parameters from MuJoCo
        camera_id = env.sim.model.camera_name2id(camera_name)
        fovy = env.sim.model.cam_fovy[camera_id]
        
        # Get image dimensions
        width = env.camera_widths[0] if hasattr(env, 'camera_widths') else 256
        height = env.camera_heights[0] if hasattr(env, 'camera_heights') else 256
        
        # Compute focal length
        f = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        
        # Intrinsic matrix
        intrinsics = np.array([
            [f, 0, width / 2],
            [0, f, height / 2],
            [0, 0, 1]
        ])
        
        return intrinsics
    
    def _get_camera_extrinsics(self, env, camera_id: int) -> np.ndarray:
        """
        Get camera extrinsic matrix (world to camera transform).
        
        Args:
            env: Robosuite environment
            camera_id: MuJoCo camera ID
            
        Returns:
            4x4 extrinsic transformation matrix
        """
        # Get camera position and orientation
        cam_pos = env.sim.data.cam_xpos[camera_id]
        cam_mat = env.sim.data.cam_xmat[camera_id].reshape(3, 3)
        
        # Create SE(3) transformation matrix
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = cam_mat
        extrinsics[:3, 3] = cam_pos
        
        return extrinsics
    
    def _backproject_depth(self, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """
        Backproject depth image to 3D points in camera frame.
        
        Args:
            depth: HxW depth image
            intrinsics: 3x3 camera intrinsic matrix
            
        Returns:
            HxWx3 array of 3D points in camera coordinates
        """
        height, width = depth.shape
        
        # Create pixel grid
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        
        # Backproject using intrinsics
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        points = np.stack([x, y, z], axis=-1)
        
        return points
    
    def _transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Apply rigid transformation to 3D points.
        
        Args:
            points: Nx3 or HxWx3 array of 3D points
            transform: 4x4 transformation matrix
            
        Returns:
            Transformed points in same shape as input
        """
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        
        # Convert to homogeneous coordinates
        ones = np.ones((points_flat.shape[0], 1))
        points_homo = np.hstack([points_flat, ones])
        
        # Apply transformation
        points_transformed = (transform @ points_homo.T).T
        
        # Convert back to 3D
        points_transformed = points_transformed[:, :3]
        
        return points_transformed.reshape(original_shape)
    
    def _within_bounds(self, points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """
        Check which points are within workspace bounds.
        
        Args:
            points: Nx3 array of points
            bounds: (3, 2) array of bounds [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
            
        Returns:
            Boolean mask of length N
        """
        mask = np.ones(len(points), dtype=bool)
        
        for dim in range(3):
            mask &= (points[:, dim] >= bounds[dim, 0])
            mask &= (points[:, dim] <= bounds[dim, 1])
        
        return mask
    
    def visualize_with_matplotlib(self, pcd: o3d.geometry.PointCloud, subsample: int = 10):
        """
        Create a matplotlib 3D scatter plot of the point cloud.
        Useful for notebooks or when Open3D window doesn't work.
        
        Args:
            pcd: Open3D PointCloud to visualize
            subsample: Show every Nth point (for performance)
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Subsample for performance
        points = points[::subsample]
        colors = colors[::subsample]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(
            points[:, 0], 
            points[:, 1], 
            points[:, 2],
            c=colors,
            s=1,
            alpha=0.6
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud Visualization')
        
        # Equal aspect ratio
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # plt.show()

        plt.tight_layout()
        plt.savefig(f'full_pcd.png', dpi=150, bbox_inches='tight')
        plt.close()

    def save_point_cloud_file(self, pcd: o3d.geometry.PointCloud, filename: str = "full_pcd.ply"):
        """
        Save point cloud to file (PLY, PCD, XYZ formats supported).
        This is thread-safe and recommended for macOS.
        
        Args:
            pcd: Open3D PointCloud to save
            filename: Output filename (e.g., 'output.ply', 'output.pcd')
        """
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved point cloud to {filename}")
        print(f"To visualize: python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('{filename}')])\"")



def example_usage():
    """Example of how to use the PointCloudGenerator with Robosuite."""
    import robosuite as suite
    
    # Create environment
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["frontview", "agentview"],
        camera_heights=256,
        camera_widths=256,
    )
    
    # Reset environment
    env.reset()
    
    # Define workspace bounds (optional)
    bounds = np.array([
        [-0.5, 0.5],   # x bounds
        [-0.5, 0.5],   # y bounds
        [0.8, 1.5]     # z bounds (table height and above)
    ])
    
    # Create point cloud generator
    pcd_generator = PointCloudGenerator(voxel_size=0.005, bounds=bounds)
    
    # Generate point cloud from multiple views
    pcd = pcd_generator.generate(env, camera_names=["frontview", "agentview"])
    
    # Visualize (optional)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, coord_frame])
    
    # Access point cloud data
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"Generated point cloud with {len(points)} points")
    
    env.close()


if __name__ == "__main__":
    example_usage()