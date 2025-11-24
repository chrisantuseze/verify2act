"""
Point Cloud Generation for Robosuite Environments

This module provides clean utilities for generating point clouds from
multiple camera views in Robosuite simulation environments.
"""
import mujoco
import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple, Dict
from robosuite.environments.base import make


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
    
    def generate_segmented(
        self, 
        env, 
        camera_names: List[str],
        object_names: Optional[List[str]] = None
    ) -> Dict[str, o3d.geometry.PointCloud]:
        """
        Generate object-specific point clouds using segmentation masks.
        
        Args:
            env: Robosuite environment instance
            camera_names: List of camera names to use
            object_names: Optional list of specific object names to extract.
                         If None, extracts all objects in the scene.
            
        Returns:
            Dictionary mapping object names to their point clouds
        """
        # Storage for each object's points across all views
        object_points = {}
        object_colors = {}
        
        for cam_name in camera_names:
            # Get observations including segmentation
            obs = self._get_camera_observation_with_segmentation(env, cam_name)
            
            if obs is None:
                print(f"Warning: Could not get observation from camera '{cam_name}'")
                continue
            
            color, depth, seg_mask, intrinsics, extrinsics = obs
            
            # Backproject depth to 3D points in camera frame
            points_cam = self._backproject_depth(depth, intrinsics)
            
            # Transform to world frame
            points_world = self._transform_points(points_cam, extrinsics)
            
            # Flatten arrays
            points_flat = points_world.reshape(-1, 3)
            colors_flat = color.reshape(-1, 3)
            seg_flat = seg_mask.flatten()
            
            # Get unique object IDs from segmentation mask
            unique_ids = np.unique(seg_flat)
            
            # Get object name mapping from environment
            obj_id_to_name = self._get_object_id_mapping(env)
            
            for obj_id in unique_ids:
                # Skip background (usually ID 0 or -1)
                if obj_id <= 0:
                    continue
                
                # Get object name
                obj_name = obj_id_to_name.get(obj_id, f"object_{obj_id}")
                
                # Filter by object names if specified
                if object_names is not None and obj_name not in object_names:
                    continue
                
                # Extract points for this object
                mask = (seg_flat == obj_id)
                obj_pts = points_flat[mask]
                obj_cols = colors_flat[mask]
                
                # Filter invalid points
                valid_mask = np.isfinite(obj_pts).all(axis=1)
                obj_pts = obj_pts[valid_mask]
                obj_cols = obj_cols[valid_mask]
                
                # Filter by bounds if specified
                if self.bounds is not None:
                    bounds_mask = self._within_bounds(obj_pts, self.bounds)
                    obj_pts = obj_pts[bounds_mask]
                    obj_cols = obj_cols[bounds_mask]
                
                # Accumulate points for this object
                if obj_name not in object_points:
                    object_points[obj_name] = []
                    object_colors[obj_name] = []
                
                object_points[obj_name].append(obj_pts)
                object_colors[obj_name].append(obj_cols)
        
        # Create point clouds for each object
        object_pcds = {}
        for obj_name in object_points:
            if not object_points[obj_name]:
                continue
            
            # Merge points from all views
            points = np.concatenate(object_points[obj_name], axis=0)
            colors = np.concatenate(object_colors[obj_name], axis=0)
            
            if len(points) == 0:
                continue
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            
            # Downsample
            if self.voxel_size > 0:
                pcd = pcd.voxel_down_sample(self.voxel_size)
            
            object_pcds[obj_name] = pcd
        
        return object_pcds
    
    def generate_single_object(
        self,
        env,
        camera_names: List[str],
        object_name: str
    ) -> Optional[o3d.geometry.PointCloud]:
        """
        Generate point cloud for a specific object.
        
        Args:
            env: Robosuite environment instance
            camera_names: List of camera names to use
            object_name: Name of the object to extract
            
        Returns:
            Point cloud of the specified object, or None if not found
        """
        object_pcds = self.generate_segmented(env, camera_names, [object_name])
        return object_pcds.get(object_name, None)
    
    def _get_camera_observation_with_segmentation(
        self, 
        env, 
        camera_name: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get camera observation including segmentation mask.
        
        Args:
            env: Robosuite environment
            camera_name: Name of the camera
            
        Returns:
            Tuple of (color, depth, segmentation, intrinsics, extrinsics) or None if camera not found
        """
        width = env.camera_widths[0] if hasattr(env, 'camera_widths') else 256
        height = env.camera_heights[0] if hasattr(env, 'camera_heights') else 256
        
        # Get RGB-D
        obs_dict = env.sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            depth=True
        )
        
        color = obs_dict[:, :, :3]  # RGB
        depth = obs_dict[:, :, 3]    # Depth channel
        
        # Get segmentation mask
        # Robosuite uses MuJoCo's segmentation which provides geom IDs
        seg_mask = env.sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            depth=False,
            segmentation=True
        )[:, :, 0]  # First channel contains object IDs
        
        # Get camera parameters
        camera_id = env.sim.model.camera_name2id(camera_name)
        intrinsics = self._get_camera_intrinsics(env, camera_name)
        extrinsics = self._get_camera_extrinsics(env, camera_id)
        
        return color, depth, seg_mask, intrinsics, extrinsics
    
    def _get_object_id_mapping(self, env) -> Dict[int, str]:
        """
        Get mapping from segmentation IDs to object names.
        
        Args:
            env: Robosuite environment
            
        Returns:
            Dictionary mapping object IDs to names
        """
        id_to_name = {}
        
        # Get all geom names and IDs from MuJoCo model
        model = env.sim.model
        
        for i in range(model.ngeom):
            geom_id = i + 1  # MuJoCo geom IDs start at 1
            geom_name = model.geom_id2name(i)
            
            if geom_name is None:
                continue
            
            # Parse object name from geom name
            # Robosuite typically names geoms like "object_geom", "cube_g0", etc.
            obj_name = self._parse_object_name(geom_name, env)
            id_to_name[geom_id] = obj_name
        
        return id_to_name
    
    def _parse_object_name(self, geom_name: str, env) -> str:
        """
        Parse a clean object name from MuJoCo geom name.
        
        Args:
            geom_name: Raw geom name from MuJoCo
            env: Robosuite environment
            
        Returns:
            Clean object name
        """
        # Common Robosuite object prefixes
        object_keywords = ['cube', 'can', 'milk', 'bread', 'cereal', 'object', 
                          'peg', 'box', 'target', 'obstacle', 'gripper', 'robot']
        
        geom_lower = geom_name.lower()
        
        # Check if this is a robot/gripper geom
        if any(kw in geom_lower for kw in ['gripper', 'robot', 'eef', 'link']):
            return 'robot'
        
        # Check if this is a table/floor geom
        if any(kw in geom_lower for kw in ['table', 'floor', 'ground', 'arena']):
            return 'table'
        
        # Try to identify the object
        for keyword in object_keywords:
            if keyword in geom_lower:
                # Remove suffixes like _g0, _visual, etc.
                clean_name = geom_name.split('_')[0]
                return clean_name
        
        # Default: use the full geom name
        return geom_name
    
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
    
    print("=== Example 1: Full Scene Point Cloud ===")
    # Generate point cloud from multiple views
    pcd = pcd_generator.generate(env, camera_names=["frontview", "agentview"])
    print(f"Generated full scene point cloud with {len(np.asarray(pcd.points))} points")
    
    print("\n=== Example 2: Segmented Object Point Clouds ===")
    # Generate point clouds for each object separately
    object_pcds = pcd_generator.generate_segmented(env, camera_names=["frontview", "agentview"])
    
    print(f"Found {len(object_pcds)} objects:")
    for obj_name, obj_pcd in object_pcds.items():
        num_points = len(np.asarray(obj_pcd.points))
        print(f"  - {obj_name}: {num_points} points")
    
    print("\n=== Example 3: Extract Specific Object ===")
    # Get point cloud for a specific object (e.g., the cube in Lift task)
    cube_pcd = pcd_generator.generate_single_object(
        env, 
        camera_names=["frontview", "agentview"],
        object_name="cube"  # Adjust based on your environment
    )
    
    if cube_pcd:
        print(f"Extracted cube point cloud with {len(np.asarray(cube_pcd.points))} points")
    else:
        print("Cube not found. Available objects:", list(object_pcds.keys()))
    
    env.close()


if __name__ == "__main__":
    example_usage()