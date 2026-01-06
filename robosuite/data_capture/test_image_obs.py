import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for macOS compatibility
import matplotlib.pyplot as plt

# Import mujoco for direct rendering
try:
    import mujoco
    MUJOCO_AVAILABLE = True
    print("MuJoCo module available ✓")
except ImportError:
    MUJOCO_AVAILABLE = False
    print("MuJoCo module not available ✗")

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def get_camera_image(sim, camera_name: str, width: int = 640, height: int = 480, 
                     camera_renderers=None):
    """
    Render RGB image from a specific camera using direct MuJoCo rendering.
    
    Args:
        sim: Robosuite sim object
        camera_name: Name of the camera to render from
        width: Image width
        height: Image height
        camera_renderers: Dict to store/reuse renderers (pass empty dict on first call)
    
    Returns:
        RGB image array (H, W, 3) uint8, or None if rendering fails
    """
    if not MUJOCO_AVAILABLE:
        print("MuJoCo not available for rendering")
        return None
    
    if camera_renderers is None:
        camera_renderers = {}
    
    try:
        # Get camera ID from name using robosuite's wrapper method
        camera_id = sim.model.camera_name2id(camera_name)
        
        if camera_id == -1:
            print(f"Camera '{camera_name}' not found in model")
            return None
        
        # Create renderer for this camera if it doesn't exist
        renderer_key = f"{camera_name}_{width}_{height}"
        
        if renderer_key not in camera_renderers:
            # Need to access the underlying MuJoCo model for the Renderer
            # Robosuite wraps the model, so we access ._model
            mj_model = sim.model._model if hasattr(sim.model, '_model') else sim.model
            camera_renderers[renderer_key] = mujoco.Renderer(
                mj_model, 
                height=height, 
                width=width
            )
            print(f"Created MuJoCo renderer for camera '{camera_name}' ({width}x{height})")
        
        renderer = camera_renderers[renderer_key]
        
        # Update renderer with current simulation state and specific camera
        # Need to access underlying MuJoCo data as well
        mj_data = sim.data._data if hasattr(sim.data, '_data') else sim.data
        renderer.update_scene(mj_data, camera=camera_id)
        
        # Render and return image
        image = renderer.render()
        return image
        
    except Exception as e:
        print(f"Failed to render camera '{camera_name}': {e}")
        import traceback
        traceback.print_exc()
        return None


def get_camera_depth(sim, camera_name: str, width: int = 640, height: int = 480, 
                     camera_renderers=None):
    """Render depth image (meters) from a specific camera using MuJoCo."""
    if not MUJOCO_AVAILABLE:
        print("MuJoCo not available for rendering")
        return None

    if camera_renderers is None:
        camera_renderers = {}

    try:
        camera_id = sim.model.camera_name2id(camera_name)
        if camera_id == -1:
            print(f"Camera '{camera_name}' not found in model")
            return None

        renderer_key = f"{camera_name}_{width}_{height}"
        if renderer_key not in camera_renderers:
            mj_model = sim.model._model if hasattr(sim.model, '_model') else sim.model
            camera_renderers[renderer_key] = mujoco.Renderer(
                mj_model,
                height=height,
                width=width,
            )
            print(f"Created MuJoCo depth renderer for camera '{camera_name}' ({width}x{height})")

        renderer = camera_renderers[renderer_key]
        mj_data = sim.data._data if hasattr(sim.data, '_data') else sim.data
        renderer.update_scene(mj_data, camera=camera_id)

        # Switch renderer to depth mode, render, then switch back
        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()

        if depth is None:
            print(f"Depth render returned None for camera '{camera_name}'")
            return None

        depth = np.asarray(depth, dtype=np.float32)
        depth = np.where(np.isfinite(depth), depth, 0.0)
        return depth

    except Exception as e:
        print(f"Failed to render depth for camera '{camera_name}': {e}")
        import traceback
        traceback.print_exc()
        return None


def get_camera_segmentation(sim, camera_name: str, width: int = 640, height: int = 480,
                            camera_renderers=None):
    """Render segmentation IDs from a camera using MuJoCo.

    Returns an int32 array of shape (H, W, 2) where the last dimension is
    (object_id, object_type). Object types follow mujoco.mjtObj (e.g., 5=geom).
    """
    if not MUJOCO_AVAILABLE:
        print("MuJoCo not available for rendering")
        return None

    if camera_renderers is None:
        camera_renderers = {}

    try:
        camera_id = sim.model.camera_name2id(camera_name)
        if camera_id == -1:
            print(f"Camera '{camera_name}' not found in model")
            return None

        renderer_key = f"{camera_name}_{width}_{height}"
        if renderer_key not in camera_renderers:
            mj_model = sim.model._model if hasattr(sim.model, '_model') else sim.model
            camera_renderers[renderer_key] = mujoco.Renderer(
                mj_model,
                height=height,
                width=width,
            )
            print(f"Created MuJoCo segmentation renderer for camera '{camera_name}' ({width}x{height})")

        renderer = camera_renderers[renderer_key]
        mj_data = sim.data._data if hasattr(sim.data, '_data') else sim.data
        renderer.update_scene(mj_data, camera=camera_id)

        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        renderer.disable_segmentation_rendering()

        if seg is None:
            print(f"Segmentation render returned None for camera '{camera_name}'")
            return None

        seg = np.asarray(seg, dtype=np.int32)
        return seg

    except Exception as e:
        print(f"Failed to render segmentation for camera '{camera_name}': {e}")
        import traceback
        traceback.print_exc()
        return None


def get_camera_intrinsics(sim, camera_name: str, width: int, height: int):
    """Compute pinhole intrinsics (fx, fy, cx, cy) from MuJoCo camera params."""
    camera_id = sim.model.camera_name2id(camera_name)
    if camera_id == -1:
        raise ValueError(f"Camera '{camera_name}' not found in model")

    fovy_deg = sim.model.cam_fovy[camera_id]
    near = sim.model.vis.map.znear
    far = sim.model.vis.map.zfar

    fovy_rad = np.deg2rad(fovy_deg)
    fy = height / (2.0 * np.tan(fovy_rad / 2.0))
    fovx_rad = 2.0 * np.arctan((width / height) * np.tan(fovy_rad / 2.0))
    fx = width / (2.0 * np.tan(fovx_rad / 2.0))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "fovx_rad": fovx_rad,
        "fovy_rad": fovy_rad,
        "near": near,
        "far": far,
        "width": width,
        "height": height,
    }

# Create environment WITHOUT offscreen renderer (to avoid GLFW issues)
controller_config = load_composite_controller_config(controller="BASIC")
env = suite.make(
    "Stack",  # Use simpler single-arm environment
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,                     # no on-screen rendering
    has_offscreen_renderer=False,           # DISABLED - we'll use direct MuJoCo rendering
    control_freq=20,
    horizon=200,
    use_object_obs=True,
    use_camera_obs=False,                   # DISABLED - we'll render manually
    reward_shaping=True,
)

print("Environment created successfully!")
print(f"Action dimension: {env.action_dim}")

# Reset environment
obs = env.reset()
print(f"\nObservation keys: {obs.keys()}")

# Test direct MuJoCo rendering
print("\n" + "="*60)
print("Testing Direct MuJoCo Camera Rendering")
print("="*60)

camera_renderers = {}  # Store renderers for reuse

# Try rendering from different cameras
cameras_to_test = ["frontview", "agentview", "birdview", "sideview"]
print("\nAvailable cameras in model:")
for i in range(env.sim.model.ncam):
    # Access the underlying MuJoCo model (robosuite wraps it)
    cam_name = env.sim.model.camera_id2name(i)
    print(f"  - {cam_name}")

print("\nRendering from primary camera...")
primary_camera = "frontview" #-------------------------------------------
img = get_camera_image(env.sim, primary_camera, width=640, height=480, 
                       camera_renderers=camera_renderers)

if img is not None:
    print(f"\n✓ Successfully rendered image!")
    print(f"  Shape: {img.shape}")
    print(f"  Dtype: {img.dtype}")
    print(f"  Value range: [{img.min()}, {img.max()}]")
else:
    print("\n✗ Failed to render image")
    exit(1)

# Test depth rendering
print("\nRendering depth from primary camera...")
depth = get_camera_depth(env.sim, primary_camera, width=640, height=480, 
                         camera_renderers=camera_renderers)

if depth is not None:
    print(f"\n✓ Successfully rendered depth!")
    print(f"  Shape: {depth.shape}")
    print(f"  Dtype: {depth.dtype}")
    print(f"  Value range: [{depth.min():.4f}, {depth.max():.4f}] meters")
else:
    print("\n✗ Failed to render depth")
    exit(1)

# Test segmentation rendering
print("\nRendering segmentation from primary camera...")
seg = get_camera_segmentation(env.sim, primary_camera, width=640, height=480,
                              camera_renderers=camera_renderers)

if seg is not None:
    seg_obj = seg[..., 0]
    seg_type = seg[..., 1]
    unique_ids = np.unique(seg_obj)
    unique_types = np.unique(seg_type)
    print(f"\n✓ Successfully rendered segmentation!")
    print(f"  Shape: {seg.shape}")
    print(f"  Dtype: {seg.dtype}")
    print(f"  Unique object ids (first 10): {unique_ids[:10]}")
    print(f"  Unique types (mjtObj): {unique_types}")
else:
    print("\n✗ Failed to render segmentation")
    exit(1)

# Compute and report camera intrinsics for policy consumption
intr = get_camera_intrinsics(env.sim, primary_camera, width=640, height=480)
print("\nCamera intrinsics (pinhole):")
print(f"  fx={intr['fx']:.2f}, fy={intr['fy']:.2f}, cx={intr['cx']:.2f}, cy={intr['cy']:.2f}")
print(f"  fovx={intr['fovx_rad']:.4f} rad, fovy={intr['fovy_rad']:.4f} rad")
print(f"  near={intr['near']:.4f}, far={intr['far']:.4f}")

# Run a few steps and collect images
print("\n" + "="*60)
print("Collecting RGB and depth images over 10 timesteps...")
print("="*60)

images_collected = []
depths_collected = []
segmentations_collected = []

for step in range(10):
    # Random action
    action = np.random.uniform(env.action_spec[0], env.action_spec[1])
    obs, reward, done, info = env.step(action)
    
    # Render camera image using direct MuJoCo rendering
    img = get_camera_image(env.sim, primary_camera, width=640, height=480,
                          camera_renderers=camera_renderers)
    
    # Render depth image
    depth = get_camera_depth(env.sim, primary_camera, width=640, height=480,
                            camera_renderers=camera_renderers)

    # Render segmentation image
    seg = get_camera_segmentation(env.sim, primary_camera, width=640, height=480,
                                  camera_renderers=camera_renderers)
    
    if img is not None:
        images_collected.append(img)
        if step == 0:
            print(f"Step {step}: Captured RGB {img.shape}")
    else:
        print(f"Step {step}: Failed to capture RGB")
    
    if depth is not None:
        depths_collected.append(depth)
        if step == 0:
            print(f"Step {step}: Captured depth {depth.shape}, range [{depth.min():.2f}, {depth.max():.2f}]")
    else:
        print(f"Step {step}: Failed to capture depth")

    if seg is not None:
        segmentations_collected.append(seg)
        if step == 0:
            obj_ids = np.unique(seg[..., 0])[:8]
            print(f"Step {step}: Captured segmentation {seg.shape}, example ids {obj_ids}")
    else:
        print(f"Step {step}: Failed to capture segmentation")
    
    if done:
        obs = env.reset()

print(f"\n✓ Collected {len(images_collected)} RGB frames, {len(depths_collected)} depth frames, and {len(segmentations_collected)} seg frames from camera '{primary_camera}'")

if len(images_collected) == 0:
    print("No images collected! Exiting.")
    env.close()
    exit(1)

# Visualize multiple frames
print("\n" + "="*60)
print("Saving visualizations...")
print("="*60)

# Show first, middle, and last frame for both RGB and depth
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# RGB images (top row)
axes[0, 0].imshow(images_collected[0])
axes[0, 0].set_title(f"RGB Frame 0 - {primary_camera}")
axes[0, 0].axis('off')

mid_idx = len(images_collected) // 2
axes[0, 1].imshow(images_collected[mid_idx])
axes[0, 1].set_title(f"RGB Frame {mid_idx} - {primary_camera}")
axes[0, 1].axis('off')

axes[0, 2].imshow(images_collected[-1])
axes[0, 2].set_title(f"RGB Frame {len(images_collected)-1} - {primary_camera}")
axes[0, 2].axis('off')

# Depth images (bottom row)
if len(depths_collected) > 0:
    axes[1, 0].imshow(depths_collected[0], cmap='viridis')
    axes[1, 0].set_title(f"Depth Frame 0 - {primary_camera}")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(depths_collected[mid_idx], cmap='viridis')
    axes[1, 1].set_title(f"Depth Frame {mid_idx} - {primary_camera}")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(depths_collected[-1], cmap='viridis')
    axes[1, 2].set_title(f"Depth Frame {len(depths_collected)-1} - {primary_camera}")
    axes[1, 2].axis('off')

# Segmentation images (third row, showing object_id channel)
if len(segmentations_collected) > 0:
    axes[2, 0].imshow(segmentations_collected[0][..., 0], cmap='tab20')
    axes[2, 0].set_title(f"Seg Frame 0 - {primary_camera}")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(segmentations_collected[mid_idx][..., 0], cmap='tab20')
    axes[2, 1].set_title(f"Seg Frame {mid_idx} - {primary_camera}")
    axes[2, 1].axis('off')

    axes[2, 2].imshow(segmentations_collected[-1][..., 0], cmap='tab20')
    axes[2, 2].set_title(f"Seg Frame {len(segmentations_collected)-1} - {primary_camera}")
    axes[2, 2].axis('off')

plt.tight_layout()
output_path = "mujoco_direct_rendering_test.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to '{output_path}'")

# Also save individual frames for inspection
from pathlib import Path
debug_dir = Path("./test_camera_images")
debug_dir.mkdir(exist_ok=True)

print("\nSaving individual frames...")
for i, img in enumerate(images_collected[:3]):  # Save first 3 frames
    frame_path = f"frame_{i:03d}.png"
    save_path = debug_dir / frame_path
    plt.imsave(save_path, img)
    print(f"  - Saved {save_path}")

for i, depth in enumerate(depths_collected[:3]):  # Save first 3 depth frames
    depth_path = f"depth_{i:03d}.png"
    save_path = debug_dir / depth_path
    plt.imsave(save_path, depth, cmap='viridis')
    print(f"  - Saved {save_path}")

for i, seg in enumerate(segmentations_collected[:3]):
    seg_path = f"seg_{i:03d}.png"
    save_path = debug_dir / seg_path
    plt.imsave(save_path, seg[..., 0], cmap='tab20')
    print(f"  - Saved {save_path}")

# Clean up
env.close()
print("\n" + "="*60)
print("✓ Test completed successfully!")
print("="*60)
print("\nSummary:")
print(f"  - MuJoCo direct rendering: {'✓ Working' if MUJOCO_AVAILABLE else '✗ Not available'}")
print(f"  - Frames collected: {len(images_collected)}")
print(f"  - Image resolution: {images_collected[0].shape if images_collected else 'N/A'}")
print(f"  - Output files: {output_path}, frame_000.png, frame_001.png, frame_002.png, seg_000.png")
print("\nThis approach avoids GLFW threading issues on macOS!")