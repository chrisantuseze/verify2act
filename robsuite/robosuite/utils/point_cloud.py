import numpy as np

from . import camera_utils as CU
from .binding_utils import MjRenderContextOffscreen


def depth_to_pointcloud(depth, K, mask=None):
    """Back-project a depth map to a point cloud in camera frame.

    Args:
        depth (np.ndarray): HxW depth map (unnormalized, in meters)
        K (np.ndarray): 3x3 camera intrinsic matrix
        mask (np.ndarray, optional): boolean HxW mask selecting pixels

    Returns:
        points_cam (np.ndarray): Nx3 points in camera frame
        idxs (np.ndarray): Nx2 integer pixel indices (v, u)
    """
    assert depth.ndim == 2, "depth must be HxW"
    H, W = depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    us = np.arange(W)
    vs = np.arange(H)
    uu, vv = np.meshgrid(us, vs)

    z = depth
    valid = np.isfinite(z) & (z > 0)
    if mask is not None:
        valid &= mask

    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

    u = uu[valid].astype(np.float32)
    v = vv[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    pix = np.stack([v.astype(np.int32), u.astype(np.int32)], axis=1)
    return pts, pix


def _seg_to_body_ids(segmentation, model):
    """Map per-pixel geom ids to body ids using MuJoCo model.

    Args:
        segmentation (np.ndarray): HxWx2, channel 1 contains geom ids
        model: mjModel from env.sim.model

    Returns:
        body_ids (np.ndarray): HxW int array of body ids (-1 for background)
    """
    assert segmentation.ndim == 3 and segmentation.shape[2] >= 2
    geom_ids = segmentation[..., 1].astype(np.int32)
    body_ids = np.full(geom_ids.shape, -1, dtype=np.int32)
    valid = geom_ids >= 0
    if np.any(valid):
        lut = model.geom_bodyid
        body_ids[valid] = lut[geom_ids[valid]]
    return body_ids


def get_pointcloud(
    sim,
    camera_name,
    camera_height,
    camera_width,
    return_world=True,
    with_rgb=False,
    with_segmentation=False,
    exclude_body_name_prefix=("robot",),
):
    """Capture a point cloud from a MuJoCo camera.

    Args:
        sim: env.sim
        camera_name (str): camera name
        camera_height (int): image height
        camera_width (int): image width
        return_world (bool): transform points to world frame if True
        with_rgb (bool): also return per-point RGB
        with_segmentation (bool): also return per-point body id and names
        exclude_body_name_prefix (tuple[str]): drop points whose body name starts with any prefix

    Returns:
        dict with keys:
            points_cam: Nx3
            points_world: Nx3 (if return_world)
            rgb: Nx3 uint8 (if with_rgb)
            body_ids: N (if with_segmentation)
            body_names: list[str] (unique names in output, if with_segmentation)
            pixels: Nx2 (v, u)
    """
    # Ensure offscreen renderer exists (some scripts may not initialize it)
    if getattr(sim, "_render_context_offscreen", None) is None:
        # Use provided size to avoid immediate resizing
        MjRenderContextOffscreen(sim, device_id=-1, max_width=camera_width, max_height=camera_height)

    # RGB
    rgb = None
    if with_rgb:
        rgb = sim.render(camera_name=camera_name, height=camera_height, width=camera_width)[::-1]

    # Depth (unnormalized)
    depth = sim.render(
        camera_name=camera_name, height=camera_height, width=camera_width, depth=True
    )[::-1]
    depth = CU.get_real_depth_map(sim=sim, depth_map=depth)

    # Intrinsics / extrinsics
    K = CU.get_camera_intrinsic_matrix(
        sim=sim, camera_name=camera_name, camera_height=camera_height, camera_width=camera_width
    )
    T_cam_world = CU.get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)

    seg = None
    body_ids_img = None
    if with_segmentation:
        seg = CU.get_camera_segmentation(
            sim=sim, camera_name=camera_name, camera_height=camera_height, camera_width=camera_width
        )
        body_ids_img = _seg_to_body_ids(seg, sim.model)

    # Optional masking to exclude certain bodies (e.g., robot)
    mask = None
    if body_ids_img is not None and exclude_body_name_prefix:
        mask = np.ones(depth.shape, dtype=bool)
        unique_ids = np.unique(body_ids_img)
        drop_ids = []
        for bid in unique_ids:
            if bid < 0:
                continue
            name = sim.model.body_id2name(int(bid))
            if any(name.startswith(prefix) for prefix in exclude_body_name_prefix):
                drop_ids.append(bid)
        for bid in drop_ids:
            mask &= (body_ids_img != bid)

    pts_cam, pix = depth_to_pointcloud(depth, K, mask=mask)

    result = {
        "points_cam": pts_cam,
        "pixels": pix,
    }

    if return_world and pts_cam.shape[0] > 0:
        ones = np.ones((pts_cam.shape[0], 1), dtype=pts_cam.dtype)
        cam_h = np.concatenate([pts_cam, ones], axis=1)
        world_h = (T_cam_world @ cam_h.T).T
        pts_world = world_h[:, :3]
        result["points_world"] = pts_world

    if with_rgb and rgb is not None and pts_cam.shape[0] > 0:
        v, u = result["pixels"].T
        result["rgb"] = rgb[v, u]

    if with_segmentation and body_ids_img is not None and pts_cam.shape[0] > 0:
        v, u = result["pixels"].T
        body_ids = body_ids_img[v, u]
        result["body_ids"] = body_ids
        # Provide unique names list for convenience
        unique = np.unique(body_ids[body_ids >= 0])
        result["body_names"] = [sim.model.body_id2name(int(b)) for b in unique]

    return result
