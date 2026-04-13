"""
Goal renderers for robosuite environments.

Two renderers are provided:

StackGoalRenderer
    For StackMulti environments (Stack, Stack3, Stack4).
    Renders the desired tower configuration (cubeB base, cubeA/C/D stacked on
    top) anchored to the base cube's settled position.

NutAssemblyGoalRenderer
    For ClutteredNutAssembly (and plain NutAssembly) environments.
    Renders the goal state where every target-type nut is placed on its
    corresponding peg, with non-target nuts left at their settled positions.

Both renderers use the same save → teleport → render → restore cycle:
    1. sim.get_state() + sim.data.ctrl.copy()   — snapshot full MuJoCo state
    2. Teleport objects to goal poses + sim.forward()
    3. render_camera(...)                        — capture goal RGB
    4. sim.set_state() + ctrl restore + sim.forward()  — exact restore
"""

import numpy as np
from typing import Any, Dict, Optional

# Canonical stacking order, bottom → top, for StackMulti environments.
# Filtered at runtime to the cubes that actually exist in the variant in use.
_STACK_ORDER = ["cubeB", "cubeA", "cubeC", "cubeD"]


class StackGoalRenderer:
    """
    Renders a goal image for StackMulti environments.

    The base cube (cubeB) is left at its settled position; all cubes above it
    are teleported into the stacked goal pose before rendering, then the scene
    is restored unconditionally.

    Args:
        env: A StackMulti environment instance (must have been reset at least
             once so cube objects are initialised).
        camera: Camera name to render from (e.g. ``"agentview"``).
        image_size: Square render resolution in pixels.
    """

    def __init__(self, env, camera: str = "agentview", image_size: int = 512):
        self.env = env
        self.camera = camera
        self.image_size = image_size
        # Renderer cache; keyed by "camera_width_height" — same pattern as
        # camera_utils so the allocation logic is identical.
        self._camera_renderers: Dict[str, Any] = {}

        # Filter to cubes present in this environment variant.
        self._stack_order = [c for c in _STACK_ORDER if hasattr(env, c)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_goal(self) -> Optional[np.ndarray]:
        """Render the goal image and return a uint8 RGB array (H×W×3).

        Must be called *after* ``env.reset()`` has settled so the base cube's
        (cubeB) position is known.  The scene is restored to the exact
        pre-render state unconditionally — even if rendering fails.

        Returns:
            uint8 ndarray of shape (image_size, image_size, 3), or ``None``
            when rendering fails.
        """
        from robosuite.utils.camera_utils import render_camera

        if len(self._stack_order) < 2:
            print("[GoalRenderer] Fewer than 2 cubes found — cannot render goal.")
            return None

        sim = self.env.sim

        # 1. Snapshot full state before any teleportation.
        saved_state = sim.get_state()
        saved_ctrl = sim.data.ctrl.copy()

        goal_rgb: Optional[np.ndarray] = None
        try:
            # 2 & 3. Teleport objects to goal poses and call sim.forward().
            self._teleport_to_goal()

            # 4. Render.
            result = render_camera(
                sim,
                self._camera_renderers,
                self.camera,
                self.image_size,
                self.image_size,
                rgb_only=True,
            )
            if result is not None:
                cam_obs, self._camera_renderers = result
                goal_rgb = cam_obs.rgb.copy()
            else:
                print(
                    f"[GoalRenderer] render_camera returned None for camera '{self.camera}'"
                )
        finally:
            # 5. Restore unconditionally.
            sim.set_state(saved_state)
            sim.data.ctrl[:] = saved_ctrl
            sim.forward()

        return goal_rgb

    def flush_renderers(self):
        """Close all cached renderers.

        Call this whenever the underlying sim/model object is replaced (e.g.
        after robosuite's hard_reset recreates the env) so that the next render
        allocates a fresh renderer against the new model.
        """
        for renderer in self._camera_renderers.values():
            try:
                renderer.close()
            except Exception:
                pass
        self._camera_renderers.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _teleport_to_goal(self):
        """Set non-base cubes to their stacked goal poses, then call forward().

        The base cube (cubeB) is left untouched.  Each subsequent cube is
        placed with its bottom surface resting flush on the top surface of
        the cube below it, centred on the base cube's (x, y).
        """
        sim = self.env.sim
        base_name = self._stack_order[0]  # "cubeB"

        base_body_id = sim.model.body_name2id(
            getattr(self.env, base_name).root_body
        )
        base_pos = sim.data.body_xpos[base_body_id].copy()

        # current_top_z = z-coordinate of the top surface of the topmost cube
        # placed so far (starts at the top surface of the base cube).
        current_top_z = base_pos[2] + self._half_z(base_name)

        for cube_name in self._stack_order[1:]:
            half_z = self._half_z(cube_name)
            # Bottom of this cube sits exactly on current_top_z.
            goal_z = current_top_z + half_z
            goal_pos = np.array([base_pos[0], base_pos[1], goal_z])
            # Upright (identity) quaternion: [qw, qx, qy, qz].
            goal_quat = np.array([1.0, 0.0, 0.0, 0.0])

            joint_name = getattr(self.env, cube_name).joints[0]
            sim.data.set_joint_qpos(
                joint_name, np.concatenate([goal_pos, goal_quat])
            )
            current_top_z = goal_z + half_z

        # Recompute all Cartesian quantities so the renderer sees goal poses.
        sim.forward()

    def _half_z(self, cube_name: str) -> float:
        """Return the collision half-extent along z for the named cube."""
        return float(getattr(self.env, cube_name).size[2])


# ---------------------------------------------------------------------------
# NutAssembly goal renderer
# ---------------------------------------------------------------------------

class NutAssemblyGoalRenderer:
    """
    Renders a goal image for ClutteredNutAssembly (and plain NutAssembly) envs.

    Goal definition: every *target-type* nut is placed on its corresponding peg
    at the correct height, with an upright orientation.  Non-target nuts are
    left at their settled positions so the goal image accurately shows the
    expected scene (obstacle nuts still on the table).

    The nut-to-peg assignment follows ``env.nut_type_to_peg``:
        squarenut → peg 0   (square peg)
        roundnut  → peg 1   (round peg)

    For ClutteredNutAssembly with multiple round nuts, all round nuts are
    stacked above peg 1 in a column so the goal image remains unambiguous.

    Height computation uses the nut's ``bottom_offset`` / ``top_offset`` XML
    sites (accessed via ``MujocoXMLObject.bottom_offset`` / ``.top_offset``)
    so there is no visual interpenetration.

    Args:
        env: A ClutteredNutAssembly or NutAssembly environment instance that
             has been reset at least once.
        camera: Camera name to render from.
        image_size: Square render resolution in pixels.
        target_nut_type: ``"roundnut"`` or ``"squarenut"``.  When ``None``
            (default), the renderer reads ``env.current_nut_type`` each time
            ``render_goal()`` is called so it follows the episode's target.
    """

    def __init__(
        self,
        env,
        camera: str = "agentview",
        image_size: int = 512,
        target_nut_type: Optional[str] = None,
    ):
        self.env = env
        self.camera = camera
        self.image_size = image_size
        self._fixed_target_type = target_nut_type
        self._camera_renderers: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_goal(self) -> Optional[np.ndarray]:
        """Render the goal image and return a uint8 RGB array (H×W×3).

        Must be called after ``env.reset()`` has settled.  The scene is
        restored to the exact pre-render state unconditionally.

        Returns:
            uint8 ndarray of shape (image_size, image_size, 3), or ``None``
            when rendering fails.
        """
        from robosuite.utils.camera_utils import render_camera

        sim = self.env.sim
        saved_state = sim.get_state()
        saved_ctrl = sim.data.ctrl.copy()

        goal_rgb: Optional[np.ndarray] = None
        try:
            self._teleport_to_goal()

            result = render_camera(
                sim,
                self._camera_renderers,
                self.camera,
                self.image_size,
                self.image_size,
                rgb_only=True,
            )
            if result is not None:
                cam_obs, self._camera_renderers = result
                goal_rgb = cam_obs.rgb.copy()
            else:
                print(
                    f"[NutAssemblyGoalRenderer] render_camera returned None "
                    f"for camera '{self.camera}'"
                )
        finally:
            sim.set_state(saved_state)
            sim.data.ctrl[:] = saved_ctrl
            sim.forward()

        return goal_rgb

    def flush_renderers(self):
        """Close all cached renderers (call after hard_reset replaces the sim)."""
        for renderer in self._camera_renderers.values():
            try:
                renderer.close()
            except Exception:
                pass
        self._camera_renderers.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _teleport_to_goal(self):
        """Teleport target nuts to their peg positions, then call sim.forward()."""
        env = self.env
        sim = env.sim

        # Resolve which nut type is the target for this episode.
        target_type = self._fixed_target_type or getattr(env, "current_nut_type", "roundnut")

        if target_type == "roundnut":
            target_nut_names = list(env.round_nut_names)
        else:
            target_nut_names = list(env.square_nut_names)

        # peg_id for the target type
        peg_id = env.nut_type_to_peg[target_type]  # int: 0 or 1
        peg_body_id = env.peg_body_ids[peg_id]
        peg_pos = sim.data.body_xpos[peg_body_id].copy()

        # Use the table surface as the starting z for nut placement.
        # Heights are computed from actual collision geom extents (not XML
        # sites) so that stacked nuts are flush with no gaps.
        table_z = float(env.table_offset[2])
        current_z = table_z  # = collision bottom surface of the next nut

        for nut_name in target_nut_names:
            nut_obj = env.nuts[nut_name]
            bot_z, top_z = self._nut_geom_z_extent(nut_name)
            # bot_z < 0: offset from joint origin to collision bottom
            # goal_z: joint origin such that collision bottom == current_z
            goal_z = current_z - bot_z
            goal_pos = np.array([peg_pos[0], peg_pos[1], goal_z])

            # Preserve the nut's settled orientation — the robot places the
            # nut onto the peg in whatever yaw it was grasped; don't override.
            joint_name = nut_obj.joints[0]
            current_qpos = sim.data.get_joint_qpos(joint_name)
            goal_quat = current_qpos[3:].copy()  # [qw, qx, qy, qz] from settled state
            sim.data.set_joint_qpos(
                joint_name, np.concatenate([goal_pos, goal_quat])
            )

            # Next nut's collision bottom = this nut's collision top
            current_z = goal_z + top_z

        sim.forward()

    def _nut_geom_z_extent(self, nut_name: str):
        """Return (bot_z, top_z) of collision geoms in the nut's joint-origin frame.

        Both values are relative to the free-joint origin (root body).
        bot_z is negative (below origin), top_z is positive (above origin).
        Falls back to (-0.01, 0.01) if no collision geoms are found.
        """
        sim = self.env.sim
        nut_obj = self.env.nuts[nut_name]

        # Locate root body via the free joint
        joint_id = sim.model.joint_name2id(nut_obj.joints[0])
        root_body_id = sim.model.jnt_bodyid[joint_id]

        # Collect all body IDs in this nut's subtree (BFS/DFS)
        body_ids = set()
        stack = [root_body_id]
        while stack:
            bid = stack.pop()
            body_ids.add(bid)
            for child_bid in range(sim.model.nbody):
                if sim.model.body_parentid[child_bid] == bid:
                    stack.append(child_bid)

        min_z, max_z = float("inf"), float("-inf")
        for gid in range(sim.model.ngeom):
            bid = sim.model.geom_bodyid[gid]
            if bid not in body_ids:
                continue
            if sim.model.geom_contype[gid] == 0:  # visual-only, skip
                continue
            # Accumulate body-frame z offsets from this body up to root
            offset_z = 0.0
            b = bid
            while b != root_body_id:
                offset_z += sim.model.body_pos[b, 2]
                b = sim.model.body_parentid[b]
            gz = sim.model.geom_pos[gid, 2] + offset_z
            sz = sim.model.geom_size[gid, 2]  # half-height (box z)
            min_z = min(min_z, gz - sz)
            max_z = max(max_z, gz + sz)

        if min_z == float("inf"):
            return (-0.01, 0.01)  # fallback
        return (min_z, max_z)

