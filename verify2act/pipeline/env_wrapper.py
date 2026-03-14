"""Environment wrapper for robosuite NutAssembly.

Adds goal-image rendering, text→action execution, and state
save/restore to the vanilla robosuite ``NutAssembly`` environment.

Usage::

    import robosuite
    from robosuite.controllers import load_composite_controller_config

    env = robosuite.make(
        "NutAssembly", robots="Panda",
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_offscreen_renderer=True, use_camera_obs=False, use_object_obs=True,
        control_freq=20, horizon=2000, ignore_done=True,
    )
    wrapper = NutAssemblyEnvWrapper(env, camera="agentview", image_size=512)
    obs, goal_img = wrapper.reset()
"""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from robosuite.robosuite.utils.camera_utils import render_camera

logger = logging.getLogger(__name__)

# ── Lazy import of render_camera (lives in the robosuite tree) ──────────

_render_camera = None


def _get_render_camera():
    global _render_camera
    if _render_camera is not None:
        return _render_camera

    # The robosuite repo sits next to verify2act in the workspace.
    # Ensure the package is importable.
    repo_root = Path(__file__).resolve().parents[2]
    robosuite_pkg = repo_root / "robosuite"
    if str(robosuite_pkg) not in sys.path:
        sys.path.insert(0, str(robosuite_pkg))

    from robosuite.utils.camera_utils import render_camera  # noqa: E402

    _render_camera = render_camera
    return _render_camera


# ── Lazy import of HeuristicNutAssemblyPolicy ──────────────────────────

_HeuristicPolicy = None


def _get_heuristic_policy_class():
    global _HeuristicPolicy
    if _HeuristicPolicy is not None:
        return _HeuristicPolicy

    repo_root = Path(__file__).resolve().parents[2]
    robosuite_dir = repo_root / "robosuite"
    if str(robosuite_dir) not in sys.path:
        sys.path.insert(0, str(robosuite_dir))

    # run_nutassembly.py defines the class at module scope.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_nutassembly", robosuite_dir / "run_nutassembly.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _HeuristicPolicy = mod.HeuristicNutAssemblyPolicy
    return _HeuristicPolicy


# ═══════════════════════════════════════════════════════════════════════
# NutAssemblyEnvWrapper
# ═══════════════════════════════════════════════════════════════════════


class NutAssemblyEnvWrapper:
    """Thin wrapper that adds goal images, text→action, and state helpers.

    Parameters
    ----------
    env :
        A live ``robosuite.NutAssembly`` environment instance.  Must have been
        created with ``has_offscreen_renderer=True`` and ``use_object_obs=True``.
    camera : str
        Camera name to render (e.g. ``"agentview"``).
    image_size : int
        Render resolution (square).
    """

    # Nut-type → peg mapping (matches env.nut_to_id in NutAssembly)
    _NUT_PEG_MAP = {"square": 0, "round": 1}

    def __init__(
        self,
        env,
        camera: str = "agentview",
        image_size: int = 512,
    ) -> None:
        self.env = env
        self.camera = camera
        self.image_size = image_size

        # Renderer cache — see EpisodeRecorder._render_rgb() pattern.
        self._camera_renderers: Dict[str, Any] = {}
        self._last_mj_model = None

        self._obs: Optional[Dict] = None

    # ── reset ──────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, np.ndarray]:
        """Reset the environment and return ``(obs, goal_image_np)``.

        Goal image is rendered by programmatically placing every nut on its
        matching peg, capturing the camera frame, then restoring the original
        (randomised) nut positions.
        """
        if seed is not None:
            self.env.seed(seed)
        self._obs = self.env.reset()

        # Invalidate renderers if MuJoCo rebuilt the model (hard_reset=True).
        self._maybe_flush_renderers()

        goal_img = self.render_goal_image()
        return self._obs, goal_img

    # ── goal image ─────────────────────────────────────────────────────

    def render_goal_image(self) -> np.ndarray:
        """Render the goal state (only active nuts on their pegs) and return ``[H, W, 3]`` uint8.

        Algorithm:
          1. Save the full simulator state.
          2. For each *active* nut this episode, teleport its body to its matching peg.
          3. Forward-step physics so the rendered scene is consistent.
          4. Render the camera frame.
          5. Restore the saved simulator state and re-forward.
        """
        sim = self.env.sim
        saved_state = deepcopy(sim.get_state())

        try:
            for nut in self._active_nuts():
                nut_name = nut.name  # e.g. "SquareNut" or "RoundNut"
                peg_id = self._peg_id_for_nut(nut_name)
                peg_body_id = self._peg_body_id(peg_id)
                peg_pos = np.array(sim.data.body_xpos[peg_body_id])

                nut_body_id = self.env.obj_body_id[nut_name]
                nut_jnt_addr = sim.model.body_jntadr[nut_body_id]

                if nut_jnt_addr < 0:
                    logger.warning("Nut %s has no joint; skipping goal placement.", nut_name)
                    continue

                # Free joints store 7 DoF in qpos: [x, y, z, qw, qx, qy, qz]
                # Place nut directly above the peg top.
                target_pos = peg_pos.copy()
                target_pos[2] = self.env.table_offset[2] + 0.02  # just above table

                sim.data.qpos[nut_jnt_addr: nut_jnt_addr + 3] = target_pos
                # Upright orientation (identity quaternion: w=1, x=y=z=0)
                sim.data.qpos[nut_jnt_addr + 3: nut_jnt_addr + 7] = [1, 0, 0, 0]

            sim.forward()
            goal_img = self._render_rgb()
        finally:
            sim.set_state(saved_state)
            sim.forward()

        return goal_img

    # ── observation helpers ────────────────────────────────────────────

    def read_image(self) -> np.ndarray:
        """Render current camera view as ``[H, W, 3]`` uint8."""
        return self._render_rgb()

    def get_obj_labels(self) -> List[str]:
        """Return human-readable object labels (e.g. ``["round nut", "square nut"]``)."""
        labels = []
        for nut in self.env.nuts:
            name = nut.name.lower()
            if "round" in name:
                labels.append("round nut")
            elif "square" in name:
                labels.append("square nut")
            else:
                labels.append(name.replace("_", " "))
        return sorted(labels)

    def is_done(self) -> bool:
        """Return ``True`` if the episode's target nut(s) are on their correct pegs."""
        return bool(self.env._check_success())

    # ── active-nut helper ──────────────────────────────────────────────

    def _active_nuts(self) -> list:
        """Return only the nuts that are active this episode.

        In ``single_object_mode`` 1 or 2 the environment picks one nut per
        episode and clears the others from the scene.  ``env.obj_to_use``
        holds that nut's name.  In mode 0 (default) every nut is active.
        """
        if getattr(self.env, "single_object_mode", 0) == 0:
            return self.env.nuts
        obj_to_use = getattr(self.env, "obj_to_use", None)
        if obj_to_use is None:
            return self.env.nuts
        return [nut for nut in self.env.nuts if nut.name == obj_to_use]

    # ── state save / restore (for oracle world model) ──────────────────

    def save_state(self) -> Any:
        """Snapshot the full simulator state (qpos + qvel)."""
        return deepcopy(self.env.sim.get_state())

    def restore_state(self, state) -> None:
        """Restore a previously-saved simulator state."""
        self.env.sim.set_state(state)
        self.env.sim.forward()

    # ── action execution ───────────────────────────────────────────────

    def execute_action(
        self, action_text: str, max_steps: int = 400
    ) -> Tuple[Dict, bool]:
        """Parse *action_text* and execute it using the heuristic policy.

        ``action_text`` has the form ``"<skill> <object>"`` (e.g.
        ``"pick round nut"``, ``"insert square nut"``).

        Returns ``(obs, skill_completed)`` where *skill_completed* is ``True``
        when the policy's stage machine naturally transitions past the
        requested skill (e.g. a pick completes at "lift_nut"→"move_to_peg").
        """
        skill, nut_query = self._parse_action_text(action_text)
        nut_name = self._resolve_nut_name(nut_query)
        peg_id = self._peg_id_for_nut(nut_name)

        HeuristicPolicy = _get_heuristic_policy_class()
        policy = HeuristicPolicy(self.env)

        # Force the policy to target the requested nut.
        policy.obs = self._obs
        policy.current_nut = nut_name
        policy.current_peg_id = peg_id
        policy.nuts_to_place = [nut_name]
        policy.grasp_attempts = 0

        # Set the starting stage based on skill.
        policy.stage = self._initial_stage(skill)

        terminal_stages = self._terminal_stages(skill)
        skill_completed = False

        for _ in range(max_steps):
            action, _done = policy.step()
            self._obs, _reward, env_done, _info = self.env.step(action)
            policy.obs = self._obs

            if policy.stage in terminal_stages:
                skill_completed = True
                break

            # If the state-machine moved to "done" or "skip_nut", stop.
            if policy.stage in ("done", "skip_nut"):
                break

        return self._obs, skill_completed

    # ── internal helpers ───────────────────────────────────────────────

    def _peg_id_for_nut(self, nut_name: str) -> int:
        name_lower = nut_name.lower()
        for key, pid in self._NUT_PEG_MAP.items():
            if key in name_lower:
                return pid
        raise ValueError(f"Cannot determine peg for nut '{nut_name}'")

    def _peg_body_id(self, peg_id: int) -> int:
        if peg_id == 0:
            return self.env.peg1_body_id
        return self.env.peg2_body_id

    @staticmethod
    def _parse_action_text(text: str) -> Tuple[str, str]:
        """Split ``'pick round nut'`` into ``('pick', 'round nut')``."""
        parts = text.strip().lower().split()
        if len(parts) < 2:
            raise ValueError(f"Cannot parse action_text: '{text}'")
        skill = parts[0]
        nut_query = " ".join(parts[1:])
        return skill, nut_query

    def _resolve_nut_name(self, nut_query: str) -> str:
        """Map an object query like ``'round nut'`` to the env's nut name
        (e.g. ``'RoundNut'``).  Ignores spatial qualifiers (front-left etc.)."""
        query_lower = nut_query.lower()
        for nut in self.env.nuts:
            name_lower = nut.name.lower()
            # Match core nut type, ignoring spatial qualifiers
            if "round" in query_lower and "round" in name_lower:
                return nut.name
            if "square" in query_lower and "square" in name_lower:
                return nut.name
        raise ValueError(
            f"No nut matching '{nut_query}' found. "
            f"Known nuts: {[n.name for n in self.env.nuts]}"
        )

    @staticmethod
    def _initial_stage(skill: str) -> str:
        """Map a skill name to the heuristic policy's starting stage."""
        skill = skill.lower()
        if skill == "pick":
            return "move_to_nut"
        if skill == "insert":
            return "move_to_peg"
        if skill in ("place", "put_down"):
            return "move_to_table"
        # Default: start from the beginning of the pick-place cycle.
        return "move_to_nut"

    @staticmethod
    def _terminal_stages(skill: str) -> frozenset:
        """Stages whose entry signals that *skill* has completed."""
        skill = skill.lower()
        if skill == "pick":
            # Pick completes when the policy transitions to "move_to_peg"
            # (nut is lifted and heading toward the peg).
            return frozenset({"move_to_peg"})
        if skill == "insert":
            # Insert completes at release or retract.
            return frozenset({"release", "retract", "reset_orientation"})
        if skill in ("place", "put_down"):
            return frozenset({"release", "retract", "reset_orientation"})
        return frozenset({"done"})

    # ── rendering ──────────────────────────────────────────────────────

    def _maybe_flush_renderers(self):
        """Detect hard-reset model replacement and clear the renderer cache."""
        current_model = id(self.env.sim.model)
        if self._last_mj_model is not None and self._last_mj_model != current_model:
            for renderer in self._camera_renderers.values():
                try:
                    renderer.close()
                except Exception:
                    pass
            self._camera_renderers.clear()
        self._last_mj_model = current_model

    def _render_rgb(self) -> np.ndarray:
        """Render via ``render_camera`` — same path as ``EpisodeRecorder``."""
        # render_camera = _get_render_camera()
        result = render_camera(
            self.env.sim,
            self._camera_renderers,
            self.camera,
            self.image_size,
            self.image_size,
        )
        if result is None:
            raise RuntimeError(
                f"render_camera returned None for camera '{self.camera}'"
            )
        cam_obs, self._camera_renderers = result
        return cam_obs.rgb
