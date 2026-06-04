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

    # run_cluttered_nutassembly.py defines the class at module scope.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_cluttered_nutassembly", robosuite_dir / "run_cluttered_nutassembly.py"
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
        # Persistent heuristic policy instance — create lazily and reuse so
        # the environment is not re-initialized every time a skill runs.
        self._policy: Optional[Any] = None
        # Track placed nuts to avoid repeats
        self.placed: Set[str] = set()

    # ── reset ──────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Dict:
        """Reset the environment and return the settled T=0 obs.

        Goal image rendering is the caller's responsibility — use
        ``NutAssemblyGoalRenderer.render_goal()`` after this returns.

        Sequence
        --------
        1. Hard-reset the env (places objects at sampled positions).
        2. Run ``settle_steps`` raw physics steps so stacked/touching objects
           reach a fully stable configuration — this is the true T=0 state.
        3. Force the on-screen viewer to refresh to T=0.
        """
        if seed is not None:
            if hasattr(self.env, "seed") and callable(self.env.seed):
                self.env.seed(seed)
            else:
                np.random.seed(seed)
        self._obs = self.env.reset()

        # Invalidate renderers if MuJoCo rebuilt the model (hard_reset=True).
        self._maybe_flush_renderers()

        # Let physics fully settle after random placement / stacking.
        self._settle_and_sync_viewer()

        # Re-read observations so self._obs reflects the settled T=0 state
        # (env.reset() returns obs from before settling sim steps).
        self._obs = self.env._get_observations(force_update=True)
        self.placed.clear()
        return self._obs

    def _settle_and_sync_viewer(self, n_steps: int = 100, sync_viewer: bool = True) -> None:
        """Run raw physics steps until objects are stable, then optionally sync the viewer.

        Uses ``sim.step()`` (dynamics integration) rather than
        ``sim.forward()`` (geometry propagation only) so that stacked nuts
        actually fall into their resting positions under gravity.

        After settling, forces ``viewer.update()`` if sync_viewer is True so the on-screen passive
        viewer window is created / refreshed to the settled state *before* we
        render the goal image or block on matplotlib.  This ensures the viewer
        and our offscreen renders (goal + current) are always in sync.
        """
        sim = self.env.sim
        for _ in range(n_steps):
            sim.step()
        sim.forward()   # propagate final geometry

        if sync_viewer:
            # Refresh on-screen viewer — launches it if not yet open, or syncs it.
            viewer = getattr(self.env, "viewer", None)
            if viewer is not None and hasattr(viewer, "update"):
                viewer.update()

    # ── observation helpers ────────────────────────────────────────────

    def read_image(self) -> np.ndarray:
        """Render current camera view as ``[H, W, 3]`` uint8."""
        return self._render_rgb()

    def get_obj_labels(self) -> List[str]:
        """Return human-readable labels with unique IDs for all workspace nuts.

        E.g. ["left round nut (id: RoundNut0)", "right round nut (id: RoundNut1)"]
        """
        labels = []
        active = self._all_workspace_nuts()
        round_nuts = [name for name in active if "round" in name.lower()]
        square_nuts = [name for name in active if "square" in name.lower()]
        
        def get_x_pos(name):
            try:
                body_id = self.env.sim.model.body_name2id(name)
                return self.env.sim.data.body_xpos[body_id][0]
            except Exception:
                try:
                    body_id = self.env.sim.model.body_name2id(f"{name}_main")
                    return self.env.sim.data.body_xpos[body_id][0]
                except Exception:
                    return 0.0

        # Sort round/square nuts by x position to generate spatial qualifiers
        round_nuts = sorted(round_nuts, key=get_x_pos)
        square_nuts = sorted(square_nuts, key=get_x_pos)
        
        def assign_labels(nuts, type_str):
            if len(nuts) == 1:
                return [f"{type_str} (id: {nuts[0]})"]
            elif len(nuts) == 2:
                return [
                    f"left {type_str} (id: {nuts[0]})",
                    f"right {type_str} (id: {nuts[1]})"
                ]
            elif len(nuts) == 3:
                return [
                    f"left {type_str} (id: {nuts[0]})",
                    f"middle {type_str} (id: {nuts[1]})",
                    f"right {type_str} (id: {nuts[2]})"
                ]
            elif len(nuts) == 4:
                return [
                    f"leftmost {type_str} (id: {nuts[0]})",
                    f"middle-left {type_str} (id: {nuts[1]})",
                    f"middle-right {type_str} (id: {nuts[2]})",
                    f"rightmost {type_str} (id: {nuts[3]})"
                ]
            else:
                res = []
                for idx, name in enumerate(nuts):
                    res.append(f"{type_str} {idx} (id: {name})")
                return res

        labels.extend(assign_labels(round_nuts, "round nut"))
        labels.extend(assign_labels(square_nuts, "square nut"))
        return labels

    def get_task_instruction(self) -> str:
        """Return a concise task instruction matching the active nut type this episode."""
        current_type = getattr(self.env, "current_nut_type", None)
        if current_type == "roundnut":
            return "Assemble all round nuts onto their matching pegs."
        if current_type == "squarenut":
            return "Assemble all square nuts onto their matching pegs."
        return "Assemble all nuts onto their matching pegs."

    def is_done(self) -> bool:
        """Return ``True`` if the episode's target nut(s) are on their correct pegs."""
        return bool(self.env._check_success())

    # ── nut name helpers ───────────────────────────────────────────────

    def _nut_names(self) -> List[str]:
        """Return all nut names as strings for both env variants.

        - ``ClutteredNutAssembly``: ``nuts`` is an ``OrderedDict``; keys are names.
        - Standard ``NutAssembly``: ``nuts`` is a list of nut objects.
        """
        nuts = self.env.nuts
        if isinstance(nuts, dict):
            return list(nuts.keys())
        return [n.name for n in nuts]

    def _active_nuts(self) -> List[str]:
        """Return name strings of nuts that are targets this episode.

        - ``ClutteredNutAssembly``: uses ``current_nut_type`` to select only the
          target nut group (round or square).
        - Standard ``NutAssembly`` with ``single_object_mode``: returns the one
          selected nut.  Mode 0 returns all nuts.
        """
        # ClutteredNutAssembly exposes current_nut_type + round/square_nut_names
        current_type = getattr(self.env, "current_nut_type", None)
        if current_type == "roundnut":
            return list(self.env.round_nut_names)
        if current_type == "squarenut":
            return list(self.env.square_nut_names)

        # Standard NutAssembly
        if getattr(self.env, "single_object_mode", 0) != 0:
            obj_to_use = getattr(self.env, "obj_to_use", None)
            if obj_to_use is not None:
                return [obj_to_use]

        return self._nut_names()

    def _all_workspace_nuts(self) -> List[str]:
        """Return name strings of all round and square nuts present in the environment workspace."""
        if hasattr(self.env, "round_nut_names") and hasattr(self.env, "square_nut_names"):
            return list(self.env.round_nut_names) + list(self.env.square_nut_names)
        return self._active_nuts()

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
        # Create the policy lazily and reuse across skill executions so the
        # wrapped env is not re-created/reset when we retry or run multiple
        # skills. Recreate only if the stored policy references a different
        # env instance or is not of the expected class.
        if self._policy is None or not isinstance(self._policy, HeuristicPolicy):
            self._policy = HeuristicPolicy(
                self.env,
                data_collection_mode=False,
                disable_reactive_blocking=True,
                obs=self._obs
            )
        policy = self._policy

        # Force the policy to target the requested nut.
        policy.obs = self._obs
        policy.current_nut = nut_name
        policy.current_peg_id = peg_id
        policy.nuts_to_place = [nut_name]
        policy.grasp_attempts = 0
        print(f"Forcing policy target: {policy.current_nut} -> peg {policy.current_peg_id}")

        is_obstacle = self._is_nut_obstacle(nut_name)

        # Set the starting stage based on skill.
        policy.stage = self._initial_stage(skill, is_obstacle=is_obstacle)

        terminal_stages = self._terminal_stages(skill, is_obstacle=is_obstacle)
        skill_completed = False

        prev_stage = policy.stage

        for _ in range(max_steps):
            try:
                action, _done = policy.step()
                self._obs, _reward, env_done, _info = self.env.step(action)
            except ValueError as exc:
                # The robosuite horizon was consumed (env.done=True) — usually
                # because the EEF-stagnation handler burned through many steps
                # before skill completion.  Treat as a clean skill failure.
                if "terminated episode" in str(exc):
                    logger.warning(
                        "execute_action: env horizon exhausted mid-skill ('%s'). "
                        "Treating as skill failure.",
                        skill,
                    )
                    break
                raise
            policy.obs = self._obs

            if policy.stage != prev_stage:
                logger.debug("Stage transition: %s → %s", prev_stage, policy.stage)
                prev_stage = policy.stage

            if policy.stage in terminal_stages:
                skill_completed = True
                print(f"Skill '{skill}' completed at stage '{policy.stage}'")
                break

            # If the state-machine moved to "done" or "skip_nut", stop.
            if policy.stage in ("done", "skip_nut"):
                if policy.stage == "done":
                    skill_completed = True
                print(f"Skill '{skill}' terminated early at stage '{policy.stage}'")
                break

        return self._obs, skill_completed

    def execute_nut_assembly(
        self, nut_name: str | dict, max_steps_per_skill: int = 700
    ) -> Tuple[Dict, bool]:
        """Execute a full pick-then-insert assembly for one nut.

        This is the primary execution primitive for Option A inference, where
        the VLM outputs nut names rather than pick/insert actions.

        Parameters
        ----------
        nut_name : str | dict
            The nut label exactly as provided by the VLM / ``get_obj_labels()``,
            e.g. ``"left round nut"`` or a paired dict
            ``{"label": "left round nut", "id": "RoundNut0"}``.
        max_steps_per_skill : int
            Maximum environment steps allowed per skill (pick and insert each).

        Returns
        -------
        (obs, success)
            *success* is ``True`` only when both pick **and** insert complete
            their terminal stages within the step budget.
        """
        if isinstance(nut_name, dict):
            nut_id = nut_name.get("id")      # e.g. "RoundNut0" — used for placed-set
            nut_label = nut_name.get("label", nut_id)  # human label for world-model / logging
            nut_query = nut_id                # Use exact ID so _resolve_nut_name matches it perfectly
        else:
            nut_label = nut_name
            nut_query = nut_name
            nut_id = self._resolve_nut_name(nut_name)  # resolve once for the placed-set

        if nut_id and nut_id in self.placed:
            print(f"Nut '{nut_id}' is already placed. Skipping execution.")
            return self._obs, True

        print(f"Nut query: {nut_query}")
        _, pick_ok = self.execute_action(f"pick {nut_query}", max_steps=max_steps_per_skill)
        print(f"Pick completed: {pick_ok}")

        if not pick_ok:
            print(f"Pick failed for nut '{nut_query}'. Aborting insert.")
            return self._obs, False

        obs, insert_ok = self.execute_action(f"insert {nut_query}", max_steps=max_steps_per_skill)
        print(f"Insert completed: {insert_ok}")

        success = pick_ok and insert_ok
        if success and nut_id:
            self.placed.add(nut_id)
        return obs, success

    def is_nut_blocked_by_any_other_nut(self, nut_id: str) -> Optional[str]:
        """Check if *nut_id* is physically blocked by another nut stacked on top of it.

        Returns the name of the blocking nut if blocked, otherwise None.
        """
        # Get target position
        try:
            body_id = self.env.sim.model.body_name2id(f"{nut_id}_main")
        except Exception:
            try:
                body_id = self.env.obj_body_id[nut_id]
            except Exception:
                try:
                    body_id = self.env.sim.model.body_name2id(nut_id)
                except Exception:
                    logger.error(f"Could not resolve body ID for {nut_id} to check blocking")
                    return None

        pos_target = self.env.sim.data.body_xpos[body_id].copy()

        # Check against all other nuts in the environment
        all_nuts = self.env.round_nut_names + self.env.square_nut_names
        for other_name in all_nuts:
            if other_name == nut_id:
                continue
            if other_name in self.placed:
                continue  # Already placed on a peg, not blocking

            try:
                other_body_id = self.env.sim.model.body_name2id(f"{other_name}_main")
            except Exception:
                try:
                    other_body_id = self.env.obj_body_id[other_name]
                except Exception:
                    try:
                        other_body_id = self.env.sim.model.body_name2id(other_name)
                    except Exception:
                        continue

            pos_other = self.env.sim.data.body_xpos[other_body_id]

            # Stacked check:
            # 1. Close in XY plane (within 0.06 meters)
            xy_dist = np.linalg.norm(pos_other[:2] - pos_target[:2])
            # 2. Significantly higher in Z (other nut is stacked on top)
            z_diff = pos_other[2] - pos_target[2]

            # Standard nut height is 0.04m, so a difference between 0.015 and 0.06 is expected.
            if xy_dist < 0.06 and 0.015 < z_diff < 0.06:
                return other_name
        return None

    def execute_nut_oracle(
        self, nut_name: str | dict
    ) -> Tuple[Dict, bool]:
        """Programmatically execute an assembly or clearance action via state teleportation.

        This bypasses the physical robotic controller and simulates a perfect execution,
        allowing clean evaluation of high-level planning and obstacle reasoning.
        """
        if isinstance(nut_name, dict):
            nut_id = nut_name.get("id")
            nut_query = nut_id
        else:
            nut_query = nut_name
            nut_id = self._resolve_nut_name(nut_name)

        if not nut_id:
            logger.error(f"[ORACLE] Could not resolve nut name for {nut_name}")
            return self._obs, False

        if nut_id in self.placed:
            print(f"Nut '{nut_id}' is already placed. Skipping execution.")
            return self._obs, True

        # Check if the nut is physically blocked by an obstacle stacked on top
        blocking_nut = self.is_nut_blocked_by_any_other_nut(nut_id)
        if blocking_nut:
            logger.warning(
                f"[ORACLE] Action FAILED: Nut '{nut_id}' is physically blocked by stacked nut '{blocking_nut}'."
            )
            return self._obs, False

        is_obstacle = self._is_nut_obstacle(nut_id)

        try:
            # 1. Get body ID in MuJoCo
            try:
                body_id = self.env.sim.model.body_name2id(f"{nut_id}_main")
            except Exception:
                body_id = self.env.sim.model.body_name2id(nut_id)
            
            # Find the starting index for this body's free joint
            qpos_start_idx = self.env.sim.model.jnt_qposadr[self.env.sim.model.body_jntadr[body_id]]

            if is_obstacle:
                # TELEPORT TO TABLE (Clear Obstacle)
                # Place at a randomized, safe table coordinate away from pegs
                new_x = np.random.uniform(-0.25, 0.25)
                new_y = np.random.uniform(-0.15, -0.25)
                new_z = 0.82  # resting on table

                self.env.sim.data.qpos[qpos_start_idx : qpos_start_idx + 3] = [new_x, new_y, new_z]
                # Reset orientation to flat
                self.env.sim.data.qpos[qpos_start_idx + 3 : qpos_start_idx + 7] = [1.0, 0.0, 0.0, 0.0]
                print(f"[ORACLE] Cleared obstacle {nut_id} to table position ({new_x:.2f}, {new_y:.2f})")
                success = True
            else:
                # TELEPORT TO PEG (Assembly)
                peg_id = self._peg_id_for_nut(nut_id)
                peg_body_id = self._peg_body_id(peg_id)
                peg_pos = self.env.sim.data.body_xpos[peg_body_id]

                # Stack height depends on how many nuts are already on this peg
                nuts_on_peg = sum(1 for p_nut in self.placed if self._peg_id_for_nut(p_nut) == peg_id)
                stack_offset = 0.035 * nuts_on_peg

                new_x = peg_pos[0]
                new_y = peg_pos[1]
                new_z = peg_pos[2] + 0.03 + stack_offset

                self.env.sim.data.qpos[qpos_start_idx : qpos_start_idx + 3] = [new_x, new_y, new_z]
                self.env.sim.data.qpos[qpos_start_idx + 3 : qpos_start_idx + 7] = [1.0, 0.0, 0.0, 0.0]

                self.placed.add(nut_id)
                print(f"[ORACLE] Teleported nut {nut_id} to peg {peg_id} (stack level {nuts_on_peg})")
                success = True

            # 2. Reset robot arm to neutral, safe pose to avoid collisions
            if hasattr(self.env, "robots") and len(self.env.robots) > 0:
                robot = self.env.robots[0]
                neutral_qpos = robot.init_qpos
                robot.set_robot_joint_positions(neutral_qpos)

            # 3. Settle physics and sync rendering
            self._settle_and_sync_viewer(n_steps=30, sync_viewer=True)
            self._obs = self.env._get_observations(force_update=True)

            return self._obs, success

        except Exception as e:
            logger.error(f"[ORACLE] Teleportation failed for {nut_id}: {e}")
            return self._obs, False

    # ── internal helpers ───────────────────────────────────────────────

    def _peg_id_for_nut(self, nut_name: str) -> int:
        name_lower = nut_name.lower()
        for key, pid in self._NUT_PEG_MAP.items():
            if key in name_lower:
                return pid
        raise ValueError(f"Cannot determine peg for nut '{nut_name}'")

    def _peg_body_id(self, peg_id: int) -> int:
        """Return the MuJoCo body id for a peg.

        - ``ClutteredNutAssembly`` stores ``peg_body_ids`` as a dict ``{0: id, 1: id}``.
        - Standard ``NutAssembly`` exposes ``peg1_body_id`` / ``peg2_body_id``.
        """
        peg_body_ids = getattr(self.env, "peg_body_ids", None)
        if peg_body_ids is not None:
            return peg_body_ids[peg_id]
        # Standard NutAssembly fallback
        if peg_id == 0:
            return self.env.peg1_body_id
        return self.env.peg2_body_id

    @staticmethod
    def _parse_action_text(text: str) -> Tuple[str, str]:
        """Split ``'pick round nut'`` into ``('pick', 'round nut')``."""
        parts = text.strip().lower().split()
        if len(parts) < 1:
            raise ValueError(f"Cannot parse action_text: '{text}'")
        skill = parts[0]
        if skill == "done" or len(parts) == 1:
            return skill, ""
        nut_query = " ".join(parts[1:])
        return skill, nut_query

    def _resolve_nut_name(self, nut_query: str) -> str:
        """Map an object query to the env's canonical nut name (e.g. ``'RoundNut0'``).

        Resolution order
        ----------------
        1. Case-insensitive exact match against env nut names (handles direct ID
           inputs like ``'RoundNut0'`` returned by the VLM).
        2. Spatial qualifier matching (resolves relative terms like ``'left round nut'``).
        3. Fuzzy type match on ``'round'`` / ``'square'`` substring — but only
           among *unplaced* nuts so we prefer the next available instance.
        4. Fuzzy type match among all nuts as final fallback.
        """
        query_lower = nut_query.lower()
        nut_names = self._nut_names()

        # 1. Case-insensitive exact match (handles env IDs like "RoundNut0")
        for name in nut_names:
            if name.lower() == query_lower:
                return name

        # 2. Spatial qualifier matching
        matched_spatial = self._resolve_spatial_nut(query_lower)
        if matched_spatial:
            return matched_spatial

        # 3. Type-based fuzzy match — prefer unplaced nuts so we pick the next available
        for name in [n for n in nut_names if n not in self.placed]:
            if self._is_type_match(query_lower, name):
                return name

        # 4. Fallback: type fuzzy match across all nuts regardless of placed state
        for name in nut_names:
            if self._is_type_match(query_lower, name):
                return name

        raise ValueError(
            f"No nut matching '{nut_query}' found. Known nuts: {nut_names}"
        )

    def _resolve_spatial_nut(self, query_lower: str) -> Optional[str]:
        """Resolve relative spatial qualifiers ('left', 'right', indices) to a canonical nut name."""
        active = self._active_nuts()
        round_nuts = [name for name in active if "round" in name.lower()]
        square_nuts = [name for name in active if "square" in name.lower()]

        def get_x_pos(name):
            try:
                body_id = self.env.sim.model.body_name2id(name)
                return self.env.sim.data.body_xpos[body_id][0]
            except Exception:
                try:
                    body_id = self.env.sim.model.body_name2id(f"{name}_main")
                    return self.env.sim.data.body_xpos[body_id][0]
                except Exception:
                    return 0.0

        # Sort round/square nuts by x position (exactly matching get_obj_labels)
        round_sorted = sorted(round_nuts, key=get_x_pos)
        square_sorted = sorted(square_nuts, key=get_x_pos)

        def match_in_list(nuts, type_str):
            if type_str not in query_lower:
                return None
            if len(nuts) == 1:
                return nuts[0]
            elif len(nuts) == 2:
                if "left" in query_lower:
                    return nuts[0]
                if "right" in query_lower:
                    return nuts[1]
            elif len(nuts) == 3:
                if "left" in query_lower:
                    return nuts[0]
                if "middle" in query_lower:
                    return nuts[1]
                if "right" in query_lower:
                    return nuts[2]
            elif len(nuts) == 4:
                if "leftmost" in query_lower or ("left" in query_lower and "middle" not in query_lower):
                    return nuts[0]
                if "middle-left" in query_lower or "left-middle" in query_lower:
                    return nuts[1]
                if "middle-right" in query_lower or "right-middle" in query_lower:
                    return nuts[2]
                if "rightmost" in query_lower or ("right" in query_lower and "middle" not in query_lower):
                    return nuts[3]
            
            # Fallback to index matching (for any number of nuts, or if they still refer to index)
            for idx, name in enumerate(nuts):
                if f"{type_str} {idx}" in query_lower or f" {idx}" in query_lower:
                    return name
            return None

        return match_in_list(round_sorted, "round") or match_in_list(square_sorted, "square")

    @staticmethod
    def _is_type_match(query_lower: str, name: str) -> bool:
        """Check if a candidate nut name matches the type substring in the query."""
        nl = name.lower()
        if "round" in query_lower and "round" in nl:
            return True
        if "square" in query_lower and "square" in nl:
            return True
        return False

    def _is_nut_obstacle(self, nut_name: str) -> bool:
        """Check if target nut is an obstacle (different type from target nut type of this episode)."""
        current_type = getattr(self.env, "current_nut_type", None)
        if current_type == "roundnut":
            return "square" in nut_name.lower()
        elif current_type == "squarenut":
            return "round" in nut_name.lower()
        return False

    def _initial_stage(self, skill: str, is_obstacle: bool = False) -> str:
        """Map a skill name to the heuristic policy's starting stage."""
        skill = skill.lower()
        # High-level skills
        if skill == "pick":
            return "move_to_nut"
        if skill == "insert":
            return "move_to_table" if is_obstacle else "move_to_peg"
        if skill in ("place", "put_down"):
            return "move_to_table"
        # Sub-skill primitives (maps mirror the subskill event-tag boundaries
        # defined in policy_wrappers._SUBSKILL_EVENT_TAG_BY_STAGE)
        if skill == "approach":
            return "move_to_nut"
        if skill == "grasp":
            return "lower_to_nut"
        if skill == "carry":
            return "lift_nut"
        if skill == "align":
            return "align_over_peg"
        if skill in ("lower", "lower_insert"):
            return "lower_to_peg"
        # Default: start from the beginning of the pick-place cycle.
        return "move_to_nut"

    def _terminal_stages(self, skill: str, is_obstacle: bool = False) -> frozenset:
        """Stages whose entry signals that *skill* has completed."""
        skill = skill.lower()
        # High-level skills
        if skill == "pick":
            # Pick completes when the policy transitions to the next high-level phase.
            # If the nut is an obstacle, it goes to "move_to_table" instead of "move_to_peg".
            return frozenset({"move_to_table"}) if is_obstacle else frozenset({"move_to_peg"})
        if skill == "insert":
            # Insert completes at release or retract.
            return frozenset({"move_to_nut"})
        if skill in ("place", "put_down"):
            return frozenset({"release", "retract", "reset_orientation"})
        # Sub-skill primitives
        if skill == "approach":
            # Approach ends when the gripper has descended to grasping height.
            return frozenset({"lower_to_nut"})
        if skill == "grasp":
            # Grasp ends when the nut is lifted clear of the table.
            return frozenset({"lift_nut"})
        if skill == "carry":
            # Carry ends when the arm is over the peg, ready to align.
            return frozenset({"align_over_peg"})
        if skill == "align":
            # Align ends when the arm begins the final downward insertion.
            return frozenset({"lower_to_peg"})
        if skill in ("lower", "lower_insert"):
            # Lower ends at nut release.
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
        render_camera = _get_render_camera()
        result = render_camera(
            self.env.sim,
            self._camera_renderers,
            self.camera,
            self.image_size,
            self.image_size,
            rgb_only=True,
        )
        if result is None:
            raise RuntimeError(
                f"render_camera returned None for camera '{self.camera}'"
            )
        cam_obs, self._camera_renderers = result
        return cam_obs.rgb
    
    def _render_rgb_(self) -> np.ndarray:
        """Render agentview camera at self.image_size × self.image_size.

        Delegates to camera_utils.render_camera() — same function P2P uses.
        Always accesses self.env.sim fresh so that hard_reset sim replacements
        are picked up automatically.
        """
        result = render_camera(
            self.env.sim,
            self._camera_renderers,
            self.camera,
            self.image_size,
            self.image_size,
            rgb_only=True,
        )
        if result is None:
            raise RuntimeError(
                f"render_camera returned None for camera '{self.camera}'"
            )
        cam_obs, self._camera_renderers = result
        return cam_obs.rgb
