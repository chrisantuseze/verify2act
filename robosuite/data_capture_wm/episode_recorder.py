"""
Episode Recorder for World Model Data Collection

Saves per-timestep RGB frames (512×512, agentview), simulator states,
and action prompts. Outputs a JSONL transition manifest per episode.

Transition output modes:
- dense: every simulator step becomes one transition (t -> t+1)
- keyframe: stage-event-aligned sparse transitions (e.g., pick_start -> pick_end)
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robosuite.utils.camera_utils import render_camera

class EpisodeRecorder:
    """
    Records episode data for world-model training.

    Per timestep it saves:
      - RGB frame (agentview, 512×512 PNG)
      - Simulator state (qpos + qvel as .npz)
    At episode end:
      - goal.png (last frame of successful episode)
      - meta.json (episode-level metadata)
      - Appends all transitions to a JSONL file
            - In keyframe mode, prunes unreferenced frame/state files
    """

    def __init__(
        self,
        env,
        output_root: Path,
        camera: str = "agentview",
        image_size: int = 512,
        transition_mode: str = "keyframe",
    ):
        self.env = env
        self.output_root = Path(output_root)
        self.camera = camera
        self.image_size = image_size
        self.transition_mode = transition_mode

        # Renderer cache keyed by "camera_width_height" — same pattern as camera_utils.
        # Cleared on start_episode() when hard_reset recreates the sim/model object.
        self._camera_renderers: Dict[str, Any] = {}
        self._last_mj_model = None  # track model identity across resets

        self.episode_counter = self._detect_episode_counter()
        self._reset_buffers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_episode(self, episode_id: Optional[str] = None):
        """Call after env.reset().  Sets up episode directory and saves frame 0.

        Flushes the renderer cache if robosuite's hard_reset replaced the sim
        model object since the last episode.
        """
        self._reset_buffers()
        # hard_reset=True (robosuite default) recreates env.sim each episode.
        # Detect model replacement and flush the renderer cache so we don't
        # render from a stale GL scene.
        sim = self.env.sim
        mj_model = sim.model._model if hasattr(sim.model, '_model') else sim.model
        if mj_model is not self._last_mj_model:
            self._flush_renderers()
            self._last_mj_model = mj_model

        self.episode_id = episode_id or f"ep_{self.episode_counter:05d}"
        self.episode_dir = self.output_root / "episodes" / self.episode_id
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        # Save frame 0 and initial sim state
        self._save_frame(0)
        self._save_sim_state(0)
        self._active = True

    def record_step(
        self,
        action: np.ndarray,
        obs: Dict[str, Any],
        done: bool,
        info: Dict[str, Any],
        skill: str,
        object_name: str,
        cartesian_target: np.ndarray,
        action_text: str,
        stage: str,
        event_tag: Optional[str],
        policy_type: str = "expert",
    ):
        """
        Record one transition (t → t+1).

        Must be called *after* env.step().

        Args:
            action: Raw action array sent to env.step().
            obs: Observation dict returned by env.step().
            done: Whether episode finished this step.
            info: Info dict from env.step().
            skill: High-level skill name ('pick', 'place', 'insert').
            object_name: Target object name (e.g. 'round nut').
            cartesian_target: 3-element Cartesian target in world frame.
            action_text: Pre-built prompt string from prompt_utils.
            policy_type: 'expert' | 'noisy_0.02' | 'noisy_0.05' | 'noisy_0.10'.
        """
        if not self._active:
            raise RuntimeError("Episode not started. Call start_episode() first.")

        t = self._step_count  # current frame index before this step
        t1 = t + 1
        self._step_count = t1

        # Save frame t+1 and sim state t+1
        self._save_frame(t1)
        self._save_sim_state(t1)

        step_record = {
            "episode_id": self.episode_id,
            "timestep": t,
            "image_t": str(self._frame_relpath(t)),
            "image_t1": str(self._frame_relpath(t1)),
            "goal_image": None,
            "goal_image_source": None,
            "action_text": action_text,
            "action_params": {
                "skill": skill,
                "object": object_name,
                "cartesian_target": [
                    round(float(cartesian_target[0]), 5),
                    round(float(cartesian_target[1]), 5),
                    round(float(cartesian_target[2]), 5),
                ],
            },
            "state_t": str(self._state_relpath(t)),
            "state_t1": str(self._state_relpath(t1)),
            "policy_type": policy_type,
            "policy_stage": stage,
            "event_tag": event_tag,
            "episode_success": None,
            "label_reachable": -1,
        }
        self._step_records.append(step_record)

        self._done = done
        self._info = info

    def end_episode(
        self,
        success: bool,
        fallback_goal: Optional[str] = None,
    ) -> List[Dict]:
        """
        Finalise the episode.

        - If success: saves last frame as goal.png.
        - If not success: uses fallback_goal path.
        - Writes all transitions to the master JSONL.
        - Writes per-episode meta.json.

        Args:
            success: Whether the episode was a task success.
            fallback_goal: Relative path to a goal.png from another successful
                           episode (used when this episode failed).

        Returns:
            List of transition dicts written.
        """
        if not self._active:
            raise RuntimeError("No active episode.")

        self._active = False

        # Goal image
        if success:
            # Copy last frame as goal.png
            src = self.episode_dir / f"frame_{self._step_count:05d}.png"
            dst = self.episode_dir / "goal.png"
            if src.exists():
                import shutil
                shutil.copy2(src, dst)
            goal_rel = str(Path("episodes") / self.episode_id / "goal.png")
            goal_source = "self"
        else:
            goal_rel = fallback_goal or ""
            goal_source = "fallback"

        transitions = self._build_transitions()

        # Resolve goal and success in buffered transitions
        for tr in transitions:
            tr["goal_image"] = goal_rel
            tr["goal_image_source"] = goal_source
            tr["episode_success"] = bool(success)

        # Keep only files referenced by emitted transitions (and goal image).
        # This keeps keyframe datasets compact even though per-step capture is
        # used online during collection.
        self._prune_episode_artifacts(transitions, keep_goal=success)

        # Append to master JSONL
        jsonl_path = self.output_root / "transitions.jsonl"
        with open(jsonl_path, "a") as f:
            for tr in transitions:
                f.write(json.dumps(tr) + "\n")

        # Episode-level metadata
        meta = {
            "episode_id": self.episode_id,
            "num_transitions": len(transitions),
            "num_step_records": len(self._step_records),
            "transition_mode": self.transition_mode,
            "success": bool(success),
            "goal_image": goal_rel,
        }
        with open(self.episode_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.episode_counter += 1
        result = list(transitions)
        self._reset_buffers()
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_episode_counter(self) -> int:
        """Scan output_root/episodes/ for existing ep_NNNNN dirs and return
        the next episode index so we never overwrite existing frame files."""
        episodes_dir = self.output_root / "episodes"
        if not episodes_dir.exists():
            return 0
        max_idx = -1
        for p in episodes_dir.iterdir():
            if p.is_dir() and p.name.startswith("ep_"):
                try:
                    idx = int(p.name[3:])
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    pass
        return max_idx + 1 if max_idx >= 0 else 0

    def _reset_buffers(self):
        self._step_records: List[Dict] = []
        self._step_count = 0
        self._active = False
        self._done = False
        self._info: Dict = {}
        self.episode_id = ""
        self.episode_dir: Optional[Path] = None

    def _render_rgb(self) -> np.ndarray:
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
        )
        if result is None:
            raise RuntimeError(
                f"render_camera returned None for camera '{self.camera}'"
            )
        cam_obs, self._camera_renderers = result
        return cam_obs.rgb

    def _save_frame(self, t: int):
        path = self.episode_dir / f"frame_{t:05d}.png"
        rgb = self._render_rgb()
        Image.fromarray(rgb).save(path)

    def _save_sim_state(self, t: int):
        state = self.env.sim.get_state()  # always use live sim reference
        path = self.episode_dir / f"state_{t:05d}.npz"
        np.savez_compressed(str(path), qpos=state.qpos, qvel=state.qvel)

    def _frame_relpath(self, t: int) -> Path:
        return Path("episodes") / self.episode_id / f"frame_{t:05d}.png"

    def _state_relpath(self, t: int) -> Path:
        return Path("episodes") / self.episode_id / f"state_{t:05d}.npz"

    def _flush_renderers(self):
        """Close and discard all cached renderers (called when model is replaced)."""
        for renderer in self._camera_renderers.values():
            try:
                renderer.close()
            except Exception:
                pass
        self._camera_renderers.clear()

    def _build_transitions(self) -> List[Dict]:
        if self.transition_mode == "dense":
            return [dict(row) for row in self._step_records]

        return self._build_keyframe_transitions()

    def _build_keyframe_transitions(self) -> List[Dict]:
        pair_rules = [
            ("pick_start", "pick_end", "pick"),
            ("insert_start", "insert_end", "insert"),
            ("place_start", "place_end", "place"),
        ]
        end_to_rule = {end: (start, skill) for start, end, skill in pair_rules}
        pending = {(start, skill): {} for start, _, skill in pair_rules}

        transitions: List[Dict] = []

        for row in self._step_records:
            raw_event_tag = row.get("event_tag")
            event_tags = []
            if raw_event_tag:
                event_tags = [tag.strip() for tag in str(raw_event_tag).split("|") if tag.strip()]
            obj = row["action_params"]["object"]

            for event_tag in event_tags:
                for start_tag, _, rule_skill in pair_rules:
                    if event_tag == start_tag:
                        pending[(start_tag, rule_skill)][obj] = row

                if event_tag in end_to_rule:
                    start_tag, rule_skill = end_to_rule[event_tag]

                    start_row = pending[(start_tag, rule_skill)].pop(obj, None)
                    if start_row is None:
                        continue

                    transitions.append(
                        {
                            "episode_id": row["episode_id"],
                            "timestep": start_row["timestep"],
                            "image_t": start_row["image_t"],
                            "image_t1": row["image_t1"],
                            "goal_image": None,
                            "goal_image_source": None,
                            "action_text": start_row["action_text"],
                            "action_params": start_row["action_params"],
                            "state_t": start_row["state_t"],
                            "state_t1": row["state_t1"],
                            "policy_type": start_row["policy_type"],
                            "policy_stage_t": start_row.get("policy_stage"),
                            "policy_stage_t1": row.get("policy_stage"),
                            "event_tag_t": start_row.get("event_tag"),
                            "event_tag_t1": row.get("event_tag"),
                            "source_timestep_t": start_row["timestep"],
                            "source_timestep_t1": row["timestep"],
                            "episode_success": None,
                            "label_reachable": -1,
                        }
                    )

        if transitions:
            return transitions

        return []

    def _prune_episode_artifacts(self, transitions: List[Dict], keep_goal: bool):
        """Delete per-step frame/state files that are not referenced by transitions."""
        if self.episode_dir is None:
            return

        keep_paths = set()
        for tr in transitions:
            for key in ("image_t", "image_t1", "state_t", "state_t1"):
                rel = tr.get(key)
                if rel:
                    keep_paths.add(self.output_root / rel)

        # If no transitions were emitted, keep the initial snapshot only.
        if not keep_paths:
            keep_paths.add(self.episode_dir / "frame_00000.png")
            keep_paths.add(self.episode_dir / "state_00000.npz")

        if keep_goal:
            keep_paths.add(self.episode_dir / "goal.png")

        for frame_path in self.episode_dir.glob("frame_*.png"):
            if frame_path not in keep_paths:
                try:
                    frame_path.unlink()
                except OSError:
                    pass

        for state_path in self.episode_dir.glob("state_*.npz"):
            if state_path not in keep_paths:
                try:
                    state_path.unlink()
                except OSError:
                    pass

    def close(self):
        """Free all cached MuJoCo renderers."""
        self._flush_renderers()
