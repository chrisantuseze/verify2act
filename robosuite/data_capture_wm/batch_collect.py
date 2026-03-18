"""
Batch Data Collection for World Model Training

Collects (image_t, action_prompt, image_t+1, goal_image, sim_state) transitions
using expert and noisy policies in robosuite.

Transition mode:
- dense: one transition per simulator step (t -> t+1)
- keyframe: sparse event-aligned transitions (recommended for nut assembly)

Usage:
    # Expert episodes
    xvfb-run -a python batch_collect.py \
        --env ClutteredNutAssembly --policy-mode expert \
        --transition-mode keyframe \
        --output-dir dataset/nut_assembly \
        --num-round 2 --num-square 2 --initial-stacking-prob 0.5 \
        --nut-type-mode random --num-episodes 1000 \
        --image-size 512 --seed 42

    # Noisy episodes (sigma=0.05)
    xvfb-run -a python batch_collect.py \
        --env ClutteredNutAssembly --policy-mode noisy --noise-sigma 0.05 \
        --transition-mode keyframe \
        --output-dir dataset/nut_assembly \
        --num-round 2 --num-square 2 --initial-stacking-prob 0.5 \
        --nut-type-mode random --num-episodes 1000 \
        --image-size 512 --seed 0
"""

# import os
# if 'MUJOCO_GL' not in os.environ:
#     os.environ['MUJOCO_GL'] = 'glx'

import sys
import time
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # robosuite/
sys.path.insert(0, str(Path(__file__).resolve().parent))         # data_capture_wm/

from episode_recorder import EpisodeRecorder
from prompt_utils import build_action_prompt
from policy_wrappers import get_policy_factory


class BatchCollector:
    """
    Collects episodes and writes world-model training data.
    """

    def __init__(
        self,
        env_factory: Callable,
        policy_factory: Callable,
        env_name: str,
        env_config: Optional[dict],
        output_dir: str,
        camera: str = "agentview",
        image_size: int = 512,
        policy_mode: str = "expert",
        noise_sigma: float = 0.0,
        transition_mode: str = "keyframe",
    ):
        self.env_factory = env_factory
        self.policy_factory = policy_factory
        self.env_name = env_name
        self.env_config = dict(env_config or {})
        self.output_root = Path(output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.image_size = image_size
        self.policy_mode = policy_mode
        self.noise_sigma = noise_sigma
        self.transition_mode = transition_mode

        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "transitions": 0,
        }
        self.last_success_goal: Optional[str] = None

    # ------------------------------------------------------------------

    def collect(self, num_episodes: int, max_timesteps: int = 1000):
        env = self.env_factory()
        recorder = EpisodeRecorder(
            env,
            output_root=self.output_root,
            camera=self.camera,
            image_size=self.image_size,
            transition_mode=self.transition_mode,
        )

        for ep_idx in range(num_episodes):
            t0 = time.time()
            try:
                obs = env.reset()
                episode_id = f"ep_{recorder.episode_counter:05d}"
                recorder.start_episode(episode_id=episode_id)

                policy = self.policy_factory(env, data_collection_mode=True)
                policy.obs = obs

                policy_type = (
                    "expert"
                    if self.policy_mode == "expert"
                    else f"noisy_{self.noise_sigma}"
                )

                done = False
                task_success = False
                t = 0

                while t < max_timesteps and not done:
                    action, policy_done = policy.step()

                    # Add noise if requested (skip gripper dimension)
                    if self.policy_mode == "noisy" and self.noise_sigma > 0:
                        noise = np.random.normal(0, self.noise_sigma, size=action.shape)
                        noise[-1] = 0.0
                        action = action + noise

                    obs, reward, env_done, info = env.step(action)
                    policy.obs = obs

                    done = env_done or policy_done
                    task_success = env_done  # robosuite returns done=True on success

                    # Skill / target from the adapter (task-agnostic)
                    ai = policy.get_action_info()
                    action_text = build_action_prompt(
                        ai.skill, ai.object_name, ai.cartesian_target
                    )

                    recorder.record_step(
                        action=action,
                        obs=obs,
                        done=done,
                        info=info,
                        skill=ai.skill,
                        object_name=ai.object_name,
                        cartesian_target=ai.cartesian_target,
                        action_text=action_text,
                        stage=ai.stage,
                        event_tag=ai.event_tag,
                        policy_type=policy_type,
                    )

                    t += 1

                # End episode
                fallback_goal = self.last_success_goal
                transitions = recorder.end_episode(
                    success=task_success, fallback_goal=fallback_goal
                )

                if task_success:
                    self.last_success_goal = transitions[0]["goal_image"] if transitions else None
                    self.stats["success"] += 1
                else:
                    self.stats["failed"] += 1

                self.stats["total"] += 1
                self.stats["transitions"] += len(transitions)
                dur = time.time() - t0

                status = "OK" if task_success else "FAIL"
                print(
                    f"  [{status}] Episode {ep_idx:04d} ({episode_id}): "
                    f"{len(transitions)} transitions, {dur:.1f}s"
                )

            except Exception as exc:  # noqa: BLE001
                dur = time.time() - t0
                print(
                    f"  [SKIP] Episode {ep_idx:04d}: render/sim error after {dur:.1f}s"
                    f" — {type(exc).__name__}: {exc}"
                )
                # Flush the renderer cache and discard the partial episode dir.
                # episode_counter is NOT incremented so the next episode reuses
                # the same ID slot and the output numbering stays contiguous.
                recorder.abort_episode()
                self.stats["skipped"] = self.stats.get("skipped", 0) + 1

        recorder.close()
        env.close()
        self._save_metadata()
        self._print_summary()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_metadata(self):
        meta = {
            "env_name": self.env_name,
            "env_config": self.env_config,
            "policy_mode": self.policy_mode,
            "noise_sigma": self.noise_sigma,
            "transition_mode": self.transition_mode,
            "camera": self.camera,
            "image_size": self.image_size,
            "stats": self.stats,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.output_root / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _print_summary(self):
        s = self.stats
        rate = s["success"] / max(1, s["total"]) * 100
        skipped = s.get("skipped", 0)
        print(f"\n{'='*50}")
        print(f"Collection complete: {s['total']} episodes attempted")
        print(f"  Success: {s['success']} ({rate:.1f}%)")
        print(f"  Failed:  {s['failed']}")
        if skipped:
            print(f"  Skipped (render error): {skipped}")
        print(f"  Total transitions: {s['transitions']}")
        print(f"{'='*50}\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def build_env_factory(args):
    """Return (env_factory, policy_factory, env_name, env_config)."""
    env_name = args.env

    if env_name == "ClutteredNutAssembly":
        from run_cluttered_nutassembly import create_environment
        from policy_wrappers import create_nut_assembly_policy

        env_config = {
            "env_name": "ClutteredNutAssembly",
            "num_round_nuts": args.num_round,
            "num_square_nuts": args.num_square,
            "initial_stacking_prob": args.initial_stacking_prob,
            "nut_type_mode": args.nut_type_mode,
            "horizon": args.max_timesteps,
        }

        def env_factory():
            return create_environment(
                env_name="ClutteredNutAssembly",
                num_round_nuts=args.num_round,
                num_square_nuts=args.num_square,
                initial_stacking_prob=args.initial_stacking_prob,
                nut_type_mode=args.nut_type_mode,
                horizon=args.max_timesteps,
                has_renderer=not args.headless,
                has_offscreen_renderer=True,
            )

        return env_factory, create_nut_assembly_policy, env_name, env_config

    elif env_name in ("Stack", "Stack3", "Stack4"):
        from run_stack import create_environment
        from policy_wrappers import create_stack_policy

        env_config = {
            "env_name": env_name,
            "horizon": args.max_timesteps,
        }

        def env_factory():
            return create_environment(
                env_name,
                has_renderer=not args.headless,
                has_offscreen_renderer=True,
            )

        return env_factory, create_stack_policy, env_name, env_config

    elif env_name == "PickPlace":
        from run_pickplace import create_environment
        from policy_wrappers import create_pickplace_policy

        env_config = {
            "env_name": "PickPlace",
            "horizon": args.max_timesteps,
        }

        def env_factory():
            return create_environment(
                "PickPlaceCan",
                has_renderer=not args.headless,
                has_offscreen_renderer=True,
            )

        return env_factory, create_pickplace_policy, env_name, env_config

    raise ValueError(f"Unsupported env: {env_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch collection for world-model training"
    )
    parser.add_argument("--env", type=str, default="ClutteredNutAssembly",
                        choices=["Stack", "Stack3", "Stack4",
                                 "ClutteredNutAssembly", "PickPlace"])
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-timesteps", type=int, default=1000)
    parser.add_argument("--output-dir", type=str,
                        default="dataset/nut_assembly")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--policy-mode", type=str, default="expert",
                        choices=["expert", "noisy"])
    parser.add_argument("--noise-sigma", type=float, default=0.05)
    parser.add_argument("--transition-mode", type=str, default="keyframe",
                        choices=["dense", "keyframe"])
    parser.add_argument("--seed", type=int, default=42)
    # Nut assembly params
    parser.add_argument("--num-round", type=int, default=6)
    parser.add_argument("--num-square", type=int, default=2)
    parser.add_argument("--initial-stacking-prob", type=float, default=0.6)
    parser.add_argument(
        "--nut-type-mode",
        type=str,
        default="roundnut",
        choices=["roundnut", "squarenut", "random", "alternate"],
        help="Nut type mode for ClutteredNutAssembly",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI viewer (faster-than-realtime). "
             "Offscreen rendering for image capture is always enabled.",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    env_factory, policy_factory, env_name, env_config = build_env_factory(args)

    collector = BatchCollector(
        env_factory=env_factory,
        policy_factory=policy_factory,
        env_name=env_name,
        env_config=env_config,
        output_dir=args.output_dir,
        camera=args.camera,
        image_size=args.image_size,
        policy_mode=args.policy_mode,
        noise_sigma=args.noise_sigma,
        transition_mode=args.transition_mode,
    )

    collector.collect(
        num_episodes=args.num_episodes,
        max_timesteps=args.max_timesteps,
    )


if __name__ == "__main__":
    main()
