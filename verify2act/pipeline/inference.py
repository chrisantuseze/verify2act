"""Inference pipeline — the full Stage 1+2+3 loop.

This module wires together the VLM planner, world model, critic, and
reflection mechanism into a single ``run_episode()`` function that
drives the receding-horizon control loop described in the pipeline plan.

Usage::

    from verify2act.pipeline.inference import run_episode

    result = run_episode(
        env_wrapper=env_wrapper,
        vae=vae,
        world_model=oracle_wm,           # or diffusion_wm
        critic=critic,
        planner=planner,
        horizon=5,
        output_dir="output/eval/run_001",
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from verify2act.critic.inference import CriticDecision, decide_replan
from verify2act.pipeline.reflection import build_reflection_context
from verify2act.utils.vae import VAE_LATENT_SCALE

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Image / latent utilities
# ═══════════════════════════════════════════════════════════════════════


def preprocess_image(img_np: np.ndarray) -> torch.Tensor:
    """Convert a ``[H, W, 3]`` uint8 numpy image to ``[1, 3, 512, 512]``
    float32 tensor normalised to ``[-1, 1]``.
    """
    img = Image.fromarray(img_np).resize((512, 512))
    arr = np.asarray(img, dtype=np.float32) / 255.0     # [512, 512, 3] in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)     # [3, 512, 512]
    tensor = tensor * 2.0 - 1.0                          # [-1, 1]
    return tensor.unsqueeze(0)                            # [1, 3, 512, 512]


@torch.no_grad()
def encode_image(
    vae: torch.nn.Module,
    img_np: np.ndarray,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Encode a ``[H, W, 3]`` uint8 image to a ``[1, 4, 64, 64]`` VAE latent.

    Applies the SD scaling factor ``0.18215``.
    """
    tensor = preprocess_image(img_np)
    if device is not None:
        tensor = tensor.to(device, dtype=next(vae.parameters()).dtype)
    return vae.encode(tensor).latent_dist.sample() * VAE_LATENT_SCALE


# ═══════════════════════════════════════════════════════════════════════
# Episode trace (logging)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class StepRecord:
    """What happened at one real timestep."""

    timestep: int
    plan: List[str]
    replan_attempts: int = 0
    action_executed: Optional[str] = None
    all_scores: List[Tuple[float, float]] = field(default_factory=list)
    failed_step: Optional[int] = None
    reflection_analyses: List[str] = field(default_factory=list)
    critic_decisions: List[str] = field(default_factory=list)


@dataclass
class EpisodeTrace:
    """Full episode-level log returned by ``run_episode()``."""

    success: bool = False
    total_steps: int = 0
    total_replans: int = 0
    history: List[str] = field(default_factory=list)
    steps: List[StepRecord] = field(default_factory=list)


def _save_trace(trace: EpisodeTrace, output_dir: Path) -> None:
    """Persist the episode trace as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "episode_trace.json"

    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(asdict(trace), f, indent=2, default=_default)
    logger.info("Episode trace saved to %s", path)


def _save_image(img_np: np.ndarray, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_np).save(output_dir / name)


# ═══════════════════════════════════════════════════════════════════════
# Main inference loop
# ═══════════════════════════════════════════════════════════════════════


def run_episode(
    env_wrapper,
    vae: torch.nn.Module,
    world_model,
    critic: torch.nn.Module,
    planner,
    *,
    horizon: int = 5,
    max_steps: int = 50,
    theta_f: float = 0.4,
    theta_u: float = 0.15,
    max_retries: int = 2,
    max_replans: int = 3,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> EpisodeTrace:
    """Run a full Verify2Act inference episode.

    Parameters
    ----------
    env_wrapper : NutAssemblyEnvWrapper
        The wrapped robosuite environment.
    vae : AutoencoderKL
        Frozen VAE encoder used by both world model and critic.
    world_model : WorldModelBase
        Either ``OracleWorldModel`` or ``DiffusionWorldModel``.
    critic : SpatialBetaPRMCritic
        Trained feasibility critic.
    planner : VLMPlanner
        GPT-4o VLM planner with propose/reflect methods.
    horizon : int
        Number of steps per VLM planning call.
    max_steps : int
        Maximum number of real-environment timesteps per episode.
    theta_f : float
        Feasibility threshold (below ⇒ potential failure).
    theta_u : float
        Uncertainty threshold (below ⇒ critic is confident).
    max_retries : int
        World-model re-samples on uncertain (requery) decisions.
    max_replans : int
        Maximum reflect-replan cycles per timestep.
    device : str
        Torch device.
    output_dir : str or None
        If set, save episode traces and images here.

    Returns
    -------
    EpisodeTrace
        Full record of the episode for analysis and debugging.
    """
    torch_device = torch.device(device)
    out_path = Path(output_dir) if output_dir else None
    critic.eval()

    # ── Reset environment ──────────────────────────────────────────────
    obs, goal_image_np = env_wrapper.reset()
    z_goal = encode_image(vae, goal_image_np, device=torch_device)
    obj_labels = env_wrapper.get_obj_labels()
    history: List[str] = []

    trace = EpisodeTrace()

    if out_path:
        _save_image(goal_image_np, out_path, "goal.png")

    # ── Main loop: one iteration per real timestep ─────────────────────
    for t in range(max_steps):
        current_image_np = env_wrapper.read_image()
        step_record = StepRecord(timestep=t)

        if out_path:
            _save_image(current_image_np, out_path / "steps", f"step_{t:03d}_current.png")

        # ── Stage 1: Plan generation (VLM call) ───────────────────────
        plan = planner.propose(
            current_image_np, goal_image_np, history, obj_labels, horizon
        )
        step_record.plan = list(plan)
        logger.info("t=%d  Proposed plan: %s", t, plan)

        # ── Stage 2+3: Imagination + Critic loop ──────────────────────
        plan_accepted = False

        for replan_attempt in range(max_replans + 1):
            all_scores: List[Tuple[float, float]] = []
            imagined_img = current_image_np
            step_failed = False

            for k, action in enumerate(plan):
                # ── Imagination ────────────────────────────────────────
                imagined_img_next = world_model.imagine(imagined_img, action)

                if out_path:
                    step_dir = out_path / "steps"
                    _save_image(
                        imagined_img_next, step_dir,
                        f"step_{t:03d}_imagine_r{replan_attempt}_k{k}.png",
                    )

                # ── Critic evaluation ──────────────────────────────────
                z_t1 = encode_image(vae, imagined_img_next, device=torch_device)
                with torch.no_grad():
                    critic_out = critic(z_t1, z_goal)
                mean_f = critic_out["mean_feasibility"].item()
                uncert = critic_out["uncertainty"].item()
                all_scores.append((mean_f, uncert))

                decision = decide_replan(mean_f, uncert, theta_f, theta_u)
                step_record.critic_decisions.append(
                    f"k={k} action='{action}' μ={mean_f:.3f} σ²={uncert:.4f} → {decision.action}"
                )
                logger.info(
                    "  k=%d  action='%s'  feasibility=%.3f  uncertainty=%.4f  → %s",
                    k, action, mean_f, uncert, decision.action,
                )

                # ── Requery: re-sample world model on high uncertainty ─
                if decision.action == "requery":
                    for retry_i in range(max_retries):
                        imagined_img_next = world_model.imagine(imagined_img, action)
                        z_t1 = encode_image(vae, imagined_img_next, device=torch_device)
                        with torch.no_grad():
                            critic_out = critic(z_t1, z_goal)
                        mean_f = critic_out["mean_feasibility"].item()
                        uncert = critic_out["uncertainty"].item()
                        all_scores[-1] = (mean_f, uncert)

                        decision = decide_replan(mean_f, uncert, theta_f, theta_u)
                        logger.info(
                            "    requery %d/%d  feasibility=%.3f  uncertainty=%.4f  → %s",
                            retry_i + 1, max_retries, mean_f, uncert, decision.action,
                        )
                        if decision.action != "requery":
                            break
                    else:
                        # Exhausted retries — escalate to reflection.
                        decision = CriticDecision(
                            action="reflect", reason="requery_exhausted"
                        )

                # ── Reflect: confident failure ─────────────────────────
                if decision.action == "reflect":
                    step_record.failed_step = k
                    step_record.replan_attempts = replan_attempt + 1

                    diff_map = z_t1 - z_goal
                    ctx = build_reflection_context(
                        imagined_state=imagined_img_next,
                        z_t1=z_t1,
                        z_goal=z_goal,
                        diff_map=diff_map,
                        critic=critic,
                        all_scores=all_scores,
                        failed_step=k,
                        full_plan=plan,
                    )
                    result = planner.reflect(
                        current_image_np, goal_image_np,
                        history, obj_labels, plan, ctx,
                    )
                    revised_plan = result["revised_plan"]
                    analysis = result.get("analysis", "")

                    step_record.reflection_analyses.append(analysis)
                    logger.info(
                        "  REFLECT at step %d: %s → revised: %s",
                        k, analysis, revised_plan,
                    )

                    plan = revised_plan
                    step_record.plan = list(plan)
                    step_failed = True
                    break

                # ── Continue: step passed — chain forward ──────────────
                imagined_img = imagined_img_next

            if not step_failed:
                plan_accepted = True
                break

        # ── If all replanning attempts exhausted, execute anyway ───────
        step_record.all_scores = all_scores

        if not plan_accepted:
            logger.warning(
                "t=%d  Max replans exhausted (%d). Executing first action anyway.",
                t, max_replans,
            )

        # ── Execute the first action on the real environment ───────────
        if not plan:
            logger.warning("t=%d  Empty plan — skipping execution.", t)
            trace.steps.append(step_record)
            continue

        action_to_execute = plan[0]

        # Handle "done" action from the VLM.
        if action_to_execute.strip().lower() == "done":
            logger.info("t=%d  VLM returned 'done'.", t)
            step_record.action_executed = "done"
            trace.steps.append(step_record)
            break

        logger.info("t=%d  EXECUTE: '%s'", t, action_to_execute)
        obs, skill_ok = env_wrapper.execute_action(action_to_execute)
        step_record.action_executed = action_to_execute
        history.append(action_to_execute)

        if out_path:
            _save_image(
                env_wrapper.read_image(),
                out_path / "steps",
                f"step_{t:03d}_after_exec.png",
            )

        trace.steps.append(step_record)

        # ── Done check ─────────────────────────────────────────────────
        if env_wrapper.is_done():
            logger.info("t=%d  Task completed!", t)
            trace.success = True
            break

    # ── Finalise trace ─────────────────────────────────────────────────
    trace.total_steps = len(trace.steps)
    trace.history = list(history)
    trace.total_replans = sum(s.replan_attempts for s in trace.steps)

    if out_path:
        _save_trace(trace, out_path)

    return trace
