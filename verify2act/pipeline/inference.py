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

    python -m verify2act.pipeline.inference \
    --critic-ckpt /path/to/critic_ckpt.pt \
    --device cuda \
    --dtype fp16 \
    --wm-mode hybrid \
    --vae-model runwayml/stable-diffusion-v1-5 \
    --output-dir verify2act/output/inference_run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))         # project root (contains verify2act/)

from verify2act.critic.inference import (
    CriticDecision,
    check_rollout_consistency,
    decide_from_proximity,
)
from verify2act.critic.model import DINOv2DualHeadCritic
from verify2act.pipeline.env_wrapper import NutAssemblyEnvWrapper
from verify2act.pipeline.planner import VLMPlanner
from verify2act.pipeline.decompose import expand_nut_plan
from verify2act.pipeline.reflection import build_reflection_context
from verify2act.pipeline.world_model import DiffusionWorldModel, OracleWorldModel, WorldModelBase
from contextlib import contextmanager, nullcontext
from verify2act.utils.vae import load_vae_encoder
from verify2act.utils.vae import VAE_LATENT_SCALE

import os
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glx'

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


def preprocess_image_for_critic(img_np: np.ndarray) -> torch.Tensor:
    """Convert a ``[H, W, 3]`` uint8 image to ``[1, 3, 224, 224]`` in ``[-1, 1]``.

    DINOv2 expects 224×224 input (patch size 14 → 16×16 patches).
    The range [-1, 1] matches how training images were stored and what
    ``DINOv2DualHeadCritic.encode()`` normalises internally.
    """
    img = Image.fromarray(img_np).resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0)                            # [1, 3, 224, 224]


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
    plan: List[str] = field(default_factory=list)
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

def run_inference_episode(   
    env_wrapper,
    vae: torch.nn.Module,
    world_model: WorldModelBase,
    critic: torch.nn.Module,
    planner: VLMPlanner,
    *,
    requery_world_model=None,
    horizon: int = 4,
    max_steps: int = 10,
    theta_c: float = 0.5,
    theta_p: float = 0.6,
    max_retries: int = 2,
    max_replans: int = 3,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> EpisodeTrace:

    torch_device = torch.device(device)
    out_path = Path(output_dir) if output_dir else None
    critic.eval()

    env_wrapper._obs = env_wrapper.env.reset()
    goal_image_np = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    import matplotlib.pyplot as plt


def run_episode(
    env_wrapper,
    vae: torch.nn.Module,
    world_model: WorldModelBase,
    critic: torch.nn.Module,
    planner: VLMPlanner,
    *,
    requery_world_model=None,
    horizon: int = 4,
    max_steps: int = 10,
    theta_c: float = 0.5,
    theta_p: float = 0.6,
    max_retries: int = 2,
    max_replans: int = 3,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> EpisodeTrace:
    """Run a full Verify2Act inference episode (DINOv2DualHeadCritic).

    Parameters
    ----------
    env_wrapper : NutAssemblyEnvWrapper
        The wrapped robosuite environment.
    vae : AutoencoderKL
        Frozen VAE encoder — used only by the diffusion world model.
    world_model : WorldModelBase
        Primary world model for imagined rollouts.
    requery_world_model : WorldModelBase or None
        Optional alternate world model used during ``requery`` retries.
        If ``None``, retries reuse ``world_model``.
    critic : DINOv2DualHeadCritic
        Trained dual-head contrastive critic.
    planner : VLMPlanner
        GPT-4o VLM planner with propose/reflect methods.
    horizon : int
        Maximum number of nuts per VLM planning call.
    max_steps : int
        Maximum nut-assembly attempts per episode.
    theta_c : float
        Temporal consistency threshold (Head 2). Frames below → requery.
    theta_p : float
        Goal proximity threshold (Head 1). Final frame below → reflect.
    max_retries : int
        World-model re-samples on ``requery`` decisions.
    max_replans : int
        Maximum reflect-replan cycles per timestep.
    device : str
        Torch device string.
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

    if not isinstance(critic, DINOv2DualHeadCritic):
        raise TypeError("run_episode expects DINOv2DualHeadCritic")

    # ── Reset environment ──────────────────────────────────────────────
    # obs, goal_image_np = env_wrapper.reset()
    env_wrapper._obs = env_wrapper.env.reset()
    goal_image_np = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    import matplotlib.pyplot as plt

    # plt.figure(figsize=(6, 6))
    # plt.imshow(goal_image_np)
    # plt.title("Goal Image")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()
         
    goal_img_224 = preprocess_image_for_critic(goal_image_np).to(torch_device)
    with torch.no_grad():
        emb_goal = critic.encode(goal_img_224)  # ProbEmbedding
    obj_labels = env_wrapper.get_obj_labels()
    task_instruction = env_wrapper.get_task_instruction()
    print(f"Task instruction: {task_instruction}")
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
        print(f"t={t}  Generating plan with VLM...")
        # plan = planner.propose(
        #     current_image_np, goal_image_np, history, obj_labels, horizon,
        #     task_instruction=task_instruction,
        # ) 
        plan = ['round nut']
        step_record.plan = list(plan)
        logger.info("t=%d  Proposed plan: %s", t, plan)

        # ── Stage 2+3: Imagination + Critic loop ──────────────────────
        plan_accepted = False

        for replan_attempt in range(max_replans + 1):
            all_scores: List[Tuple[float, float]] = []
            imagined_img = current_image_np
            step_failed = False

            # Expand nut names to (nut_name, sub_skill_prompt) pairs.
            imagination_steps = expand_nut_plan(plan)

            # Track previous frame embedding for Head 2 (temporal consistency)
            cur_img_224 = preprocess_image_for_critic(current_image_np).to(torch_device)
            with torch.no_grad():
                emb_prev = critic.encode(cur_img_224)  # ProbEmbedding

            # Wrap the k-loop in rollout_context so the oracle WM chains states:
            # S0 → S1 → S2 … instead of always branching from S0.
            _wm_ctx = (
                world_model.rollout_context()
                if isinstance(world_model, OracleWorldModel)
                else nullcontext()
            )
            with _wm_ctx:
              for k, (hl_action, imagine_action) in enumerate(imagination_steps):
                # ── Imagination ────────────────────────────────────────
                imagined_img_next = world_model.imagine(imagined_img, imagine_action)

                if out_path:
                    step_dir = out_path / "steps"
                    _save_image(
                        imagined_img_next, step_dir,
                        f"step_{t:03d}_imagine_r{replan_attempt}_k{k}.png",
                    )

                fig, ax = plt.subplots(1, 2)
                fig.suptitle(f"Step {k + 1}/{len(imagination_steps)}: '{imagine_action}'")
                ax[0].imshow(imagined_img)
                ax[0].set_title(f"Before '{imagine_action}'")
                ax[1].imshow(imagined_img_next)
                ax[1].set_title(f"After '{imagine_action}'")
                plt.show()

                # ── Critic evaluation ──────────────────────────────────
                img_224 = preprocess_image_for_critic(imagined_img_next).to(torch_device)
                with torch.no_grad():
                    emb_next = critic.encode(img_224)  # ProbEmbedding
                    mean_tc, std_tc = critic.temporal_sim_with_uncertainty(emb_prev, emb_next)
                    tc_score = mean_tc.item()
                    tc_uncertainty = std_tc.item()
                all_scores.append((tc_score, 0.0))

                decision = check_rollout_consistency(tc_score, theta_c, uncertainty=tc_uncertainty)
                step_record.critic_decisions.append(
                    f"k={k} action='{imagine_action}' tc={tc_score:.3f}(unc={tc_uncertainty:.3f}) → {decision.action}"
                )
                logger.info(
                    "  k=%d  action='%s'  temporal_sim=%.3f(unc=%.3f)  → %s",
                    k, imagine_action, tc_score, tc_uncertainty, decision.action,
                )

                if decision.action == "requery":
                    retry_wm = requery_world_model or world_model
                    for retry_i in range(max_retries):
                        # If retrying with the oracle WM inside a chain, roll the sim
                        # back to the pre-action state before each attempt.
                        if isinstance(retry_wm, OracleWorldModel) and retry_wm._rollout_state is not None:
                            retry_wm.rollback_step()
                        imagined_img_next = retry_wm.imagine(imagined_img, imagine_action)
                        
                        fig, ax = plt.subplots(1, 2)
                        fig.suptitle(f"Retry {retry_i + 1}/{max_retries} for action '{imagine_action}'")
                        ax[0].imshow(imagined_img)
                        ax[0].set_title(f"Before '{imagine_action}'")
                        ax[1].imshow(imagined_img_next)
                        ax[1].set_title(f"After '{imagine_action}'")
                        plt.show()

                        img_224 = preprocess_image_for_critic(imagined_img_next).to(torch_device)
                        with torch.no_grad():
                            emb_next = critic.encode(img_224)
                            mean_tc, std_tc = critic.temporal_sim_with_uncertainty(emb_prev, emb_next)
                            tc_score = mean_tc.item()
                            tc_uncertainty = std_tc.item()
                        all_scores[-1] = (tc_score, 0.0)
                        decision = check_rollout_consistency(tc_score, theta_c, uncertainty=tc_uncertainty)
                        logger.info(
                            "    requery %d/%d  tc=%.3f(unc=%.3f)  → %s",
                            retry_i + 1, max_retries, tc_score, tc_uncertainty, decision.action,
                        )
                        if decision.action != "requery":
                            break
                    else:
                        decision = CriticDecision(action="reflect", reason="requery_exhausted")

                if decision.action == "reflect":
                    step_record.failed_step = k
                    step_record.replan_attempts = replan_attempt + 1
                    reflect_plan = [s for _, s in imagination_steps]
                    ctx = build_reflection_context(
                        imagined_state=imagined_img_next,
                        all_scores=all_scores,
                        consistency_scores=[s for s, _ in all_scores],
                        proximity_score=None,
                        failed_step=k,
                        full_plan=reflect_plan,
                    )
                    ctx["failed_highlevel_action"] = hl_action
                    result = planner.reflect(
                        current_image_np, goal_image_np,
                        history, obj_labels, plan, ctx,
                        task_instruction=task_instruction,
                    )
                    plan = result["revised_plan"]
                    step_record.reflection_analyses.append(result.get("analysis", ""))
                    step_record.plan = list(plan)
                    step_failed = True
                    break

                emb_prev = emb_next
                imagined_img = imagined_img_next

            # ── Head 1 gate: final proximity check (DINOv2 only) ──────────
            if not step_failed:
                # emb_prev is now the final imagined frame embedding
                with torch.no_grad():
                    mean_prox, std_prox = critic.goal_sim_with_uncertainty(emb_prev, emb_goal)
                    prox_score = mean_prox.item()
                    prox_uncertainty = std_prox.item()

                # Update last score tuple with the proximity score
                if all_scores:
                    last_tc = all_scores[-1][0]
                    all_scores[-1] = (last_tc, prox_score)

                prox_decision = decide_from_proximity(prox_score, theta_p, uncertainty=prox_uncertainty)
                logger.info(
                    "  HEAD1 proximity=%.3f(unc=%.3f)  → %s",
                    prox_score, prox_uncertainty, prox_decision.action
                )
                step_record.critic_decisions.append(
                    f"HEAD1 proximity={prox_score:.3f}(unc={prox_uncertainty:.3f}) → {prox_decision.action}"
                )

                if prox_decision.action == "requery":
                    logger.info("  HEAD1 uncertain failure; rerolling imagination with same plan")
                    continue

                if prox_decision.action == "reflect":
                    step_record.failed_step = len(imagination_steps) - 1
                    step_record.replan_attempts = replan_attempt + 1
                    reflect_plan = [s for _, s in imagination_steps]
                    ctx = build_reflection_context(
                        imagined_state=imagined_img,
                        all_scores=all_scores,
                        consistency_scores=[s for s, _ in all_scores],
                        proximity_score=prox_score,
                        failed_step=step_record.failed_step,
                        full_plan=reflect_plan,
                    )
                    ctx["failed_highlevel_action"] = imagination_steps[-1][0]
                    result = planner.reflect(
                        current_image_np, goal_image_np,
                        history, obj_labels, plan, ctx,
                        task_instruction=task_instruction,
                    )
                    plan = result["revised_plan"]
                    step_record.reflection_analyses.append(result.get("analysis", ""))
                    step_record.plan = list(plan)
                    step_failed = True

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

        logger.info("t=%d  EXECUTE nut: '%s'", t, action_to_execute)
        obs, skill_ok = env_wrapper.execute_nut_assembly(action_to_execute)
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


def _ensure_workspace_robosuite_on_path() -> None:
    """Ensure the local workspace robosuite package is importable."""
    repo_root = Path(__file__).resolve().parents[2]
    robosuite_dir = repo_root / "robosuite"
    if robosuite_dir.exists() and str(robosuite_dir) not in sys.path:
        sys.path.insert(0, str(robosuite_dir))


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def _build_env(args: argparse.Namespace):
    _robosuite_root = Path(__file__).resolve().parents[2] / "robosuite"
    if str(_robosuite_root) not in sys.path:
        sys.path.insert(0, str(_robosuite_root))
    from run_cluttered_nutassembly import create_environment

    env = create_environment(
        env_name="ClutteredNutAssembly",
        num_round_nuts=args.num_round,
        num_square_nuts=args.num_square,
        initial_stacking_prob=args.initial_stacking_prob,
        nut_type_mode=args.nut_type_mode,
        has_offscreen_renderer=True,
        # render_camera=args.camera,
        use_camera_obs=False,
        horizon=args.env_horizon,
    )
    return NutAssemblyEnvWrapper(env, camera=args.camera, image_size=args.image_size)


def _build_critic(
    args: argparse.Namespace, device: torch.device
) -> DINOv2DualHeadCritic:
    """Load DINOv2DualHeadCritic checkpoint."""
    ckpt = torch.load(args.critic_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    critic = DINOv2DualHeadCritic(pretrained=False).to(device)
    missing, unexpected = critic.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(
            "Checkpoint missing keys (log_var heads will be randomly initialised): %s",
            missing,
        )
    if unexpected:
        logger.warning("Checkpoint had unexpected keys: %s", unexpected)
    critic.eval()
    return critic


def _build_world_models(args: argparse.Namespace):
    mode = args.wm_mode
    wm = None
    requery_wm = None

    if mode in ("oracle", "hybrid"):
        wm = OracleWorldModel(args.env_wrapper)

    if mode in ("diffusion", "hybrid"):
        # If the requested decoder directory is missing or doesn't contain
        # the expected files, ignore it and let the pipeline use the
        # pretrained VAE decoder (matches demo_wm.py behaviour).
        decoder_dir = args.wm_decoder_dir
        if decoder_dir is not None:
            dec_path = Path(decoder_dir)
            if not dec_path.exists() or not (
                (dec_path / "config.json").exists() or (dec_path / "decoder_state_dict.pt").exists()
            ):
                logger.warning(
                    "Requested wm-decoder-dir=%s missing config.json or decoder_state_dict.pt; "
                    "falling back to pretrained VAE decoder.",
                    decoder_dir,
                )
                decoder_dir = None

        diff_wm = DiffusionWorldModel(
            pretrained_model=args.wm_model,
            adapter_dir=args.wm_adapter_dir,
            decoder_dir=decoder_dir,
            vae_model=args.vae_model,
            vae_subfolder=args.vae_subfolder,
            device=args.device,
            torch_dtype=_dtype_from_name(args.dtype),
            num_inference_steps=args.wm_steps,
            image_guidance_scale=args.wm_image_guidance,
            guidance_scale=args.wm_text_guidance,
            seed=args.wm_seed,
        )
        if mode == "diffusion":
            wm = diff_wm
        else:
            requery_wm = diff_wm

    if wm is None:
        raise ValueError(f"Unsupported wm_mode: {mode}")

    return wm, requery_wm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Verify2Act inference episode")
    parser.add_argument("--env-name", default="ClutteredNutAssembly")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--has-renderer", action="store_true")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--env-horizon", type=int, default=2000)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--vae-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--vae-subfolder", default="vae")

    parser.add_argument("--critic-ckpt", default="verify2act/output/contrastive/best_contrastive_critic.pt")

    parser.add_argument("--prompt-config", default="verify2act/configs/prompts/planner.yaml")
    parser.add_argument("--planner-model", default="gpt-4o")
    parser.add_argument("--planner-max-tokens", type=int, default=512)
    parser.add_argument("--planner-temperature", type=float, default=0.2)

    parser.add_argument("--wm-mode", choices=["oracle", "diffusion", "hybrid"], default="hybrid")
    parser.add_argument("--wm-model", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--wm-adapter-dir", default="verify2act/output/wm/best/unet_lora")
    parser.add_argument("--wm-decoder-dir", default="verify2act/output/decoder")
    parser.add_argument("--wm-steps", type=int, default=30)
    parser.add_argument("--wm-image-guidance", type=float, default=1.5)
    parser.add_argument("--wm-text-guidance", type=float, default=7.5)
    parser.add_argument("--wm-seed", type=int, default=None)

    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--theta-c", type=float, default=0.5)
    parser.add_argument("--theta-p", type=float, default=0.6)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-replans", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="verify2act/output/inference_run")

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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    # fp16 LayerNorm is only supported on CUDA; fall back to fp32 on CPU/MPS
    if device.type != "cuda" and args.dtype == "fp16":
        logger.warning("fp16 is not supported on %s; falling back to fp32", device.type)
        args.dtype = "fp32"

    args.env_wrapper = _build_env(args)
    vae, _resolved = load_vae_encoder(
        model_name_or_path=args.vae_model,
        device=device,
        torch_dtype=_dtype_from_name(args.dtype),
        subfolder=args.vae_subfolder,
        local_files_only=args.local_files_only,
    )
    critic = _build_critic(args, device)
    planner = VLMPlanner.from_yaml(
        args.prompt_config,
        model=args.planner_model,
        max_tokens=args.planner_max_tokens,
        temperature=args.planner_temperature,
    )
    world_model, requery_world_model = _build_world_models(args)

    trace = run_episode(
        env_wrapper=args.env_wrapper,
        vae=vae,
        world_model=world_model,
        requery_world_model=requery_world_model,
        critic=critic,
        planner=planner,
        horizon=args.horizon,
        max_steps=args.max_steps,
        theta_c=args.theta_c,
        theta_p=args.theta_p,
        max_retries=args.max_retries,
        max_replans=args.max_replans,
        device=args.device,
        output_dir=args.output_dir,
    )
    logger.info(
        "Episode complete: success=%s total_steps=%d total_replans=%d",
        trace.success,
        trace.total_steps,
        trace.total_replans,
    )

    args.env_wrapper.env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
