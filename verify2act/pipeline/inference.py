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
    --wm-mode v2a_wm \
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

import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence
collections.Iterable = collections.abc.Iterable
collections.Set = collections.abc.Set
import math
import fractions
fractions.gcd = math.gcd
import numpy as np
np.float = float

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
from verify2act.pipeline.planner import BeamSearchPlanner
from verify2act.pipeline.world_model import LatentWorldModel

import os
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glx'

logger = logging.getLogger(__name__)

# Silence the repetitive robosuite controller-config warnings.
# These fire on every policy episode init ("left/torso/head/base/legs
# controller not found") and add no useful information during inference.
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)


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
    # True when the real robot skill (pick/insert) failed during execution.
    # Distinguishes execution failures from critic-imagined failures.
    execution_failed: bool = False


@dataclass
class EpisodeTrace:
    """Full episode-level log returned by ``run_episode()``."""

    success: bool = False
    total_steps: int = 0
    total_replans: int = 0
    nuts_placed: int = 0
    total_target_nuts: int = 0
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
    goal_renderer,
    vae: Optional[torch.nn.Module],
    world_model: Optional[WorldModelBase],
    critic: Optional[torch.nn.Module],
    planner: VLMPlanner,
    *,
    beam_planner: Optional[BeamSearchPlanner] = None,
    decoder: Optional[torch.nn.Module] = None,
    wm_mode: str = "v2a_wm",
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
    goal_renderer : NutAssemblyGoalRenderer
        Goal renderer bound to the same env instance.
    vae : AutoencoderKL
        Frozen VAE encoder — used only by the diffusion world model.
    world_model : WorldModelBase
        Unified world model for imagined rollouts and requery retries.
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
    wm_mode : str
        The world model mode being run ("oracle", "diffusion", "v2a_wm", "rla_wm").
    output_dir : str or None
        If set, save episode traces and images here.

    Returns
    -------
    EpisodeTrace
        Full record of the episode for analysis and debugging.
    """
    torch_device = torch.device(device)
    out_path = Path(output_dir) if output_dir else None
    if critic is not None:
        critic.eval()
        if not isinstance(critic, DINOv2DualHeadCritic):
            raise TypeError("run_episode expects DINOv2DualHeadCritic")

    # ── Read current environment state ────────────────────────────────────
    # The caller is responsible for resetting the env with the desired seed
    # before calling run_episode().  We just read the current obs.
    obs = env_wrapper._obs
    if obs is None:
        # Fallback: reset if no prior state (e.g. standalone usage)
        obs = env_wrapper.reset()
    print(f"Environment ready. Observation keys: {list(obs.keys())}")

    # # Sync the on-screen viewer to the settled T=0 state.
    # try:
    #     env_wrapper._settle_and_sync_viewer(n_steps=0)
    #     print("Viewer synced to initial state.")
    # except Exception as e:
    #     logger.warning("Could not sync viewer: %s", e)
            
    obj_labels = env_wrapper.get_obj_labels()
    language_goal = env_wrapper.get_task_instruction()
    print(f"Language goal: {language_goal}")
    history: List[str] = []

    trace = EpisodeTrace()

    # ── Stagnation tracking ─────────────────────────────────────────────
    # Counts consecutive timesteps where nothing progressed (empty plan or
    # execution failure).  When this hits the limit we strip [FAILED] tags
    # from history so the VLM can see those nuts again and retry them.
    consecutive_failures: int = 0
    STAGNATION_LIMIT: int = 3

    # ── Environment-reset tracking ────────────────────────────────
    # When EEF stagnation exhausts the robosuite env horizon (env.done=True)
    # we reset the physics episode in-place (preserving history / labels /
    # stagnation state) so the outer loop can keep retrying.  After
    # MAX_ENV_RESETS such resets we give up and end the episode gracefully.
    env_resets: int = 0
    MAX_ENV_RESETS: int = 5
    _episode_exhausted: bool = False  # set True to break the outer t-loop

    # ── Main loop: one iteration per real timestep ─────────────────────
    for t in range(max_steps):
        if _episode_exhausted:
            break
        current_image_np = env_wrapper.read_image()
        # Keep the MuJoCo viewer in sync with the real sim state at every
        # timestep.  In diffusion WM mode env.step() is never called during
        # imagination, so without this the viewer freezes after reset.
        env_wrapper._settle_and_sync_viewer(n_steps=0)
        print(f"t={t}  Current image shape: {current_image_np.shape}, dtype: {current_image_np.dtype}")
        step_record = StepRecord(timestep=t)

        if out_path:
            _save_image(current_image_np, out_path / "steps", f"step_{t:03d}_current.png")

        # ── Stage 1-3: Plan generation, Imagination, and Critic (BeamSearch) ──
        if beam_planner:
            print(f"t={t}  Generating plan with BeamSearchPlanner (Mode: {wm_mode})...")
            res = beam_planner.plan(
                current_image_np=current_image_np,
                history=history,
                obj_labels=obj_labels,
                horizon=horizon,
                language_goal=language_goal,
                timestep=t,
                output_dir=out_path,
                decoder=decoder,
            )
            plan = res["plan"]
            step_record.plan = list(plan)
            step_record.replan_attempts = res["replan_attempts"]
            step_record.all_scores = res["all_scores"]
            step_record.failed_step = res["failed_step"]
            step_record.reflection_analyses = res["reflection_analyses"]
            step_record.critic_decisions = res["critic_decisions"]
            plan_accepted = res["plan_accepted"]
        else:
            print(f"t={t}  Generating plan with VLM (Mode: {wm_mode})...")
            plan = planner.propose(
                current_image_np=current_image_np,
                history=history,
                obj_labels=obj_labels,
                horizon=horizon,
                language_goal=language_goal,
            )
            step_record.plan = list(plan)
            logger.info("t=%d  Proposed plan: %s", t, plan)
            plan_accepted = True


        if not plan_accepted:
            logger.warning(
                "t=%d  Max replans exhausted (%d). Executing first action anyway.",
                t, max_replans,
            )

        # ── Execute the first action on the real environment ───────────
        if not plan:
            logger.warning("t=%d  Empty plan — skipping execution.", t)
            consecutive_failures += 1
            if consecutive_failures >= STAGNATION_LIMIT:
                # The VLM has been returning empty plans for several rounds,
                # almost certainly because it sees [FAILED] tags for every
                # remaining nut and thinks there is nothing left to do.
                # Purge failure tags so it can see those nuts again and retry.
                logger.warning(
                    "t=%d  Stagnation detected (%d consecutive empty/failed timesteps). "
                    "Clearing execution-failure tags from history to allow retries.",
                    t, consecutive_failures,
                )
                history = [
                    (a["original"] if isinstance(a, dict) and "original" in a else a)
                    for a in history
                ]
                consecutive_failures = 0
            trace.steps.append(step_record)
            continue

        for idx, action in enumerate(plan):
            logger.info("t=%d  plan[%d]: '%s'", t, idx, action)

            # Resolve the human-readable label (supports both str and dict outputs).
            action_label = action.get("label", "") if isinstance(action, dict) else action

            # Handle "done" action from the VLM.
            if action_label.strip().lower() == "done":
                logger.info("t=%d  VLM returned 'done'.", t)
                step_record.action_executed = "done"
                trace.steps.append(step_record)
                break

            logger.info("t=%d  EXECUTE nut: '%s'", t, action)
            # Snapshot placed set BEFORE execution so we can restore it after
            # an in-place physics reset (env_wrapper.reset() clears placed).
            _placed_before_reset = set(env_wrapper.placed)
            obs, skill_ok = env_wrapper.execute_nut_assembly(action)
            step_record.action_executed = action
            # Record the attempt in history so the VLM knows what was tried.
            # On success: plain entry (VLM treats it as assembled → will skip).
            # On failure: tagged entry so the VLM knows to RETRY, not skip.
            # Actual tagging happens below after we know skill_ok.
            history.append(action)

            if out_path:
                _save_image(
                    env_wrapper.read_image(),
                    out_path / "steps",
                    f"step_{t:03d}_after_exec.png",
                )

            if not skill_ok:
                # ── Execution failure: abort this plan ─────────────────────
                # The task is sequential — subsequent actions depend on this
                # one succeeding (e.g. insert requires a successful pick).
                # Continuing would execute sub-plans built on a failed premise.
                # Instead we break so the outer loop re-plans from a fresh
                # observation of the actual (failed) scene state.
                #
                # Tag the most-recently-appended history entry as [FAILED].
                # This tells the VLM the nut is still unassembled and must
                # be RETRIED — not skipped as if it were already placed.
                # We preserve the original action under an "original" key so
                # that the stagnation-reset can strip the tag later.
                _failed_tag = (
                    {"label": f"[FAILED] {action.get('label', '')}",
                     "id": action.get("id", ""),
                     "original": action}
                    if isinstance(action, dict)
                    else f"[FAILED] {action}"
                )
                history[-1] = _failed_tag  # replace the plain entry just appended
                consecutive_failures += 1

                if consecutive_failures >= STAGNATION_LIMIT:
                    # Several consecutive kinematic failures on the same nut.
                    # Purge failure tags so the VLM is not permanently blocked.
                    logger.warning(
                        "t=%d  Stagnation limit reached (%d consecutive failures). "
                        "Clearing failure tags from history to allow fresh retries.",
                        t, consecutive_failures,
                    )
                    history = [
                        (a["original"] if isinstance(a, dict) and "original" in a else a)
                        for a in history
                    ]
                    consecutive_failures = 0

                logger.warning(
                    "t=%d  plan[%d]: skill execution FAILED for action '%s'. "
                    "Aborting remaining %d action(s) in this plan and re-planning.",
                    t, idx, action_label, len(plan) - idx - 1,
                )
                step_record.execution_failed = True
                trace.steps.append(step_record)

                if getattr(env_wrapper.env, "done", False):
                    logger.warning(
                        "t=%d  Environment horizon has been exhausted (env.done=True). "
                        "Terminating episode gracefully.",
                        t,
                    )
                    _episode_exhausted = True
                    break

                break

            # Successful execution — reset the stagnation counter.
            consecutive_failures = 0

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

    # Track partial success: how many nuts were placed out of the total
    try:
        active_targets = set(env_wrapper._active_nuts())
        trace.nuts_placed = len(env_wrapper.placed.intersection(active_targets))
        trace.total_target_nuts = len(active_targets)
    except Exception:
        trace.nuts_placed = len(env_wrapper.placed)
        trace.total_target_nuts = 0

    if trace.nuts_placed >= trace.total_target_nuts and trace.total_target_nuts > 0:
        trace.success = True

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
    from goal_renderer import NutAssemblyGoalRenderer

    env = create_environment(
        env_name="ClutteredNutAssembly",
        num_round_nuts=args.num_round,
        num_square_nuts=args.num_square,
        initial_stacking_prob=args.initial_stacking_prob,
        nut_type_mode=args.nut_type_mode,
        # has_offscreen_renderer=True,
        # render_camera=args.camera,
        use_camera_obs=False,
        horizon=args.env_horizon,
    )
    wrapper = NutAssemblyEnvWrapper(env, camera=args.camera, image_size=args.image_size)
    goal_renderer = NutAssemblyGoalRenderer(env, camera=args.camera, image_size=args.image_size)
    return wrapper, goal_renderer


def _randomize_environment(args: argparse.Namespace, world_model: Optional[Any], ep_idx: int) -> None:
    """Randomize the number of round and square nuts for this episode and re-create env."""
    total_nuts = args.num_round + args.num_square
    # Range from 1 to total_nuts - 1
    ep_num_round = int(np.random.randint(1, total_nuts))
    ep_num_square = total_nuts - ep_num_round
    
    logger.info(
        "Varying nut counts for Episode %d: Round=%d, Square=%d",
        ep_idx, ep_num_round, ep_num_square
    )
    
    # Close previous environment to avoid resource leaks
    if args.env_wrapper is not None:
        try:
            args.env_wrapper.env.close()
        except Exception as e:
            logger.warning("Error closing previous environment: %s", e)
        
    # Build new environment with the randomized counts
    # Temporarily modify args for _build_env
    orig_round = args.num_round
    orig_square = args.num_square
    args.num_round = ep_num_round
    args.num_square = ep_num_square
    
    args.env_wrapper, args.goal_renderer = _build_env(args)
    
    # Restore original args values
    args.num_round = orig_round
    args.num_square = orig_square
    
    # Update world model if it's Oracle
    if world_model is not None and hasattr(world_model, "env_wrapper"):
        world_model.env_wrapper = args.env_wrapper


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

    dino_channels = 1024  # default fallback
    if isinstance(ckpt, dict) and "args" in ckpt and "dino_channels" in ckpt["args"]:
        dino_channels = ckpt["args"]["dino_channels"]
    else:
        for key in ["head1.weight", "head1.0.weight"]:
            if key in state_dict:
                dino_channels = state_dict[key].shape[1]
                break

    critic = DINOv2DualHeadCritic(pretrained=False, dino_channels=dino_channels).to(device)
    missing, unexpected = critic.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(
            "Checkpoint missing keys (log_var heads will be randomly initialised): %s",
            missing,
        )
    if unexpected:
        logger.debug("Checkpoint had unexpected keys (safe to ignore): %s", unexpected)
    critic.eval()
    return critic


def _build_world_model(args: argparse.Namespace):
    mode = args.wm_mode
    if mode == "vlm_only":
        return None

    if mode == "v2a_wm":
        from verify2act.pipeline.world_model import LatentWorldModel
        wm = LatentWorldModel(
            device=args.device, 
            dynamics_weights_path=args.latent_wm_ckpt,
            encoder_ckpt=args.encoder_ckpt,
            history_len=args.history_len,
        )
        return wm
        
    if mode == "rla_wm":
        from verify2act.pipeline.world_model import RLAWorldModel
        wm = RLAWorldModel(
            device=args.device, 
            dynamics_weights_path=args.latent_wm_ckpt,
            encoder_ckpt=args.encoder_ckpt,
            history_len=args.history_len,
        )
        return wm

    if mode == "diffusion":
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

        wm = DiffusionWorldModel(
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
        return wm

    if mode == "oracle":
        wm = OracleWorldModel(args.env_wrapper)
        return wm
    raise ValueError(f"Unsupported wm_mode: {mode}")


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

    parser.add_argument("--wm-mode", choices=["oracle", "diffusion", "v2a_wm", "rla_wm", "vlm_only"], default="v2a_wm")
    parser.add_argument("--wm-model", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--wm-adapter-dir", default="verify2act/output/wm/best/unet_lora")
    parser.add_argument("--wm-decoder-dir", default="verify2act/output/decoder")
    parser.add_argument("--latent-wm-ckpt", default=None, help="Path to LatentDynamicsModel checkpoint")
    parser.add_argument("--encoder-ckpt", default=None, help="Path to pre-trained DeltaEncoder checkpoint (encoder_only_best.pt)")
    parser.add_argument("--history-len", type=int, default=3, help="Number of historical frames for world model context")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--wm-steps", type=int, default=30)
    parser.add_argument("--wm-image-guidance", type=float, default=2.8)
    parser.add_argument("--wm-text-guidance", type=float, default=7.5)
    parser.add_argument("--wm-seed", type=int, default=None)

    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--theta-c", type=float, default=0.7)
    parser.add_argument("--theta-p", type=float, default=0.7)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-replans", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="verify2act/output/eval")

    # Multi-episode eval
    parser.add_argument("--num-episodes", type=int, default=15,
                        help="Number of evaluation episodes to run")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base seed; episode i uses base_seed + i")

    # Nut assembly params
    parser.add_argument("--num-round", type=int, default=3)
    parser.add_argument("--num-square", type=int, default=2)
    parser.add_argument("--initial-stacking-prob", type=float, default=0.6)
    parser.add_argument(
        "--nut-type-mode",
        type=str,
        default="random",
        choices=["roundnut", "squarenut", "random", "alternate"],
        help="Nut type mode for ClutteredNutAssembly",
    )
    parser.add_argument(
        "--randomize-nut-counts",
        action="store_true",
        help="If set, the number of round and square nuts is randomized per episode "
             "such that the total number remains constant (num-round + num-square) "
             "with at least 1 of each type present."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("robosuite_logs").setLevel(logging.ERROR)
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()

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

    args.env_wrapper, args.goal_renderer = _build_env(args)
    planner = VLMPlanner.from_yaml(
        args.prompt_config,
        model=args.planner_model,
        max_tokens=args.planner_max_tokens,
        temperature=args.planner_temperature,
    )
    # Load VAE encoder unless in VLM-Only mode
    if args.wm_mode == "vlm_only":
        vae = None
    else:
        vae, _resolved = load_vae_encoder(
            model_name_or_path=args.vae_model,
            device=device,
            torch_dtype=_dtype_from_name(args.dtype),
            subfolder=args.vae_subfolder,
            local_files_only=args.local_files_only,
        )

    # Load Critic unless in VLM-Only or Diffusion modes
    if args.wm_mode in ["vlm_only", "diffusion"]:
        logger.info(f"Bypassing Critic loading for wm_mode={args.wm_mode}.")
        critic = None
    else:
        critic = _build_critic(args, device)

    # Load World Model unless in VLM-Only mode
    if args.wm_mode == "vlm_only":
        world_model = None
    else:
        world_model = _build_world_model(args)

    # Initialize BeamSearchPlanner unless in VLM-Only mode
    if args.wm_mode == "vlm_only":
        beam_planner = None
    else:
        beam_planner = BeamSearchPlanner(
            vlm_planner=planner,
            world_model=world_model,
            critic=critic,
            beam_width=args.beam_width,
            goal_threshold=args.theta_p,
            plan_expander=expand_nut_plan,
            temporal_threshold=args.theta_c,
            max_retries=args.max_retries,
            max_replans=args.max_replans,
            wm_mode=args.wm_mode,
        )
    
    decoder = None
    if args.wm_mode in ["v2a_wm", "rla_wm"]:
        # Load decoder for visual reflection
        from verify2act.latent_wm.decoder import FeatureDecoder
        logger.info("Loading FeatureDecoder for visual reflection...")
        decoder = FeatureDecoder(dino_channels=1024).to(device)
        decoder.eval()
        if args.wm_decoder_dir:
            dec_path = Path(args.wm_decoder_dir) / "latent_decoder_best.pt"
            if dec_path.exists():
                state_dict = torch.load(dec_path, map_location=device)
                # If keys don't have "decoder." prefix, add it (checkpoint was saved from inner decoder)
                if "decoder.input_proj.0.weight" not in state_dict and "input_proj.0.weight" in state_dict:
                    state_dict = {f"decoder.{k}": v for k, v in state_dict.items()}
                decoder.load_state_dict(state_dict)
            else:
                logger.warning(f"Decoder checkpoint not found at {dec_path}")

    # ── Multi-episode evaluation loop ────────────────────────────────────
    eval_dir = Path(args.output_dir) / args.wm_mode
    eval_dir.mkdir(parents=True, exist_ok=True)
    num_episodes = args.num_episodes
    base_seed = args.base_seed
    traces: List[EpisodeTrace] = []

    logger.info(
        "Starting evaluation: mode=%s, episodes=%d, base_seed=%d",
        args.wm_mode, num_episodes, base_seed,
    )

    for ep_idx in range(num_episodes):
        ep_seed = base_seed + ep_idx
        ep_dir = eval_dir / f"episode_{ep_idx:03d}"

        logger.info(
            "\n" + "=" * 60 + "\n"
            "  Episode %d / %d  (seed=%d, mode=%s)\n" + "=" * 60,
            ep_idx + 1, num_episodes, ep_seed, args.wm_mode,
        )

        # Seed everything for reproducibility
        np.random.seed(ep_seed)
        torch.manual_seed(ep_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ep_seed)

        if args.randomize_nut_counts:
            _randomize_environment(args, world_model, ep_idx)

        # Reset env (policy instance is invalidated so the next
        # execute_nut_assembly call creates a fresh one).
        args.env_wrapper._policy = None
        args.env_wrapper.reset(seed=ep_seed)

        trace = run_episode(
            env_wrapper=args.env_wrapper,
            goal_renderer=args.goal_renderer,
            vae=vae,
            world_model=world_model,
            critic=critic,
            planner=planner,
            beam_planner=beam_planner,
            decoder=decoder,
            wm_mode=args.wm_mode,
            horizon=args.horizon,
            max_steps=args.max_steps,
            theta_c=args.theta_c,
            theta_p=args.theta_p,
            max_retries=args.max_retries,
            max_replans=args.max_replans,
            device=args.device,
            output_dir=str(ep_dir),
        )
        traces.append(trace)

        logger.info(
            "Episode %d/%d complete: success=%s  nuts_placed=%d/%d  "
            "steps=%d  replans=%d",
            ep_idx + 1, num_episodes,
            trace.success, trace.nuts_placed, trace.total_target_nuts,
            trace.total_steps, trace.total_replans,
        )

    # ── Aggregate metrics ────────────────────────────────────────────────
    num_eps = len(traces)
    successes = sum(1 for t in traces if t.success)
    total_nuts_placed = sum(t.nuts_placed for t in traces)
    total_target = sum(t.total_target_nuts for t in traces)

    summary = {
        "wm_mode": args.wm_mode,
        "num_episodes": num_eps,
        "base_seed": base_seed,
        "success_rate": successes / num_eps if num_eps else 0.0,
        "nut_completion_rate": total_nuts_placed / total_target if total_target else 0.0,
        "avg_steps": np.mean([t.total_steps for t in traces]).item() if traces else 0.0,
        "avg_replans": np.mean([t.total_replans for t in traces]).item() if traces else 0.0,
        "avg_nuts_placed": np.mean([t.nuts_placed for t in traces]).item() if traces else 0.0,
        "per_episode": [
            {
                "episode": i,
                "seed": base_seed + i,
                "success": t.success,
                "nuts_placed": t.nuts_placed,
                "total_target_nuts": t.total_target_nuts,
                "total_steps": t.total_steps,
                "total_replans": t.total_replans,
            }
            for i, t in enumerate(traces)
        ],
    }

    summary_path = eval_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Eval summary saved to %s", summary_path)

    # Print final report
    print("\n" + "=" * 60)
    print(f"  EVALUATION COMPLETE — {args.wm_mode}")
    print("=" * 60)
    print(f"  Episodes:            {num_eps}")
    print(f"  Success Rate:        {summary['success_rate']:.1%} ({successes}/{num_eps})")
    print(f"  Nut Completion Rate: {summary['nut_completion_rate']:.1%} ({total_nuts_placed}/{total_target})")
    print(f"  Avg Nuts Placed:     {summary['avg_nuts_placed']:.2f}")
    print(f"  Avg Steps:           {summary['avg_steps']:.2f}")
    print(f"  Avg Replans:         {summary['avg_replans']:.2f}")
    print(f"  Results:             {summary_path}")
    print("=" * 60 + "\n")

    args.env_wrapper.env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
