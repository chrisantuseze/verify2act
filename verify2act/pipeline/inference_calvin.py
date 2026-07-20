"""CALVIN Inference pipeline - Stage 1+2+3 loop for CALVIN.

This script wires together the VLM planner, world model, critic, and MCIL baseline
policy to run multi-step evaluation sequences on CALVIN's PlayTableSimEnv.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# GPU memory optimization
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# PyTorch 2.6+ compatibility: patch lightning_fabric._load to disable weights_only
try:
    import lightning_fabric.utilities.cloud_io as cloud_io_module
    _original_load = cloud_io_module._load
    def _patched_load(path, map_location=None, weights_only=None):
        # Always use weights_only=False for checkpoint loading
        import torch
        return torch.load(path, map_location=map_location, weights_only=False)
    cloud_io_module._load = _patched_load
except (ImportError, AttributeError):
    pass

# Ensure verify2act, calvin_agent, and calvin_env are in PYTHONPATH
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
import torch
np.float = float

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
calvin_models = repo_root / "calvin/calvin_models"
if calvin_models.exists() and str(calvin_models) not in sys.path:
    sys.path.insert(0, str(calvin_models))
calvin_env_path = repo_root / "calvin/calvin_env"
if calvin_env_path.exists() and str(calvin_env_path) not in sys.path:
    sys.path.insert(0, str(calvin_env_path))
tacto_path = calvin_env_path / "tacto"
if tacto_path.exists() and str(tacto_path) not in sys.path:
    sys.path.insert(0, str(tacto_path))

from calvin_env.envs.play_table_env import get_env
from calvin_agent.evaluation.multistep_sequences import get_sequences
import time
from termcolor import colored
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from verify2act.pipeline.calvin_agent_wrapper import (
    CalvinEpisodeTrace,
    Verify2ActCalvinAgent,
    _save_calvin_trace,
)
from verify2act.pipeline.planner import VLMPlanner
from verify2act.pipeline.world_model import LatentWorldModel
from verify2act.critic.model import DINOv2DualHeadCritic
import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


EP_LEN = 360
NUM_SEQUENCES = 1000


def _evaluate_calvin(
    agent: Verify2ActCalvinAgent,
    env,
    eval_log_dir: Path,
    debug: bool = False,
    num_sequences: int = NUM_SEQUENCES,
) -> tuple:
    """Custom CALVIN evaluation loop that mirrors evaluate_policy() but hooks
    agent.start_sequence() / flush_trace() at each sequence boundary so that
    rich per-step metrics are captured.

    Returns
    -------
    results : List[int]
        Number of subtasks completed per sequence (0-5).
    traces : List[CalvinEpisodeTrace]
        One per sequence, containing critic/VLM/reflection metrics.
    """
    conf_dir = (
        Path(__file__).absolute().parents[2]
        / "calvin/calvin_models/conf"
    )
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )

    eval_sequences = get_sequences(num_sequences)
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    results = []
    traces: list = []

    for seq_idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        # ── Reset per-sequence accumulator ─────────────────────────
        agent.start_sequence(seq_idx, output_dir=eval_log_dir)

        # ── Reset env to the initial scene state ─────────────────────
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        if debug:
            time.sleep(1)
            print()
            print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
            print("Subtask: ", end="")

        success_counter = 0
        for subtask in eval_sequence:
            lang_annotation = val_annotations[subtask][0]
            agent.reset()
            start_info = env.get_info()
            obs = env.get_obs()

            if debug:
                print(f"{subtask} ", end="", flush=True)
                time.sleep(0.5)

            subtask_success = False
            for step in range(EP_LEN):
                action = agent.step(obs, lang_annotation)
                obs, _, _, current_info = env.step(action)
                if debug:
                    img = env.render(mode="rgb_array")
                    join_vis_lang(img, lang_annotation)
                task_info = task_oracle.get_task_info_for_set(
                    start_info, current_info, {subtask}
                )
                if len(task_info) > 0:
                    subtask_success = True
                    break

            if debug:
                if subtask_success:
                    print(colored("success", "green"), end=" ", flush=True)
                else:
                    print(colored("fail", "red"), end=" ", flush=True)

            # Commit subtask outcome into agent's running trace
            agent._close_subtask(subtask_success)

            if subtask_success:
                success_counter += 1
            else:
                break  # CALVIN stops the sequence on first failure

        if debug:
            print()  # newline after the subtask success/fail line

        # ── Flush trace after sequence completes ────────────────────
        seq_dir = eval_log_dir / f"sequence_{seq_idx:04d}"
        trace = agent.flush_trace(
            subtasks_completed=success_counter,
            output_dir=eval_log_dir,
        )
        traces.append(trace)
        results.append(success_counter)

        if not debug:
            eval_sequences.set_description(
                " ".join(
                    [f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]
                ) + "|"
            )

    return results, traces


def _build_critic(args: argparse.Namespace, device: torch.device) -> DINOv2DualHeadCritic:
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
    # Filter out _clip_model keys since CLIP is lazily loaded and not part of the model structure at init
    unexpected = [k for k in unexpected if not k.startswith("_clip_model.")]
    if unexpected:
        logger.info("Checkpoint had unexpected keys: %s", unexpected)
    critic.eval()
    return critic


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Verify2Act on CALVIN")
    parser.add_argument("--device", default="cuda", help="Torch device (e.g. cuda, cpu)")
    
    # Model checkpoints
    parser.add_argument("--critic-ckpt", default="verify2act/output/contrastive/calvin/best_contrastive_critic.pt")
    parser.add_argument("--latent-wm-ckpt", default="verify2act/output/v2a_wm/calvin/wm/ckpt/latent_dynamics_best.pt")
    parser.add_argument("--encoder-ckpt", default="verify2act/output/v2a_wm/calvin/encoder_vitl/ckpt/encoder_only_best.pt")
    parser.add_argument("--wm-decoder-dir", default="verify2act/output/v2a_wm/calvin/decoder")
    
    # CALVIN directories & policies
    parser.add_argument("--train-folder", default="calvin/models/hulc_baseline", help="MCIL/HULC baseline logs/checkpoints dir")
    parser.add_argument("--low-level-policy", choices=["hulc", "diffusion", "mode"], default="hulc", help="Type of low-level policy to use")
    parser.add_argument("--low-level-policy-ckpt", default=None, help="Path to checkpoint for diffusion low-level policy (if different from train-folder)")
    parser.add_argument("--dataset-path", default="calvin/dataset/task_ABC_D", help="Primary CALVIN dataset dir (used for env init and statistics)")
    parser.add_argument(
        "--full-dataset-path",
        default=None,
        help="Optional path to the full (unfiltered) CALVIN dataset. When --dataset-path points to "
             "task_ABC_D_filtered, pass 'calvin/dataset/task_ABC_D' here to merge its language "
             "embeddings and eliminate nearest-neighbour fallbacks for missing eval sentences.",
    )
    
    # Planner configuration
    parser.add_argument("--prompt-config", default="verify2act/configs/prompts/planner.yaml")
    parser.add_argument("--planner-model", default="gemini-2.5-flash")
    parser.add_argument("--planner-max-tokens", type=int, default=2048)
    parser.add_argument("--planner-temperature", type=float, default=0.2)
    parser.add_argument(
        "--no-gemini-retry-warn",
        action="store_true",
        default=False,
        help="Suppress rate-limit retry warnings when using Gemini (equivalent to GEMINI_WARN_ON_RETRY=0).",
    )
    parser.add_argument(
        "--gcp-project",
        default="verify2act",
        metavar="PROJECT_ID",
        help=(
            "GCP project ID to use when calling Vertex AI (sets GOOGLE_CLOUD_PROJECT). "
            "Required when ADC credentials do not embed a project (e.g. user credentials "
            "obtained via 'gcloud auth application-default login')."
        ),
    )
    parser.add_argument(
        "--planner-call-delay",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="Seconds to sleep before each Gemini API call to reduce RPM pressure on the free tier (e.g. 3.0).",
    )
    
    # Hyperparameters
    parser.add_argument("--theta-c", type=float, default=0.6, help="Head 2 temporal consistency threshold (cosine sim). Observed CALVIN range: 0.35-0.84; default 0.5 separates poor from plausible transitions.")
    parser.add_argument(
        "--theta-p", type=float, default=0.2,
        help=(
            "Head 1 goal proximity threshold (cosine sim, DINO-to-CLIP cross-modal). "
            "Observed CALVIN range is 0.1-0.44; default 0.2 accepts well-aligned imaginations "
            "while rejecting clearly off-goal ones. Tune this separately from --theta-c."
        ),
    )
    parser.add_argument(
        "--critic-unc-threshold",
        type=float,
        default=0.08,
        metavar="UNC",
        help=(
            "MC uncertainty gate for the critic (confidence_threshold). "
            "Predictions with std > this value are treated as 'requery' regardless of score. "
            "Observed CALVIN critic std is 0.030-0.065; default 0.08 clears this range. "
            "Lower → stricter gating; raise if critic still returns all requery."
        ),
    )
    parser.add_argument("--max-replans", type=int, default=2, help="Max replanning rounds per failure")
    parser.add_argument("--history-len", type=int, default=3, help="Number of historical frames for world model context")
    parser.add_argument("--token-dim", type=int, default=128, help="Compact latent token dimension")
    parser.add_argument("--num-latent-tokens", type=int, default=32, help="Number of compact latent tokens")
    parser.add_argument("--action-conditioning", choices=["cross_attn", "adaln"], default="cross_attn", help="Action conditioning strategy for latent world model")
    parser.add_argument("--wm-mode", choices=["v2a_wm", "rla_wm", "dino_wm", "diffusion", "vlm_only"], default="v2a_wm", help="World Model mode")
    
    # Diffusion world model arguments
    parser.add_argument("--vae-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--vae-subfolder", default="vae")
    parser.add_argument("--wm-model", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--wm-adapter-dir", default="verify2act/output/wm/best/unet_lora")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--wm-steps", type=int, default=30)
    parser.add_argument("--wm-image-guidance", type=float, default=2.8)
    parser.add_argument("--wm-text-guidance", type=float, default=7.5)
    parser.add_argument("--wm-seed", type=int, default=None)
    
    # Evaluation config
    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment")
    parser.add_argument("--output-dir", default="verify2act/output/eval", help="Base directory where evaluation logs are saved")
    parser.add_argument("--num-sequences", type=int, default=20, help="Number of evaluation sequences to run")

    args = parser.parse_args()

    # Apply GCP project ID before any VLM/Vertex AI calls are made.
    if args.gcp_project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.gcp_project
        logging.getLogger(__name__).info(
            "GCP project set to '%s' (via --gcp-project)", args.gcp_project
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    # Environment will be initialized automatically inside MCILLowLevelPolicy
    # as a wrapped CalvinEnvWrapper.

    logger.info("Loading VLM Planner...")
    planner = VLMPlanner.from_yaml(
        args.prompt_config,
        model=args.planner_model,
        max_tokens=args.planner_max_tokens,
        temperature=args.planner_temperature,
        warn_on_retry=not args.no_gemini_retry_warn,
        call_delay=args.planner_call_delay,
    )

    if args.wm_mode == "vlm_only":
        logger.info("VLM-Only mode selected. Bypassing World Model and Critic loading.")
        world_model = None
        critic = None
    elif args.wm_mode == "diffusion":
        logger.info("Loading Diffusion World Model (ReflectVLM)...")
        from verify2act.pipeline.world_model import DiffusionWorldModel
        
        def _dtype_from_name(name: str) -> torch.dtype:
            name = name.lower()
            if name == "fp16":
                return torch.float16
            if name == "bf16":
                return torch.bfloat16
            return torch.float32

        world_model = DiffusionWorldModel(
            pretrained_model=args.wm_model,
            adapter_dir=args.wm_adapter_dir,
            decoder_dir=args.wm_decoder_dir,
            vae_model=args.vae_model,
            vae_subfolder=args.vae_subfolder,
            device=args.device,
            torch_dtype=_dtype_from_name(args.dtype),
            num_inference_steps=args.wm_steps,
            image_guidance_scale=args.wm_image_guidance,
            guidance_scale=args.wm_text_guidance,
            seed=args.wm_seed,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        logger.info("ReflectVLM (Diffusion mode) selected. Bypassing Critic loading.")
        critic = None
    elif args.wm_mode == "rla_wm":
        logger.info("Loading RLA World Model...")
        from verify2act.pipeline.world_model import RLAWorldModel
        world_model = RLAWorldModel(
            device=args.device,
            dynamics_weights_path=args.latent_wm_ckpt,
            encoder_ckpt=args.encoder_ckpt,
            history_len=args.history_len,
            token_dim=args.token_dim,
            num_latent_tokens=args.num_latent_tokens,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elif args.wm_mode == "dino_wm":
        logger.info("Loading DINO World Model...")
        from verify2act.pipeline.world_model import DINOWorldModel
        world_model = DINOWorldModel(
            device=args.device,
            dynamics_weights_path=args.latent_wm_ckpt,
            history_len=args.history_len,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        logger.info("Loading Latent World Model...")
        world_model = LatentWorldModel(
            device=args.device,
            dynamics_weights_path=args.latent_wm_ckpt,
            encoder_ckpt=args.encoder_ckpt,
            history_len=args.history_len,
            token_dim=args.token_dim,
            num_latent_tokens=args.num_latent_tokens,
            action_conditioning=args.action_conditioning,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.wm_mode not in ["vlm_only", "diffusion"]:
        logger.info("Loading Critic...")
        critic = _build_critic(args, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Note: verify2act's latent_wm has its own FeatureDecoder class
    # but we only load feature decoder if the directories are provided.
    decoder = None
    if args.wm_decoder_dir and args.wm_mode not in ["vlm_only", "diffusion"]:
        from verify2act.latent_wm.decoder import FeatureDecoder
        logger.info("Loading FeatureDecoder for visual reflection...")
        decoder = FeatureDecoder(dino_channels=1024).to(device)
        decoder.eval()
        dec_path = Path(args.wm_decoder_dir) / "latent_decoder_best.pt"
        if dec_path.exists():
            ckpt = torch.load(dec_path, map_location=device)
            # Unwrap checkpoint wrapper keys, matching the logic in visualize_wm.py.
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            # If keys don't have "decoder." prefix, add it (checkpoint was saved from inner decoder)
            if "decoder.input_proj.0.weight" not in state_dict and "input_proj.0.weight" in state_dict:
                state_dict = {f"decoder.{k}": v for k, v in state_dict.items()}
            decoder.load_state_dict(state_dict)
        else:
            logger.warning(f"DECODER CHECKPOINT NOT FOUND AT {dec_path}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Set up the V2A CALVIN Agent
    logger.info("Setting up Verify2Act CALVIN Agent...")
    agent = Verify2ActCalvinAgent(
        vlm_planner=planner,
        world_model=world_model,
        critic=critic,
        device=device,
        train_folder=args.low_level_policy_ckpt if args.low_level_policy_ckpt else args.train_folder,
        dataset_path=args.dataset_path,
        theta_c=args.theta_c,
        theta_p=args.theta_p,
        max_replans=args.max_replans,
        extra_dataset_path=args.full_dataset_path,
        low_level_policy_type=args.low_level_policy,
        critic_unc_threshold=args.critic_unc_threshold,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Set external attributes on agent if needed (e.g. decoder)
    agent.decoder = decoder

    # Retrieve the wrapped environment from the low-level policy wrapper
    env = agent.low_level_policy.env

    # ── Evaluation ──────────────────────────────────────────────────────────
    eval_dir = Path(args.output_dir) / args.wm_mode / "calvin"
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting CALVIN evaluation: mode=%s, sequences=%d, output_dir=%s",
        args.wm_mode, args.num_sequences, eval_dir,
    )

    results, traces = _evaluate_calvin(
        agent=agent,
        env=env,
        eval_log_dir=eval_dir,
        debug=args.debug,
        num_sequences=args.num_sequences,
    )

    # ── Aggregate metrics (mirrors nut-assembly eval_summary) ────────────
    import json
    num_seqs  = len(results)
    successes = sum(1 for r in results if r == 5)
    total_subtasks = sum(results)
    max_subtasks   = num_seqs * 5

    # VLM / critic roll-ups
    total_vlm_calls    = sum(t.total_vlm_calls    for t in traces)
    total_reflections  = sum(t.total_reflections  for t in traces)
    total_critic_acc   = sum(t.critic_accepts     for t in traces)
    total_critic_rej   = sum(t.critic_rejects     for t in traces)
    total_critic_tp    = sum(t.critic_tp          for t in traces)
    total_critic_fp    = sum(t.critic_fp          for t in traces)

    # Critic precision: TP and FP are credited at subtask granularity (one per subtask).
    # Use (TP+FP) as denominator — the number of subtasks that had a pending critic accept
    # and were subsequently resolved. total_critic_acc is raw accept-events and is a
    # misleading denominator since multiple accepts can occur within one subtask.
    critic_classified  = total_critic_tp + total_critic_fp   # subtasks with resolved accept
    critic_precision   = total_critic_tp / critic_classified if critic_classified else None
    critic_fp_rate     = total_critic_fp / critic_classified if critic_classified else None
    # Raw event-level reject rate (still meaningful as a throughput metric)
    critic_reject_rate = (
        total_critic_rej / (total_critic_acc + total_critic_rej)
        if (total_critic_acc + total_critic_rej) else None
    )
    vlm_calls_per_subtask = total_vlm_calls / total_subtasks if total_subtasks else None

    # ── Chain success rates (the standard CALVIN metric) ────────────────
    # SR(k) = fraction of sequences where at least k subtasks were completed in a row.
    # This is the primary metric reported in CALVIN benchmark papers.
    chain_sr = count_success(results)  # [SR1, SR2, SR3, SR4, SR5]

    summary = {
        "wm_mode": args.wm_mode,
        "num_sequences": num_seqs,
        # ── Primary CALVIN chain success rates ──
        # SR(k): % of sequences completing at least k consecutive subtasks.
        # Report all five in your paper table.
        "chain_success_rates": {
            f"SR_{k+1}": round(v, 4) for k, v in enumerate(chain_sr)
        },
        "avg_subtasks_completed": round(total_subtasks / num_seqs, 4) if num_seqs else 0.0,
        # ── Derived / legacy metrics ──
        "success_rate_5of5": successes / num_seqs if num_seqs else 0.0,
        "subtask_completion_rate": total_subtasks / max_subtasks if max_subtasks else 0.0,
        # Planning efficiency
        "total_vlm_calls": total_vlm_calls,
        "total_reflections": total_reflections,
        "vlm_calls_per_subtask_completed": vlm_calls_per_subtask,
        # Critic quality
        "critic_total_accepts": total_critic_acc,
        "critic_total_rejects": total_critic_rej,
        "critic_tp": total_critic_tp,
        "critic_fp": total_critic_fp,
        "critic_precision": critic_precision,
        "critic_fp_rate_of_accepts": critic_fp_rate,
        "critic_reject_rate": critic_reject_rate,
        # Per-sequence breakdown
        "per_sequence": [
            {
                "sequence": i,
                "success": r == 5,
                "subtasks_completed": r,
                "total_target_subtasks": 5,
                "total_vlm_calls": t.total_vlm_calls,
                "total_reflections": t.total_reflections,
                "critic_accepts": t.critic_accepts,
                "critic_rejects": t.critic_rejects,
                "critic_tp": t.critic_tp,
                "critic_fp": t.critic_fp,
            }
            for i, (r, t) in enumerate(zip(results, traces))
        ],
    }

    summary_path = eval_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Eval summary saved to %s", summary_path)

    _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
    _flt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    print("\n" + "=" * 60)
    print(f"  EVALUATION COMPLETE — {args.wm_mode} (CALVIN)")
    print("=" * 60)
    print(f"  Sequences evaluated:  {num_seqs}")
    print(f"")
    print(f"  --- Chain Success Rates (primary CALVIN metric) ---")
    print(f"  {'Tasks':>5}  {'SR':>8}   (# sequences completing at least N tasks)")
    for k, v in enumerate(chain_sr):
        n_success = sum(1 for r in results if r >= k + 1)
        print(f"  {k+1:>5}/5  {_pct(v):>8}   ({n_success}/{num_seqs})")
    print(f"")
    print(f"  Avg subtasks completed:  {total_subtasks / num_seqs if num_seqs else 0.0:.2f} / 5.0")
    print(f"  --- Planning Efficiency ---")
    print(f"  Total VLM Calls:               {total_vlm_calls}")
    print(f"  Total Reflections:             {total_reflections}")
    print(f"  VLM Calls / Subtask Completed: {_flt(vlm_calls_per_subtask)}")
    if total_critic_acc + total_critic_rej > 0:
        print(f"  --- Critic Quality (online) ---")
        print(f"  Critic Accepts:                {total_critic_acc}")
        print(f"  Critic Rejects:                {total_critic_rej}")
        print(f"  Critic Reject Rate:            {_pct(critic_reject_rate)}")
        print(f"  Critic Precision (TP/classified): {_pct(critic_precision)}  [TP={total_critic_tp} FP={total_critic_fp} of {critic_classified} subtasks]")
        print(f"  Critic FP Rate   (FP/classified): {_pct(critic_fp_rate)}")
    print(f"  Results:                       {summary_path}")
    print("=" * 60 + "\n")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
