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
from calvin_agent.evaluation.evaluate_policy import evaluate_policy
from verify2act.pipeline.calvin_agent_wrapper import Verify2ActCalvinAgent
from verify2act.pipeline.planner import VLMPlanner
from verify2act.pipeline.world_model import LatentWorldModel
from verify2act.critic.model import DINOv2DualHeadCritic

logger = logging.getLogger(__name__)


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
    parser.add_argument("--dataset-path", default="calvin/dataset/task_ABC_D", help="CALVIN dataset dir")
    
    # Planner configuration
    parser.add_argument("--prompt-config", default="verify2act/configs/prompts/planner.yaml")
    parser.add_argument("--planner-model", default="gpt-4o")
    parser.add_argument("--planner-max-tokens", type=int, default=512)
    parser.add_argument("--planner-temperature", type=float, default=0.2)
    
    # Hyperparameters
    parser.add_argument("--theta-c", type=float, default=0.7, help="Consistency threshold")
    parser.add_argument("--max-replans", type=int, default=2, help="Max replanning rounds per failure")
    parser.add_argument("--history-len", type=int, default=3, help="Number of historical frames for world model context")
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
    parser.add_argument("--output-dir", default="verify2act/output/inference_run", help="Base directory where evaluation logs are saved")

    args = parser.parse_args()

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
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
            state_dict = torch.load(dec_path, map_location=device)
            # If keys don't have "decoder." prefix, add it (checkpoint was saved from inner decoder)
            if "decoder.input_proj.0.weight" not in state_dict and "input_proj.0.weight" in state_dict:
                state_dict = {f"decoder.{k}": v for k, v in state_dict.items()}
            decoder.load_state_dict(state_dict)
        else:
            logger.warning(f"Decoder checkpoint not found at {dec_path}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Set up the V2A CALVIN Agent
    logger.info("Setting up Verify2Act CALVIN Agent...")
    agent = Verify2ActCalvinAgent(
        vlm_planner=planner,
        world_model=world_model,
        critic=critic,
        device=device,
        train_folder=args.train_folder,
        dataset_path=args.dataset_path,
        theta_c=args.theta_c,
        max_replans=args.max_replans,
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
        "Starting standard CALVIN evaluation sequence: mode=%s, output_dir=%s",
        args.wm_mode, eval_dir
    )
    
    results = evaluate_policy(
        model=agent,
        env=env,
        epoch=0,
        eval_log_dir=str(eval_dir),
        debug=args.debug,
    )

    # ── Aggregate metrics ────────────────────────────────────────────────
    num_seqs = len(results)
    successes = sum(1 for r in results if r == 5)
    total_subtasks = sum(results)
    max_subtasks = num_seqs * 5

    summary = {
        "wm_mode": args.wm_mode,
        "num_sequences": num_seqs,
        "success_rate": successes / num_seqs if num_seqs else 0.0,
        "subtask_completion_rate": total_subtasks / max_subtasks if max_subtasks else 0.0,
        "avg_subtasks_completed": total_subtasks / num_seqs if num_seqs else 0.0,
        "per_sequence": [
            {
                "sequence": i,
                "success": r == 5,
                "subtasks_completed": r,
                "total_target_subtasks": 5,
            }
            for i, r in enumerate(results)
        ],
    }

    summary_path = eval_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        import json
        json.dump(summary, f, indent=2)
    logger.info("Eval summary saved to %s", summary_path)

    # Print final report
    print("\n" + "=" * 60)
    print(f"  EVALUATION COMPLETE — {args.wm_mode} (CALVIN)")
    print("=" * 60)
    print(f"  Sequences:                 {num_seqs}")
    print(f"  Success Rate (5/5):        {summary['success_rate']:.1%} ({successes}/{num_seqs})")
    print(f"  Subtask Completion Rate:   {summary['subtask_completion_rate']:.1%} ({total_subtasks}/{max_subtasks})")
    print(f"  Avg Subtasks Completed:    {summary['avg_subtasks_completed']:.2f}")
    print(f"  Results:                   {summary_path}")
    print("=" * 60 + "\n")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
