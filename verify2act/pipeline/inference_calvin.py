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

import torch

# Ensure verify2act, calvin_agent, and calvin_env are in PYTHONPATH
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
calvin_models = repo_root / "calvin/calvin_models"
if calvin_models.exists() and str(calvin_models) not in sys.path:
    sys.path.insert(0, str(calvin_models))
calvin_env_path = repo_root / "calvin/calvin_env"
if calvin_env_path.exists() and str(calvin_env_path) not in sys.path:
    sys.path.insert(0, str(calvin_env_path))

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
    if unexpected:
        logger.warning("Checkpoint had unexpected keys: %s", unexpected)
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
    
    # Evaluation config
    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment")
    parser.add_argument("--eval-log-dir", default="verify2act/output/inference_run/v2a_wm/calvin", help="Directory where evaluation logs are saved")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    logger.info("Initializing CALVIN environment...")
    env = get_env(dataset_path=args.dataset_path, show_gui=args.debug)

    logger.info("Loading VLM Planner...")
    planner = VLMPlanner.from_yaml(
        args.prompt_config,
        model=args.planner_model,
        max_tokens=args.planner_max_tokens,
        temperature=args.planner_temperature,
    )

    logger.info("Loading Latent World Model...")
    world_model = LatentWorldModel(
        device=args.device,
        dynamics_weights_path=args.latent_wm_ckpt,
        encoder_ckpt=args.encoder_ckpt,
    )

    logger.info("Loading Critic...")
    critic = _build_critic(args, device)

    # Note: verify2act's latent_wm has its own FeatureDecoder class
    # but we only load feature decoder if the directories are provided.
    decoder = None
    if args.wm_decoder_dir:
        from verify2act.latent_wm.decoder import FeatureDecoder
        logger.info("Loading FeatureDecoder for visual reflection...")
        decoder = FeatureDecoder().to(device)
        decoder.eval()
        dec_path = Path(args.wm_decoder_dir) / "decoder.pt"
        if dec_path.exists():
            decoder.load_state_dict(torch.load(dec_path, map_location=device))
        else:
            logger.warning(f"Decoder checkpoint not found at {dec_path}")

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

    # Set external attributes on agent if needed (e.g. decoder)
    agent.decoder = decoder

    logger.info("Starting standard CALVIN evaluation sequence...")
    results = evaluate_policy(
        model=agent,
        env=env,
        epoch=0,
        eval_log_dir=args.eval_log_dir,
        debug=args.debug,
    )

    logger.info("Evaluation complete! Results: %s", results)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
