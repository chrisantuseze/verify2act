import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser(desc: str = "Phase 3 Demo: Full Integration") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--task",
        type=str,
        default="ClutteredNutAssembly",
        choices=["Stack", "Stack3", "Stack4", "PickPlace", "ClutteredNutAssembly"],
        help="Task to run"
    )
    # ClutteredNutAssembly specific arguments
    parser.add_argument(
        "--num-round",
        type=int,
        default=6,
        help="Number of round nuts (ClutteredNutAssembly only)"
    )
    parser.add_argument(
        "--num-square",
        type=int,
        default=2,
        help="Number of square nuts (ClutteredNutAssembly only)"
    )
    parser.add_argument(
        "--initial-stacking-prob",
        type=float,
        default=0.6,
        help="Probability of initial nut stacking (ClutteredNutAssembly only)"
    )
    parser.add_argument(
        "--nut-type-mode",
        type=str,
        default="roundnut",
        choices=["roundnut", "squarenut"],
        help="Which nut type to target (ClutteredNutAssembly only)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../../Points2Plans/ckpt/checkpoint/cp_1.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="../../Points2Plans/LLM/configs/models/pretrained/generative/gpt_4_cot.yaml",
        help="Path to model configuration"
    )
    parser.add_argument(
        "--prompt-config-path",
        type=str,
        default="configs/prompts/tasks/stack_task.yaml",
        help="Path to prompt configuration"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable on-screen rendering (requires display)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch evaluation instead of single episode"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of trials for batch evaluation"
    )
    parser.add_argument(
        "--max-primitives",
        type=int,
        default=5,
        help="Maximum primitives per episode"
    )
    parser.add_argument(
        "--max-replans-per-primitive",
        type=int,
        default=3,
        help="Maximum replans per primitive execution"
    )
    parser.add_argument(
        "--goal-threshold",
        type=float,
        default=0.8,
        help="Threshold for goal achievement (predicate difference)"
    )
    parser.add_argument(
        "--num-planning-samples",
        type=int,
        default=50,
        help="Number of action samples for rejection sampling"
    )
    parser.add_argument(
        "--delta-forward",
        type=str2bool, 
        default=False, #Trained with True
        help="Use delta forward prediction in dynamics model"
    )
    parser.add_argument(
        "--latent-forward",
        type=str2bool, 
        default=True, # Trained with False
        help="Use latent space forward prediction in dynamics model"
    )
    parser.add_argument(
        "--demo-recovery",
        type=str2bool, 
        default=False,
        help="Run failure recovery demo"
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Number of primitives to simulate ahead (1=greedy, 2-3=multi-step)"
    )
    parser.add_argument(
        "--predicate-threshold",
        type=float,
        default=0.3,
        help="Threshold for predicate matching (default 0.3, use lower for undertrained models)"
    )
    parser.add_argument(
        "--enable-trajectory-tracking",
        type=str2bool,
        default=False,
        help="Enable trajectory tracking during planning"
    )
    parser.add_argument(
        "--collect-data",
        type=str2bool,
        default=True,
        help="Enable critic data collection"
    )

    return parser

