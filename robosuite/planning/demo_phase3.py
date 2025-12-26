"""
Phase 3 Demo: Full Closed-Loop Controller with Robosuite Environment

This demonstrates the complete integration:
1. Real robosuite environment (Stack3 or PickPlace)
2. LLM task planning (goals + high-level plan)
3. Closed-loop planning (dynamics model with rejection sampling)
4. Primitive execution (actual robosuite control)
5. Automatic replanning on failures
6. Goal achievement checking

Usage:
    # Stack3 task (default)
    xvfb-run -a python demo_phase3.py
    
    # PickPlace task
    xvfb-run -a python demo_phase3.py --task pickplace
    
    # With visualization (requires display)
    python demo_phase3.py --render
    
    # Batch evaluation
    python demo_phase3.py --batch --num-trials 5
    
    # Custom checkpoint
    xvfb-run -a python demo_phase3.py --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_1.pth
"""

import os
# Set rendering backend before robosuite imports
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glx'

import argparse
import sys
import numpy as np
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from closed_loop_controller import ClosedLoopController, BatchController


def create_environment(task: str = "stack3", render: bool = False):
    """
    Create robosuite environment.
    
    Args:
        task: Task name ("stack3" or "pickplace")
        render: Whether to enable on-screen rendering
    
    Returns:
        Robosuite environment
    """
    # Task-specific parameters
    if task.lower() == "stack3":
        env_name = "Stack"
        horizon = 1000
        env_kwargs = {}
    elif task.lower() == "pickplace":
        env_name = "PickPlace"
        horizon = 1000
        env_kwargs = {
            'single_object_mode': 0,  # Multiple objects
            'object_type': "milk",     # Start with milk
        }
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create environment
    controller_config = load_composite_controller_config(controller="BASIC")
    env = suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=render,  # Disable on-screen rendering
        has_offscreen_renderer=True,  # Enable offscreen for point cloud generation
        use_camera_obs=True,  # Enable camera observations
        use_object_obs=True,
        camera_names=["frontview", "agentview"],
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,               # Enable depth
        control_freq=20,
        horizon=horizon,
        ignore_done=True,
        reward_shaping=True,
    )
    
    return env


def demo_single_episode(
    args,
    task: str = "stack3",
    checkpoint_path: str = "../../Points2Plans/ckpt/checkpoint/cp_1.pth",
    render: bool = False,
    max_primitives: int = 20
):
    """
    Run a single episode with closed-loop controller.
    
    Args:
        task: Task name
        checkpoint_path: Path to trained model checkpoint
        render: Whether to render environment
        max_primitives: Maximum primitives to execute
    
    Returns:
        success: Whether task was completed
        stats: Episode statistics
    """
    print("\n" + "=" * 80)
    print("PHASE 3: FULL CLOSED-LOOP INTEGRATION")
    print("=" * 80)
    
    # Create environment
    print(f"\n[1/3] Creating {task} environment...")
    env = create_environment(task, render)
    print("  ✓ Environment created")
    
    # Create controller
    print("\n[2/3] Initializing closed-loop controller...")
    
    # Get lookahead depth from args if available
    lookahead_depth = getattr(args, 'lookahead_depth', 2)  # Default to 2-step
    
    controller = ClosedLoopController(
        args,
        env=env,
        checkpoint_path=checkpoint_path,
        num_planning_samples=50,
        goal_threshold=0.2,
        max_replans_per_primitive=3,
        lookahead_depth=lookahead_depth,
        enable_collision_checking=True,
        verbose=True
    )
    print("  ✓ Controller initialized")
    
    # Define task
    if task.lower() == "stack3":
        task_description = "Stack all cubes on top of each other"
        initial_predicates = None  # Auto-detect
    elif task.lower() == "pickplace":
        task_description = "Put all objects in the bin"
        initial_predicates = None  # Auto-detect
    else:
        task_description = "Complete the task"
        initial_predicates = None
    
    # Run episode
    print("\n[3/3] Running episode...")
    success, stats = controller.run_episode(
        task_description=task_description,
        max_primitives=max_primitives,
        initial_predicates=initial_predicates
    )
    
    # Clean up
    env.close()
    
    return success, stats


def demo_batch_evaluation(
    task: str = "stack3",
    checkpoint_path: str = "../../Points2Plans/ckpt/checkpoint/cp_1.pth",
    num_trials: int = 5,
    max_primitives: int = 20
):
    """
    Run batch evaluation with multiple trials.
    
    Args:
        task: Task name
        checkpoint_path: Path to trained model checkpoint
        num_trials: Number of trials per task
        max_primitives: Max primitives per episode
    
    Returns:
        results: Aggregated results dictionary
    """
    print("\n" + "=" * 80)
    print("PHASE 3: BATCH EVALUATION")
    print("=" * 80)
    
    # Create environment
    print(f"\nCreating {task} environment...")
    env = create_environment(task, render=False)
    print("✓ Environment created")
    
    # Create batch controller
    batch_controller = BatchController(
        env=env,
        checkpoint_path=checkpoint_path,
        num_planning_samples=50,
        goal_threshold=0.2,
        max_replans_per_primitive=3,
        verbose=False  # Disable verbose for batch
    )
    
    # Define tasks
    if task.lower() == "stack3":
        tasks = [
            ("Stack all cubes on top of each other", None),
        ]
    elif task.lower() == "pickplace":
        tasks = [
            ("Put all objects in the bin", None),
        ]
    else:
        tasks = [("Complete the task", None)]
    
    # Run batch
    results = batch_controller.run_batch(
        tasks=tasks,
        num_trials_per_task=num_trials,
        max_primitives=max_primitives
    )
    
    # Clean up
    env.close()
    
    return results


def demo_failure_recovery(
    checkpoint_path: str = "../../Points2Plans/ckpt/checkpoint/cp_1.pth",
):
    """
    Demonstrate failure recovery with intentional errors.
    
    Shows how the controller handles:
    - Primitive execution failures
    - Replanning on failure
    - Recovery strategies
    """
    print("\n" + "=" * 80)
    print("PHASE 3: FAILURE RECOVERY DEMO")
    print("=" * 80)
    
    # Create environment
    env = create_environment("pickplace", render=False)
    
    # Create controller with aggressive replanning
    controller = ClosedLoopController(
        env=env,
        checkpoint_path=checkpoint_path,
        num_planning_samples=20,  # Fewer samples = higher failure rate
        goal_threshold=0.2,
        max_replans_per_primitive=5,  # More replans
        verbose=True
    )
    
    # Run challenging task
    success, stats = controller.run_episode(
        task_description="Put all objects in the bin",
        max_primitives=30,  # Allow more attempts
        initial_predicates=None
    )
    
    # Analyze failure recovery
    print("\n" + "=" * 80)
    print("FAILURE RECOVERY ANALYSIS")
    print("=" * 80)
    print(f"Total primitives executed: {stats['num_primitives_executed']}")
    print(f"Primitives failed: {stats['num_primitives_failed']}")
    print(f"Replans triggered: {stats['num_replans']}")
    
    if stats['num_primitives_failed'] > 0:
        recovery_rate = 1 - (stats['num_primitives_failed'] / stats['num_primitives_executed'])
        print(f"Recovery rate: {recovery_rate:.1%}")
    
    print("=" * 80)
    
    env.close()
    
    return success, stats


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Demo: Full Integration")
    parser.add_argument(
        "--task",
        type=str,
        default="stack3",
        choices=["stack3", "pickplace"],
        help="Task to run"
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
        default=1,#20,
        help="Maximum primitives per episode"
    )
    parser.add_argument(
        "--demo-recovery",
        action="store_true",
        help="Run failure recovery demo"
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Number of primitives to simulate ahead (1=greedy, 2-3=multi-step)"
    )
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try relative to script location
        checkpoint_path = Path(__file__).parent.parent.parent / args.checkpoint
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            return
    
    try:
        if args.demo_recovery:
            # Failure recovery demo
            success, stats = demo_failure_recovery(str(checkpoint_path))
        elif args.batch:
            # Batch evaluation
            results = demo_batch_evaluation(
                task=args.task,
                checkpoint_path=str(checkpoint_path),
                num_trials=args.num_trials,
                max_primitives=args.max_primitives
            )
            print(f"\nBatch evaluation complete!")
            print(f"Success rate: {results['success_rate']:.1%}")
        else:
            # Single episode
            success, stats = demo_single_episode(
                args,
                task=args.task,
                checkpoint_path=str(checkpoint_path),
                render=args.render,
                max_primitives=args.max_primitives
            )
            print(f"\nEpisode complete: {'SUCCESS' if success else 'FAILED'}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
