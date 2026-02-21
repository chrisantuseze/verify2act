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
    xvfb-run -a python demo_phase3.py --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_99.pth

    xvfb-run -a python demo_phase3.py \
    --task Stack3 \
    --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_99.pth

    xvfb-run -a python demo_phase3.py \
    --task ClutteredNutAssembly \
    --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_99.pth \
    --num-round 2 --num-square 1 --initial-stacking-prob 0.5 \
    --nut-type-mode roundnut
"""

# import os
# # Set rendering backend before robosuite imports
# if 'MUJOCO_GL' not in os.environ:
#     os.environ['MUJOCO_GL'] = 'glx'

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

import parse_util

def create_env_factory(env_name: str, args):
    """
    Create environment factory function based on environment name.
    
    Args:
        env_name: Name of the environment
        args: Command-line arguments
        
    Returns:
        Factory function that creates environment instance
    """
    if env_name == "ClutteredNutAssembly":
        sys.path.insert(0, str(current_dir.parent))
        from run_cluttered_nutassembly import create_environment
        
        def env_factory():
            return create_environment(
                env_name="ClutteredNutAssembly",
                num_round_nuts=args.num_round,
                num_square_nuts=args.num_square,
                initial_stacking_prob=args.initial_stacking_prob,
                nut_type_mode=args.nut_type_mode,
                horizon=1000
            )
        return env_factory
        
    elif env_name in ["Stack", "Stack3", "Stack4"]:
        sys.path.insert(0, str(current_dir.parent))
        from run_stack import create_environment
        
        def env_factory():
            return create_environment(env_name)
        return env_factory
        
    elif env_name == "PickPlace":
        sys.path.insert(0, str(current_dir.parent))
        from run_pickplace import create_environment
        
        def env_factory():
            return create_environment("PickPlaceCan")
        return env_factory
        
    else:
        raise ValueError(f"Unsupported environment: {env_name}")


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
    env_factory = create_env_factory(task, args)
    env = env_factory()
    print("  ✓ Environment created")
    
    # Create controller
    print("\n[2/3] Initializing closed-loop controller...")
    
    # Determine task type for predicate filtering
    if task.lower() in ["clutterednutassembly", "nutassembly"]:
        task_type = "assembly"
    elif task.lower() in ["stack", "stack3", "stack4"]:
        task_type = "stacking"
    elif task.lower() == "pickplace":
        task_type = "pickplace"
    else:
        task_type = "all"  # No filtering
    
    controller = ClosedLoopController(
        args,
        env=env,
        checkpoint_path=checkpoint_path,
        lookahead_depth=args.lookahead_depth,
        enable_collision_checking=True,
        predicate_threshold=args.predicate_threshold,
        enable_trajectory_tracking=args.enable_trajectory_tracking,
        delta_forward=args.delta_forward,
        latent_forward=args.latent_forward,
        verbose=True,
        task_type=task_type
    )
    print("  ✓ Controller initialized")
    
    # Define task
    if task.lower() == "stack3":
        task_description = "Stack all cubes on top of each other"
        initial_predicates = None  # Auto-detect
    elif task.lower() == "pickplace":
        task_description = "Put all objects in the bin"
        initial_predicates = None  # Auto-detect
    elif task.lower() == "clutterednutassembly":
        task_description = "Assemble the cluttered nuts"
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
    # Create a minimal args object for env_factory
    args_obj = type('Args', (), {})()  
    env_factory = create_env_factory(task, args_obj)
    env = env_factory()
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
    args=None
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
    args_obj = type('Args', (), {})() if not args else args
    env_factory = create_env_factory("PickPlace", args_obj)
    env = env_factory()
    
    # Get predicate threshold from args if available
    predicate_threshold = getattr(args, 'predicate_threshold', 0.3) if args else 0.3
    
    # Create controller with aggressive replanning
    controller = ClosedLoopController(
        args if args else type('obj', (object,), {
            'model_config_path': '../../Points2Plans/LLM/configs/models/pretrained/generative/gpt_4_cot.yaml',
            'prompt_config_path': 'configs/prompts/tasks/pickplace_task.yaml'
        })(),
        env=env,
        checkpoint_path=checkpoint_path,
        predicate_threshold=predicate_threshold,
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
    parser = parse_util.get_parser(desc="Phase 3 Demo: Full Closed-Loop Controller with Robosuite Environment")
    args = parser.parse_args()


    # ''' @Chris: Uncomment
    
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
            success, stats = demo_failure_recovery(str(checkpoint_path), args)
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

    # '''

    '''
    print("\n" + "=" * 80)
    success, stats = demo_single_episode(
        args,
        task=args.task,
        checkpoint_path=None,
        render=args.render,
        max_primitives=args.max_primitives
    )
    print(f"\nEpisode complete: {'SUCCESS' if success else 'FAILED'}")
    '''


if __name__ == "__main__":
    main()
