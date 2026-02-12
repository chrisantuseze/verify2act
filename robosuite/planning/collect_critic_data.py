"""
Critic Data Collection Script

Runs multiple episodes to collect training data for the critic model.
Automatically labels successful and failed trajectories.

Usage:
    # Collect 50 episodes on Stack3 task
    xvfb-run -a python collect_critic_data.py --num-episodes 50 --task Stack3
    
    # Collect with specific checkpoint
    xvfb-run -a python collect_critic_data.py \
        --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_99.pth \
        --num-episodes 100 \
        --save-interval 25
    
    # PickPlace task
    xvfb-run -a python collect_critic_data.py --task PickPlace --num-episodes 50

    xvfb-run -a python collect_critic_data.py \
    --task ClutteredNutAssembly \
    --checkpoint ../../Points2Plans/ckpt/checkpoint/cp_50.pth \
    --num-round 2 --num-square 1 --initial-stacking-prob 0.5 \
    --nut-type-mode roundnut --num-episodes 50 \
    --save-interval 1

"""

import os
# Set rendering backend before robosuite imports
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glx'

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from closed_loop_controller import ClosedLoopController
from dynamics_model_data_collector import DynamicsModelDataCollector


def create_env_factory(env_name: str, args=None):
    """
    Create environment factory function based on environment name.
    
    Args:
        env_name: Name of the environment
        args: Command-line arguments (optional)
        
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


def collect_data(
    args,
    checkpoint_path: str,
    num_episodes: int,
    save_dir: str,
    save_interval: int = 50,
    max_primitives: int = 20,
    verbose: bool = True,
):
    """
    Collect critic training data from multiple episodes.
    
    Args:
        args: Command-line arguments
        checkpoint_path: Path to trained dynamics model checkpoint
        num_episodes: Number of episodes to collect
        save_dir: Directory to save collected data
        save_interval: Save dataset every N episodes
        max_primitives: Max primitives per episode
        verbose: Whether to print progress
    
    Returns:
        collector: DataCollector with collected samples
    """
    print("\n" + "=" * 80)
    print("CRITIC DATA COLLECTION")
    print("=" * 80)
    print(f"Task:             {args.task}")
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Target episodes:  {num_episodes}")
    print(f"Save directory:   {save_dir}")
    print(f"Save interval:    {save_interval} episodes")
    print("=" * 80)
    
    # Create environment
    print(f"\n[1/3] Creating environment...")
    # Create minimal args object for env factory
    env_factory = create_env_factory(args.task, args)
    env = env_factory()
    print("  ✓ Environment created")
    
    # Create controller with data collection enabled
    print(f"\n[2/3] Initializing controller with data collector...")
    
    # Create args object for controller
    controller_args = type('Args', (), {
        'model_config_path': '../../Points2Plans/LLM/configs/models/pretrained/generative/gpt_4_cot.yaml',
        'prompt_config_path': 'configs/prompts/tasks/stack_task.yaml',
        'lookahead_depth': 2,
        'predicate_threshold': 0.3,
        'enable_trajectory_tracking': False,
        'num_planning_samples': 50,
    })()
    
    # Determine task type for predicate filtering
    if args.task.lower() in ["clutterednutassembly", "nutassembly"]:
        task_type = "assembly"
    elif args.task.lower() in ["stack", "stack3", "stack4"]:
        task_type = "stacking"
    elif args.task.lower() == "pickplace":
        task_type = "pickplace"
    else:
        task_type = "all"
    
    # Create controller
    controller = ClosedLoopController(
        controller_args,
        env=env,
        checkpoint_path=checkpoint_path,
        lookahead_depth=2,
        enable_collision_checking=True,
        predicate_threshold=0.3,
        enable_trajectory_tracking=False,
        delta_forward=True,
        latent_forward=False,
        verbose=False,  # Disable controller verbosity for cleaner output
        task_type=task_type
    )
    
    # Create data collector
    data_collector = DynamicsModelDataCollector(
        dynamics_planner=controller.dynamics_planner,
        save_dir=save_dir,
        enable_hard_negatives=True,
        verbose=verbose,
    )
    
    # Attach collector to controller's planner
    controller.dynamics_planner.data_collector = data_collector
    
    print("  ✓ Controller and data collector initialized")
    
    # Define task descriptions
    task_descriptions = {
        "Stack3": "Stack all cubes on top of each other",
        "Stack4": "Stack all cubes on top of each other",
        "PickPlace": "Put all objects in the bin",
        "ClutteredNutAssembly": "Assemble the cluttered nuts",
    }
    task_description = task_descriptions.get(args.task, "Complete the task")
    
    # Run episodes
    print(f"\n[3/3] Collecting data from {args.num_episodes} episodes...")
    print("-" * 80)
    
    for episode_idx in range(args.num_episodes):
        print(f"\nEpisode {episode_idx + 1}/{args.num_episodes}")
        
        # Start episode in collector
        # Note: We'll need to integrate this with the controller's run_episode
        # For now, run episode and collect data through the planner's callback
        
        try:
            success, stats = controller.run_episode(
                task_description=task_description,
                max_primitives=args.max_primitives,
                initial_predicates=None
            )
            
            # Determine failure step if episode failed
            failure_step = None
            failure_type = "predicate"
            
            if not success:
                # Estimate failure step from stats
                failure_step = stats.get('num_primitives_executed', 0) - 1
                failure_step = max(0, failure_step)
                
                # Determine failure type from stats
                if stats.get('collision_detected', False):
                    failure_type = "feasibility"
                else:
                    failure_type = "predicate"
            
            print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
            print(f"  Primitives executed: {stats.get('num_primitives_executed', 0)}")
            
            # Save periodically
            if (episode_idx + 1) % args.save_interval == 0:
                print(f"\n  💾 Saving checkpoint at episode {episode_idx + 1}...")
                filename = f"critic_data_ep{episode_idx + 1}.pkl"
                data_collector.save_dataset(filename)
                print(f"  ✓ Checkpoint saved")
        
        except Exception as e:
            print(f"  ⚠ Episode failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final save
    print(f"\n💾 Saving final dataset...")
    data_collector.save_dataset("critic_data_final.pkl")
    
    # Print statistics
    data_collector.print_statistics()
    
    # Clean up
    env.close()
    
    return data_collector


def main():
    parser = argparse.ArgumentParser(description="Collect critic training data")
    parser.add_argument(
        "--task",
        type=str,
        default="ClutteredNutAssembly",
        choices=["Stack", "Stack3", "Stack4", "PickPlace", "ClutteredNutAssembly"],
        help="Task to collect data from"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../../Points2Plans/ckpt/checkpoint/cp_99.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes to collect"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./data/critic",
        help="Directory to save collected data"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=25,
        help="Save dataset every N episodes"
    )
    parser.add_argument(
        "--max-primitives",
        type=int,
        default=20,
        help="Maximum primitives per episode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    # ClutteredNutAssembly specific arguments
    parser.add_argument(
        '--num-round',
        type=int,
        default=6,
        help='Number of round nuts (ClutteredNutAssembly only)'
    )
    
    parser.add_argument(
        '--num-square',
        type=int,
        default=2,
        help='Number of square nuts (ClutteredNutAssembly only)'
    )
    
    parser.add_argument(
        '--initial-stacking-prob',
        type=float,
        default=0.6,
        help='Probability of initial nut stacking (ClutteredNutAssembly only)'
    )
    
    parser.add_argument(
        '--nut-type-mode',
        type=str,
        default='roundnut',
        choices=['roundnut', 'squarenut'],
        help='Which nut type to target (ClutteredNutAssembly only)'
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
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        collector = collect_data(
            args=args,
            checkpoint_path=str(checkpoint_path),
            num_episodes=args.num_episodes,
            save_dir=str(save_dir),
            save_interval=args.save_interval,
            max_primitives=args.max_primitives,
            verbose=args.verbose,
        )
        
        print("\n" + "=" * 80)
        print("DATA COLLECTION COMPLETE")
        print("=" * 80)
        print(f"Dataset saved to: {save_dir}")
        print(f"Ready for training!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
