"""
Batch Data Collection Script for Points2Plans Dataset

Integrates with HeuristicStackPolicy from run_stack.py to automatically
collect multiple episodes with progress tracking and error recovery.

xvfb-run -a python data_capture/batch_collect.py \
    --env Stack3 \
    --num-episodes 1 \
    --output-dir data_capture/dataset/stack_v1

xvfb-run -a python data_capture/batch_collect.py \
    --env ClutteredNutAssembly \
    --max-timesteps 3000 --output-dir data_capture/dataset/nut_assembly \
    --num-round 2 --num-square 1 --initial-stacking-prob 0.5 \
    --nut-type-mode roundnut --num-episodes 100 --seed 42

xvfb-run -a python data_capture/batch_collect.py \
    --env Stack3 \
    --max-timesteps 1000 --output-dir data_capture/dataset/stack3 \
    --num-episodes 100 --seed 42

Phase 4: Batch Collection ✓
"""

# import os
# # Set rendering backend before robosuite imports
# if 'MUJOCO_GL' not in os.environ:
#     os.environ['MUJOCO_GL'] = 'glx'

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import json

# Add parent directory to path to import run_stack
sys.path.append(str(Path(__file__).parent.parent))

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from episode_recorder import EpisodeRecorder
from policy_wrappers import get_policy_factory

class BatchCollector:
    """
    Automated batch collection of episodes using heuristic policy.
    
    Features:
    - Multiple episode collection with progress tracking
    - Automatic error recovery and retry
    - Dataset organization and metadata
    - Success rate statistics
    - Resumable collection sessions
    """
    
    def __init__(self, 
                 args,
                 env_factory: Callable,
                 policy_factory: Callable,
                 env_name: str = "Stack",
                 output_dir: str = "./data_capture/dataset",
                 camera_names: Optional[List[str]] = None,
                 num_points: int = 128,
                 voxel_size: float = 0.005,
                 data_collection_mode: bool = True):
        """
        Initialize batch collector.
        
        Args:
            args: Parsed command-line arguments
            env_factory: Callable that creates and returns environment instance
            policy_factory: Callable that takes env and returns policy instance
            env_name: Environment name (for logging and metadata)
            output_dir: Root directory for dataset
            camera_names: Camera names for point cloud capture
            num_points: Points per object point cloud
            voxel_size: Voxel size for downsampling
            data_collection_mode: If True, disable policy retries for clean trajectories
        """
        self.args = args
        self.env_factory = env_factory
        self.policy_factory = policy_factory
        self.env_name = env_name
        self.output_dir = Path(output_dir)
        self.camera_names = camera_names or ["sideview", "frontview", "agentview", "robot0_eye_in_hand"]
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.data_collection_mode = data_collection_mode
        
        # Statistics
        self.stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'failed_episodes': 0,
            'total_timesteps': 0,
            'start_time': None,
            'end_time': None,
            'episode_durations': [],
        }
        
        # Error tracking
        self.error_log = []
        
        # Setup directories
        self._setup_directories()
        
        print(f"\n{'='*60}")
        print(f"Batch Collector Initialized")
        print(f"{'='*60}")
        print(f"Environment: {env_name}")
        print(f"Output: {self.output_dir}")
        print(f"Cameras: {self.camera_names}")
        print(f"Points/object: {num_points}")
        print(f"{'='*60}\n")
    
    def _setup_directories(self):
        """Create directory structure for dataset."""
        self.episodes_dir = self.output_dir / "episodes"
        self.metadata_dir = self.output_dir / "metadata"
        self.logs_dir = self.output_dir / "logs"
        
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Dataset structure:")
        print(f"  Episodes: {self.episodes_dir}")
        print(f"  Metadata: {self.metadata_dir}")
        print(f"  Logs: {self.logs_dir}")

    def _create_env(self): #for linux/os_mesa
        controller_config = load_composite_controller_config(controller="BASIC")
        env = suite.make(
            env_name=self.env_name,
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=True,  # Disable on-screen rendering
            has_offscreen_renderer=True,  # Enable offscreen for point cloud generation
            use_camera_obs=True,  # Enable camera observations
            use_object_obs=True,
            camera_names=self.camera_names,
            camera_heights=256,
            camera_widths=256,
            camera_depths=True,               # Enable depth
            control_freq=20,
            horizon=1000,
            ignore_done=True,
            reward_shaping=True,
        )
        return env
    
    def collect(self, 
                num_episodes: int,
                max_timesteps: int = 1000,
                max_retries: int = 3,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Collect multiple episodes using heuristic policy.
        
        Args:
            num_episodes: Number of episodes to collect
            max_timesteps: Maximum timesteps per episode
            max_retries: Maximum retry attempts for failed episodes
            verbose: Print detailed progress
            
        Returns:
            Collection statistics
        """
        self.stats['start_time'] = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"Starting Collection: {num_episodes} episodes")
        print(f"{'='*60}\n")
        
        # Create environment and recorder using factories
        env = self.env_factory()
        recorder = EpisodeRecorder(
            env, 
            camera_names=self.camera_names,
            num_points=self.num_points,
            voxel_size=self.voxel_size,
            key_timesteps_only=True  # Enable key timestep mode
        )
        
        for episode_idx in range(num_episodes):
            episode_success = False
            retry_count = 0
            
            while not episode_success and retry_count <= max_retries:
                try:
                    episode_start = time.time()
                    
                    # Collect single episode
                    episode_success = self._collect_episode(
                        env, 
                        recorder, 
                        episode_idx, 
                        max_timesteps,
                        verbose
                    )

                    print(f"   Episode {episode_idx} completed. Success: {episode_success}")
                    
                    if episode_success:
                        episode_duration = time.time() - episode_start
                        self.stats['episode_durations'].append(episode_duration)
                        self.stats['successful_episodes'] += 1
                        
                        if verbose:
                            self._print_episode_summary(episode_idx, episode_duration, True)
                    elif self.data_collection_mode:
                        print(f"   Episode {episode_idx} unsuccessful.")
                        self.stats['failed_episodes'] += 1
                        if verbose:
                            self._print_episode_summary(episode_idx, 0, False)
                        break  # No retries in data collection mode

                    else: # since its data collection script, this else block is even useless
                        print(f"   Episode {episode_idx} unsuccessful, but in inference mode allowing retries.")
                        retry_count += 1
                        error_msg = f"Episode {episode_idx} unsuccessful (attempt {retry_count}/{max_retries}): task not completed"
                        self.error_log.append(error_msg)
                        
                        if verbose:
                            print(f"⚠️  {error_msg}")
                        
                        if retry_count <= max_retries:
                            if verbose:
                                print(f"   Retrying...")
                            # Reset environment
                            env.reset()
                        else:
                            self.stats['failed_episodes'] += 1
                            if verbose:
                                self._print_episode_summary(episode_idx, 0, False)
                    
                except Exception as e:
                    print(f"⚠️  Error during episode {episode_idx} collection: {e}")
                    self.stats['failed_episodes'] += 1
                    break
            
            self.stats['total_episodes'] += 1
            
            # Print progress
            self._print_progress(episode_idx + 1, num_episodes)
    
        env.close()
        self.stats['end_time'] = datetime.now()
        
        # Save final statistics
        self._save_collection_metadata()
        
        return self.stats
    
    def _collect_episode(self, 
                         env, 
                         recorder: EpisodeRecorder,
                         episode_idx: int,
                         max_timesteps: int,
                         verbose: bool) -> bool:
        """Collect a single episode using heuristic policy.
        
        Returns:
            bool: True if episode was successful and saved, False otherwise
        """
        # Reset environment
        obs = env.reset()
        
        # Start recording
        recorder.start_episode()
        
        # Create policy using factory (pass data_collection_mode to disable retries)
        policy = self.policy_factory(env, data_collection_mode=self.data_collection_mode)
        policy.obs = obs
        
        # Run episode
        timestep = 0
        done, episode_successful = False, False
        
        while timestep < max_timesteps and not done:
            # Get action from policy
            action, policy_end = policy.step()
            
            # Execute action
            obs, reward, task_done, info = env.step(action)

            done = task_done or policy_end # success only if task_done is True
            
            # Record timestep
            recorder.record_step(action, obs, done=done)
            
            # Update policy observations
            policy.obs = obs
            
            timestep += 1
            
            # Check if episode complete (policy signals done)
            if done:
                episode_successful = task_done
                if verbose:
                    print(f"   Episode {episode_idx}: Task complete at timestep {timestep}")
        

        # End recording
        # data_dict, attrs_dict = recorder.end_episode() # we already end the episode in save_episode
        
        # Only save episode if it was successful
        if episode_successful:
            # Save episode (no need for save_subsampled since we're already in key timestep mode)
            saved_path = recorder.save_episode(str(self.episodes_dir))
            
            # Update statistics
            episode_stats = recorder.get_statistics()
            self.stats['total_timesteps'] += episode_stats['num_timesteps']
            
            # Save episode metadata
            self._save_episode_metadata(recorder.episode_counter, episode_stats, saved_path)
            
            if verbose:
                print(f"   ✓ Episode {episode_idx} saved successfully")
            return True
        else:
            if verbose:
                print(f"   ✗ Episode {episode_idx} not saved (unsuccessful)")
            return False
    
    def _save_episode_metadata(self, episode_idx: int, stats: Dict, filepath: str):
        """Save metadata for individual episode."""
        metadata = {
            'episode_idx': episode_idx,
            'filepath': filepath,
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
        }
        
        metadata_file = self.metadata_dir / f"episode_{episode_idx:05d}_meta.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_collection_metadata(self):
        """Save overall collection statistics."""
        total_duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        # Build env params dict
        env_params = {}
        if self.env_name == "ClutteredNutAssembly":
            env_params = {
                'num_round_nuts': self.args.num_round,
                'num_square_nuts': self.args.num_square,
                'initial_stacking_prob': self.args.initial_stacking_prob,
                'nut_type_mode': self.args.nut_type_mode,
            }
        
        metadata = {
            'env_name': self.env_name,
            'env_params': env_params,
            'seed': self.args.seed,
            'collection_date': self.stats['start_time'].isoformat(),
            'duration_seconds': total_duration,
            'total_episodes': self.stats['total_episodes'],
            'successful_episodes': self.stats['successful_episodes'],
            'failed_episodes': self.stats['failed_episodes'],
            'total_timesteps': self.stats['total_timesteps'],
            'avg_timesteps_per_episode': self.stats['total_timesteps'] / max(1, self.stats['successful_episodes']),
            'avg_duration_per_episode': np.mean(self.stats['episode_durations']) if self.stats['episode_durations'] else 0,
            'success_rate': self.stats['successful_episodes'] / max(1, self.stats['total_episodes']),
            'camera_names': self.camera_names,
            'num_points': self.num_points,
            'voxel_size': self.voxel_size,
            'error_log': self.error_log,
        }
        
        metadata_file = self.metadata_dir / "collection_summary.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved collection metadata: {metadata_file}")
    
    def _print_episode_summary(self, episode_idx: int, duration: float, success: bool):
        """Print summary for completed episode."""
        status = "✓" if success else "✗"
        if success:
            print(f"{status} Episode {episode_idx:05d}: {duration:.1f}s")
        else:
            print(f"{status} Episode {episode_idx:05d}: FAILED")
    
    def _print_progress(self, completed: int, total: int):
        """Print overall progress."""
        progress = completed / total * 100
        success_rate = self.stats['successful_episodes'] / max(1, completed) * 100
        
        print(f"\n{'─'*60}")
        print(f"Progress: {completed}/{total} ({progress:.1f}%) | Success: {success_rate:.1f}%")
        print(f"{'─'*60}\n")
    
    def print_final_summary(self):
        """Print final collection summary."""
        if self.stats['start_time'] is None:
            print("No collection data available.")
            return
        
        total_duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        success_rate = self.stats['successful_episodes'] / max(1, self.stats['total_episodes']) * 100
        avg_timesteps = self.stats['total_timesteps'] / max(1, self.stats['successful_episodes'])
        avg_duration = np.mean(self.stats['episode_durations']) if self.stats['episode_durations'] else 0
        
        print(f"\n{'='*60}")
        print(f"Collection Complete!")
        print(f"{'='*60}")
        print(f"Total Episodes: {self.stats['total_episodes']}")
        print(f"  ✓ Successful: {self.stats['successful_episodes']}")
        print(f"  ✗ Failed: {self.stats['failed_episodes']}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"")
        print(f"Total Timesteps: {self.stats['total_timesteps']}")
        print(f"Avg Timesteps/Episode: {avg_timesteps:.1f}")
        print(f"Avg Duration/Episode: {avg_duration:.1f}s")
        print(f"")
        print(f"Total Duration: {total_duration/60:.1f} minutes")
        print(f"Output Directory: {self.output_dir}")
        print(f"{'='*60}\n")


def create_env_and_policy_factories(args):
    """
    Create environment and policy factory functions based on CLI arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple of (env_factory, policy_factory, env_name)
    """
    env_name = args.env

    print(f"env_name: {env_name}")
    
    # Import appropriate modules
    if env_name == "ClutteredNutAssembly":
        # Import create_environment from run_cluttered_nutassembly
        from run_cluttered_nutassembly import create_environment
        from policy_wrappers import create_nut_assembly_policy
        
        # Create factory that passes env params
        def env_factory():
            return create_environment(
                env_name="ClutteredNutAssembly",
                num_round_nuts=args.num_round,
                num_square_nuts=args.num_square,
                initial_stacking_prob=args.initial_stacking_prob,
                nut_type_mode=args.nut_type_mode,
                horizon=args.max_timesteps
            )
        
        policy_factory = create_nut_assembly_policy
        
    elif env_name in ["Stack", "Stack3", "Stack4"]:
        from run_stack import create_environment
        from policy_wrappers import create_stack_policy
        
        def env_factory():
            return create_environment(env_name)
        
        policy_factory = create_stack_policy
        
    elif env_name == "PickPlace":
        from run_pickplace import create_environment
        from policy_wrappers import create_pickplace_policy
        
        def env_factory():
            return create_environment("PickPlaceCan")
        
        policy_factory = create_pickplace_policy
        
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    
    return env_factory, policy_factory, env_name


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main():
    parser = argparse.ArgumentParser(
        description="Batch collection of robosuite episodes for Points2Plans dataset"
    )
    
    parser.add_argument(
        '--env',
        type=str,
        default='Stack3',
        choices=['Stack', 'Stack3', 'Stack4', 'ClutteredNutAssembly', 'PickPlace'],
        help='Environment name'
    )
    
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=4,
        help='Number of episodes to collect'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./dataset',
        help='Output directory for dataset'
    )
    
    parser.add_argument(
        '--max-timesteps',
        type=int,
        default=1000,
        help='Maximum timesteps per episode'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retry attempts for failed episodes'
    )
    
    parser.add_argument(
        '--num-points',
        type=int,
        default=128,
        help='Number of points per object point cloud'
    )
    
    parser.add_argument(
        '--cameras',
        type=str,
        nargs='+',
        default=['sideview', 'frontview', 'agentview', 'robot0_eye_in_hand'],
        help='Camera names for point cloud capture'
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
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--inference-mode',
        type=str2bool, 
        default=False,
        help='Enable inference mode with policy retries (default is data collection mode with no retries)'
    )
    
    parser.add_argument(
        '--quiet',
        type=str2bool, 
        default=False,
        help='Suppress verbose output'
    )

    parser.add_argument(
        '--save-subsampled',
        type=str2bool, 
        default=True,
        help='Save a subsampled version of the episode with only key states'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create environment and policy factories
    env_factory, policy_factory, env_name = create_env_and_policy_factories(args)
    
    # Create collector
    collector = BatchCollector(
        args=args,
        env_factory=env_factory,
        policy_factory=policy_factory,
        env_name=env_name,
        output_dir=args.output_dir,
        camera_names=args.cameras,
        num_points=args.num_points,
        data_collection_mode=not args.inference_mode  # Default is data collection mode
    )
    
    # Collect episodes
    try:
        stats = collector.collect(
            num_episodes=args.num_episodes,
            max_timesteps=args.max_timesteps,
            max_retries=args.max_retries,
            verbose=not args.quiet
        )
        
        # Print summary
        collector.print_final_summary()
        
        # Exit with success
        return 0
    
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")
        collector.print_final_summary()
        return 1
    
    except Exception as e:
        print(f"\n\nCollection failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
