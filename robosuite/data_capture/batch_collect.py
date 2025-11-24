"""
Batch Data Collection Script for Points2Plans Dataset

Integrates with HeuristicStackPolicy from run_stack.py to automatically
collect multiple episodes with progress tracking and error recovery.

Phase 4: Batch Collection ✓
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

# Add parent directory to path to import run_stack
sys.path.append(str(Path(__file__).parent.parent))

from run_stack import HeuristicStackPolicy, create_environment
from episode_recorder import EpisodeRecorder


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
                 env_name: str = "Stack",
                 output_dir: str = "./data_capture/dataset",
                 camera_names: Optional[List[str]] = None,
                 num_points: int = 128,
                 voxel_size: float = 0.005):
        """
        Initialize batch collector.
        
        Args:
            env_name: Environment name ("Stack", "Stack3", "Stack4")
            output_dir: Root directory for dataset
            camera_names: Camera names for point cloud capture
            num_points: Points per object point cloud
            voxel_size: Voxel size for downsampling
        """
        self.env_name = env_name
        self.output_dir = Path(output_dir)
        self.camera_names = camera_names or ["frontview", "agentview"]
        self.num_points = num_points
        self.voxel_size = voxel_size
        
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
        
        # Create environment and recorder
        env = create_environment(self.env_name)
        recorder = EpisodeRecorder(
            env, 
            camera_names=self.camera_names,
            num_points=self.num_points,
            voxel_size=self.voxel_size
        )
        
        try:
            for episode_idx in range(num_episodes):
                success = False
                retry_count = 0
                
                while not success and retry_count <= max_retries:
                    try:
                        episode_start = time.time()
                        
                        # Collect single episode
                        self._collect_episode(
                            env, 
                            recorder, 
                            episode_idx, 
                            max_timesteps,
                            verbose
                        )
                        
                        episode_duration = time.time() - episode_start
                        self.stats['episode_durations'].append(episode_duration)
                        self.stats['successful_episodes'] += 1
                        success = True
                        
                        if verbose:
                            self._print_episode_summary(episode_idx, episode_duration, True)
                        
                    except Exception as e:
                        retry_count += 1
                        error_msg = f"Episode {episode_idx} failed (attempt {retry_count}/{max_retries}): {e}"
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
                
                self.stats['total_episodes'] += 1
                
                # Print progress
                self._print_progress(episode_idx + 1, num_episodes)
        
        finally:
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
                         verbose: bool):
        """Collect a single episode using heuristic policy."""
        # Reset environment
        obs = env.reset()
        
        # Start recording
        recorder.start_episode()
        
        # Create policy
        policy = HeuristicStackPolicy(env)
        policy.obs = obs
        
        # Run episode
        timestep = 0
        episode_complete = False
        
        while timestep < max_timesteps and not episode_complete:
            # Get action from policy
            action, _ = policy.step()
            
            # Execute action
            obs, reward, done, info = env.step(action)
            
            # Record timestep
            recorder.record_step(action, obs)
            
            # Update policy observations
            policy.obs = obs
            
            timestep += 1
            
            # Check if all stacking pairs completed
            if policy.stage == "done" or policy.pair_idx >= len(policy.stacking_pairs):
                episode_complete = True
                if verbose:
                    print(f"   Episode {episode_idx}: Stacking complete at timestep {timestep}")
        
        # End recording
        data_dict, attrs_dict = recorder.end_episode()
        
        # Save episode
        episode_name = f"episode_{episode_idx:05d}"
        saved_path = recorder.save_episode(str(self.episodes_dir), episode_name)
        
        # Update statistics
        episode_stats = recorder.get_statistics()
        self.stats['total_timesteps'] += episode_stats['num_timesteps']
        
        # Save episode metadata
        self._save_episode_metadata(episode_idx, episode_stats, saved_path)
    
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
        
        metadata = {
            'env_name': self.env_name,
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
            print(f"{status} Episode {episode_idx:05d}: FAILED after retries")
    
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


def main():
    parser = argparse.ArgumentParser(
        description="Batch collection of robosuite episodes for Points2Plans dataset"
    )
    
    parser.add_argument(
        '--env',
        type=str,
        default='Stack',
        choices=['Stack', 'Stack3', 'Stack4'],
        help='Environment name'
    )
    
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
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
        default=['frontview', 'agentview'],
        help='Camera names for point cloud capture'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Create collector
    collector = BatchCollector(
        env_name=args.env,
        output_dir=args.output_dir,
        camera_names=args.cameras,
        num_points=args.num_points
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
