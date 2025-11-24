"""
Dataset Inspection and Quality Assurance Tools

Provides utilities for validating, visualizing, and analyzing
collected Points2Plans datasets.

Phase 4: Quality Assurance ✓
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt


class DatasetInspector:
    """
    Quality assurance tools for Points2Plans datasets.
    
    Features:
    - Dataset validation
    - Statistics computation
    - Visualization
    - Anomaly detection
    """
    
    def __init__(self, dataset_dir: str):
        """
        Initialize dataset inspector.
        
        Args:
            dataset_dir: Root directory of dataset
        """
        self.dataset_dir = Path(dataset_dir)
        self.episodes_dir = self.dataset_dir / "episodes"
        self.metadata_dir = self.dataset_dir / "metadata"
        
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
        self.episode_files = sorted(list(self.episodes_dir.glob("*.pkl")))
        self.num_episodes = len(self.episode_files)
        
        print(f"Dataset Inspector initialized")
        print(f"  Directory: {self.dataset_dir}")
        print(f"  Episodes: {self.num_episodes}")
    
    def validate_dataset(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Validate all episodes in dataset.
        
        Args:
            verbose: Print detailed validation results
            
        Returns:
            Validation report
        """
        print(f"\n{'='*60}")
        print("Validating Dataset")
        print(f"{'='*60}\n")
        
        validation_report = {
            'total_episodes': self.num_episodes,
            'valid_episodes': 0,
            'invalid_episodes': 0,
            'errors': [],
            'warnings': [],
        }
        
        for i, episode_file in enumerate(self.episode_files):
            try:
                # Load episode
                with open(episode_file, 'rb') as f:
                    episode_data = pickle.load(f)
                
                # Validate format
                errors, warnings = self._validate_episode_format(episode_data, episode_file.name)
                
                if errors:
                    validation_report['invalid_episodes'] += 1
                    validation_report['errors'].extend(errors)
                else:
                    validation_report['valid_episodes'] += 1
                
                validation_report['warnings'].extend(warnings)
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"  Validated {i+1}/{self.num_episodes} episodes...")
            
            except Exception as e:
                error_msg = f"{episode_file.name}: Failed to load - {e}"
                validation_report['invalid_episodes'] += 1
                validation_report['errors'].append(error_msg)
                if verbose:
                    print(f"  ✗ {error_msg}")
        
        # Print summary
        print(f"\n{'─'*60}")
        print(f"Validation Complete")
        print(f"{'─'*60}")
        print(f"Valid Episodes: {validation_report['valid_episodes']}/{self.num_episodes}")
        print(f"Invalid Episodes: {validation_report['invalid_episodes']}")
        print(f"Warnings: {len(validation_report['warnings'])}")
        print(f"Errors: {len(validation_report['errors'])}")
        
        if validation_report['errors'] and verbose:
            print(f"\nFirst 5 errors:")
            for error in validation_report['errors'][:5]:
                print(f"  - {error}")
        
        print(f"{'─'*60}\n")
        
        return validation_report
    
    def _validate_episode_format(self, episode_data: Tuple, filename: str) -> Tuple[List[str], List[str]]:
        """Validate single episode format."""
        errors = []
        warnings = []
        
        # Check tuple structure
        if not isinstance(episode_data, tuple) or len(episode_data) != 2:
            errors.append(f"{filename}: Invalid format - not (data_dict, attrs_dict) tuple")
            return errors, warnings
        
        data_dict, attrs_dict = episode_data
        
        # Check dictionaries
        if not isinstance(data_dict, dict):
            errors.append(f"{filename}: data_dict is not a dictionary")
        if not isinstance(attrs_dict, dict):
            errors.append(f"{filename}: attrs_dict is not a dictionary")
        
        if errors:
            return errors, warnings
        
        # Get object keys
        object_keys = [k for k in data_dict.keys() if k.startswith('block_')]
        
        if len(object_keys) == 0:
            warnings.append(f"{filename}: No objects found")
        
        # Validate each object
        for obj_key in object_keys:
            # Check data_dict structure
            if obj_key not in data_dict:
                errors.append(f"{filename}: {obj_key} missing from data_dict")
                continue
            
            obj_data = data_dict[obj_key]
            
            required_keys = ['positions', 'orientations', 'point_cloud', 'hidden_label']
            for key in required_keys:
                if key not in obj_data:
                    errors.append(f"{filename}: {obj_key} missing '{key}' in data_dict")
            
            # Check shapes
            if 'positions' in obj_data:
                pos_shape = obj_data['positions'].shape
                if len(pos_shape) != 2 or pos_shape[1] != 3:
                    errors.append(f"{filename}: {obj_key} positions shape invalid: {pos_shape}")
            
            if 'orientations' in obj_data:
                ori_shape = obj_data['orientations'].shape
                if len(ori_shape) != 2 or ori_shape[1] != 4:
                    errors.append(f"{filename}: {obj_key} orientations shape invalid: {ori_shape}")
            
            # Check attrs_dict
            if obj_key not in attrs_dict:
                errors.append(f"{filename}: {obj_key} missing from attrs_dict")
            else:
                obj_attrs = attrs_dict[obj_key]
                if 'extents' not in obj_attrs:
                    warnings.append(f"{filename}: {obj_key} missing 'extents' in attrs_dict")
                if 'fix_base_link' not in obj_attrs:
                    warnings.append(f"{filename}: {obj_key} missing 'fix_base_link' in attrs_dict")
        
        return errors, warnings
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        print(f"\n{'='*60}")
        print("Computing Dataset Statistics")
        print(f"{'='*60}\n")
        
        stats = {
            'num_episodes': self.num_episodes,
            'total_timesteps': 0,
            'total_objects': 0,
            'total_contacts': 0,
            'timesteps_per_episode': [],
            'objects_per_episode': [],
            'contacts_per_episode': [],
            'episode_sizes_mb': [],
            'point_cloud_sizes': [],
        }
        
        for episode_file in self.episode_files:
            try:
                # Load episode
                with open(episode_file, 'rb') as f:
                    episode_data = pickle.load(f)
                
                data_dict, attrs_dict = episode_data
                
                # Get object keys
                object_keys = [k for k in data_dict.keys() if k.startswith('block_')]
                
                # Count timesteps
                if object_keys:
                    num_timesteps = len(data_dict[object_keys[0]]['positions'])
                    stats['timesteps_per_episode'].append(num_timesteps)
                    stats['total_timesteps'] += num_timesteps
                
                # Count objects
                num_objects = len(object_keys)
                stats['objects_per_episode'].append(num_objects)
                stats['total_objects'] += num_objects
                
                # File size
                file_size_mb = episode_file.stat().st_size / (1024 * 1024)
                stats['episode_sizes_mb'].append(file_size_mb)
                
                # Point cloud sizes
                for obj_key in object_keys:
                    if 'point_cloud' in data_dict[obj_key]:
                        pcd = data_dict[obj_key]['point_cloud']
                        stats['point_cloud_sizes'].append(pcd.shape)
            
            except Exception as e:
                print(f"  Warning: Failed to process {episode_file.name}: {e}")
        
        # Compute averages
        stats['avg_timesteps'] = np.mean(stats['timesteps_per_episode']) if stats['timesteps_per_episode'] else 0
        stats['avg_objects'] = np.mean(stats['objects_per_episode']) if stats['objects_per_episode'] else 0
        stats['avg_size_mb'] = np.mean(stats['episode_sizes_mb']) if stats['episode_sizes_mb'] else 0
        stats['total_size_mb'] = sum(stats['episode_sizes_mb'])
        
        # Print statistics
        print(f"Episodes: {stats['num_episodes']}")
        print(f"Total Timesteps: {stats['total_timesteps']}")
        print(f"Avg Timesteps/Episode: {stats['avg_timesteps']:.1f}")
        print(f"")
        print(f"Total Objects: {stats['total_objects']}")
        print(f"Avg Objects/Episode: {stats['avg_objects']:.1f}")
        print(f"")
        print(f"Total Size: {stats['total_size_mb']:.1f} MB")
        print(f"Avg Size/Episode: {stats['avg_size_mb']:.2f} MB")
        print(f"{'='*60}\n")
        
        return stats
    
    def visualize_statistics(self, stats: Optional[Dict] = None, save_path: Optional[str] = None):
        """Create visualization of dataset statistics."""
        if stats is None:
            stats = self.compute_statistics()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Dataset Statistics', fontsize=16)
        
        # Timesteps per episode
        if stats['timesteps_per_episode']:
            axes[0, 0].hist(stats['timesteps_per_episode'], bins=20, edgecolor='black')
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Timesteps per Episode')
            axes[0, 0].axvline(stats['avg_timesteps'], color='r', linestyle='--', label=f'Mean: {stats["avg_timesteps"]:.1f}')
            axes[0, 0].legend()
        
        # Objects per episode
        if stats['objects_per_episode']:
            axes[0, 1].hist(stats['objects_per_episode'], bins=10, edgecolor='black')
            axes[0, 1].set_xlabel('Number of Objects')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Objects per Episode')
        
        # Episode sizes
        if stats['episode_sizes_mb']:
            axes[1, 0].hist(stats['episode_sizes_mb'], bins=20, edgecolor='black')
            axes[1, 0].set_xlabel('Size (MB)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Episode File Sizes')
            axes[1, 0].axvline(stats['avg_size_mb'], color='r', linestyle='--', label=f'Mean: {stats["avg_size_mb"]:.2f} MB')
            axes[1, 0].legend()
        
        # Dataset summary
        axes[1, 1].axis('off')
        summary_text = f"""
        Dataset Summary
        ───────────────────
        Episodes: {stats['num_episodes']}
        Total Timesteps: {stats['total_timesteps']}
        Total Size: {stats['total_size_mb']:.1f} MB
        
        Avg Timesteps: {stats['avg_timesteps']:.1f}
        Avg Objects: {stats['avg_objects']:.1f}
        Avg Size: {stats['avg_size_mb']:.2f} MB
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved visualization: {save_path}")
        else:
            plt.show()
    
    def inspect_episode(self, episode_idx: int):
        """Inspect specific episode in detail."""
        if episode_idx >= self.num_episodes:
            print(f"Error: Episode {episode_idx} not found (total: {self.num_episodes})")
            return
        
        episode_file = self.episode_files[episode_idx]
        
        print(f"\n{'='*60}")
        print(f"Inspecting Episode {episode_idx}: {episode_file.name}")
        print(f"{'='*60}\n")
        
        # Load episode
        with open(episode_file, 'rb') as f:
            episode_data = pickle.load(f)
        
        data_dict, attrs_dict = episode_data
        
        # Get object keys
        object_keys = [k for k in data_dict.keys() if k.startswith('block_')]
        
        print(f"File: {episode_file}")
        print(f"Size: {episode_file.stat().st_size / 1024:.1f} KB")
        print(f"Objects: {len(object_keys)}")
        
        if object_keys:
            num_timesteps = len(data_dict[object_keys[0]]['positions'])
            print(f"Timesteps: {num_timesteps}")
        
        print(f"\nObject Details:")
        for obj_key in object_keys:
            print(f"\n  {obj_key}:")
            
            # Data dict
            if 'positions' in data_dict[obj_key]:
                print(f"    positions: {data_dict[obj_key]['positions'].shape}")
            if 'orientations' in data_dict[obj_key]:
                print(f"    orientations: {data_dict[obj_key]['orientations'].shape}")
            if 'point_cloud' in data_dict[obj_key]:
                print(f"    point_cloud: {data_dict[obj_key]['point_cloud'].shape}")
            if 'hidden_label' in data_dict[obj_key]:
                print(f"    hidden_label: {data_dict[obj_key]['hidden_label']}")
            
            # Attrs dict
            if obj_key in attrs_dict:
                if 'extents' in attrs_dict[obj_key]:
                    print(f"    extents: {attrs_dict[obj_key]['extents']}")
                if 'fix_base_link' in attrs_dict[obj_key]:
                    print(f"    fix_base_link: {attrs_dict[obj_key]['fix_base_link']}")
        
        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and validate Points2Plans datasets"
    )
    
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate all episodes'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Compute dataset statistics'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create statistics visualization'
    )
    
    parser.add_argument(
        '--inspect',
        type=int,
        metavar='N',
        help='Inspect specific episode by index'
    )
    
    parser.add_argument(
        '--save-viz',
        type=str,
        metavar='PATH',
        help='Save visualization to file'
    )
    
    args = parser.parse_args()
    
    # Create inspector
    inspector = DatasetInspector(args.dataset_dir)
    
    # Run requested operations
    if args.validate:
        inspector.validate_dataset()
    
    if args.stats:
        stats = inspector.compute_statistics()
    
    if args.visualize:
        stats = inspector.compute_statistics() if not args.stats else stats
        inspector.visualize_statistics(stats, save_path=args.save_viz)
    
    if args.inspect is not None:
        inspector.inspect_episode(args.inspect)
    
    # If no operations specified, run all
    if not any([args.validate, args.stats, args.visualize, args.inspect is not None]):
        print("Running full inspection...\n")
        inspector.validate_dataset()
        stats = inspector.compute_statistics()
        inspector.visualize_statistics(stats, save_path=args.save_viz)


if __name__ == "__main__":
    main()
