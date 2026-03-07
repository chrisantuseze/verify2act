#!/usr/bin/env python3
"""
Example training script for the Verify2Act critic model.
This demonstrates the full training pipeline from data loading to evaluation.
"""

import argparse
import torch
from pathlib import Path
import json

from .critic_config import CriticConfig
from .critic_trainer import CriticTrainer
from .critic_data_collector import CriticDataCollector, split_dataset
from .critic_evaluator import CriticEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train Verify2Act Critic")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to collected training data (pickle file)")
    parser.add_argument("--train_split", type=float, default=0.7,
                        help="Training split ratio")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Test split ratio")
    
    # Model
    parser.add_argument("--encoder_type", type=str, default="mlp",
                        choices=["mlp", "transformer"],
                        help="Encoder architecture type")
    parser.add_argument("--ensemble_size", type=int, default=5,
                        help="Number of ensemble members")
    parser.add_argument("--use_mc_dropout", action="store_true",
                        help="Use MC dropout instead of deep ensemble")
    
    # Active heads (phased implementation)
    parser.add_argument("--use_predicate_head", action="store_true", default=True,
                        help="Enable predicate satisfaction head")
    parser.add_argument("--use_feasibility_head", action="store_true",
                        help="Enable action feasibility head")
    parser.add_argument("--use_noninterference_head", action="store_true",
                        help="Enable non-interference head")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Early stopping patience")
    
    # Paths
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/critic",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs/critic",
                        help="Directory to save logs")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create configuration
    config = CriticConfig()
    
    # Update config from args
    config.model.encoder_type = args.encoder_type
    config.model.ensemble_size = args.ensemble_size
    config.model.use_mc_dropout = args.use_mc_dropout
    config.model.use_predicate_head = args.use_predicate_head
    config.model.use_feasibility_head = args.use_feasibility_head
    config.model.use_noninterference_head = args.use_noninterference_head
    
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.num_epochs = args.num_epochs
    config.training.early_stopping_patience = args.early_stopping_patience
    
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    config.device = args.device
    config.seed = args.seed
    
    print("="*80)
    print("VERIFY2ACT CRITIC TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Encoder: {config.model.encoder_type}")
    print(f"  Ensemble size: {config.model.ensemble_size}")
    print(f"  MC Dropout: {config.model.use_mc_dropout}")
    print(f"  Active heads:")
    print(f"    - Predicate: {config.model.use_predicate_head}")
    print(f"    - Feasibility: {config.model.use_feasibility_head}")
    print(f"    - Non-interference: {config.model.use_noninterference_head}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Device: {config.device}")
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    collector = CriticDataCollector()
    collector.load_dataset(args.data_path)
    
    # Get statistics
    stats = collector.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Positive: {stats['num_positive']}")
    print(f"  Negative: {stats['num_negative']}")
    
    # Balance and split data
    print("\nBalancing dataset...")
    balanced_data = collector.balance_dataset(
        negative_augmentation=True,
        target_ratio=1.0,
    )
    
    print("\nSplitting dataset...")
    train_data, val_data, test_data = split_dataset(
        balanced_data,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = CriticTrainer(config)
    
    # Train
    print("\nStarting training...")
    print("="*80)
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATION ON TEST SET")
    print("="*80)
    
    evaluator = CriticEvaluator(trainer.model, config, config.device)
    results = evaluator.evaluate_dataset(test_data)
    evaluator.print_report(results)
    
    # Save final results
    results_path = Path(args.checkpoint_dir) / "final_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved final results to {results_path}")
    
    # Save config
    config_path = Path(args.checkpoint_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Saved config to {config_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
