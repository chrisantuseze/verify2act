"""
Demo script for LLM Task Planner

Shows two usage modes:
1. YAML-based prompts (recommended): Uses BehaviorPromptManager with few-shot examples
2. Fallback mode: Uses hardcoded prompts when no YAML config provided

Usage:
    # With YAML config (recommended)
    python demo_llm_planner.py --prompt-config planning/prompts/robosuite_pickplace.yaml
    
    # Fallback mode (no YAML config)
    python demo_llm_planner.py
"""

import argparse
from pathlib import Path
from llm_task_planner import LLMTaskPlanner


def demo_yaml_mode():
    """Demo using YAML-based prompt configuration (recommended)."""
    print("=" * 80)
    print("DEMO: YAML-based Prompt Mode (Recommended)")
    print("=" * 80)
    
    # Initialize planner with YAML config
    planner = LLMTaskPlanner(
        prompt_config_path="planning/prompts/robosuite_pickplace.yaml",
        use_examples=True  # Enable few-shot learning
    )
    
    # Task details
    task = "Put all objects in the bin"
    objects = ["milk", "cereal", "bread", "can", "bin"]
    initial_predicates = [
        "On(milk, table)",
        "On(cereal, table)",
        "On(bread, table)",
        "On(can, table)"
    ]
    
    print(f"\nTask: {task}")
    print(f"Objects: {objects}")
    print(f"Initial State: {initial_predicates}")
    
    # Generate goals and plans
    goals, plans = planner.generate_goals_and_plans(
        task_description=task,
        objects=objects,
        initial_predicates=initial_predicates
    )
    
    print(f"\n✓ Goals: {goals}")
    print(f"✓ Plans: {plans}")


def demo_fallback_mode():
    """Demo using fallback hardcoded prompts (no YAML config)."""
    print("=" * 80)
    print("DEMO: Fallback Mode (No YAML Config)")
    print("=" * 80)
    
    # Initialize planner without YAML config
    planner = LLMTaskPlanner()
    
    # Task details
    task = "Put the milk in the bin"
    objects = ["milk", "bin", "table"]
    initial_predicates = ["On(milk, table)"]
    
    print(f"\nTask: {task}")
    print(f"Objects: {objects}")
    print(f"Initial State: {initial_predicates}")
    
    # Generate goals and plans
    goals, plans = planner.generate_goals_and_plans(
        task_description=task,
        objects=objects,
        initial_predicates=initial_predicates
    )
    
    print(f"\n✓ Goals: {goals}")
    print(f"✓ Plans: {plans}")


def main():
    parser = argparse.ArgumentParser(description="Demo LLM Task Planner")
    parser.add_argument(
        "--prompt-config",
        type=str,
        default=None,
        help="Path to YAML prompt config (if None, uses fallback mode)"
    )
    parser.add_argument(
        "--use-examples",
        action="store_true",
        default=True,
        help="Use few-shot examples from YAML config"
    )
    
    args = parser.parse_args()
    
    if args.prompt_config:
        # YAML mode
        planner = LLMTaskPlanner(
            prompt_config_path=args.prompt_config,
            use_examples=args.use_examples
        )
        
        # Example task
        task = "Put all objects in the bin"
        objects = ["milk", "cereal", "bread", "can", "bin"]
        initial_predicates = [
            "On(milk, table)",
            "On(cereal, table)",
            "On(bread, table)",
            "On(can, table)"
        ]
        
        print(f"\nTask: {task}")
        print(f"Objects: {objects}")
        
        goals, plans = planner.generate_goals_and_plans(
            task_description=task,
            objects=objects,
            initial_predicates=initial_predicates
        )
        
        print(f"\n✓ Goals: {goals}")
        print(f"✓ Plans: {plans}")
    else:
        # Fallback mode
        demo_fallback_mode()


if __name__ == "__main__":
    main()
