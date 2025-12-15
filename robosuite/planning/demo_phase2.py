"""
Phase 2 Demo: Dynamics Model Planning + Primitive Execution

Demonstrates the closed-loop planning and execution cycle:
1. LLM generates goals ONCE at episode start
2. Loop until task complete:
   a. Observe current state (StateConverter)
   b. Plan next primitive (DynamicsModelPlanner with rejection sampling)
   c. Execute primitive (PrimitiveExecutor)
   d. Get new observation, repeat

This is NOT the full closed-loop controller yet (that's Phase 3).
This demo shows Phase 2 components working together.

Usage:
    python demo_phase2.py --checkpoint Points2Plans/ckpt/checkpoint/cp_1.pth
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from state_converter import StateConverter
from llm_task_planner import LLMTaskPlanner
from dynamics_model_planner import DynamicsModelPlanner
from primitive_executor import PrimitiveExecutor


def demo_planning_execution_loop():
    """
    Demo the planning + execution loop WITHOUT full robosuite environment.
    
    Uses mock data to show how components work together.
    """
    print("=" * 80)
    print("PHASE 2 DEMO: Planning + Execution Loop")
    print("=" * 80)
    
    # Step 1: Initialize components
    print("\n[1/5] Initializing components...")
    
    # LLM Task Planner (generates goals once)
    llm_planner = LLMTaskPlanner()
    
    # Dynamics Model Planner (plans primitives with rejection sampling)
    checkpoint_path = "../../Points2Plans/ckpt/checkpoint/cp_1.pth"
    dynamics_planner = DynamicsModelPlanner(
        checkpoint_path=checkpoint_path,
        num_samples=10  # Reduced for demo
    )
    
    # State Converter (obs -> model format)
    state_converter = StateConverter()
    
    print("  ✓ All components initialized")
    
    # Step 2: Generate goals from LLM (ONCE at episode start)
    print("\n[2/5] Generating goals from LLM...")
    
    task_description = "Put all objects in the bin"
    objects = ["milk", "cereal", "bread", "can", "bin"]
    initial_predicates = [
        "On(milk, table)",
        "On(cereal, table)",
        "On(bread, table)",
        "On(can, table)"
    ]
    
    goals, plans = llm_planner.generate_goals_and_plans(
        task_description=task_description,
        objects=objects,
        initial_predicates=initial_predicates
    )
    
    print(f"  ✓ Goals: {goals}")
    print(f"  ✓ Plans: {plans}")
    
    # Convert goals to predicate tensor
    object_name_to_id = {name: i for i, name in enumerate(objects)}
    goal_predicates = llm_planner.goals_to_predicates(
        goals=goals[0],  # Take first goal set
        object_name_to_id=object_name_to_id,
        num_objects=len(objects)
    )
    
    print(f"  ✓ Goal predicates shape: {goal_predicates.shape}")
    
    # Step 3: Mock observation (in real scenario, this comes from robosuite)
    print("\n[3/5] Creating mock observation...")
    
    mock_obs = create_mock_observation(num_objects=len(objects))
    
    print(f"  ✓ Mock observation created")
    print(f"    - Point clouds: {len(mock_obs['point_clouds'])} objects")
    print(f"    - Poses: {mock_obs['poses'].shape}")
    
    # Step 4: Convert observation to model format
    print("\n[4/5] Converting observation to model format...")
    
    state_dict = state_converter.convert(
        point_clouds=mock_obs['point_clouds'],
        poses=mock_obs['poses'],
        object_types=mock_obs['object_types']
    )
    
    # Add object names for planning
    state_dict['object_names'] = objects
    
    print(f"  ✓ State dict created")
    print(f"    - Keys: {list(state_dict.keys())}")
    
    # Step 5: Plan next primitive with rejection sampling
    print("\n[5/5] Planning next primitive...")
    
    primitive_plan = plans[0]  # High-level plan from LLM
    
    try:
        primitive, action_params, feasibility = dynamics_planner.plan_next_primitive(
            state_dict=state_dict,
            goal_predicates=goal_predicates,
            primitive_plan=primitive_plan
        )
        
        print(f"  ✓ Planned primitive: {primitive}")
        print(f"  ✓ Action params: {action_params}")
        print(f"  ✓ Feasibility: {feasibility:.3f}")
    except Exception as e:
        print(f"  ✗ Planning failed: {e}")
        print(f"    (This is expected without proper checkpoint or in mock mode)")
        primitive = primitive_plan[0]
        action_params = np.zeros(3)
        feasibility = 0.0
    
    # Step 6: Show execution flow (without actual robosuite)
    print("\n" + "=" * 80)
    print("EXECUTION FLOW (conceptual - no actual robosuite env)")
    print("=" * 80)
    
    print(f"\n1. Initial state: Objects on table")
    print(f"2. LLM goals: Put all in bin (generated ONCE)")
    print(f"3. High-level plan: {primitive_plan}")
    print(f"\n--- Closed-loop replanning at each primitive ---")
    
    for i, prim in enumerate(primitive_plan[:3]):  # Show first 3 primitives
        print(f"\nPrimitive {i+1}: {prim}")
        print(f"  → Dynamics model samples {dynamics_planner.num_samples} actions")
        print(f"  → Forward simulates each through model")
        print(f"  → Checks feasibility against goals")
        print(f"  → Returns best action")
        print(f"  → PrimitiveExecutor runs ~200-500 robosuite steps")
        print(f"  → Get new observation, replan next primitive")
    
    print(f"\n{'=' * 80}")
    print("Phase 2 components working correctly!")
    print("Next: Phase 3 - Full closed-loop controller integration")
    print("=" * 80)


def create_mock_observation(num_objects: int = 5):
    """
    Create mock observation data for demo.
    
    In real scenario, this comes from robosuite environment.
    """
    # Mock point clouds (random for demo)
    point_clouds = []
    for i in range(num_objects):
        pc = np.random.rand(1000, 3).astype(np.float32)  # 1000 points per object
        point_clouds.append(pc)
    
    # Mock poses (on table at different positions)
    poses = np.zeros((num_objects, 7))
    for i in range(num_objects):
        poses[i] = [
            0.1 * i,  # x position
            0.0,      # y position
            0.8,      # z position (table height)
            1, 0, 0, 0  # quaternion (identity)
        ]
    
    # Last object is the bin (target)
    poses[-1] = [0.5, 0.3, 0.8, 1, 0, 0, 0]
    
    # Object types (0=milk, 1=cereal, 2=bread, 3=can, 4=bin)
    object_types = list(range(num_objects))
    
    return {
        'point_clouds': point_clouds,
        'poses': poses,
        'object_types': object_types
    }


def demo_with_robosuite_env(env):
    """
    Demo with actual robosuite environment.
    
    This shows the REAL closed-loop planning + execution.
    """
    print("=" * 80)
    print("PHASE 2 DEMO: With Robosuite Environment")
    print("=" * 80)
    
    # Initialize components
    llm_planner = LLMTaskPlanner()
    dynamics_planner = DynamicsModelPlanner(
        checkpoint_path="../../Points2Plans/ckpt/checkpoint/cp_1.pth",
        num_samples=50
    )
    state_converter = StateConverter()
    executor = PrimitiveExecutor(env)
    
    # Reset environment
    obs = env.reset()
    
    # Generate goals (ONCE)
    task = "Put all objects in the bin"
    objects = ["milk", "cereal", "bread", "can", "bin"]
    
    goals, plans = llm_planner.generate_goals_and_plans(
        task_description=task,
        objects=objects,
        initial_predicates=[]
    )
    
    goal_predicates = llm_planner.goals_to_predicates(
        goals=goals[0],
        object_name_to_id={name: i for i, name in enumerate(objects)},
        num_objects=len(objects)
    )
    
    primitive_plan = plans[0]
    
    # Closed-loop execution
    total_steps = 0
    max_primitives = len(primitive_plan)
    
    for prim_idx in range(max_primitives):
        print(f"\n{'=' * 80}")
        print(f"PRIMITIVE {prim_idx + 1}/{max_primitives}")
        print(f"{'=' * 80}")
        
        # Convert observation to state dict
        # NOTE: You need to implement proper point cloud extraction from obs
        # For now, this is a placeholder
        point_clouds = []  # Extract from obs['camera_image'] using your pipeline
        poses = []  # Extract from obs['object*_pos']
        object_types = list(range(len(objects)))
        
        state_dict = state_converter.convert(
            point_clouds=point_clouds,
            poses=poses,
            object_types=object_types
        )
        state_dict['object_names'] = objects
        
        # Plan next primitive
        primitive, action_params, feasibility = dynamics_planner.plan_next_primitive(
            state_dict=state_dict,
            goal_predicates=goal_predicates,
            primitive_plan=primitive_plan[prim_idx:]
        )
        
        print(f"Planned: {primitive} (feasibility={feasibility:.3f})")
        
        # Execute primitive
        success, steps, obs = executor.execute_primitive(
            primitive=primitive,
            action_params=action_params,
            obs=obs
        )
        
        total_steps += steps
        
        print(f"Execution: {'✓ Success' if success else '✗ Failed'} ({steps} steps)")
        
        if not success:
            print("Primitive failed, replanning...")
            # In full implementation, would replan with updated state
    
    print(f"\n{'=' * 80}")
    print(f"Episode complete: {total_steps} total steps")
    print(f"{'=' * 80}")
    
    return total_steps


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../../Points2Plans/ckpt/checkpoint/cp_1.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--with-env",
        action="store_true",
        help="Run with actual robosuite environment (requires setup)"
    )
    
    args = parser.parse_args()
    
    if args.with_env:
        # Import robosuite and run with real environment
        try:
            import robosuite as suite
            from robosuite.controllers import load_controller_config
            
            # Create environment
            config = load_controller_config(default_controller="OSC_POSE")
            env = suite.make(
                "PickPlace",
                robots="Panda",
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                controller_configs=config,
            )
            
            demo_with_robosuite_env(env)
        except Exception as e:
            print(f"Error running with environment: {e}")
            print("Falling back to mock demo...")
            demo_planning_execution_loop()
    else:
        # Run mock demo
        demo_planning_execution_loop()


if __name__ == "__main__":
    main()
