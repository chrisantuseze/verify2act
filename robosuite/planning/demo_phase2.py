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
    # With display:
    python demo_phase2.py --checkpoint Points2Plans/ckpt/checkpoint/cp_1.pth

    xvfb-run -a python demo_phase2.py --checkpoint Points2Plans/ckpt/checkpoint/cp_1.pth --use-real-converter
    
    # Headless (no display) - use xvfb:
    MUJOCO_GL=glfw xvfb-run -a python demo_phase2.py --checkpoint Points2Plans/ckpt/checkpoint/cp_1.pth --use-real-converter
"""

import os
# CRITICAL: Set rendering backend BEFORE any robosuite imports
# Must be set before binding_utils.py is imported (which happens during robosuite.__init__)
# Use 'glx' to bypass the MUJOCO_GPU_RENDERING override in binding_utils.py (line 37)
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glx'  # GLX works with X11/xvfb for headless rendering

import argparse
import numpy as np
from pathlib import Path
import sys

from llm_task_planner import LLMTaskPlanner
from dynamics_model_planner import DynamicsModelPlanner
from state_converter import StateConverter

# Don't import other components yet - they'll be imported when needed
# to avoid dependency issues in mock mode


class MockStateConverter:
    """Mock state converter for demo without full robosuite dependencies."""
    
    def convert(self, point_clouds, poses, object_types):
        """Convert mock observation to model format."""
        num_objects = len(point_clouds)
        
        # Create mock voxelized point clouds
        batch_voxel_list = []
        for pc in point_clouds:
            # Downsample to 512 points
            if len(pc) > 512:
                indices = np.random.choice(len(pc), 512, replace=False)
                pc_sampled = pc[indices]
            else:
                pc_sampled = pc
            batch_voxel_list.append(pc_sampled)
        
        # Stack into batch
        batch_voxel = np.stack(batch_voxel_list, axis=0)
        
        # One-hot encoding for object types
        batch_one_hot = np.zeros((num_objects, num_objects))
        for i, obj_type in enumerate(object_types):
            if obj_type < num_objects:
                batch_one_hot[i, obj_type] = 1.0
        
        # Edge attributes (pairwise relations)
        batch_edge_attr = np.zeros((num_objects * (num_objects - 1), 3))
        edge_idx = 0
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    # Relative position
                    batch_edge_attr[edge_idx] = poses[j][:3] - poses[i][:3]
                    edge_idx += 1
        
        return {
            'batch_voxel_list_single': [batch_voxel],
            'batch_one_hot_encoding': batch_one_hot,
            'batch_6DOF_pose': poses,
            'batch_edge_attr': batch_edge_attr,
            'batch_num_objects': num_objects
        }


def demo_planning_execution_loop(checkpoint_path="../../Points2Plans/ckpt/checkpoint/cp_1.pth", use_real_converter=False):
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
    # State Converter (obs -> model format)
    if use_real_converter:
        # Create a minimal robosuite environment for StateConverter
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config
        
        print("  Creating robosuite environment for StateConverter...")
        controller_config = load_composite_controller_config(controller="BASIC")
        env = suite.make(
            env_name="Stack3",
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,  # Disable on-screen rendering
            has_offscreen_renderer=True,  # Enable offscreen for point cloud generation
            use_camera_obs=True,  # Enable camera observations
            use_object_obs=True,
            control_freq=20,
            horizon=1000,
            ignore_done=True,
        )

        state_converter = StateConverter(env)
        print("  ✓ Using real StateConverter with robosuite environment")
    else:
        # Use mock version to avoid robosuite dependencies
        state_converter = MockStateConverter()
        print("  ✓ Using MockStateConverter")
    
    # Dynamics Model Planner (plans primitives with rejection sampling)
    # Pass state_converter for consistent object ID lookup
    dynamics_planner = DynamicsModelPlanner(
        checkpoint_path=checkpoint_path,
        num_samples=10,  # Reduced for demo
        state_converter=state_converter if use_real_converter else None
    )
    
    print("  ✓ All components initialized")
    
    # Step 2: Get observation first (needed to detect objects)
    print("\n[2/5] Creating observation...")
    
    if use_real_converter:
        # Real StateConverter gets observation from environment directly
        print("  ✓ Using environment observation (StateConverter will capture it)")
        state_dict = state_converter.convert()  # No arguments needed
    else:
        # Mock StateConverter needs explicit data
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
        state_dict['object_names'] = objects
   
    print(f"  ✓ State dict created")
    print(f"    - Keys: {list(state_dict.keys())}")
    
    # Step 3: Generate goals from LLM (ONCE at episode start)
    print("\n[3/5] Generating goals from LLM...")
    
    task_description = "Stack all objects on top of each other"
    
    # Get actual objects from environment/converter
    if use_real_converter:
        objects = state_dict['object_names']
        print(f"  Detected objects from environment: {objects}")
        
        # Compute initial predicates from actual state
        # For Stack3: all objects start on table
        initial_predicates = []
        for obj in objects:
            # Only include stackable objects (exclude bins and table)
            if obj not in ['table', 'bin1', 'bin2'] and not obj.startswith('bin'):
                initial_predicates.append(f"On({obj}, table)")
        print(f"  Initial predicates: {initial_predicates}")
    else:
        # Mock mode - use hardcoded values for Stack3
        objects = ["milk", "cereal", "bread"]
        initial_predicates = [
            "On(milk, table)",
            "On(cereal, table)",
            "On(bread, table)"
        ]
        print(f"  Using mock objects: {objects}")
    
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
        goals=goals,  # Already extracted first goal set in llm_task_planner
        object_name_to_id=object_name_to_id,
        num_objects=len(objects)
    )
    print(f"  ✓ Goal predicates shape: {goal_predicates.shape}")
    
    # Step 4: Plan next primitive with rejection sampling
    print("\n[4/5] Planning next primitive...")
    
    primitive_plan = plans  # Full high-level plan from LLM
    
    primitive, action_params, feasibility = dynamics_planner.plan_next_primitive(
        state_dict=state_dict,
        goal_predicates=goal_predicates,
        primitive_plan=primitive_plan
    )
    
    print(f"  ✓ Planned primitive: {primitive}")
    print(f"  ✓ Action params: {action_params}")
    print(f"  ✓ Feasibility: {feasibility:.3f}")
    
    # Step 5: Show execution flow (without actual robosuite)
    print("\n" + "=" * 80)
    print("EXECUTION FLOW (Stack3 Task)")
    print("=" * 80)
    
    print(f"\n1. Initial state: Objects on table - {', '.join([o for o in objects if 'bin' not in o.lower() and o != 'table'])}")
    print(f"2. LLM goals: Stack objects (generated ONCE)")
    print(f"3. High-level plan: {primitive_plan}")
    print(f"\n--- Closed-loop replanning at each primitive ---")
    
    for i, prim in enumerate(primitive_plan[:3]):  # Show first 3 primitives
        print(f"\nPrimitive {i+1}: {prim}")
        num_samples = dynamics_planner.num_samples if dynamics_planner else 50
        print(f"  → Dynamics model samples {num_samples} actions")
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
    parser.add_argument(
        "--use-real-converter",
        action="store_true",
        help="Use real StateConverter instead of mock (requires robosuite env)"
    )
    
    args = parser.parse_args()

    args.with_env = False
    args.use_real_converter = True
    
    if args.with_env:
        # Import robosuite and run with real environment
        try:
            import robosuite as suite
            from robosuite.controllers import load_composite_controller_config
            
            # Create environment
            config = load_composite_controller_config(controller="OSC_POSE")
            env = suite.make(
                "PickPlace",
                robots="Panda",
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                controller_configs=config,
            )
            
            demo_with_robosuite_env(env, args.checkpoint)
        except Exception as e:
            print(f"Error running with environment: {e}")
            print("Falling back to mock demo...")
            demo_planning_execution_loop(args.checkpoint, use_real_converter=False)
    else:
        # Run mock demo
        demo_planning_execution_loop(args.checkpoint, use_real_converter=args.use_real_converter)


if __name__ == "__main__":
    main()
