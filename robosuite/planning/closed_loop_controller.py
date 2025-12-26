"""
Closed-Loop Controller for Points2Plans + Robosuite Integration

This is the main controller that orchestrates:
1. LLM Task Planning (ONCE at episode start)
2. Closed-Loop Planning + Execution:
   - Observe current state (StateConverter)
   - Plan next primitive (DynamicsModelPlanner with rejection sampling)
   - Execute primitive (PrimitiveExecutor)
   - Check goal achievement
   - Repeat until goal achieved or max primitives reached

Key features:
- Replanning at primitive boundaries (closed-loop control)
- Automatic failure recovery
- Goal achievement checking
- Episode statistics tracking

Usage:
    from closed_loop_controller import ClosedLoopController
    
    controller = ClosedLoopController(
        env=robosuite_env,
        checkpoint_path="Points2Plans/ckpt/checkpoint/cp_1.pth"
    )
    
    success, stats = controller.run_episode(
        task_description="Stack all objects",
        max_primitives=20
    )
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from llm_task_planner import LLMTaskPlanner
from dynamics_model_planner import DynamicsModelPlanner
from state_converter import StateConverter
from primitive_executor import PrimitiveExecutor


class ClosedLoopController:
    """
    Closed-loop controller integrating all planning and execution components.
    
    This is the main entry point for running complete episodes with
    Points2Plans + Robosuite integration.
    """
    
    def __init__(
        self,
        args,
        env,
        checkpoint_path: str = "../../Points2Plans/ckpt/checkpoint/cp_1.pth",
        num_planning_samples: int = 50,
        goal_threshold: float = 0.1,
        max_replans_per_primitive: int = 3,
        lookahead_depth: int = 1,
        enable_collision_checking: bool = True,
        verbose: bool = True
    ):
        """
        Initialize closed-loop controller.
        
        Args:
            args: Command-line arguments or configuration
            env: Robosuite environment instance
            checkpoint_path: Path to trained dynamics model checkpoint
            num_planning_samples: Number of action samples for rejection sampling
            goal_threshold: Threshold for goal achievement (predicate difference)
            max_replans_per_primitive: Max replanning attempts if execution fails
            lookahead_depth: Number of primitives to simulate ahead (1-3)
            enable_collision_checking: Whether to enable collision detection
            verbose: Whether to print detailed logs
        """
        self.args = args
        self.env = env
        self.goal_threshold = goal_threshold
        self.max_replans_per_primitive = max_replans_per_primitive
        self.verbose = verbose
        
        # Initialize components
        if self.verbose:
            print("Initializing closed-loop controller components...")
        
        self.llm_planner = LLMTaskPlanner(args.model_config_path, args.prompt_config_path)
        self.state_converter = StateConverter(env)
        self.dynamics_planner = DynamicsModelPlanner(
            checkpoint_path=checkpoint_path,
            num_samples=num_planning_samples,
            state_converter=self.state_converter,
            lookahead_depth=lookahead_depth,
            enable_collision_checking=enable_collision_checking
        )
        self.executor = PrimitiveExecutor(env)
        
        if self.verbose:
            print("✓ All components initialized")
        
        # Episode statistics
        self.reset_stats()
    
    def reset_stats(self):
        """Reset episode statistics."""
        self.stats = {
            'num_primitives_executed': 0,
            'num_primitives_failed': 0,
            'num_replans': 0,
            'total_steps': 0,
            'start_time': None,
            'end_time': None,
            'primitive_history': [],
            'feasibility_history': []
        }
    
    def run_episode(
        self,
        task_description: str,
        max_primitives: int = 20,
        initial_predicates: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run a complete episode with closed-loop planning and execution.
        
        Args:
            task_description: Natural language task description
            max_primitives: Maximum number of primitives to execute
            initial_predicates: Optional list of initial scene predicates
                              (auto-detected if None)
        
        Returns:
            success: Whether the task was completed successfully
            stats: Dictionary of episode statistics
        """
        self.reset_stats()
        self.stats['start_time'] = time.time()
        
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"STARTING EPISODE: {task_description}")
            print("=" * 80)
        
        # Reset environment
        obs = self.env.reset()
        
        # Step 1: Get initial state and detect objects
        if self.verbose:
            print("\n[1/4] Detecting objects in scene...")
        
        state_dict = self.state_converter.convert()
        objects = state_dict['object_names']
        
        if self.verbose:
            print(f"  ✓ Detected {len(objects)} objects: {objects}")
        
        # Generate initial predicates if not provided
        if initial_predicates is None:
            initial_predicates = self._generate_initial_predicates(state_dict, objects)
        
        if self.verbose:
            print(f"  ✓ Initial predicates: {initial_predicates}")
        
        # Step 2: Generate goals from LLM (ONCE)
        if self.verbose:
            print("\n[2/4] Generating goals from LLM...")
        
        goals, plans = self.llm_planner.generate_goals_and_plans(
            task_description=task_description,
            objects=objects,
            initial_predicates=initial_predicates
        )
        
        if not goals or not plans:
            print("ERROR: LLM failed to generate goals/plans")
            self.stats['end_time'] = time.time()
            return False, self.stats
        
        if self.verbose:
            print(f"  ✓ Goals: {goals}")
            print(f"  ✓ Plan: {plans[0] if plans else 'No plan'}")
        
        # Convert goals to predicate tensor
        object_name_to_id = {name: i for i, name in enumerate(objects)}
        goal_predicates = self.llm_planner.goals_to_predicates(
            goals=goals,  # Already extracted first goal set in llm_task_planner
            object_name_to_id=object_name_to_id,
            num_objects=len(objects)
        )
        
        primitive_plan = plans  # Full plan from LLM
        
        # Step 3: Closed-loop planning and execution
        if self.verbose:
            print("\n[3/4] Starting closed-loop execution...")
            print(f"  Max primitives: {max_primitives}")
        
        for prim_idx in range(max_primitives):
            if self.verbose:
                print(f"\n{'─' * 80}")
                print(f"PRIMITIVE {prim_idx + 1}/{max_primitives}")
                print(f"{'─' * 80}")
            
            # Get current state
            state_dict = self.state_converter.convert()
            state_dict['object_names'] = objects
            
            # Check if goal already achieved
            if self._check_goal_achieved(state_dict, goal_predicates):
                if self.verbose:
                    print("✓ Goal achieved!")
                self.stats['end_time'] = time.time()
                return True, self.stats
            
            # Plan next primitive with replanning on failure
            success = False
            for replan_attempt in range(self.max_replans_per_primitive):
                if replan_attempt > 0:
                    if self.verbose:
                        print(f"  Replanning (attempt {replan_attempt + 1}/{self.max_replans_per_primitive})...")
                    self.stats['num_replans'] += 1
                
                # Plan
                primitive, action_params, feasibility = self.dynamics_planner.plan_next_primitive(
                    state_dict=state_dict,
                    goal_predicates=goal_predicates,
                    primitive_plan=primitive_plan
                )
                
                if primitive is None:
                    if self.verbose:
                        print("  ✗ Planning failed (no feasible action found)")
                    continue
                
                if self.verbose:
                    print(f"  → Planned: {primitive} (feasibility={feasibility:.3f})")
                
                # Get current observation for execution
                obs = self.env._get_observations()
                
                # Execute primitive
                exec_success, num_steps = self._execute_primitive_with_monitoring(
                    primitive, action_params, obs
                )
                
                self.stats['total_steps'] += num_steps
                self.stats['num_primitives_executed'] += 1
                self.stats['primitive_history'].append(primitive)
                self.stats['feasibility_history'].append(feasibility)
                
                if exec_success:
                    if self.verbose:
                        print(f"  ✓ Execution succeeded ({num_steps} steps)")
                    success = True
                    break
                else:
                    if self.verbose:
                        print(f"  ✗ Execution failed ({num_steps} steps)")
                    self.stats['num_primitives_failed'] += 1
            
            if not success:
                if self.verbose:
                    print(f"  ✗ Failed after {self.max_replans_per_primitive} replan attempts")
                # Continue to next primitive (failure recovery)
        
        # Step 4: Final goal check
        if self.verbose:
            print("\n[4/4] Final goal check...")
        
        state_dict = self.state_converter.convert()
        state_dict['object_names'] = objects
        
        success = self._check_goal_achieved(state_dict, goal_predicates)
        
        self.stats['end_time'] = time.time()
        
        if self.verbose:
            self._print_episode_summary(success)
        
        return success, self.stats
    
    def _generate_initial_predicates(
        self,
        state_dict: Dict[str, Any],
        objects: List[str]
    ) -> List[str]:
        """
        Generate initial scene predicates from current state using decoder.
        
        Uses the trained decoder to predict current predicates from observations.
        This is more accurate than manual heuristics and consistent with goal checking.
        """
        # Use decoder to predict current predicates from state
        current_predicates = self.dynamics_planner.predict_predicates(state_dict)
        print(f"Decoded predicates shape: {current_predicates.shape if current_predicates is not None else 'None'}")
        
        if current_predicates is None:
            # Fallback to heuristics only if prediction fails
            if self.verbose:
                print("  Warning: Decoder prediction failed, using fallback heuristics")
            return self._fallback_initial_predicates(objects)
        
        # Convert predicted predicate matrix to string format for LLM
        predicate_strings = self._predicates_to_strings(
            current_predicates, 
            objects,
            threshold=0.5  # Only include predicates with confidence > 0.5
        )
        
        return predicate_strings
    
    def _fallback_initial_predicates(self, objects: List[str]) -> List[str]:
        """
        Fallback heuristics for initial predicates when decoder fails.
        
        Assumes all non-table/non-bin objects start on table.
        """
        predicates = []
        for obj in objects:
            if obj == 'table' or 'bin' in obj.lower():
                continue
            predicates.append(f"On({obj}, table)")
        return predicates
    
    def _predicates_to_strings(
        self,
        predicate_matrix: np.ndarray,
        objects: List[str],
        threshold: float = 0.5
    ) -> List[str]:
        """
        Convert predicate matrix to string format for LLM.
        
        Args:
            predicate_matrix: [num_objects, num_objects, num_predicates] array
            objects: List of object names
            threshold: Minimum confidence to include predicate
        
        Returns:
            List of predicate strings like ["On(cubeA, table)", "Stacked(cubeA, cubeB)"]
        """
        # Define predicate names matching the system prompts
        # Order matches the 9-predicate system: On, Inside, Left, Right, Front, Behind, Near, Touching, Grasped
        # But we only use predicates that the LLM prompt says the robot can detect
        predicate_names = [
            'On',        # 0: Object on another (table, bin, etc)
            'Inside',    # 1: Object inside container
            None,        # 2: Left (spatial - not used for LLM)
            None,        # 3: Right (spatial - not used for LLM) 
            None,        # 4: Front (spatial - not used for LLM)
            None,        # 5: Behind (spatial - not used for LLM)
            None,        # 6: Near (not in our prompt)
            None,        # 7: Touching (not in our prompt)
            'Grasped'    # 8: Currently held by robot
        ]
        
        predicate_strings = []
        num_objects = len(objects)
        
        for i in range(num_objects):
            for j in range(num_objects):
                if i == j:
                    print("Skipping self-predicate")
                    continue
                
                # Check each predicate type
                for pred_idx in range(min(len(predicate_names), predicate_matrix.shape[2])):
                    pred_name = predicate_names[pred_idx]
                    if pred_name is None:
                        print("Skipping unused predicate index:", pred_idx)
                        continue
                    
                    confidence = predicate_matrix[i, j, pred_idx]
                    print(f"Predicate {pred_name}({objects[i]}, {objects[j]}) confidence: {confidence:.3f}")
                    
                    if confidence > threshold:
                        # Special case: On(cube, cube) should be Stacked for clarity
                        if pred_name == 'On' and objects[j] != 'table' and 'bin' not in objects[j].lower():
                            pred_str = f"Stacked({objects[i]}, {objects[j]})"
                        else:
                            pred_str = f"{pred_name}({objects[i]}, {objects[j]})"
                        predicate_strings.append(pred_str)
        
        return predicate_strings
    
    def _check_goal_achieved(
        self,
        state_dict: Dict[str, Any],
        goal_predicates: np.ndarray,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if current state satisfies goal predicates.
        
        Uses dynamics model's decoder to predict current predicates,
        then compares with goal predicates.
        """
        if threshold is None:
            threshold = self.goal_threshold
        
        # Get current predicates from dynamics model
        current_predicates = self.dynamics_planner.predict_predicates(state_dict)
        
        if current_predicates is None:
            return False
        
        # Only compare predicates that are set in goals (where goal_predicates > 0)
        # This handles dimension mismatch and focuses on relevant predicates
        goal_mask = goal_predicates > 0.5
        
        if goal_mask.sum() == 0:
            # No goals set, consider achieved
            return True
        
        # Extract only the goal dimensions we care about
        # Ensure current_predicates matches goal_predicates shape
        min_pred_dim = min(current_predicates.shape[2], goal_predicates.shape[2])
        current_subset = current_predicates[:, :, :min_pred_dim]
        goal_subset = goal_predicates[:, :, :min_pred_dim]
        
        # Compare only where goals are set
        goal_mask_subset = goal_subset > 0.5
        matches = np.logical_and(
            goal_mask_subset,
            current_subset > 0.5
        )
        
        # Success if most goals are satisfied
        satisfaction_rate = matches.sum() / goal_mask_subset.sum()
        
        if self.verbose:
            print(f"  Goal satisfaction: {satisfaction_rate:.2%} (threshold={1-threshold:.2%})")
        
        return satisfaction_rate >= (1 - threshold)
    
    def _execute_primitive_with_monitoring(
        self,
        primitive: str,
        action_params: Dict[str, Any],
        obs: Dict
    ) -> Tuple[bool, int]:
        """
        Execute primitive with monitoring and early failure detection.
        
        Args:
            primitive: Primitive action string
            action_params: Action parameters from planner
            obs: Current robosuite observation
        
        Returns:
            success: Whether execution succeeded
            num_steps: Number of robosuite steps taken
        """
        try:
            success, num_steps, final_obs = self.executor.execute_primitive(
                primitive=primitive,
                action_params=action_params,
                obs=obs
            )
            return success, num_steps
        
        except Exception as e:
            if self.verbose:
                print(f"  Exception during execution: {e}")
            return False, 0
    
    def _print_episode_summary(self, success: bool):
        """Print summary of episode statistics."""
        duration = self.stats['end_time'] - self.stats['start_time']
        
        print("\n" + "=" * 80)
        print("EPISODE SUMMARY")
        print("=" * 80)
        print(f"Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
        print(f"Duration: {duration:.2f}s")
        print(f"Total steps: {self.stats['total_steps']}")
        print(f"Primitives executed: {self.stats['num_primitives_executed']}")
        print(f"Primitives failed: {self.stats['num_primitives_failed']}")
        print(f"Replans: {self.stats['num_replans']}")
        
        if self.stats['primitive_history']:
            print(f"\nPrimitive history:")
            for i, (prim, feas) in enumerate(zip(
                self.stats['primitive_history'],
                self.stats['feasibility_history']
            )):
                print(f"  {i+1}. {prim} (feasibility={feas:.3f})")
        
        print("=" * 80)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        return self.stats.copy()


class BatchController:
    """
    Controller for running multiple episodes with evaluation.
    
    Useful for benchmarking and testing.
    """
    
    def __init__(
        self,
        env,
        checkpoint_path: str = "../../Points2Plans/ckpt/checkpoint/cp_1.pth",
        **controller_kwargs
    ):
        """
        Initialize batch controller.
        
        Args:
            env: Robosuite environment instance
            checkpoint_path: Path to trained dynamics model
            **controller_kwargs: Additional arguments for ClosedLoopController
        """
        self.env = env
        self.checkpoint_path = checkpoint_path
        self.controller_kwargs = controller_kwargs
        
        # Results tracking
        self.results = []
    
    def run_batch(
        self,
        tasks: List[Tuple[str, Optional[List[str]]]],
        num_trials_per_task: int = 1,
        max_primitives: int = 20
    ) -> Dict[str, Any]:
        """
        Run multiple episodes and collect statistics.
        
        Args:
            tasks: List of (task_description, initial_predicates) tuples
            num_trials_per_task: Number of trials per task
            max_primitives: Max primitives per episode
        
        Returns:
            results: Dictionary with aggregated results
        """
        print(f"\nRunning batch evaluation: {len(tasks)} tasks, "
              f"{num_trials_per_task} trials each")
        
        self.results = []
        
        for task_idx, (task_description, initial_predicates) in enumerate(tasks):
            print(f"\n{'=' * 80}")
            print(f"TASK {task_idx + 1}/{len(tasks)}: {task_description}")
            print(f"{'=' * 80}")
            
            task_results = []
            
            for trial in range(num_trials_per_task):
                print(f"\nTrial {trial + 1}/{num_trials_per_task}")
                
                # Create controller for this trial
                controller = ClosedLoopController(
                    env=self.env,
                    checkpoint_path=self.checkpoint_path,
                    verbose=(num_trials_per_task == 1),  # Verbose only if single trial
                    **self.controller_kwargs
                )
                
                # Run episode
                success, stats = controller.run_episode(
                    task_description=task_description,
                    max_primitives=max_primitives,
                    initial_predicates=initial_predicates
                )
                
                result = {
                    'task': task_description,
                    'trial': trial,
                    'success': success,
                    **stats
                }
                
                task_results.append(result)
                self.results.append(result)
                
                if num_trials_per_task > 1:
                    print(f"  Result: {'SUCCESS' if success else 'FAILED'} "
                          f"({stats['num_primitives_executed']} primitives, "
                          f"{stats['total_steps']} steps)")
            
            # Print task summary
            if num_trials_per_task > 1:
                self._print_task_summary(task_description, task_results)
        
        # Print overall summary
        self._print_overall_summary()
        
        return self._compute_aggregate_results()
    
    def _print_task_summary(self, task: str, results: List[Dict]):
        """Print summary for a single task."""
        num_success = sum(1 for r in results if r['success'])
        success_rate = num_success / len(results)
        
        avg_primitives = np.mean([r['num_primitives_executed'] for r in results])
        avg_steps = np.mean([r['total_steps'] for r in results])
        
        print(f"\nTask Summary: {task}")
        print(f"  Success rate: {success_rate:.1%} ({num_success}/{len(results)})")
        print(f"  Avg primitives: {avg_primitives:.1f}")
        print(f"  Avg steps: {avg_steps:.1f}")
    
    def _print_overall_summary(self):
        """Print overall batch summary."""
        if not self.results:
            return
        
        num_success = sum(1 for r in self.results if r['success'])
        success_rate = num_success / len(self.results)
        
        avg_primitives = np.mean([r['num_primitives_executed'] for r in self.results])
        avg_steps = np.mean([r['total_steps'] for r in self.results])
        avg_duration = np.mean([
            r['end_time'] - r['start_time'] for r in self.results
        ])
        
        print("\n" + "=" * 80)
        print("BATCH EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total episodes: {len(self.results)}")
        print(f"Overall success rate: {success_rate:.1%} ({num_success}/{len(self.results)})")
        print(f"Avg primitives per episode: {avg_primitives:.1f}")
        print(f"Avg steps per episode: {avg_steps:.1f}")
        print(f"Avg duration per episode: {avg_duration:.2f}s")
        print("=" * 80)
    
    def _compute_aggregate_results(self) -> Dict[str, Any]:
        """Compute aggregate statistics."""
        if not self.results:
            return {}
        
        return {
            'num_episodes': len(self.results),
            'num_success': sum(1 for r in self.results if r['success']),
            'success_rate': sum(1 for r in self.results if r['success']) / len(self.results),
            'avg_primitives': float(np.mean([r['num_primitives_executed'] for r in self.results])),
            'avg_steps': float(np.mean([r['total_steps'] for r in self.results])),
            'avg_duration': float(np.mean([r['end_time'] - r['start_time'] for r in self.results])),
            'avg_replans': float(np.mean([r['num_replans'] for r in self.results])),
            'avg_failures': float(np.mean([r['num_primitives_failed'] for r in self.results])),
            'results': self.results
        }
