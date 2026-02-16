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

from llm_task_planner import LLMTaskPlanner
from dynamics_model_planner import DynamicsModelPlanner
from state_converter import StateConverter
from primitive_executor import PrimitiveExecutor
from predicate_registry import PREDICATE_NAMES, should_include_predicate
from dynamics_model_data_collector import DynamicsModelDataCollector


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
        lookahead_depth: int = 1,
        enable_collision_checking: bool = True,
        predicate_threshold: float = 0.3,
        enable_trajectory_tracking: bool = True,
        delta_forward: bool = True,
        latent_forward: bool = False,
        verbose: bool = True,
        task_type: str = "all"
    ):
        """
        Initialize closed-loop controller.
        
        Args:
            args: Command-line arguments or configuration
            env: Robosuite environment instance
            checkpoint_path: Path to trained dynamics model checkpoint
            lookahead_depth: Number of primitives to simulate ahead (1-3)
            enable_collision_checking: Whether to enable collision detection
            predicate_threshold: Threshold for predicate matching (default 0.3, use lower for undertrained models)
            enable_trajectory_tracking: Whether to track trajectory during planning
            delta_forward: Whether to use delta state forward model
            latent_forward: Whether to use latent state forward model
            verbose: Whether to print detailed logs
        """
        self.args = args
        self.env = env
        self.verbose = verbose
        self.predicate_threshold = predicate_threshold
        self.enable_trajectory_tracking = enable_trajectory_tracking
        self.task_type = task_type  # For predicate filtering
        
        # Initialize components
        if self.verbose:
            print("Initializing closed-loop controller components...")
        
        self.llm_planner = LLMTaskPlanner(args.model_config_path, args.prompt_config_path)
        self.state_converter = StateConverter(env)
        self.dynamics_planner = DynamicsModelPlanner(
            checkpoint_path=checkpoint_path,
            num_samples=self.args.num_planning_samples,
            state_converter=self.state_converter,
            lookahead_depth=lookahead_depth,
            enable_collision_checking=enable_collision_checking,
            predicate_threshold=predicate_threshold,
            delta_forward=delta_forward,
            latent_forward=latent_forward,
        )
        self.executor = PrimitiveExecutor(env)
        
        # Data collector (optional, set externally)
        self.data_collector: Optional[DynamicsModelDataCollector] = None
        
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
        
        # Convert goals to predicate tensor (will be set after getting goals)
        object_name_to_id = {name: i for i, name in enumerate(objects)}
        
        # goals, plans = self.llm_planner.generate_goals_and_plans(
        #     task_description=task_description,
        #     objects=objects,
        #     initial_predicates=initial_predicates
        # )

        goals = ['Stacked(cubeB, cubeA)', 'Stacked(cubeC, cubeB)']
        plans = ['Pick(cubeB, table)', 'Place(cubeB, cubeA)', 'Pick(cubeC, table)', 'Place(cubeC, cubeB)']
        
        if not goals or not plans:
            print("ERROR: LLM failed to generate goals/plans")
            self.stats['end_time'] = time.time()
            return False, self.stats
        
        if self.verbose:
            print(f"  ✓ Goals: {goals}")
            print(f"  ✓ Plan: {plans[0] if plans else 'No plan'}")
        
        goal_predicates = self.llm_planner.goals_to_predicates(
            goals=goals,
            object_name_to_id=object_name_to_id,
            num_objects=len(objects)
        )
        
        primitive_plan = plans  # Full plan from LLM
        max_primitives = len(primitive_plan)  # Limit to length of LLM plan for testing
        
        # Initialize data collector if enabled
        if self.data_collector is not None:
            self.data_collector.start_episode(
                goal_predicates=goal_predicates,
                primitive_plan=primitive_plan
            )
        
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
            
            # Check if plan is exhausted
            if not primitive_plan:
                if self.verbose:
                    print("✓ All primitives in plan executed")
                break
            
            # Check if goal already achieved #@Chris: Uncomment
            if self._check_goal_achieved(state_dict, goal_predicates):
                if self.verbose:
                    print("✓ Goal achieved!")
                if self.data_collector is not None:
                    self.data_collector.end_episode(success=True)
                self.stats['end_time'] = time.time()
                return True, self.stats
            
            # Plan next primitive with replanning on failure
            success = False
            for replan_attempt in range(self.args.max_replans_per_primitive):
                if replan_attempt > 0:
                    if self.verbose:
                        print(f"  Replanning (attempt {replan_attempt + 1}/{self.args.max_replans_per_primitive})...")
                    self.stats['num_replans'] += 1
                
                # Plan with trajectory tracking for failure analysis
                primitive, action_params, feasibility, failure_analysis = self.dynamics_planner.plan_next_primitive(
                    state_dict=state_dict,
                    goal_predicates=goal_predicates,
                    primitive_plan=primitive_plan,
                    enable_trajectory_tracking=self.enable_trajectory_tracking
                )
                
                if self.verbose:
                    print(f"  Primitive: {primitive}")
                    print(f"  Action params: {action_params}")
                    print(f"  Feasibility: {feasibility:.3f}")
                
                # Check both: primitive must exist AND feasibility must meet threshold
                if primitive is None or feasibility < self.dynamics_planner.feasibility_threshold:
                    if self.verbose:
                        if primitive is None:
                            print("  ✗ Planning failed (no action found)")
                        else:
                            print(f"  ✗ Planning failed (feasibility {feasibility:.3f} < threshold {self.dynamics_planner.feasibility_threshold})")
                    
                    # Log failure analysis if available
                    if failure_analysis is not None:
                        self._log_failure_analysis(failure_analysis)

                        # Trigger LLM reflection to suggest revised plans
                        try:
                            revised_plans, suggestions = self.llm_planner.reflect_on_failure(
                                primitive_plan=primitive_plan,
                                task_goal=goals,
                                failure_info=failure_analysis['reflection_info'],
                                task_description=task_description,
                                objects=objects,
                                initial_predicates=initial_predicates,
                            )

                            if suggestions and self.verbose:
                                print("  LLM Suggestions:")
                                for s in suggestions[:3]:
                                    print(f"    - {s}")

                            if revised_plans:
                                # Use first revised plan candidate
                                new_plan = revised_plans[0]
                                primitive_plan = new_plan
                                if self.verbose:
                                    print(f"  → Updated plan from LLM reflection: {primitive_plan}")
                                # Count this as a replan attempt
                                self.stats['num_replans'] += 1

                        except Exception as e:
                            if self.verbose:
                                print(f"  Exception during LLM reflection: {e}")
                    continue
                
                if self.verbose:
                    print(f"  → Planned: {primitive} (feasibility={feasibility:.3f})")
                
                # Record step for data collection (if enabled)
                if self.data_collector is not None and primitive is not None:
                    # Parse primitive to get obj_id and target_id
                    obj_id, target_id = self._parse_primitive_for_collection(
                        primitive, state_dict
                    )
                    
                    # Record the step (before execution)
                    self.data_collector.record_step(
                        step_idx=prim_idx,
                        state_dict=state_dict,
                        action_params=action_params,
                        obj_id=obj_id,
                        target_id=target_id,
                        next_state_dict=None,  # Will be filled after execution
                        feasibility_score=feasibility,
                    )
                
                # Get current observation for execution
                obs = self.env._get_observations()
                
                # Execute primitive
                exec_success, num_steps = self._execute_primitive_with_monitoring(
                    primitive, action_params, obs
                )

                # Update collected step with observed next state (if enabled)
                if self.data_collector is not None and primitive is not None:
                    next_state_dict = self.state_converter.convert()
                    next_state_dict['object_names'] = objects
                    self.data_collector.update_last_step_next_state(
                        next_state_dict=next_state_dict,
                        execution_success=exec_success,
                        num_steps=num_steps,
                    )
                
                self.stats['total_steps'] += num_steps
                self.stats['num_primitives_executed'] += 1
                self.stats['primitive_history'].append(primitive)
                self.stats['feasibility_history'].append(feasibility)
                
                if exec_success:
                    if self.verbose:
                        print(f"  ✓ Execution succeeded ({num_steps} steps)")
                    success = True
                    # Advance to next primitive in plan
                    primitive_plan = primitive_plan[1:]
                    if self.verbose and primitive_plan:
                        print(f"  Remaining plan: {primitive_plan}")
                    break
                else:
                    if self.verbose:
                        print(f"  ✗ Execution failed ({num_steps} steps)")
                    self.stats['num_primitives_failed'] += 1
            
            if not success:
                if self.verbose:
                    print(f"  ✗ Failed after {self.args.max_replans_per_primitive} replan attempts")
                    print(f"  → Aborting episode (sequential dependency broken)")
                break  # Abort episode - subsequent primitives depend on this one
        
        # Step 4: Final goal check
        if self.verbose:
            print("\n[4/4] Final goal check...")
        
        state_dict = self.state_converter.convert()
        state_dict['object_names'] = objects
        
        success = self._check_goal_achieved(state_dict, goal_predicates)
        
        # End data collection episode (if enabled)
        if self.data_collector is not None:
            failure_step = None
            failure_type = "predicate"
            
            if not success:
                # Determine failure step
                failure_step = self.stats['num_primitives_executed'] - 1
                failure_step = max(0, failure_step)
                
                # Determine failure type
                if self.stats.get('collision_detected', False):
                    failure_type = "feasibility"
                else:
                    failure_type = "predicate"
            
            self.data_collector.end_episode(
                success=success,
                failure_step=failure_step,
                failure_type=failure_type
            )
        
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
        # Use decoder to predict current predicates from state #@Chris: Uncomment
        current_predicates = self.dynamics_planner.predict_predicates(state_dict, debug=True)

        # current_predicates = np.random.rand(len(objects), len(objects), 9)  # Placeholder random predicates for testing
        if self.verbose:
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
        
        Assumes all non-table/non-peg/non-bin objects start on table.
        """
        predicates = []
        for obj in objects:
            obj_lower = obj.lower()
            # Skip static objects (table, pegs, bins)
            if 'table' in obj_lower or 'peg' in obj_lower or 'bin' in obj_lower:
                continue
            predicates.append(f"On({obj}, table)")
        
        if self.verbose:
            print(f"  Using fallback predicates: {predicates}")
        
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
        # Define predicate names matching the training data format from Points2Plans
        # Order from get_predicates() in dataloader.py:
        # Index 0: Left, 1: Right, 2: Below, 3: Above, 4: Front, 5: Behind, 6: Contact, 7: Boundary, 8: Inside
        # Use centralized registry
        predicate_names = PREDICATE_NAMES
        
        predicate_strings = []
        num_objects = len(objects)
        
        for i in range(num_objects):
            for j in range(num_objects):
                if i == j:
                    continue
                
                # Check each predicate type
                for pred_idx in range(min(len(predicate_names), predicate_matrix.shape[2])):
                    pred_name = predicate_names[pred_idx]
                    if pred_name is None:
                        continue
                    
                    # Apply task-specific filtering to reduce noise
                    if not should_include_predicate(pred_idx, objects[i], objects[j], self.task_type):
                        continue
                    
                    confidence = predicate_matrix[i, j, pred_idx]
                    # Print predicates we care about (Above at index 3, On at index 6)
                    if self.verbose and pred_idx in [2, 3, 6]:  # Print Below, Above, and On
                        print(f"Predicate {pred_name}({objects[i]}, {objects[j]}) confidence: {confidence:.3f}")
                    
                    if confidence > threshold:
                        obj_i_lower = objects[i].lower()
                        obj_j_lower = objects[j].lower()
                        
                        # Filter out nonsensical directional predicates
                        # Skip "Below" if object i is a static/support surface (table, peg)
                        if pred_name == 'Below':
                            if 'table' in obj_i_lower or 'peg' in obj_i_lower:
                                continue  # Skip Below(table, X) or Below(peg, X)
                        
                        # Filter out nonsensical "Above" predicates
                        if pred_name == 'Above':
                            if 'table' in obj_j_lower:
                                continue  # Skip Above(X, table) - redundant with On(X, table)
                        
                        # Handle On predicate with physical constraint:
                        # On(A, B) is only valid if A is also Above B (physical consistency)
                        if pred_name == 'On':
                            # Skip nonsensical On relationships (support surface "on" object)
                            if 'table' in obj_i_lower or 'peg' in obj_i_lower:
                                continue  # Skip On(table, X) or On(peg, X)
                            
                            # Check physical constraint: A can only be "on" B if A is above B
                            above_confidence = predicate_matrix[i, j, 3]  # Above is at index 3
                            if above_confidence <= threshold:
                                # Skip On if not Above - prevents On(table, cube) and other invalid configs
                                if self.verbose and confidence > 0.4:  # Only warn for strong On predictions
                                    print(f"  Skipping On({objects[i]}, {objects[j]}): On={confidence:.3f} but Above={above_confidence:.3f} (below threshold)")
                                continue
                            
                            # Valid On predicate - check if this is stacking (nut on nut)
                            if 'nut' in obj_i_lower and 'nut' in obj_j_lower:
                                # Nut on nut = stacking
                                pred_str = f"Stacked({objects[i]}, {objects[j]})"
                            else:
                                pred_str = f"On({objects[i]}, {objects[j]})"
                        elif pred_name == 'Above' and 'nut' in obj_j_lower:
                            # Above(nut, nut) indicates potential stacking
                            pred_str = f"{pred_name}({objects[i]}, {objects[j]})"
                        else:
                            pred_str = f"{pred_name}({objects[i]}, {objects[j]})"
                        
                        predicate_strings.append(pred_str)
        
        return predicate_strings
    
    def _check_goal_achieved(
        self,
        state_dict: Dict[str, Any],
        goal_predicates: np.ndarray
    ) -> bool:
        """
        Check if current state satisfies goal predicates.
        
        Uses dynamics model's decoder to predict current predicates,
        then compares with goal predicates.
        """
        
        # Get current predicates from dynamics model
        current_predicates = self.dynamics_planner.predict_predicates(state_dict) #@Chris: Uncomment
        # current_predicates = np.random.rand(4, 4, 9)  # Placeholder random predicates for testing - This is the cause of the occassional goal reached even when no manipulation has happened
        
        if current_predicates is None:
            return False
        
        # Only compare predicates that are set in goals (where goal_predicates > 0)
        # This handles dimension mismatch and focuses on relevant predicates
        goal_mask = goal_predicates > self.args.goal_threshold
        
        if goal_mask.sum() == 0:
            # No goals set, consider achieved
            return True
        
        # Extract only the goal dimensions we care about
        # Ensure current_predicates matches goal_predicates shape
        min_pred_dim = min(current_predicates.shape[2], goal_predicates.shape[2])
        current_subset = current_predicates[:, :, :min_pred_dim]
        goal_subset = goal_predicates[:, :, :min_pred_dim]
        
        # Compare only where goals are set
        goal_mask_subset = goal_subset > self.args.goal_threshold
        matches = np.logical_and(
            goal_mask_subset,
            current_subset > self.args.goal_threshold
        )
        
        # Success if most goals are satisfied
        satisfaction_rate = matches.sum() / goal_mask_subset.sum()
        
        if self.verbose:
            print(f"  Goal satisfaction: {satisfaction_rate:.2%} (threshold={self.args.goal_threshold:.2%})")
        
        return satisfaction_rate >= self.args.goal_threshold
    
    def _log_failure_analysis(self, failure_analysis: Dict[str, Any]):
        """
        Log failure analysis for debugging and future LLM reflection.
        
        Args:
            failure_analysis: Failure analysis dict from dynamics planner
        """
        if 'reflection_info' not in failure_analysis:
            return
        
        reflection_info = failure_analysis['reflection_info']
        
        if self.verbose:
            print("\n  ┌─────────────────────────────────────────────────────")
            print("  │ FAILURE ANALYSIS (for future LLM reflection)")
            print("  ├─────────────────────────────────────────────────────")
            
            if reflection_info['failed_step_number'] is not None:
                print(f"  │ Failed Step: {reflection_info['failed_step_number']} - {reflection_info['failed_primitive']}")
            else:
                print(f"  │ Failed Step: Unknown")
            
            print(f"  │ Failure Rate: {reflection_info['failure_rate']:.1%} ({reflection_info['failure_count_at_step']}/{reflection_info['total_samples']} samples)")
            print(f"  │ Best Score: {reflection_info['best_score_achieved']:.3f}")
            
            if reflection_info['top_failure_reasons']:
                print(f"  │ Top Reasons:")
                for reason, count in reflection_info['top_failure_reasons'][:3]:
                    print(f"  │   - {reason}: {count} occurrences")
            
            if reflection_info['suggestions']:
                print(f"  │ Suggestions:")
                for suggestion in reflection_info['suggestions'][:2]:
                    print(f"  │   → {suggestion}")
            
            print("  └─────────────────────────────────────────────────────\n")
        
        # Store in stats for later analysis
        if 'failure_analyses' not in self.stats:
            self.stats['failure_analyses'] = []
        self.stats['failure_analyses'].append(reflection_info)
    
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
    def _parse_primitive_for_collection(
        self,
        primitive: str,
        state_dict: Dict
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse primitive string to extract object IDs for data collection.
        
        Args:
            primitive: Primitive string like "Pick(cubeA, table)" or "Place(cubeA, cubeB)"
            state_dict: State dictionary with object_names
        
        Returns:
            (obj_id, target_id) tuple
        """
        try:
            if '(' not in primitive or ')' not in primitive:
                return None, None
            
            # Extract content between parentheses
            content = primitive.split('(')[1].split(')')[0]
            parts = [p.strip() for p in content.split(',')]
            
            if len(parts) != 2:
                return None, None
            
            obj_name, target_name = parts
            
            # Get object names from state_dict
            object_names = state_dict.get('object_names', [])
            
            # Find object IDs
            obj_id = None
            target_id = None
            
            for idx, name in enumerate(object_names):
                if name.lower() == obj_name.lower():
                    obj_id = idx
                if name.lower() == target_name.lower():
                    target_id = idx
            
            return obj_id, target_id
        
        except Exception as e:
            print(f"Warning: Failed to parse primitive '{primitive}': {e}")
            return None, None