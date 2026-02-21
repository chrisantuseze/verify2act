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
        self.state_converter = StateConverter(
            env,
            training_compatible_one_hot=getattr(args, 'training_compatible_one_hot', False),
            one_hot_seed=getattr(args, 'one_hot_seed', None),
        )
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
            print(f"\n{'=' * 80}")
            print(f"STARTING EPISODE: {task_description}")
            print(f"{'=' * 80}")
        
        # Step 1: Observe initial scene
        obs = self.env.reset()
        state_dict = self.state_converter.convert()
        objects = state_dict['object_names']
        
        if self.verbose:
            print(f"\n[1/4] Detected {len(objects)} objects: {objects}")
        
        if initial_predicates is None:
            initial_predicates = self._generate_initial_predicates(state_dict, objects)
        if self.verbose:
            print(f"  Initial predicates: {initial_predicates}")
        
        # Step 2: Generate goals and plan from LLM (once per episode)
        if self.verbose:
            print("\n[2/4] Generating goals from LLM...")
        
        object_name_to_id = {name: i for i, name in enumerate(objects)}
        goals, plans = self.llm_planner.generate_goals_and_plans(
            task_description=task_description,
            objects=objects,
            initial_predicates=initial_predicates,
        )
        goal_predicates, primitive_plan, valid_llm_plan = self._setup_llm_plan(
            goals, plans, object_name_to_id, objects,
        )
        collector_started = self._start_data_collection(goal_predicates, primitive_plan) # start critic data collection episode if collector is attached

        goal_achieved = False
        execution_succeeded = True
        plan_succeeded = False
        failure_type = "predicate"
        failed_step_idx = None

        if not valid_llm_plan:
            self._record_planning_failure_step(0, state_dict, 0.0) # save a synthetic failure step for critic training
        else:
            # Step 3: Execute plan in closed loop
            max_primitives = len(primitive_plan)
            if self.verbose:
                print(f"\n[3/4] Executing plan ({max_primitives} primitives)...")
            
            for prim_idx in range(max_primitives):
                if self.verbose:
                    print(f"\n{'─' * 80}")
                    print(f"PRIMITIVE {prim_idx + 1}/{max_primitives}")
                    print(f"{'─' * 80}")
                
                state_dict = self.state_converter.convert()
                
                if not primitive_plan:
                    if self.verbose:
                        print("✓ All primitives in plan executed")
                    break
                
                if self._check_goal_achieved(state_dict, goal_predicates): # this should be replaced with the call to the critic model
                    if self.verbose:
                        print("✓ Goal achieved!")
                    goal_achieved = True
                    break
                
                # Attempt planning + execution with replanning on failure
                result_dict = self._attempt_primitive_with_replans(
                            prim_idx, state_dict, goal_predicates, primitive_plan,
                            goals, task_description, objects, initial_predicates,
                )
                plan_success, exec_success, feasibility, primitive_plan = result_dict['plan_success'], result_dict['exec_success'], result_dict['last_feasibility'], result_dict['updated_primitive_plan']

                plan_succeeded = plan_success
                execution_succeeded = exec_success
                
                if not exec_success: # planning (using RD Model) or execution failed after all replanning attempts
                    print(f"Execution of primitive {prim_idx + 1} failed ({plan_success}) after replanning attempts (feasibility={feasibility:.3f})")
                    failed_step_idx = prim_idx
                    if not plan_success:
                        failure_type = "feasibility"
                        self._record_planning_failure_step(
                            prim_idx, state_dict, feasibility,
                        )
                    else:
                        failure_type = "predicate"
                    if self.verbose:
                        print("  ✗ Aborting episode (sequential dependency broken)")
                    break
        
        # Step 4: Finalize episode — goal check, data collection, summary
        success = self._finalize_episode(
            goal_achieved, plan_succeeded, execution_succeeded,
            failure_type, failed_step_idx, goal_predicates, objects,
            collector_started, goals
        )
        return success, self.stats

    def _setup_llm_plan(
        self,
        goals: Optional[List[str]],
        plans: Optional[List[str]],
        object_name_to_id: Dict[str, int],
        objects: List[str],
    ) -> Tuple[np.ndarray, List[str], bool]:
        """
        Process LLM output into goal predicates and a primitive plan.
        
        Returns:
            (goal_predicates, primitive_plan, planning_succeeded)
        """
        if not goals or not plans:
            print("ERROR: LLM failed to generate goals/plans")
            fallback = np.zeros(
                (len(objects), len(objects), len(PREDICATE_NAMES)),
                dtype=np.float32,
            )
            return fallback, [], False
        
        if self.verbose:
            print(f"  ✓ Goals: {goals}")
            print(f"  ✓ Plan: {plans}")
        
        goal_predicates = self.llm_planner.goals_to_predicates(
            goals=goals,
            object_name_to_id=object_name_to_id,
            num_objects=len(objects),
        )
        return goal_predicates, list(plans), True
    
    def _start_data_collection(
        self,
        goal_predicates: np.ndarray,
        primitive_plan: List[str],
    ) -> bool:
        """Start data collection episode if collector is attached."""
        if self.data_collector is None:
            return False
        self.data_collector.start_episode(
            goal_predicates=goal_predicates,
            primitive_plan=primitive_plan,
        )
        return True
    
    def _attempt_primitive_with_replans(
        self,
        prim_idx: int,
        state_dict: Dict[str, Any],
        goal_predicates: np.ndarray,
        primitive_plan: List[str],
        goals: List[str],
        task_description: str,
        objects: List[str],
        initial_predicates: List[str],
    ) -> Tuple[bool, bool, float, List[str]]:
        """
        Attempt planning and execution of a single primitive with replanning.
        
        Tries up to max_replans_per_primitive attempts. Each attempt plans
        a new action via rejection sampling and executes it if feasible.
        
        Args:
            prim_idx: Index of current primitive in the episode
            state_dict: Current scene state
            goal_predicates: Target goal predicate tensor
            primitive_plan: Remaining plan primitives
            goals: Goal predicate strings (for LLM reflection)
            task_description: Task description (for LLM reflection)
            objects: Object name list
            initial_predicates: Initial predicate strings (for LLM reflection)
        
        Returns:
            (exec_success, planning_failed, last_feasibility, updated_primitive_plan)
        """
        planning_success = False
        last_feasibility = 0.0
        
        for replan_attempt in range(self.args.max_replans_per_primitive):
            if replan_attempt > 0:
                if self.verbose:
                    print(f"  Replanning (attempt {replan_attempt + 1}"
                          f"/{self.args.max_replans_per_primitive})...")
                self.stats['num_replans'] += 1
            
            primitive, action_params, feasibility, failure_analysis = (
                self.dynamics_planner.plan_next_primitive(
                    state_dict=state_dict,
                    goal_predicates=goal_predicates,
                    primitive_plan=primitive_plan,
                    enable_trajectory_tracking=self.enable_trajectory_tracking,
                )
            )
            last_feasibility = feasibility
            
            if self.verbose:
                print(f"  Primitive: {primitive}")
                print(f"  Action params: {action_params}")
                print(f"  Feasibility: {feasibility:.3f}")

            # Planning failed: no action found or below feasibility threshold
            if primitive is None or feasibility < self.dynamics_planner.feasibility_threshold:
                if self.verbose:
                    reason = ("no action found" if primitive is None
                              else f"feasibility {feasibility:.3f} < "
                                   f"threshold {self.dynamics_planner.feasibility_threshold}")
                    print(f"  ✗ Planning failed ({reason})")
                
                if failure_analysis is not None: # Checks if reflection was enabled
                    self._log_failure_analysis(failure_analysis)
                    primitive_plan = self._try_llm_reflection(
                        primitive_plan, goals, failure_analysis,
                        task_description, objects, initial_predicates,
                    )
                    print(f"  Updated primitive plan after LLM reflection: {primitive_plan}")
                continue
            
            # Planning succeeded
            planning_success = True
            if self.verbose:
                print(f"  → Planned: {primitive} (feasibility={feasibility:.3f})")
            
            # Record step, execute, and update data collection
            exec_success, num_steps = self._execute_and_record(
                prim_idx, primitive, action_params, feasibility,
                state_dict, objects,
            )
            
            self.stats['total_steps'] += num_steps
            self.stats['num_primitives_executed'] += 1
            self.stats['primitive_history'].append(primitive)
            self.stats['feasibility_history'].append(feasibility)

            print(f"  Execution success: {exec_success} (steps taken: {num_steps})")
            
            if exec_success:
                if self.verbose:
                    print(f"  ✓ Execution succeeded ({num_steps} steps)")
                primitive_plan = primitive_plan[1:]
                if self.verbose and primitive_plan:
                    print(f"  Remaining plan: {primitive_plan}")
                return {
                    'plan_success': True,
                    'exec_success': True,
                    'last_feasibility': feasibility,
                    'updated_primitive_plan': primitive_plan,
                }
            else:
                if self.verbose:
                    print(f"  ✗ Execution failed ({num_steps} steps)")
                self.stats['num_primitives_failed'] += 1
                # Refresh state before the next replan attempt so the planner
                # reasons from the actual post-failure scene, not the pre-execution one.
                state_dict = self.state_converter.convert()
        
        # All replan attempts exhausted
        return {
            'plan_success': planning_success,
            'exec_success': False,
            'last_feasibility': last_feasibility,
            'updated_primitive_plan': primitive_plan,
        }
    
    def _execute_and_record(
        self,
        prim_idx: int,
        primitive: str,
        action_params: Any,
        feasibility: float,
        state_dict: Dict[str, Any],
        objects: List[str],
    ) -> Tuple[bool, int]:
        """Record step for data collection, execute primitive, and update with result."""
        # Record step before execution
        if self.data_collector is not None:
            obj_id, target_id = self.dynamics_planner.retrieve_ids_from_primitive(primitive, state_dict)

            self.data_collector.record_step(
                step_idx=prim_idx,
                state_dict=state_dict,
                action_params=action_params,
                obj_id=obj_id,
                target_id=target_id,
                next_state_dict=None,
                feasibility_score=feasibility,
            )
        
        # Execute primitive
        exec_success = False
        num_steps = 0
        try:
            obs = self.env._get_observations()
            exec_success, num_steps = self._execute_primitive_with_monitoring(
                primitive, action_params, obs,
            )
        finally:
            # Always update the step record so the dataset never contains a
            # dangling entry with next_state_dict=None, even on exception.
            if self.data_collector is not None:
                next_state_dict = self.state_converter.convert()
                next_state_dict['object_names'] = objects
                self.data_collector.update_last_step_next_state(
                    next_state_dict=next_state_dict,
                    execution_success=exec_success,
                    num_steps=num_steps,
                )
        
        return exec_success, num_steps
    
    def _try_llm_reflection(
        self,
        primitive_plan: List[str],
        goals: List[str],
        failure_analysis: Dict[str, Any],
        task_description: str,
        objects: List[str],
        initial_predicates: List[str],
    ) -> List[str]:
        """Attempt LLM reflection on planning failure and return (possibly updated) plan."""
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
                primitive_plan = revised_plans[0]
                if self.verbose:
                    print(f"  → Updated plan from LLM reflection: {primitive_plan}")
                self.stats['num_replans'] += 1
        except Exception as e:
            if self.verbose:
                print(f"  Exception during LLM reflection: {e}")
        
        return primitive_plan
    
    def _record_planning_failure_step(
        self,
        step_idx: int,
        state_dict: Dict[str, Any],
        feasibility: float = 0.0,
    ) -> None:
        """
        Record a synthetic step when planning fails before execution.
        
        Ensures failed episodes with no executed steps still appear in the
        critic dataset as negative samples. Uses the current state as both
        z_t and z_next, with a zero action vector.
        
        Args:
            step_idx: Primitive index where planning failed
            state_dict: Current scene state (used as both z_t and z_next)
            feasibility: Feasibility score that caused rejection
        """
        if self.data_collector is None:
            return
        
        self.data_collector.record_step(
            step_idx=step_idx,
            state_dict=state_dict,
            action_params=np.zeros(3, dtype=np.float32),
            obj_id=None,
            target_id=None,
            next_state_dict=state_dict,
            feasibility_score=feasibility,
        )
        self.data_collector.update_last_step_next_state(
            next_state_dict=state_dict,
            execution_success=False,
            num_steps=0,
        )
    
    def _finalize_episode(
        self,
        goal_achieved: bool,
        planning_succeeded: bool,
        execution_succeeded: bool,
        failure_type: str,
        failed_step_idx: Optional[int],
        goal_predicates: np.ndarray,
        objects: List[str],
        collector_started: bool,
        goals: List[str],
    ) -> bool:
        """
        Final goal check, data collection finalization, and episode summary.
        
        Success requires all three conditions:
            planning succeeded AND execution succeeded AND goal achieved.
        
        Data is always saved at episode end regardless of outcome.
        Only labeled positive when both planning and execution succeed
        AND the goal is achieved.
        
        Args:
            goal_achieved: Whether goal was detected during execution loop
            planning_succeeded: Whether all planning attempts succeeded
            execution_succeeded: Whether all executions succeeded
            failure_type: Type of failure for critic labeling
            failed_step_idx: Explicit index of the primitive that failed
            goal_predicates: Target goal predicate tensor
            objects: Object name list
            collector_started: Whether data collection was started
        
        Returns:
            success: Whether the episode succeeded
        """
        if self.verbose:
            print("\n[4/4] Final goal check...")
        
        # Check goal if not already achieved during execution
        if not goal_achieved and planning_succeeded:
            state_dict = self.state_converter.convert()
            state_dict['object_names'] = objects
            goal_achieved = self._check_goal_achieved(state_dict, goal_predicates, goals)
        
        success = planning_succeeded and execution_succeeded and goal_achieved
        print(f"  Episode success: {success} (Goal achieved: {goal_achieved}, Planning succeeded: {planning_succeeded}, Execution succeeded: {execution_succeeded})")
        
        # Always finalize data collection — save regardless of outcome
        if self.data_collector is not None and collector_started:
            failure_step = None
            if not success:
                # Use tracked failure index; fall back to last executed step
                if failed_step_idx is not None:
                    failure_step = failed_step_idx
                else:
                    failure_step = max(0, self.stats['num_primitives_executed'] - 1)
            
            self.data_collector.end_episode(
                success=success,
                failure_step=failure_step,
                failure_type=failure_type,
            )
        
        self.stats['end_time'] = time.time()
        
        if self.verbose:
            self._print_episode_summary(success)
        
        return success
    
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
        current_predicates = self.dynamics_planner.predict_predicates(state_dict, debug=False)

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
    
    def _format_predicate(
        self,
        pred_name: str,
        obj_i: str,
        obj_j: str,
    ) -> Optional[str]:
        """Return a predicate string for (pred_name, obj_i, obj_j), or *None* to skip.

        Applies semantic sanity filters:
        - ``Below(support, X)`` and ``On(support, X)`` are nonsensical for tables/pegs.
        - ``Above(X, table)`` is redundant with ``On(X, table)``.
        - Nut-on-nut stacking is labelled ``Stacked`` instead of ``On``.
        """
        li, lj = obj_i.lower(), obj_j.lower()
        is_support_i = 'table' in li or 'peg' in li

        if pred_name == 'Below' and is_support_i:
            return None                                 # Below(table/peg, X) makes no sense
        if pred_name == 'Above' and 'table' in lj:
            return None                                 # Above(X, table) is redundant
        if pred_name == 'On':
            if is_support_i:
                return None                             # On(table/peg, X) makes no sense
            if 'nut' in li and 'nut' in lj:
                return f"Stacked({obj_i}, {obj_j})"    # nut-on-nut → Stacked
        return f"{pred_name}({obj_i}, {obj_j})"

    def _predicates_to_strings(
        self,
        predicate_matrix: np.ndarray,
        objects: List[str],
        threshold: float = 0.5
    ) -> List[str]:
        """Convert a predicate confidence matrix to human-readable strings for the LLM.

        Args:
            predicate_matrix: ``[num_objects, num_objects, num_predicates]`` array.
            objects: Object name list.
            threshold: Minimum confidence to include a predicate (default 0.5).

        Returns:
            Predicate strings like ``["On(cubeA, table)", "Stacked(cubeA, cubeB)"]``.
        """
        predicate_strings = []
        num_preds = min(len(PREDICATE_NAMES), predicate_matrix.shape[2])

        for i, obj_i in enumerate(objects):
            for j, obj_j in enumerate(objects):
                if i == j:
                    continue
                for pred_idx in range(num_preds):
                    pred_name = PREDICATE_NAMES[pred_idx]
                    if pred_name is None:
                        continue
                    if not should_include_predicate(pred_idx, obj_i, obj_j, self.task_type):
                        continue

                    confidence = predicate_matrix[i, j, pred_idx]
                    if self.verbose and pred_idx in (2, 3, 6):   # Below, Above, On
                        print(f"Predicate {pred_name}({obj_i}, {obj_j}) confidence: {confidence:.3f}")

                    if confidence > threshold:
                        pred_str = self._format_predicate(pred_name, obj_i, obj_j)
                        if pred_str is not None:
                            predicate_strings.append(pred_str)

        return predicate_strings

    def _check_goal_achieved(
        self,
        state_dict: Dict[str, Any],
        goal_predicates: np.ndarray,
        goals: Optional[List[str]] = None
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
        
        # For debugging
        predicate_strings = self._predicates_to_strings(
            current_predicates, 
            state_dict['object_names'],
            threshold=self.predicate_threshold
        )
        print(f"  Current predicates: {predicate_strings}") # I got: ['On(cubeA, table)', 'On(cubeB, table)', 'On(cubeC, table)']
        print(f"  Goal predicates: {goals if goals else 'N/A'}") # It is: ['Stacked(cubeB, cubeA)', 'Stacked(cubeC, cubeB)']
        
        # Only compare predicates that are set in goals (where goal_predicates > 0)
        # This handles dimension mismatch and focuses on relevant predicates
        goal_mask = goal_predicates > self.predicate_threshold
        
        if goal_mask.sum() == 0:
            # No goals set, consider achieved
            return True
        
        # Extract only the goal dimensions we care about
        # Ensure current_predicates matches goal_predicates shape
        min_pred_dim = min(current_predicates.shape[2], goal_predicates.shape[2])
        current_subset = current_predicates[:, :, :min_pred_dim]
        goal_subset = goal_predicates[:, :, :min_pred_dim]
        
        # A goal entry is satisfied when the decoder predicts it above threshold.
        # All goal entries must be satisfied (not a fraction).
        goal_mask_subset = goal_subset > self.predicate_threshold
        matches = np.logical_and(
            goal_mask_subset,
            current_subset > self.predicate_threshold
        )

        num_goals = int(goal_mask_subset.sum())
        num_satisfied = int(matches.sum())
        satisfaction_rate = num_satisfied / num_goals

        if self.verbose:
            print(f"  Goal satisfaction: {num_satisfied}/{num_goals} ({satisfaction_rate:.1%})")

        return num_satisfied == num_goals

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