"""
Integration script for the Verify2Act critic with Points2Plans dynamics model.
This shows how to plug the critic into the imagination loop.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from .critic_config import CriticConfig
from .critic_model import CriticEnsemble
from .critic_inference import CriticInference, TrajectoryDiagnostics


class VerifiedPlanner:
    """
    Integrates LLM planner, dynamics model, critic, and executor.
    
    Pipeline:
        1. LLM generates primitive plan
        2. For each candidate plan:
            a. Dynamics model imagines rollout
            b. Critic verifies each step
            c. Track failures and diagnostics
        3. If all candidates fail, generate reflection prompt
        4. Otherwise, select best candidate and execute
    """
    
    def __init__(
        self,
        llm_planner,
        dynamics_model,
        critic_model: torch.nn.Module,
        critic_config: CriticConfig,
        executor,
        device: str = "cuda",
    ):
        self.llm_planner = llm_planner
        self.dynamics_model = dynamics_model
        self.executor = executor
        self.device = device
        
        # Critic inference engine
        self.critic = CriticInference(critic_model, critic_config, device)
        
        # Configuration
        self.config = critic_config
        self.max_reflection_iterations = 3
        self.num_candidate_samples = 10  # Number of rollouts per plan
    
    def plan_and_verify(
        self,
        initial_state: Dict,
        goal_description: str,
        scene_context: Dict,
    ) -> Tuple[List[str], bool, Optional[str]]:
        """
        Generate and verify a plan using the critic.
        
        Args:
            initial_state: Initial robot/scene state
            goal_description: Natural language goal
            scene_context: Scene information (objects, positions, etc.)
        
        Returns:
            (primitive_plan, is_verified, reflection_prompt)
        """
        reflection_context = None
        
        for iteration in range(self.max_reflection_iterations):
            print(f"\n{'='*80}")
            print(f"Planning iteration {iteration + 1}/{self.max_reflection_iterations}")
            print(f"{'='*80}")
            
            # 1. Generate primitive plan from LLM
            if iteration == 0:
                primitive_plan = self.llm_planner.generate_plan(
                    initial_state=initial_state,
                    goal=goal_description,
                    scene=scene_context,
                )
            else:
                # Reflection iteration - use previous failure context
                primitive_plan = self.llm_planner.replan_with_reflection(
                    initial_state=initial_state,
                    goal=goal_description,
                    scene=scene_context,
                    reflection_prompt=reflection_context,
                )
            
            print(f"\nGenerated plan: {primitive_plan}")
            
            # 2. Verify plan with dynamics model + critic
            all_trajectories, should_reflect = self.verify_plan_with_critic(
                initial_state=initial_state,
                primitive_plan=primitive_plan,
                scene_context=scene_context,
            )
            
            # 3. Check if verification passed
            if not should_reflect:
                print("\n✓ Plan verified successfully!")
                return primitive_plan, True, None
            
            # 4. Generate reflection prompt for next iteration
            print("\n✗ Plan verification failed")
            failure_analysis = self.critic.aggregate_failure_analysis(all_trajectories)
            reflection_context = self.critic.generate_reflection_prompt(
                primitive_plan=primitive_plan,
                failure_analysis=failure_analysis,
                trajectory_diagnostics=all_trajectories[0] if all_trajectories else None,
            )
            
            print(f"\nReflection prompt:")
            print(reflection_context)
        
        # Failed after max iterations
        print(f"\n✗ Failed to find verified plan after {self.max_reflection_iterations} iterations")
        return primitive_plan, False, reflection_context
    
    def verify_plan_with_critic(
        self,
        initial_state: Dict,
        primitive_plan: List[str],
        scene_context: Dict,
    ) -> Tuple[List[TrajectoryDiagnostics], bool]:
        """
        Verify a plan by sampling multiple rollouts and using the critic.
        
        Args:
            initial_state: Initial state
            primitive_plan: List of primitive actions
            scene_context: Scene context
        
        Returns:
            (list_of_trajectory_diagnostics, should_reflect)
        """
        print(f"\nVerifying plan with {self.num_candidate_samples} sampled trajectories...")
        
        all_trajectory_diagnostics = []
        num_failures = 0
        
        for sample_idx in range(self.num_candidate_samples):
            # Imagine trajectory using dynamics model
            trajectory_data = self.imagine_trajectory(
                initial_state=initial_state,
                primitive_plan=primitive_plan,
                scene_context=scene_context,
                sample_idx=sample_idx,
            )
            
            # Evaluate with critic
            traj_diag = self.critic.evaluate_trajectory(trajectory_data)
            all_trajectory_diagnostics.append(traj_diag)
            
            if traj_diag.should_reflect:
                num_failures += 1
            
            # Print progress
            status = "✗ FAIL" if traj_diag.should_reflect else "✓ PASS"
            print(f"  Sample {sample_idx + 1}/{self.num_candidate_samples}: {status} "
                  f"(score: {traj_diag.terminal_score:.3f})")
            
            if traj_diag.should_reflect:
                print(f"    Failure at step {traj_diag.first_failure_step}: "
                      f"{traj_diag.failure_reason.value}")
        
        # Decision: reflect if majority of samples failed
        failure_rate = num_failures / self.num_candidate_samples
        should_reflect = failure_rate > 0.5  # Threshold: 50% failure rate
        
        print(f"\nVerification summary:")
        print(f"  Failures: {num_failures}/{self.num_candidate_samples} ({failure_rate*100:.1f}%)")
        print(f"  Decision: {'REFLECT' if should_reflect else 'PROCEED'}")
        
        return all_trajectory_diagnostics, should_reflect
    
    def imagine_trajectory(
        self,
        initial_state: Dict,
        primitive_plan: List[str],
        scene_context: Dict,
        sample_idx: int = 0,
    ) -> List[Dict]:
        """
        Use dynamics model to imagine a trajectory.
        
        Args:
            initial_state: Initial state
            primitive_plan: List of primitives
            scene_context: Scene context
            sample_idx: Sample index (for diversity)
        
        Returns:
            List of step dicts for critic evaluation
        """
        trajectory_data = []
        
        # Get initial latent state
        z_t = self.dynamics_model.encode_state(initial_state)
        
        for step_idx, primitive in enumerate(primitive_plan):
            # Parse primitive to get action
            action_dict = self.parse_primitive(primitive, scene_context)
            a_t = self.dynamics_model.encode_action(action_dict)
            
            # Predict next state (with sampling for diversity)
            z_next = self.dynamics_model.predict_next_state(
                z_t, a_t, sample=True, temperature=0.1 * (sample_idx + 1)
            )
            
            # Get target predicate for this step
            target_predicate = self.extract_target_predicate(primitive)
            predicate_embed = self.embed_predicate(target_predicate)
            
            # Get remaining plan summary
            remaining_plan = primitive_plan[step_idx + 1:]
            plan_summary = self.embed_plan_summary(remaining_plan)
            
            # Store step data
            step_data = {
                "z_t": z_t.cpu(),
                "a_t": a_t.cpu(),
                "z_next": z_next.cpu(),
                "predicate_embed": predicate_embed.cpu(),
                "plan_summary": plan_summary.cpu(),
                "target_predicate": target_predicate,
                "predicted_predicates": self.decode_predicates(z_next),
            }
            
            trajectory_data.append(step_data)
            
            # Advance state
            z_t = z_next
        
        return trajectory_data
    
    def parse_primitive(self, primitive: str, scene_context: Dict) -> Dict:
        """Parse primitive string to action dictionary."""
        # Example: "pickplace(cup, table)" -> {obj: cup, target: table}
        # This should be implemented based on your primitive format
        parts = primitive.replace(")", "").split("(")
        action_type = parts[0]
        args = parts[1].split(",") if len(parts) > 1 else []
        
        return {
            "action_type": action_type.strip(),
            "args": [arg.strip() for arg in args],
        }
    
    def extract_target_predicate(self, primitive: str) -> str:
        """Extract target predicate from primitive."""
        # Example: "pickplace(cup, table)" -> "ON(cup, table)"
        action_dict = self.parse_primitive(primitive, {})
        
        if action_dict["action_type"] == "pickplace":
            obj, target = action_dict["args"]
            return f"ON({obj}, {target})"
        elif action_dict["action_type"] == "pick":
            obj = action_dict["args"][0]
            return f"HOLDING({obj})"
        else:
            return primitive
    
    def embed_predicate(self, predicate: str) -> torch.Tensor:
        """Embed predicate string to vector."""
        # TODO: Implement predicate embedding
        # For now, return random embedding
        return torch.randn(self.config.model.predicate_embed_dim)
    
    def embed_plan_summary(self, remaining_plan: List[str]) -> torch.Tensor:
        """Embed remaining plan to summary vector."""
        # TODO: Implement plan summary embedding
        # For now, return random embedding
        return torch.randn(self.config.model.plan_summary_dim)
    
    def decode_predicates(self, z: torch.Tensor) -> Dict:
        """Decode predicates from latent state."""
        # TODO: Implement predicate decoder
        # For now, return placeholder
        return {"predicates": []}
    
    def execute_verified_plan(
        self,
        primitive_plan: List[str],
        initial_state: Dict,
    ) -> bool:
        """
        Execute a verified plan in the real environment.
        
        Args:
            primitive_plan: Verified primitive plan
            initial_state: Initial state
        
        Returns:
            success (bool)
        """
        print(f"\n{'='*80}")
        print("EXECUTING VERIFIED PLAN")
        print(f"{'='*80}")
        
        for step_idx, primitive in enumerate(primitive_plan):
            print(f"\nStep {step_idx + 1}/{len(primitive_plan)}: {primitive}")
            
            # Execute primitive
            success = self.executor.execute_primitive(primitive)
            
            if not success:
                print(f"✗ Execution failed at step {step_idx + 1}")
                return False
            
            print(f"✓ Step {step_idx + 1} completed")
        
        print(f"\n✓ Plan execution completed successfully!")
        return True


# Example usage
if __name__ == "__main__":
    print("Example usage of VerifiedPlanner:")
    print("\n1. Initialize components:")
    print("   - LLM planner")
    print("   - Points2Plans dynamics model")
    print("   - Critic model (trained)")
    print("   - Executor")
    print("\n2. Create VerifiedPlanner")
    print("\n3. Call plan_and_verify():")
    print("   - Generates plan from LLM")
    print("   - Imagines rollouts with dynamics model")
    print("   - Verifies with critic")
    print("   - Reflects if needed")
    print("\n4. Execute verified plan")
    
    # Pseudo-code example:
    """
    # Load critic
    critic_model = CriticEnsemble(config.model)
    critic_model.load_state_dict(torch.load("best_model.pt"))
    
    # Create planner
    planner = VerifiedPlanner(
        llm_planner=llm,
        dynamics_model=points2plans,
        critic_model=critic_model,
        critic_config=config,
        executor=robot_executor,
    )
    
    # Plan and verify
    plan, verified, _ = planner.plan_and_verify(
        initial_state=env.get_state(),
        goal_description="Stack cup on plate",
        scene_context=env.get_scene(),
    )
    
    # Execute if verified
    if verified:
        success = planner.execute_verified_plan(plan, initial_state)
    """
