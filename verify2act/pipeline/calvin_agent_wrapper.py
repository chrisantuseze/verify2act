"""CALVIN Agent Wrapper for Verify2Act.

This module implements the CustomModel interface expected by CALVIN's evaluation script.
It embeds the Verify2Act pipeline (VLM, Latent WM, Critic) as an "Obstacle-Resolving
Cognitive Layer" on top of a baseline low-level continuous policy.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch
from pathlib import Path

from calvin_agent.models.calvin_base_model import CalvinBaseModel
from verify2act.critic.inference import check_rollout_consistency
from verify2act.pipeline.inference import preprocess_image_for_critic
from calvin_agent.utils.utils import get_last_checkpoint
from calvin_agent.evaluation.utils import get_default_model_and_env

logger = logging.getLogger(__name__)


class MCILLowLevelPolicy:
    """Wrapper for the pre-trained CALVIN baseline policy (MCIL/HULC)."""
    
    def __init__(self, train_folder: str, dataset_path: str, device: torch.device):
        logger.info(f"Loading MCIL baseline policy from {train_folder}...")
        
        train_folder_path = Path(train_folder)
        checkpoint = get_last_checkpoint(train_folder_path)
        
        # Load the model using CALVIN's built-in loader
        self.model, _, _ = get_default_model_and_env(
            train_folder=train_folder,
            dataset_path=dataset_path,
            checkpoint=checkpoint,
            device_id=device.index if device.index is not None else 0
        )
        self.model.eval()
        logger.info("MCIL baseline policy loaded successfully.")
        
    def reset(self):
        self.model.reset()
        
    def step(self, obs: Dict[str, np.ndarray], text_instruction: str) -> np.ndarray:
        # Returns a 7-DoF continuous action
        action = self.model.step(obs, text_instruction)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        return action

    def propose_trajectory(self, obs: Dict[str, np.ndarray], text_instruction: str, steps: int = 10) -> List[np.ndarray]:
        """
        Propose a sequence of actions to achieve the instruction.
        Note: MCIL is autoregressive on observations. It's difficult to propose a true
        closed-loop trajectory without stepping the environment.
        For Verify2Act's imagination phase, we might just need the text_instruction,
        but we provide this interface for future action-conditioned world models.
        """
        return [self.step(obs, text_instruction) for _ in range(steps)]


class Verify2ActCalvinAgent(CalvinBaseModel):
    """Wrapper that integrates Verify2Act into CALVIN's evaluation loop."""

    def __init__(
        self,
        vlm_planner: Any,
        world_model: Any,
        critic: Any,
        device: torch.device,
        train_folder: str,
        dataset_path: str,
        theta_c: float = 0.7,
        max_replans: int = 2,
    ):
        self.vlm_planner = vlm_planner
        self.world_model = world_model
        self.critic = critic
        self.device = device
        self.theta_c = theta_c
        self.max_replans = max_replans
        
        # Load the actual MCIL pre-trained policy
        self.low_level_policy = MCILLowLevelPolicy(train_folder, dataset_path, device)
        
        self.current_subgoal: str = ""
        self.plan_queue: List[str] = []

    def reset(self):
        """Called at the beginning of each new evaluation sequence."""
        self.current_subgoal = ""
        self.plan_queue = []
        if hasattr(self.vlm_planner, "reset_history"):
            self.vlm_planner.reset_history()

    def step(self, obs: Dict[str, Any], goal: str) -> np.ndarray:
        """
        CALVIN's environment loop calls this function every step.
        
        Args:
            obs: dict with keys 'rgb_obs', 'depth_obs', 'robot_obs'
            goal: string containing the current natural language instruction
            
        Returns:
            action: 7-DoF continuous action array
        """
        # 1. Update the active goal
        if goal != self.current_subgoal:
            logger.info(f"New CALVIN Goal Received: {goal}")
            self.current_subgoal = goal
            self.plan_queue = [goal]  # Initially, the plan is just to execute the goal directly

        # 2. Pop the current sub-goal we are trying to achieve
        active_instruction = self.plan_queue[0] if self.plan_queue else self.current_subgoal

        # 3. We only run the Verify2Act imagination phase periodically or at the start of a new sub-goal.
        # For simplicity in this wrapper, let's assume we verify the first step of a new instruction.
        # In a full implementation, you might verify every N steps or when the policy's confidence drops.
        
        # FIXME: Add logic to decide WHEN to imagine and verify.
        # For now, we just pass the instruction to the low-level policy.
        action = self.low_level_policy.step(obs, active_instruction)
        
        # ---
        # Verify2Act Imagination & Reflection logic (The Cognitive Layer)
        # ---
        # Define a frequency to verify (e.g., step 0 and every 10 steps)
        # We assume `obs['rgb_obs']['rgb_static']` gives the numpy image.
        
        should_verify = getattr(self, "_step_count", 0) % 10 == 0
        self._step_count = getattr(self, "_step_count", 0) + 1

        if should_verify:
            # 1. Get current state image
            img_np = obs['rgb_obs']['rgb_static']
            
            # 2. Propose a trajectory from the low-level policy
            # Note: Dummy policy just returns 10 copies of the single action.
            # A real policy might autoregressively generate a sequence.
            traj = self.low_level_policy.propose_trajectory(obs, active_instruction)
            
            # 3. Imagine the outcome using the Latent WM
            # Our Latent WM expects an image and a text action.
            # In the future, we could condition on the continuous `traj` instead.
            imagined_img_next = self.world_model.imagine(img_np, active_instruction) 
            if isinstance(imagined_img_next, tuple):
                # Handle LatentWorldModel which returns (features, image) or similar
                imagined_img_next = imagined_img_next[1] if len(imagined_img_next) > 1 else imagined_img_next[0]
            
            # 4. Check temporal/physical consistency (Critic Head 2)
            img_224_prev = preprocess_image_for_critic(img_np).to(self.device)
            img_224_next = preprocess_image_for_critic(imagined_img_next).to(self.device)
            
            with torch.no_grad():
                emb_prev = self.critic.encode(img_224_prev)
                emb_next = self.critic.encode(img_224_next)
                mean_tc, std_tc = self.critic.temporal_sim_with_uncertainty(emb_prev, emb_next)
                
            decision = check_rollout_consistency(mean_tc.item(), self.theta_c, uncertainty=std_tc.item())
            
            # 5. Semantic Check (VLM Goal Verification)
            vlm_verification = {"achieved": True, "reason": "Not checked"}
            if decision.action != "reflect":
                # Only check semantics if physics didn't break
                vlm_verification = self.vlm_planner.verify_goal(imagined_img_next, active_instruction)
                if not vlm_verification["achieved"]:
                    decision.action = "reflect"
                    decision.reason = f"Goal not achieved: {vlm_verification.get('reason')}"

            # 6. Reflection Loop
            if decision.action == "reflect":
                # Trigger VLM to reflect on the failure and decompose the goal
                logger.warning(f"Rejected imagined outcome of '{active_instruction}'. Reason: {decision.reason}")
                
                # We need a reflection context similar to inference.py
                ctx = {
                    "all_scores": [(mean_tc.item(), 0.0)],
                    "imagined_state": imagined_img_next,
                    "failed_step": 0,
                    "failed_action": active_instruction,
                    "failed_highlevel_action": active_instruction,
                    "failure_pattern": decision.reason,
                }
                
                # Ask VLM to revise the plan
                reflect_result = self.vlm_planner.reflect(
                    current_image_np=img_np,
                    goal_image_np=img_np, # No goal image in CALVIN
                    history=self.plan_queue,
                    obj_labels=[], # Not strict for CALVIN, VLM can see image
                    full_plan=self.plan_queue,
                    ctx=ctx,
                    task_instruction=self.current_subgoal
                )
                
                new_plan = reflect_result.get("revised_plan", self.plan_queue)
                if new_plan and isinstance(new_plan, list) and len(new_plan) > 0:
                    self.plan_queue = new_plan
                    logger.info(f"VLM Proposed new decomposed plan: {self.plan_queue}")
                    active_instruction = self.plan_queue[0]
                    # Update action based on new active instruction
                    action = self.low_level_policy.step(obs, active_instruction)
        
        return action
