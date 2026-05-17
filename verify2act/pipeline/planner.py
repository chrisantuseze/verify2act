"""VLM Planner — wraps GPT-4o for propose and reflect calls.

Usage::

    planner = VLMPlanner.from_yaml("configs/prompts/planner.yaml")
    plan = planner.propose(current_img, goal_img, history, obj_labels, horizon=5)
    # plan: ["pick round nut", "insert round nut", ...]

    revised = planner.reflect(current_img, goal_img, history, obj_labels, old_plan, ctx)
    # revised: {"analysis": "...", "revised_plan": [...]}
"""

from __future__ import annotations

import os
import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openai
import httpx

from verify2act.pipeline.prompt_utils import PromptManager

logger = logging.getLogger(__name__)


class VLMPlanner:
    """GPT-4o based planner with YAML-driven prompt templates."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        model: str = "gpt-4o",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self._pm = prompt_manager
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

        self._api_key = os.environ.get("OPENAI_API_KEY")
        if self._api_key is None:
            raise ValueError("API key must be provided.")

        self._client = openai.OpenAI(api_key=self._api_key, http_client=httpx.Client())  # uses OPENAI_API_KEY env var

    @classmethod
    def from_yaml(
        cls,
        prompt_config: Union[str, pathlib.Path],
        model: str = "gpt-4o",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> "VLMPlanner":
        pm = PromptManager.from_yaml(prompt_config)
        return cls(pm, model=model, max_tokens=max_tokens, temperature=temperature)

    # -- internal -----------------------------------------------------------

    def _call(self, messages: List[Dict[str, Any]]) -> str:
        print("Calling GPT-4o with messages:", messages[0])
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        content = resp.choices[0].message.content.strip()
        logger.debug("GPT-4o raw response: %s", content)
        return content

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract JSON from a response that might contain extra text/fences."""
        text = raw.strip()
        if not text:
            raise ValueError(f"Empty response from model, cannot parse JSON: {raw!r}")

        if text.startswith("```"):
            # strip ```json ... ``` fences
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: extract the first JSON object or array substring.
            match = re.search(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", text)
            if match is None:
                raise ValueError(f"Failed to parse JSON from model response: {text!r}")
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"Extracted JSON substring is invalid: {match.group(0)!r}") from e

    # -- public API ---------------------------------------------------------

    def propose(
        self,
        current_image_np: np.ndarray,
        goal_image_np: np.ndarray,
        history: List[str],
        obj_labels: List[str],
        horizon: int = 4,
        task_instruction: str = "Assemble the target nuts onto their matching pegs.",
        use_examples: bool = True,
    ) -> List[str]:
        """Return the nuts in assembly order (Option A: nut-name list, not sub-skills).

        Returns a list of nut-name strings, length <= ``horizon``.
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        messages = self._pm.build_propose_messages(
            current_image_np=current_image_np,
            goal_image_np=goal_image_np,
            history=history,
            obj_labels=obj_labels,
            horizon=horizon,
            task_instruction=task_instruction,
            use_examples=use_examples,
        )
        raw = self._call(messages)
        result = self._parse_json(raw)
        plan = result["plan"]
        if not isinstance(plan, list) or not all(isinstance(a, str) for a in plan):
            raise ValueError(f"Invalid 'plan' output from model: {plan}")
        if len(plan) > horizon:
            plan = plan[:horizon]
        logger.info("Proposed plan: %s", plan)
        return plan

    def reflect(
        self,
        current_image_np: np.ndarray,
        goal_image_np: np.ndarray,
        history: List[str],
        obj_labels: List[str],
        full_plan: List[str],
        ctx: Dict[str, Any],
        task_instruction: str = "Assemble the target nuts onto their matching pegs.",
        use_examples: bool = True,
    ) -> Dict[str, Any]:
        """Diagnose a critic failure and return a revised plan.

        Returns ``{"analysis": str, "revised_plan": list[str]}``.
        """
        messages = self._pm.build_reflect_messages(
            current_image_np=current_image_np,
            goal_image_np=goal_image_np,
            history=history,
            obj_labels=obj_labels,
            full_plan=full_plan,
            ctx=ctx,
            task_instruction=task_instruction,
            use_examples=use_examples,
        )
        raw = self._call(messages)
        result = self._parse_json(raw)
        revised = result.get("revised_plan")
        if not isinstance(revised, list) or not all(isinstance(a, str) for a in revised):
            raise ValueError(f"Invalid 'revised_plan' output from model: {revised}")
        if "analysis" not in result:
            result["analysis"] = ""
        logger.info("Reflect analysis: %s", result.get("analysis"))
        logger.info("Revised plan: %s", result.get("revised_plan"))
        return result


class BeamSearchPlanner:
    """
    Coordinates the VLM, World Model, and Critic for multi-step planning
    using a beam search / trajectory sampling approach.
    """
    def __init__(
        self,
        vlm_planner: VLMPlanner,
        world_model,
        critic,
        beam_width: int = 3,
        goal_threshold: float = 0.85,
    ):
        self.vlm = vlm_planner
        self.world_model = world_model
        self.critic = critic
        self.beam_width = beam_width
        self.goal_threshold = goal_threshold

    def _sample_candidate_plans(
        self,
        current_image_np: np.ndarray,
        goal_image_np: np.ndarray,
        history: List[str],
        obj_labels: List[str],
        horizon: int,
        task_instruction: str,
        num_candidates: int
    ) -> List[List[str]]:
        """Sample multiple distinct plans from the VLM."""
        candidates = []
        attempts = 0
        max_attempts = num_candidates * 2
        
        while len(candidates) < num_candidates and attempts < max_attempts:
            try:
                # Use the VLM to propose a plan
                plan = self.vlm.propose(
                    current_image_np=current_image_np,
                    goal_image_np=goal_image_np,
                    history=history,
                    obj_labels=obj_labels,
                    horizon=horizon,
                    task_instruction=task_instruction,
                )
                if plan and plan not in candidates:
                    candidates.append(plan)
            except Exception as e:
                logger.warning(f"VLM proposal failed: {e}")
            attempts += 1
            
        if not candidates:
            # Fallback to an empty plan if VLM completely fails
            candidates.append([])
        return candidates

    def plan(
        self,
        current_image_np: np.ndarray,
        goal_image_np: np.ndarray,
        history: List[str],
        obj_labels: List[str],
        horizon: int,
        task_instruction: str,
    ) -> Tuple[List[str], float, Any, List[Tuple[float, float]], List[Tuple[str, str]]]:
        """
        Executes a trajectory search.
        
        Returns
        -------
        best_plan : List[str]
        best_score : float
        final_state : Any (Latent tensor or RGB image)
        all_scores : List[Tuple[float, float]] (Mocked/captured scores for backward compat)
        imagination_steps : List[Tuple[str, str]]
        """
        import torch
        from verify2act.pipeline.inference import preprocess_image_for_critic
        from verify2act.pipeline.decompose import expand_nut_plan
        from verify2act.pipeline.world_model import LatentWorldModel

        device = next(self.critic.parameters()).device
        is_latent_wm = isinstance(self.world_model, LatentWorldModel)

        logger.info(f"BeamSearchPlanner: Sampling up to {self.beam_width} candidate plans...")
        candidate_plans = self._sample_candidate_plans(
            current_image_np, goal_image_np, history, obj_labels, horizon, task_instruction, self.beam_width
        )
        
        logger.info(f"Generated {len(candidate_plans)} distinct candidate plans.")
        
        goal_img_224 = preprocess_image_for_critic(goal_image_np).to(device)
        with torch.no_grad():
            emb_goal = self.critic.encode(goal_img_224)

        best_plan = candidate_plans[0]
        best_score = -float('inf')
        best_final_state = current_image_np
        best_all_scores = []
        best_imagination_steps = []

        for i, plan in enumerate(candidate_plans):
            if not plan:
                continue
                
            logger.info(f"Evaluating Candidate {i+1}: {plan}")
            
            if is_latent_wm:
                self.world_model.initialize_history(current_image_np)
                
            imagination_steps = expand_nut_plan(plan)
            final_latent = None
            final_img = current_image_np
            
            for k, (nut_name, imagine_action) in enumerate(imagination_steps):
                if is_latent_wm:
                    F_next, _ = self.world_model.imagine(current_image_np, imagine_action)
                    final_latent = F_next
                else:
                    final_img = self.world_model.imagine(final_img, imagine_action)
            
            with torch.no_grad():
                if is_latent_wm and final_latent is not None:
                    emb_final = self.critic.encode_features(final_latent)
                    final_state = final_latent
                else:
                    img_224 = preprocess_image_for_critic(final_img).to(device)
                    emb_final = self.critic.encode(img_224)
                    final_state = final_img
                    
                mean_prox, std_prox = self.critic.goal_sim_with_uncertainty(emb_final, emb_goal)
                score = mean_prox.item()
                # Mock temporal consistency as 1.0 for the search phase, just taking terminal proximity
                all_scores = [(1.0, 0.0)] * (len(imagination_steps) - 1) + [(1.0, score)]
                
            logger.info(f"  -> Terminal Goal Score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_plan = plan
                best_final_state = final_state
                best_all_scores = all_scores
                best_imagination_steps = imagination_steps
                
            if score >= self.goal_threshold:
                logger.info("  -> Goal Threshold reached! Short-circuiting search.")
                break
                
        logger.info(f"BeamSearchPlanner Selected Plan: {best_plan} (Score: {best_score:.3f})")
        return best_plan, best_score, best_final_state, best_all_scores, best_imagination_steps
