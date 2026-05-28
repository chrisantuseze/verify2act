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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
        # Allow mock mode for testing if MOCK_API_KEY is set
        if self._api_key is None:
            if os.environ.get("MOCK_API_KEY"):
                logger.warning("Using mock API key for testing purposes")
                self._api_key = "mock-key-for-testing"
            else:
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

    def _call(self, messages: List[Dict[str, Any]], temperature: Optional[float] = None) -> str:
        # Mock mode: return dummy JSON response
        if self._api_key == "mock-key-for-testing":
            logger.warning("Mock mode: returning dummy response for testing")
            # Return a dummy JSON response that matches expected format
            return '{"plan": ["mock_action_1", "mock_action_2"], "analysis": "Mock response for testing"}'
        
        # print("Calling GPT-4o with messages:", messages[0])
        temp = self._temperature if temperature is None else temperature
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=temp,
        )
        content = resp.choices[0].message.content.strip()
        # logger.debug("GPT-4o raw response: %s", content)
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
        language_goal: str,
        history: List[str],
        obj_labels: List[str],
        horizon: int = 4,
        use_examples: bool = True,
        temperature: Optional[float] = None,
        exclude_plans: Optional[List[List[str]]] = None,
    ) -> List[str]:
        """Return the nuts in assembly order (Option A: nut-name list, not sub-skills).

        Returns a list of nut-name strings, length <= ``horizon``.
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        if exclude_plans:
            exclude_strs = [str(p) for p in exclude_plans]
            exclude_instruction = (
                f"\n\nCRITICAL: Do NOT propose any of the following plan(s) as they have already "
                f"been tried and verified to be incorrect: {', '.join(exclude_strs)}. Propose "
                f"a different valid plan."
            )
            language_goal = language_goal + exclude_instruction

        messages = self._pm.build_propose_messages(
            current_image_np=current_image_np,
            language_goal=language_goal,
            history=history,
            obj_labels=obj_labels,
            horizon=horizon,
            use_examples=use_examples,
        )
        raw = self._call(messages, temperature=temperature)
        result = self._parse_json(raw)
        plan = result.get("plan", [])
        if not isinstance(plan, list) or not all(isinstance(a, (str, dict)) for a in plan):
            raise ValueError(f"Invalid 'plan' output from model: {plan}")
        if len(plan) > horizon:
            plan = plan[:horizon]
        logger.info("Proposed plan: %s", plan)
        return plan

    def propose_candidates(
        self,
        current_image_np: np.ndarray,
        language_goal: str,
        history: List[str],
        obj_labels: List[str],
        horizon: int = 4,
        use_examples: bool = True,
        num_candidates: int = 3,
        temperature: Optional[float] = None,
    ) -> List[List[str]]:
        """Propose multiple distinct and diverse candidate plans in a single VLM call.

        Returns a list of candidate plans, each being a list of nut-name strings.
        """
        if num_candidates <= 1:
            plan = self.propose(
                current_image_np=current_image_np,
                language_goal=language_goal,
                history=history,
                obj_labels=obj_labels,
                horizon=horizon,
                use_examples=use_examples,
                temperature=temperature,
            )
            return [plan]

        messages = self._pm.build_propose_messages(
            current_image_np=current_image_np,
            language_goal=language_goal,
            history=history,
            obj_labels=obj_labels,
            horizon=horizon,
            use_examples=use_examples,
            num_candidates=num_candidates,
        )
        raw = self._call(messages, temperature=temperature)
        result = self._parse_json(raw)
        plans = result.get("plans")

        if not plans and "plan" in result:
            plans = [result["plan"]]

        if not isinstance(plans, list) or not all(isinstance(p, list) for p in plans):
            raise ValueError(f"Invalid 'plans' output from model: {plans}")

        cleaned_plans = []
        for p in plans:
            if all(isinstance(step, (str, dict)) for step in p):
                if len(p) > horizon:
                    p = p[:horizon]
                cleaned_plans.append(p)

        if not cleaned_plans:
            raise ValueError(f"No valid candidate plans parsed from response: {result}")

        logger.info("Proposed candidate plans: %s", cleaned_plans)
        return cleaned_plans

    def reflect(
        self,
        current_image_np: np.ndarray,
        language_goal: str,
        history: List[str],
        obj_labels: List[str],
        full_plan: List[str],
        ctx: Dict[str, Any],
        use_examples: bool = True,
    ) -> Dict[str, Any]:
        """Diagnose a critic failure and return a revised plan.

        Returns ``{"analysis": str, "revised_plan": list[str]}``.
        """
        messages = self._pm.build_reflect_messages(
            current_image_np=current_image_np,
            language_goal=language_goal,
            history=history,
            obj_labels=obj_labels,
            full_plan=full_plan,
            ctx=ctx,
            use_examples=use_examples,
        )
        raw = self._call(messages)
        result = self._parse_json(raw)
        revised = result.get("revised_plan")
        if not isinstance(revised, list) or not all(isinstance(a, (str, dict)) for a in revised):
            raise ValueError(f"Invalid 'revised_plan' output from model: {revised}")
        if "analysis" not in result:
            result["analysis"] = ""
        logger.info("Reflect analysis: %s", result.get("analysis"))
        logger.info("Revised plan: %s", result.get("revised_plan"))
        return result

    def verify_goal(
        self,
        imagined_image_np: np.ndarray,
        text_goal: str,
    ) -> Dict[str, Any]:
        """Verify if a language goal is met in the imagined image.

        Returns ``{"achieved": bool, "reason": str}``.
        """
        messages = self._pm.build_verify_goal_messages(
            imagined_image_np=imagined_image_np,
            text_goal=text_goal,
        )
        raw = self._call(messages)
        result = self._parse_json(raw)
        
        achieved = result.get("achieved")
        if not isinstance(achieved, bool):
            raise ValueError(f"Invalid 'achieved' output from model (must be bool): {achieved}")
            
        logger.info("Goal Verification for '%s': Achieved=%s, Reason=%s", 
                    text_goal, achieved, result.get("reason", ""))
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
        plan_expander: Optional[Callable[[List[str]], List[Tuple[str, str]]]] = None,
        temporal_threshold: float = 0.5,
        max_retries: int = 2,
        max_replans: int = 3,
        wm_mode: str = "v2a_wm",
    ):
        self.vlm = vlm_planner
        self.world_model = world_model
        self.critic = critic
        self.beam_width = beam_width
        self.goal_threshold = goal_threshold
        self.temporal_threshold = temporal_threshold
        self.max_retries = max_retries
        self.max_replans = max_replans
        self.wm_mode = wm_mode

        if plan_expander is None:
            # Default to no subskill expansion (identity expander)
            self.plan_expander = lambda plan: [(step, step) for step in plan if step.strip().lower() != "done"]
        else:
            self.plan_expander = plan_expander

    def _evaluate_diffusion_trajectory(
        self,
        plan: List[str],
        current_image_np: np.ndarray,
        language_goal: str = "",
        timestep: int = 0,
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        replan_attempt: int = 0,
    ) -> Tuple[float, Any, List[Tuple[float, float]], List[Tuple[str, str]], bool, Optional[int], List[str]]:
        """Roll out the proposed plan to the end of the horizon sequentially without a critic (ReflectVLM)."""
        import pathlib
        imagination_steps = self.plan_expander(plan)
        logger.info("ReflectVLM (Diffusion mode): Rolling out plan to end of horizon: %s", imagination_steps)
        imagined_state = current_image_np
        final_state = current_image_np
        
        for k, (hl_action, imagine_action) in enumerate(imagination_steps):
            imagined_state_next = self.world_model.imagine(imagined_state, imagine_action)
            imagined_state = imagined_state_next
            final_state = imagined_state_next
            
            if output_dir:
                step_dir = pathlib.Path(output_dir) / "steps"
                step_dir.mkdir(parents=True, exist_ok=True)
                from PIL import Image
                Image.fromarray(imagined_state_next).save(
                    step_dir / f"step_{timestep:03d}_imagine_r{replan_attempt}_k{k}.png"
                )
        
        # Since there is no critic, we use dummy values for scores
        all_scores = [(1.0, 1.0)] * len(imagination_steps)
        critic_decisions = ["ReflectVLM (no critic check)"]
        
        # We always trigger reflection at the end of the horizon, so step_failed = True
        step_failed = True
        failed_step = len(imagination_steps) - 1 if imagination_steps else 0
        score = 1.0
        
        return (
            score,
            final_state,
            all_scores,
            imagination_steps,
            step_failed,
            failed_step,
            critic_decisions,
        )

    def _evaluate_trajectory(
        self,
        plan: List[str],
        current_image_np: np.ndarray,
        language_goal: str,
        timestep: int = 0,
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        replan_attempt: int = 0,
        decoder: Optional[torch.nn.Module] = None,
    ) -> Tuple[float, Any, List[Tuple[float, float]], List[Tuple[str, str]], bool, Optional[int], List[str]]:
        """
        Evaluates a single candidate plan against the critic.

        Latent WM path (LatentWorldModel / RLAWorldModel)
        --------------------------------------------------
        ``current_image_np`` is used exactly ONCE — to seed the initial rolling
        latent history before the step loop.  Inside the loop every ``imagine()``
        call receives ``None`` because the WM advances its own internal window
        automatically, exactly like how ``imagined_state`` is advanced explicitly
        for the RGB path.

        ``step_history`` is a *cloned* snapshot of the WM state taken BEFORE
        each step.  It is restored when a per-step requery retry is triggered
        so the WM re-samples from the same context (not from a half-advanced
        state).

        ``start_history`` is a *cloned* snapshot of the t=0 WM state.  It is
        restored at the start of each outer-loop attempt (HEAD1 re-rolls) and
        again at the very end so that candidate evaluations are isolated.

        RGB WM path (DiffusionWorldModel / OracleWorldModel)
        ------------------------------------------------------
        ``imagined_state`` tracks the current imagined RGB frame and is passed
        to every ``imagine()`` call, then updated with the returned next frame.

        Outer ``for attempt`` loop
        --------------------------
        Retries the FULL trajectory ONLY when HEAD1 (goal proximity) returns
        ``'requery'`` — the critic is uncertain, so we re-imagine the whole
        plan and re-evaluate.  It does NOT retry on HEAD2 (temporal consistency)
        step failures; those are already retried inline by the inner
        ``for retry_i`` loop.

        Returns
        -------
        score, final_state, all_scores, imagination_steps,
        step_failed, failed_step_index, critic_decisions
        """

        import torch
        from verify2act.pipeline.inference import preprocess_image_for_critic
        from verify2act.critic.inference import CriticDecision, check_rollout_consistency, decide_from_proximity
        from verify2act.pipeline.world_model import LatentWorldModel, OracleWorldModel
        from contextlib import nullcontext
        from verify2act.pipeline.inference import _save_image

        if self.wm_mode == "diffusion":
            return self._evaluate_diffusion_trajectory(
                plan=plan,
                current_image_np=current_image_np,
                language_goal=language_goal,
                timestep=timestep,
                output_dir=output_dir,
                replan_attempt=replan_attempt,
            )

        device = next(self.critic.parameters()).device
        is_latent_wm = isinstance(self.world_model, LatentWorldModel)

        start_history = None
        if is_latent_wm:
            self.world_model.initialize_history(current_image_np)
            start_history = self.world_model.get_history().clone()  # clone to avoid aliasing

        imagination_steps = self.plan_expander(plan)
        
        best_eval_score = -float('inf')
        best_eval_final_state = current_image_np
        best_eval_all_scores: List[Tuple[float, float]] = []
        best_eval_step_failed = True
        best_eval_failed_step = None
        best_eval_critic_decisions: List[str] = []

        for attempt in range(self.max_retries):
            # Restore the WM to the start-of-plan state so each outer-loop
            # attempt (HEAD1 re-roll) evaluates a fresh independent trajectory.
            if is_latent_wm and start_history is not None:
                self.world_model.set_history(start_history.clone())

            all_scores: List[Tuple[float, float]] = []
            critic_decisions: List[str] = []

            # Encode the initial real frame so the first HEAD2 check compares
            # t=0 → t=1 (rather than t=1 → t=2).
            with torch.no_grad():
                if is_latent_wm:
                    # The last slot in the history window is the current frame.
                    F_start = self.world_model.get_history()[:, -1]  # (1, 256, 768)
                    emb_prev = self.critic.encode_features(F_start)
                else:
                    cur_img_224 = preprocess_image_for_critic(current_image_np).to(device)
                    emb_prev = self.critic.encode(cur_img_224)

            # For the RGB path, track the evolving imagined frame explicitly.
            imagined_state = current_image_np
            final_state = current_image_np
            step_failed = False
            failed_step = None

            _wm_ctx = (
                self.world_model.rollout_context()
                if isinstance(self.world_model, OracleWorldModel)
                else nullcontext()
            )

            with _wm_ctx:
                for k, (hl_action, imagine_action) in enumerate(imagination_steps):

                    # Checkpoint the WM state BEFORE this step so per-step
                    # requery retries can rewind to the same context.
                    # For latent WM: clone avoids aliasing with subsequent
                    # set_history() calls that replace self._history.
                    step_history = self.world_model.get_history().clone() if is_latent_wm else None

                    # ── 1. Imagine the next state ──────────────────────────────
                    if is_latent_wm:
                        # history is already initialised; imagine() advances it
                        # internally — current_image_np is not used after init.
                        F_next, _ = self.world_model.imagine(None, imagine_action)
                        final_state = F_next
                    else:
                        # RGB path: chain imagined frames explicitly.
                        imagined_state_next = self.world_model.imagine(imagined_state, imagine_action)
                        final_state = imagined_state_next

                    if output_dir:
                        step_dir = pathlib.Path(output_dir) / "steps"
                        if not is_latent_wm:
                            _save_image(
                                imagined_state_next, step_dir,
                                f"step_{timestep:03d}_imagine_r{replan_attempt}_k{k}.png",
                            )
                        elif decoder is not None:
                            image_from_latent = self.decode_dino_features(final_state, decoder)
                            _save_image(
                                image_from_latent, step_dir,
                                f"step_{timestep:03d}_imagine_r{replan_attempt}_k{k}_latent.png",
                            )

                    # ── 2. HEAD2: temporal consistency check ───────────────────
                    with torch.no_grad():
                        if is_latent_wm:
                            emb_next = self.critic.encode_features(final_state)
                        else:
                            img_224 = preprocess_image_for_critic(final_state).to(device)
                            emb_next = self.critic.encode(img_224)

                        mean_tc, std_tc = self.critic.temporal_sim_with_uncertainty(emb_prev, emb_next)
                        tc_score = mean_tc.item()
                        tc_uncertainty = std_tc.item()

                    all_scores.append((tc_score, 0.0))

                    decision = check_rollout_consistency(tc_score, self.temporal_threshold, uncertainty=tc_uncertainty)
                    decision_msg = f"k={k} action='{imagine_action}' tc={tc_score:.3f}(unc={tc_uncertainty:.3f}) → {decision.action}"
                    critic_decisions.append(decision_msg)
                    logger.info("  " + decision_msg)

                    # ── 2a. Per-step requery: re-sample from the same context ──
                    # Re-imagines only this step — does NOT restart the whole plan.
                    if decision.action == "requery":
                        for retry_i in range(self.max_retries):
                            if isinstance(self.world_model, OracleWorldModel) and self.world_model._rollout_state is not None:
                                self.world_model.rollback_step()

                            if is_latent_wm:
                                # Rewind to the pre-step latent state so the WM
                                # samples a different transition from the same context.
                                if step_history is not None:
                                    self.world_model.set_history(step_history.clone())
                                F_next, _ = self.world_model.imagine(None, imagine_action)
                                final_state = F_next
                            else:
                                # imagined_state still holds the pre-step RGB frame
                                # (it is updated only at the bottom of this k-loop).
                                imagined_state_next = self.world_model.imagine(imagined_state, imagine_action)
                                final_state = imagined_state_next

                            if output_dir:
                                if not is_latent_wm:
                                    _save_image(
                                        imagined_state_next, step_dir,
                                        f"step_{timestep:03d}_imagine_r{replan_attempt}_k{k}_retry{retry_i}.png",
                                    )
                                elif decoder is not None:
                                    image_from_latent = self.decode_dino_features(final_state, decoder)
                                    _save_image(
                                        image_from_latent, step_dir,
                                        f"step_{timestep:03d}_imagine_r{replan_attempt}_k{k}_retry{retry_i}_latent.png",
                                    )

                            with torch.no_grad():
                                if is_latent_wm:
                                    emb_next = self.critic.encode_features(final_state)
                                else:
                                    img_224 = preprocess_image_for_critic(final_state).to(device)
                                    emb_next = self.critic.encode(img_224)

                                mean_tc, std_tc = self.critic.temporal_sim_with_uncertainty(emb_prev, emb_next)
                                tc_score = mean_tc.item()
                                tc_uncertainty = std_tc.item()

                            all_scores[-1] = (tc_score, 0.0)
                            decision = check_rollout_consistency(tc_score, self.temporal_threshold, uncertainty=tc_uncertainty)
                            logger.info(
                                f"    requery {retry_i + 1}/{self.max_retries}  tc={tc_score:.3f}(unc={tc_uncertainty:.3f})  → {decision.action}"
                            )
                            if decision.action != "requery":
                                break
                        else:
                            # All per-step retries exhausted — escalate to reflect.
                            decision = CriticDecision(action="reflect", reason="requery_exhausted")

                    # ── 2b. Step failure → abort this trajectory ───────────────
                    if decision.action == "reflect":
                        step_failed = True
                        failed_step = k
                        break

                    # Advance the embedding chain and (RGB path) the imagined frame.
                    emb_prev = emb_next
                    if not is_latent_wm:
                        imagined_state = imagined_state_next

            # ── 3. HEAD1: goal proximity gate ──────────────────────────────────
            if not step_failed:
                with torch.no_grad():
                    mean_prox, std_prox = self.critic.goal_sim_from_text_with_uncertainty(emb_prev, language_goal)
                    prox_score = mean_prox.item()
                    prox_uncertainty = std_prox.item()

                if all_scores:
                    last_tc = all_scores[-1][0]
                    all_scores[-1] = (last_tc, prox_score)

                prox_decision = decide_from_proximity(prox_score, self.goal_threshold, uncertainty=prox_uncertainty)
                decision_msg = f"HEAD1 proximity={prox_score:.3f}(unc={prox_uncertainty:.3f}) → {prox_decision.action}"
                critic_decisions.append(decision_msg)
                logger.info("  " + decision_msg)

                if prox_decision.action == "requery":
                    # Critic is uncertain — re-roll the entire imagined trajectory.
                    # This is the ONLY case where the outer loop keeps iterating.
                    logger.info("  HEAD1 uncertain; re-rolling full imagined trajectory...")
                    continue

                if prox_decision.action == "reflect":
                    step_failed = True
                    failed_step = len(imagination_steps) - 1

                best_eval_score = prox_score
                best_eval_final_state = final_state
                best_eval_all_scores = all_scores
                best_eval_step_failed = step_failed
                best_eval_failed_step = failed_step
                best_eval_critic_decisions = critic_decisions
                break

            else:
                # HEAD2 step failure after exhausting per-step retries.
                # Record best-so-far and stop — the inner loop already handled
                # retries; re-rolling the whole trajectory won't help here.
                best_eval_score = -float('inf')
                best_eval_final_state = final_state
                best_eval_all_scores = all_scores
                best_eval_step_failed = step_failed
                best_eval_failed_step = failed_step
                best_eval_critic_decisions = critic_decisions
                break

        # Restore the WM to the initial state so subsequent candidate evaluations
        # (and the next real timestep) start from a clean, uncontaminated slate.
        if is_latent_wm and start_history is not None:
            self.world_model.set_history(start_history)

        return (
            best_eval_score,
            best_eval_final_state,
            best_eval_all_scores,
            imagination_steps,
            best_eval_step_failed,
            best_eval_failed_step,
            best_eval_critic_decisions,
        )

    def _reflect_and_replan(
        self,
        current_image_np: np.ndarray,
        history: List[str],
        obj_labels: List[str],
        language_goal: str,
        best_plan: List[str],
        best_score: float,
        final_state: Any,
        all_scores: List[Tuple[float, float]],
        imagination_steps: List[Tuple[str, str]],
        failed_step: int,
        decoder: Optional[torch.nn.Module] = None,
        timestep: int = 0,
        output_dir: Optional[Union[str, pathlib.Path]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the reflection-replanning loop (up to self.max_replans times).
        """
        import torch
        from verify2act.pipeline.reflection import build_reflection_context
        from verify2act.pipeline.world_model import LatentWorldModel
        
        is_latent_wm = isinstance(self.world_model, LatentWorldModel)
        
        reflection_analyses = []
        plan_accepted = False
        replan_attempts = 0
        current_plan = list(best_plan)
        current_score = best_score
        current_final_state = final_state
        current_all_scores = all_scores
        current_imagination_steps = imagination_steps
        current_failed_step = failed_step
        
        reflection_critic_decisions = []
        
        for attempt in range(1, self.max_replans + 1):
            replan_attempts = attempt
            logger.info(f"Reflection attempt {attempt}/{self.max_replans} for plan: {current_plan}")
            
            # 1. Reconstruct the image if using LatentWorldModel
            if is_latent_wm and decoder is not None:
                imagined_img_next = self.decode_dino_features(current_final_state, decoder)
            else:
                imagined_img_next = current_final_state
                
            # 2. Build the reflection context dict.
            # Use the high-level nut-name label (first element of tuple), NOT the
            # expanded sub-skill string (second element).  The VLM reflect() prompt
            # needs to reason about nuts, not low-level sub-skill prompts.
            reflect_plan = [hl for hl, _ in current_imagination_steps]
 
            if current_failed_step is None or current_failed_step < 0 or current_failed_step >= len(reflect_plan):
                current_failed_step = max(0, len(reflect_plan) - 1)
                
            ctx = build_reflection_context(
                imagined_state=imagined_img_next,
                all_scores=current_all_scores,
                consistency_scores=[s for s, _ in current_all_scores],
                proximity_score=current_score if current_score != -float('inf') else None,
                failed_step=current_failed_step,
                full_plan=reflect_plan,
            )
            ctx["failed_highlevel_action"] = current_imagination_steps[current_failed_step][0] if current_imagination_steps else "none"
            
            if self.wm_mode == "diffusion":
                ctx["failure_pattern"] = "ReflectVLM evaluation: The action sequence was fully simulated. Please inspect the final imagined scene and verify if the goal is met or if the plan needs correction."
            
            # 3. Call the planner's reflect method to get a revised plan
            try:
                result = self.vlm.reflect(
                    current_image_np=current_image_np,
                    language_goal=language_goal,
                    history=history,
                    obj_labels=obj_labels,
                    full_plan=current_plan,
                    ctx=ctx,
                )
                revised_plan = result["revised_plan"]
                reflection_analyses.append(result.get("analysis", ""))
            except Exception as e:
                logger.error(f"VLM reflection failed: {e}")
                break
                            
            # 4. Evaluate the revised plan
            eval_score, eval_final_state, eval_all_scores, eval_imag_steps, eval_step_failed, eval_failed_step, eval_critic_decisions = self._evaluate_trajectory(
                plan=revised_plan,
                current_image_np=current_image_np,
                language_goal=language_goal,
                timestep=timestep,
                output_dir=output_dir,
                replan_attempt=attempt,
                decoder=decoder,
            )
            
            reflection_critic_decisions.extend(eval_critic_decisions)
            
            current_plan = revised_plan
            current_score = eval_score
            current_final_state = eval_final_state
            current_all_scores = eval_all_scores
            current_imagination_steps = eval_imag_steps
            current_failed_step = eval_failed_step
            
            if self.wm_mode == "diffusion":
                logger.info("ReflectVLM (Diffusion mode): Reflection completed. Accepting plan.")
                plan_accepted = True
                break
                
            if not eval_step_failed and eval_score >= self.goal_threshold:
                logger.info(f"Revised plan succeeded with score {eval_score:.3f} >= {self.goal_threshold:.3f}")
                plan_accepted = True
                break
                
        return {
            "plan": current_plan,
            "score": current_score,
            "final_state": current_final_state,
            "all_scores": current_all_scores,
            "imagination_steps": current_imagination_steps,
            "failed_step": current_failed_step,
            "replan_attempts": replan_attempts,
            "reflection_analyses": reflection_analyses,
            "critic_decisions": reflection_critic_decisions,
            "plan_accepted": plan_accepted,
        }

    def plan(
        self,
        current_image_np: np.ndarray,
        history: List[str],
        obj_labels: List[str],
        horizon: int,
        language_goal: str,
        timestep: int = 0,
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        decoder: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        Executes a trajectory search, evaluates candidates, selects the best plan,
        and coordinates reflection/replanning if needed.
        """
        import torch
        logger.info(f"BeamSearchPlanner: Sampling up to {self.beam_width} candidate plans...")
        
        # 1. Propose candidates using propose_candidates
        if self.wm_mode == "diffusion":
            logger.info("ReflectVLM (Diffusion mode): Proposing a single plan upfront...")
            try:
                plan = self.vlm.propose(
                    current_image_np=current_image_np,
                    language_goal=language_goal,
                    history=history,
                    obj_labels=obj_labels,
                    horizon=horizon,
                )
                candidate_plans = [plan]
            except Exception as e:
                logger.error(f"VLM propose failed in ReflectVLM: {e}")
                candidate_plans = [[]]
        else:
            try:
                candidate_plans = self.vlm.propose_candidates(
                    current_image_np=current_image_np,
                    language_goal=language_goal,
                    history=history,
                    obj_labels=obj_labels,
                    horizon=horizon,
                    num_candidates=self.beam_width,
                )
            except Exception as e:
                logger.warning(f"VLM candidate proposal failed: {e}. Falling back to default plan Propose.")
                try:
                    plan = self.vlm.propose(
                        current_image_np=current_image_np,
                        language_goal=language_goal,
                        history=history,
                        obj_labels=obj_labels,
                        horizon=horizon,
                    )
                    candidate_plans = [plan]
                except Exception as ex:
                    logger.error(f"VLM propose fallback failed: {ex}")
                    candidate_plans = [[]]

        logger.info(f"Generated {len(candidate_plans)} candidate plans.")

        best_plan = None
        best_score = -float('inf')
        best_final_state = None
        best_all_scores = []
        best_imagination_steps = []
        best_failed_step = None
        best_step_failed = True
        best_critic_decisions = []

        # 2. Evaluate each candidate
        for i, plan in enumerate(candidate_plans):
            if not plan:
                continue
            
            logger.info(f"Evaluating Candidate {i+1}/{len(candidate_plans)}: {plan}")
            score, final_state, all_scores, imag_steps, step_failed, failed_step, critic_decisions = self._evaluate_trajectory(
                plan=plan,
                current_image_np=current_image_np,
                language_goal=language_goal,
                timestep=timestep,
                output_dir=output_dir,
                replan_attempt=0,
                decoder=decoder,
            )
            
            is_better = False
            if best_plan is None:
                is_better = True
            elif not step_failed and best_step_failed:
                is_better = True
            elif step_failed and not best_step_failed:
                is_better = False
            else:
                is_better = score > best_score

            if is_better:
                best_plan = plan
                best_score = score
                best_final_state = final_state
                best_all_scores = all_scores
                best_imagination_steps = imag_steps
                best_failed_step = failed_step
                best_step_failed = step_failed
                best_critic_decisions = critic_decisions

            if not step_failed and score >= self.goal_threshold:
                logger.info(f"Candidate {i+1} reached goal threshold {self.goal_threshold:.3f}. Short-circuiting search.")
                break

        # Fallback if no plan was evaluated/chosen
        if best_plan is None:
            best_plan = candidate_plans[0] if candidate_plans else []
            best_score = -float('inf')
            best_final_state = current_image_np
            best_all_scores = []
            best_imagination_steps = self.plan_expander(best_plan)
            best_failed_step = 0
            best_step_failed = True
            best_critic_decisions = []

        logger.info(f"Selected Best Plan: {best_plan} (Score: {best_score:.3f}, Failed: {best_step_failed})")

        # 3. Handle Reflection and Replanning if best plan failed or is below threshold
        if best_step_failed or best_score < self.goal_threshold:
            logger.info("Best candidate score is below threshold or failed. Triggering reflect-replan loop...")
            reflection_result = self._reflect_and_replan(
                current_image_np=current_image_np,
                history=history,
                obj_labels=obj_labels,
                language_goal=language_goal,
                best_plan=best_plan,
                best_score=best_score,
                final_state=best_final_state,
                all_scores=best_all_scores,
                imagination_steps=best_imagination_steps,
                failed_step=best_failed_step,
                decoder=decoder,
                timestep=timestep,
                output_dir=output_dir,
            )
            return reflection_result
        
        # 4. If the best plan succeeded and met the threshold, return it directly
        return {
            "plan": best_plan,
            "score": best_score,
            "final_state": best_final_state,
            "all_scores": best_all_scores,
            "imagination_steps": best_imagination_steps,
            "failed_step": None,
            "replan_attempts": 0,
            "reflection_analyses": [],
            "critic_decisions": best_critic_decisions,
            "plan_accepted": True,
        }

    def decode_dino_features(
        self,
        dino_features: torch.Tensor,
        decoder: torch.nn.Module,
        as_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Decodes predicted DINO features back to an RGB image.
        
        Args:
            dino_features (torch.Tensor): Predicted DINO features tensor of shape
                (num_patches, dino_channels) or (1, num_patches, dino_channels).
            decoder (torch.nn.Module): An instance of FeatureDecoder.
            as_numpy (bool): If True, returns a uint8 numpy array of shape (H, W, 3) in [0, 255].
                If False, returns a float tensor of shape (1, 3, H, W) in [0, 1].
                
        Returns:
            Union[np.ndarray, torch.Tensor]: Decoded RGB image.
        """
        import torch
        device = next(self.critic.parameters()).device
        # Convert numpy array to torch tensor if necessary
        if isinstance(dino_features, np.ndarray):
            dino_features = torch.from_numpy(dino_features)
        # Ensure features have a batch dimension (B, num_patches, dino_channels)
        if dino_features.ndim == 2:
            dino_features = dino_features.unsqueeze(0)
            
        with torch.no_grad():
            # Forward pass through FeatureDecoder
            rec_img = decoder.decode(dino_features.to(device))  # (B, 3, H, W) in range [-1, 1]
            
            # Map from [-1, 1] to [0, 1] and clamp
            rec_img = (rec_img + 1.0) / 2.0
            rec_img = torch.clamp(rec_img, 0.0, 1.0)
            
            # Format output
            if as_numpy:
                rec_img = rec_img.squeeze(0).cpu().numpy()  # (3, H, W)
                rec_img = (rec_img * 255.0).astype(np.uint8)
                rgb_img = np.transpose(rec_img, (1, 2, 0))  # (H, W, 3)
                return rgb_img
            else:
                return rec_img
