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
