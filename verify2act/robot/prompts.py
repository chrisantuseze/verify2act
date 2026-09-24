"""Prompt building for the real DOFBOT block tasks.

The sim ``PromptManager`` hard-codes nut-assembly wording and ``{"label", "id"}`` plan items in its user
messages. The robot plans are plain subtask strings that the Jetson's skill parser executes, so this
subclass overrides only the two user-message builders; system prompts still come from YAML and the
``VLMPlanner`` / ``BeamSearchPlanner`` code is unchanged.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from verify2act.pipeline.prompt_utils import PromptManager, _img_block, _text_block, format_openai

COLORS = ("red", "green", "blue", "yellow")
_C = "(red|green|blue|yellow)"

# The subtask vocabulary the Jetson skills execute (lang_color_grasp.py via parse_step()).
SUBTASK_PATTERNS = (
    re.compile(rf"^pick and place {_C} block into the bin$"),
    re.compile(rf"^pick {_C} block$"),
    re.compile(rf"^place {_C} block on {_C} block$"),
    re.compile(rf"^place {_C} block to the (left|right) of {_C} block$"),
)
DONE = "done"

SUBTASK_TEMPLATES = (
    "pick and place <c> block into the bin",
    "pick <c> block",
    "place <c> block on <b> block",
    "place <c> block to the left of <b> block",
    "place <c> block to the right of <b> block",
)


def step_text(step: Any) -> str:
    """Plan item (str, or dict from a VLM that ignored the format) -> normalised subtask string."""
    if isinstance(step, dict):
        step = step.get("label") or step.get("action") or step.get("id") or ""
    return " ".join(str(step).strip().lower().rstrip(".").split())


def is_valid_subtask(text: str) -> bool:
    return text == DONE or any(p.match(text) for p in SUBTASK_PATTERNS)


def expand_subtask_plan(plan: List[Any]) -> List[Tuple[str, str]]:
    """``BeamSearchPlanner`` plan_expander: one subtask = one imagination step, and the WM action text is
    the subtask itself. ``done`` is not imagined."""
    steps = [step_text(s) for s in plan]
    return [(s, s) for s in steps if s and s != DONE]


def _history_str(history: List[Any]) -> str:
    if not history:
        return "  (none — start of episode)"
    return "\n".join(f"  {i + 1}. {a.strip() if isinstance(a, str) else step_text(a)}"
                     for i, a in enumerate(history[-10:]))


class RobotPromptManager(PromptManager):
    """User messages in the robot's subtask vocabulary."""

    def build_propose_messages(
        self,
        current_image_np: np.ndarray,
        language_goal: str,
        history: List[str],
        obj_labels: List[str],
        horizon: int,
        use_examples: bool = True,
        num_candidates: int = 1,
    ) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = [self._system["propose"].message()]

        if num_candidates > 1:
            plan_req = (f"Propose exactly {num_candidates} distinct, diverse and plausible candidate plans "
                        f"(at most {horizon} subtasks each).")
            req_format = 'Respond with JSON only: {"plans": [["<subtask>", ...], ...]}'
        else:
            plan_req = f"Propose the subtask sequence (at most {horizon} subtasks)."
            req_format = 'Respond with JSON only: {"plan": ["<subtask>", ...]}'

        msgs.append(format_openai(role="user", content=[
            _text_block(f"### Goal (language instruction)\n{language_goal}"),
            _text_block("### Current state (robot camera)"),
            _img_block(current_image_np),
            _text_block(
                "### Executed subtasks (history)\n"
                "(entries marked [FAILED] did not complete; the block is still on the table and must be retried)\n"
                f"{_history_str(history)}\n\n"
                f"### Planning request\n{plan_req}\n"
                f"Blocks in the scene: {', '.join(obj_labels)}\n"
                f"Allowed subtasks (exact wording; <c>, <b> are block colours): {'; '.join(SUBTASK_TEMPLATES)}\n"
                'If the goal is already satisfied in the current image, return ["done"].\n\n'
                f"{req_format}"
            ),
        ]))
        return msgs

    def build_reflect_messages(
        self,
        current_image_np: np.ndarray,
        language_goal: str,
        history: List[str],
        obj_labels: List[str],
        full_plan: List[str],
        ctx: Dict[str, Any],
        use_examples: bool = True,
    ) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = [self._system["reflect"].message()]

        imagined = ctx["imagined_state"]
        try:
            import torch
            if isinstance(imagined, torch.Tensor):
                imagined = imagined.detach().cpu().numpy()
        except ImportError:
            pass

        plan_str = "\n".join(f"  step {i}: {step_text(a)}" for i, a in enumerate(full_plan))
        scores_str = ", ".join(f"step {i}: {s:.2f}" for i, (s, _) in enumerate(ctx["all_scores"])) or "(none)"
        failed_hl = ctx.get("failed_highlevel_action")

        msgs.append(format_openai(role="user", content=[
            _text_block("### 1. Task context"),
            _text_block(f"Goal (language instruction):\n{language_goal}"),
            _text_block("Current real state (robot camera):"),
            _img_block(current_image_np),
            _text_block(
                "### 2. Executed subtasks (history)\n"
                "(entries marked [FAILED] did not complete; the block is still on the table)\n"
                f"{_history_str(history)}"
            ),
            _text_block(f"### 3. Proposed plan\n{plan_str}"),
            _text_block(
                "### 4. Critic diagnosis\n"
                + (f"- Failed subtask: {failed_hl}\n" if failed_hl else "")
                + f"- Failed step index: {ctx['failed_step']}\n"
                + f"- Temporal-consistency scores: {scores_str}\n"
                + f"- Failure pattern: {ctx['failure_pattern']}\n"
            ),
            _text_block(f"### 5. World-model imagined scene after step {ctx['failed_step']}"),
            _img_block(np.array(imagined)),
            _text_block(
                "### 6. Replanning instruction\n"
                "Identify the root cause and output a revised subtask plan that avoids it.\n"
                f"Blocks in the scene: {', '.join(obj_labels)}\n"
                f"Allowed subtasks (exact wording): {'; '.join(SUBTASK_TEMPLATES)}\n\n"
                'Respond with JSON only: {"analysis": "one-sentence diagnosis", "revised_plan": ["<subtask>", ...]}'
            ),
        ]))
        return msgs
