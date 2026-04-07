"""Prompt utilities for the Verify2Act VLM planner.

Follows the Points2Plans pattern: YAML-driven system prompts, few-shot
examples serialised as named messages, and a BehaviorPromptManager that
assembles complete OpenAI Chat Completion message lists.

Key difference from Points2Plans: GPT-4o receives images as base64
``image_url`` content blocks, so the ``propose`` and ``reflect`` user
messages are built as lists of content items (text + images) rather
than plain strings.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import pathlib
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import yaml
from PIL import Image


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    """Normalize an image array to uint8 HxWx3 for PIL encoding."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got shape={arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected channel-last RGB image, got shape={arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0)
        maxv = float(arr.max()) if arr.size else 0.0
        minv = float(arr.min()) if arr.size else 0.0
        if minv >= 0.0 and maxv <= 1.0:
            arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def img_np_to_b64(img_np: np.ndarray, size: int = 512) -> str:
    """Encode an image array to a PNG base64 data-URL."""
    pil = Image.fromarray(_to_uint8_image(img_np)).resize((size, size))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _img_block(img_np: np.ndarray) -> dict:
    """Return an OpenAI ``image_url`` content block."""
    b64 = img_np_to_b64(img_np)
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _load_image_np(path: str) -> np.ndarray:
    """Load an image from disk into a uint8 numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def _resolve_path(path_like: Union[str, pathlib.Path], base_dir: pathlib.Path) -> pathlib.Path:
    """Resolve YAML-referenced paths relative to the config file directory."""
    p = pathlib.Path(path_like)
    if p.is_absolute():
        return p
    candidate = (base_dir / p).resolve()
    if candidate.exists():
        return candidate
    return p


def format_openai(
    role: str,
    content: Union[str, list],
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one OpenAI message dict."""
    msg: Dict[str, Any] = {"role": role, "content": content}
    if name is not None:
        msg["name"] = name
    return msg


# ---------------------------------------------------------------------------
# SystemPrompt  (loaded from configs/prompts/system/*.yaml)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SystemPrompt:
    content: str
    behavior: str
    behavior_kwargs: Optional[Dict[str, Any]] = None
    role: Optional[str] = None
    name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Union[str, pathlib.Path]) -> "SystemPrompt":
        path = pathlib.Path(path)
        with path.open("r") as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg)

    def message(self) -> Dict[str, Any]:
        """Return the system message dict for the OpenAI API."""
        assert self.role is not None
        return format_openai(role=self.role, content=self.content, name=self.name)


# ---------------------------------------------------------------------------
# ExamplePrompt  (loaded from configs/prompts/examples/*.yaml)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ExamplePrompt:
    # Task context
    task: Optional[str] = None
    instruction: Optional[str] = None
    objects: Optional[List[str]] = None
    history: Optional[List[str]] = None
    # Propose behaviour
    plan: Optional[List[str]] = None
    horizon: Optional[int] = None
    # Reflect behaviour
    analysis: Optional[str] = None
    revised_plan: Optional[List[str]] = None
    failed_step: Optional[int] = None
    failed_nut: Optional[str] = None      # nut name that owns the failed sub-skill
    failed_action: Optional[str] = None  # the failed sub-skill string
    mean_feasibility: Optional[float] = None
    uncertainty: Optional[float] = None
    score_trajectory: Optional[List[List[float]]] = None
    failure_pattern: Optional[str] = None
    worst_region: Optional[str] = None
    # Image paths (resolved at load time; enables multimodal few-shot examples)
    current_image: Optional[str] = None
    goal_image: Optional[str] = None
    imagined_image: Optional[str] = None   # world-model output for reflect
    gradcam_image: Optional[str] = None    # critic attention map for reflect
    # OpenAI
    role: Optional[str] = None
    name: Optional[str] = None
    name_query: Optional[str] = None
    name_response: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Union[str, pathlib.Path]) -> "ExamplePrompt":
        path = pathlib.Path(path)
        with path.open("r") as f:
            cfg = yaml.safe_load(f)
        base_dir = path.parent
        for key in ("current_image", "goal_image", "imagined_image", "gradcam_image"):
            if cfg.get(key) is not None:
                cfg[key] = str(_resolve_path(cfg[key], base_dir))
        return cls(**cfg)

    # -- query / response builders per behaviour ----------------------------

    def _propose_query(self) -> Union[str, list]:
        history_str = (
            "\n".join(f"  {i+1}. {a}" for i, a in enumerate(self.history))
            if self.history else "  (none — start of episode)"
        )
        obj_str = ", ".join(self.objects or [])
        horizon = self.horizon or 10

        if (
            self.goal_image and self.current_image
            and pathlib.Path(self.goal_image).exists()
            and pathlib.Path(self.current_image).exists()
        ):
            goal_np = _load_image_np(self.goal_image)
            current_np = _load_image_np(self.current_image)
            return [
                _text_block("### Goal state (target configuration)"),
                _img_block(goal_np),
                _text_block("### Current state (robot's current observation)"),
                _img_block(current_np),
                _text_block(
                    f"### Assembled nuts (history)\n{history_str}\n\n"
                    f"### Planning request\n"
                    f"List the available nuts in the order they should be assembled.\n"
                    f"Available nuts: {obj_str}\n\n"
                    f'Respond with JSON only: {{"plan": ["nut_label_1", "nut_label_2", ...]}}'
                ),
            ]

        # Text-only fallback (no images in YAML)
        return (
            f"Assembled nuts (history):\n{history_str}\n\n"
            f"Available nuts: {obj_str}\n\n"
            f"Plan: "
        )

    def _propose_response(self) -> str:
        if self.plan is None:
            raise ValueError("Example 'plan' is required for propose behavior.")
        return json.dumps({"plan": self.plan})

    def _reflect_query(self) -> Union[str, list]:
        history_str = (
            "\n".join(f"  {i+1}. {a}" for i, a in enumerate(self.history))
            if self.history else "  (none — start of episode)"
        )
        plan_str = "\n".join(f"  step {i}: {a}" for i, a in enumerate(self.plan or []))
        scores_str = ", ".join(
            f"step {i}: {s:.2f}" for i, (s, _) in enumerate(self.score_trajectory or [])
        )
        obj_str = ", ".join(self.objects or [])
        mean_f = self.mean_feasibility if self.mean_feasibility is not None else 0.0
        uncert = self.uncertainty if self.uncertainty is not None else 0.0

        if (
            self.goal_image and self.current_image
            and self.imagined_image and self.gradcam_image
            and pathlib.Path(self.goal_image).exists()
            and pathlib.Path(self.current_image).exists()
            and pathlib.Path(self.imagined_image).exists()
            and pathlib.Path(self.gradcam_image).exists()
        ):
            goal_np = _load_image_np(self.goal_image)
            current_np = _load_image_np(self.current_image)
            imagined_np = _load_image_np(self.imagined_image)
            gradcam_np = _load_image_np(self.gradcam_image)
            return [
                _text_block("### 1. Task images"),
                _text_block("Goal state (target configuration):"),
                _img_block(goal_np),
                _text_block("Current real state (robot's current observation):"),
                _img_block(current_np),
                _text_block(f"### 2. Execution history\n{history_str}"),
                _text_block(f"### 3. Proposed nut ordering\n{plan_str}"),
                _text_block(
                    "### 4. Critic diagnosis\n"
                    + (f"- Failed nut: {self.failed_nut}\n" if self.failed_nut else "")
                    + f"- Failed sub-skill: {self.failed_action}\n"
                    + f"- Feasibility at failure: {mean_f:.1%} "
                    + f"  (critic confidence: {1 - uncert:.1%})\n"
                    + f"- Score trajectory: {scores_str}\n"
                    + f"- Failure pattern: {self.failure_pattern}\n"
                    + f"- Region of highest goal mismatch: {self.worst_region}"
                ),
                _text_block(
                    f"### 5. World model output at step {self.failed_step}\n"
                    f"Imagined scene (what the world model predicted after the failed action):"
                ),
                _img_block(imagined_np),
                _text_block("Critic attention map (red = region with highest goal mismatch):"),
                _img_block(gradcam_np),
                _text_block(
                    f"### 6. Replanning instruction\n"
                    f"Identify the root cause of the failure. Revise the nut assembly "
                    f"ordering to avoid repeating the failure.\n"
                    f"Available nuts: {obj_str}\n\n"
                    f"Respond with JSON only:\n"
                    f'  {{"analysis": "one-sentence diagnosis", '
                    f'"revised_plan": ["nut_label_1", ...]}}'
                ),
            ]

        # Text-only fallback (no images in YAML)
        failed_nut_line = f"  - Failed nut: {self.failed_nut}\n" if self.failed_nut else ""
        return (
            f"Assembled nuts (history):\n{history_str}\n\n"
            f"Proposed nut ordering:\n{plan_str}\n\n"
            f"Critic diagnosis:\n"
            + failed_nut_line
            + f"  - Failed sub-skill: {self.failed_action}\n"
            + f"  - Feasibility: {mean_f:.1%} (confidence: {1 - uncert:.1%})\n"
            + f"  - Failure pattern: {self.failure_pattern}\n"
            + f"  - Region of highest goal mismatch: {self.worst_region}\n\n"
            + f"Available nuts: {obj_str}\n\n"
            + f"Revised nut ordering: "
        )

    def _reflect_response(self) -> str:
        if self.analysis is None or self.revised_plan is None:
            raise ValueError("Reflect examples require 'analysis' and 'revised_plan'.")
        return json.dumps({"analysis": self.analysis, "revised_plan": self.revised_plan})

    def messages(self, behavior: str) -> List[Dict[str, Any]]:
        """Return [query, response] message dicts for the given behaviour."""
        if self.role is None:
            raise ValueError("Example 'role' is required in YAML prompt examples.")

        if behavior == "propose":
            query = self._propose_query()
            response = self._propose_response()
        elif behavior == "reflect":
            if self.analysis is None or self.revised_plan is None:
                return []  # this example has no reflect data — skip
            query = self._reflect_query()
            response = self._reflect_response()
        else:
            raise ValueError(f"Unknown behavior: {behavior}")

        return [
            format_openai(role=self.role, content=query, name=self.name_query),
            format_openai(role=self.role, content=response, name=self.name_response),
        ]


# ---------------------------------------------------------------------------
# PromptManager — assembles [system, *examples, user] message lists
# ---------------------------------------------------------------------------

class PromptManager:
    """Assembles OpenAI message lists for ``propose`` and ``reflect`` calls.

    Mirrors ``BehaviorPromptManager`` from Points2Plans but adds multimodal
    (image) support in the user message.
    """

    def __init__(
        self,
        system_prompts: Dict[str, Union[str, pathlib.Path]],
        example_prompts: Optional[List[Union[str, pathlib.Path]]] = None,
    ) -> None:
        self._system: Dict[str, SystemPrompt] = {
            sp.behavior: sp
            for sp in (SystemPrompt.from_yaml(p) for p in system_prompts.values())
        }
        self._examples: List[ExamplePrompt] = [
            ExamplePrompt.from_yaml(p) for p in (example_prompts or [])
        ]

    @classmethod
    def from_yaml(cls, path: Union[str, pathlib.Path]) -> "PromptManager":
        path = pathlib.Path(path)
        with path.open("r") as f:
            cfg = yaml.safe_load(f)
        assert cfg["prompt"] == "PromptManager"
        kw = cfg["prompt_kwargs"]

        base_dir = path.parent
        system_prompts = {
            k: str(_resolve_path(v, base_dir))
            for k, v in kw["system_prompts"].items()
        }
        example_prompts = kw.get("example_prompts")
        if example_prompts is not None:
            example_prompts = [str(_resolve_path(v, base_dir)) for v in example_prompts]

        return cls(
            system_prompts=system_prompts,
            example_prompts=example_prompts,
        )

    @property
    def behaviors(self) -> Set[str]:
        return set(self._system.keys())

    # -- public builders ----------------------------------------------------

    def build_propose_messages(
        self,
        current_image_np: np.ndarray,
        goal_image_np: np.ndarray,
        history: List[str],
        obj_labels: List[str],
        horizon: int,
        task_instruction: str = "Assemble the target nuts onto their matching pegs.",
        use_examples: bool = True,
    ) -> List[Dict[str, Any]]:
        """Build the full message list for a ``propose`` call to GPT-4o."""
        msgs: List[Dict[str, Any]] = []

        if "propose" not in self._system:
            raise KeyError("Missing 'propose' system prompt configuration.")

        # 1. System prompt
        msgs.append(self._system["propose"].message())

        # 2. Few-shot examples (text-only, no images)
        if use_examples:
            for ex in self._examples:
                msgs.extend(ex.messages("propose"))

        # 3. User message (multimodal)
        history_str = (
            "\n".join(f"  {i+1}. {a}" for i, a in enumerate(history[-10:]))
            if history else "  (none — start of episode)"
        )
        obj_str = ", ".join(obj_labels)

        user_content = [
            _text_block("### Goal state (target configuration)"),
            _img_block(goal_image_np),
            _text_block("### Current state (robot's current observation)"),
            _img_block(current_image_np),
            _text_block(
                f"### Task instruction\n{task_instruction}\n\n"
                f"### Assembled nuts (history)\n{history_str}\n\n"
                f"### Planning request\n"
                f"List the available nuts in the order they should be assembled.\n"
                f"Available nuts: {obj_str}\n\n"
                f'Respond with JSON only: {{"plan": ["nut_label_1", "nut_label_2", ...]}}'
            ),
        ]
        msgs.append(format_openai(role="user", content=user_content))
        return msgs

    def build_reflect_messages(
        self,
        current_image_np: np.ndarray,
        goal_image_np: np.ndarray,
        history: List[str],
        obj_labels: List[str],
        full_plan: List[str],
        ctx: Dict[str, Any],
        task_instruction: str = "Assemble the target nuts onto their matching pegs.",
        use_examples: bool = True,
    ) -> List[Dict[str, Any]]:
        """Build the full message list for a ``reflect`` call to GPT-4o.

        ``ctx`` is the dict returned by ``build_reflection_context()``
        (see ``reflection.py``).
        """
        msgs: List[Dict[str, Any]] = []

        if "reflect" not in self._system:
            raise KeyError("Missing 'reflect' system prompt configuration.")

        required_ctx = [
            "all_scores",
            "imagined_state",
            # "gradcam_overlay",
            "failed_step",
            "failed_action",
            # "mean_feasibility",
            # "uncertainty",
            "failure_pattern",
            # "worst_region",
        ]
        missing_ctx = [k for k in required_ctx if k not in ctx]
        if missing_ctx:
            raise KeyError(f"Missing keys in reflection context: {missing_ctx}")

        # 1. System prompt
        msgs.append(self._system["reflect"].message())

        # 2. Few-shot examples (text-only, no images)
        if use_examples:
            for ex in self._examples:
                msgs.extend(ex.messages("reflect"))

        # 3. User message (multimodal)
        history_str = (
            "\n".join(f"  {i+1}. {a}" for i, a in enumerate(history[-10:]))
            if history else "  (none — start of episode)"
        )
        plan_str = "\n".join(f"  step {i}: {a}" for i, a in enumerate(full_plan))
        scores_str = ", ".join(
            f"step {i}: {s:.2f}" for i, (s, _) in enumerate(ctx["all_scores"])
        )
        obj_str = ", ".join(obj_labels)

        imagined_np = np.array(ctx["imagined_state"])
        # gradcam_np = np.array(ctx["gradcam_overlay"])

        user_content = [
            # 1. Task images
            _text_block("### 1. Task images"),
            _text_block("Goal state (target configuration):"),
            _img_block(goal_image_np),
            _text_block("Current real state (robot's current observation):"),
            _img_block(current_image_np),
            # 2. Execution history
            _text_block(
                f"### 2. Execution history\n{history_str}"
            ),
            # 3. Original proposed plan
            _text_block(
                f"### 3. Proposed nut ordering\n{plan_str}"
            ),
            # 4. Critic diagnosis
            _text_block(
                "### 4. Critic diagnosis\n"
                + (f"- Failed nut: {ctx['failed_highlevel_action']}\n" if ctx.get('failed_highlevel_action') else "")
                + f"- Failed sub-skill: {ctx['failed_action']}\n"
                # + f"- Feasibility at failure: {ctx['mean_feasibility']:.1%} "
                # + f"  (critic confidence: {1 - ctx['uncertainty']:.1%})\n"
                + f"- Failure pattern: {ctx['failure_pattern']}\n"
                # + f"- Region of highest goal mismatch: {ctx['worst_region']}"
            ),
            # 5. Imagined state + attention map
            _text_block(
                f"### 5. World model output at step {ctx['failed_step']}\n"
                f"Imagined scene (what the world model predicted after the failed action):"
            ),
            _img_block(imagined_np),
            # _text_block("Critic attention map (red = region with highest goal mismatch):"),
            # _img_block(gradcam_np),
            # 6. Replanning instruction
            _text_block(
                f"### 6. Replanning instruction\n"
                f"Task: {task_instruction}\n"
                f"Identify the root cause of the failure. Revise the nut assembly "
                f"ordering to avoid repeating the failure.\n"
                f"Available nuts: {obj_str}\n\n"
                f"Respond with JSON only:\n"
                f'  {{"analysis": "one-sentence diagnosis", '
                f'"revised_plan": ["nut_label_1", ...]}}'
            ),
        ]
        msgs.append(format_openai(role="user", content=user_content))
        return msgs
