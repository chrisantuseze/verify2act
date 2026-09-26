"""Verify2Act models for the real robot: VLM planner + latent world model + dual-head critic.

``plan()`` is the sim loop, unchanged: ``BeamSearchPlanner.plan`` (propose candidates -> imagine each
horizon -> temporal head per horizon -> goal head at the final horizon -> reflect/replan with a fixed
budget). The only robot-specific parts are the prompt manager (subtask vocabulary) and the plan
expander (one subtask = one imagination step).

One server process runs one verification variant (``--wm-mode``), built as ``pipeline/inference_calvin.py`` builds it:
  v2a_wm        latent WM + critic (the method)
  rla_wm        RLA-WM baseline latent WM + the same critic
  diffusion_wm  InstructPix2Pix WM, no critic (ReflectVLM; the sim's ``diffusion`` mode)
  vlm_only      one VLM propose, no imagination, no critic, no reflection (the sim's ``beam_planner=None`` branch)
"""

import json
import logging
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from verify2act.pipeline.planner import BeamSearchPlanner, VLMPlanner
from verify2act.robot.prompts import (COLORS, DONE, RobotPromptManager, expand_subtask_plan, is_valid_subtask,
                                      step_text)

logger = logging.getLogger(__name__)

WM_MODES = ("v2a_wm", "rla_wm", "diffusion_wm", "vlm_only")
# Name on the wire / CLI -> BeamSearchPlanner / inference_calvin.py name.
SIM_WM_MODE = {"diffusion_wm": "diffusion"}


def _json_float(x) -> Optional[float]:
    x = float(x)
    return x if math.isfinite(x) else None


def load_feature_decoder(decoder_dir: Optional[str], device: torch.device):
    """FeatureDecoder (DINO features -> RGB) used to show the VLM the imagined scene when reflecting.
    Same loading as ``pipeline/inference.py::main``."""
    from verify2act.latent_wm.decoder import FeatureDecoder

    decoder = FeatureDecoder(dino_channels=1024).to(device).eval()
    path = Path(decoder_dir) / "latent_decoder_best.pt" if decoder_dir else None
    if path is None or not path.exists():
        logger.warning("Decoder checkpoint not found at %s; reflection will see an untrained decoder.", path)
        return decoder
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            state_dict = ckpt[key]
            break
    if "decoder.input_proj.0.weight" not in state_dict and "input_proj.0.weight" in state_dict:
        state_dict = {f"decoder.{k}": v for k, v in state_dict.items()}
    decoder.load_state_dict(state_dict)
    return decoder


def build_world_model(args):
    """The world model of ``args.wm_mode`` (not vlm_only), with the arguments ``inference_calvin.py`` uses."""
    from verify2act.pipeline.world_model import DiffusionWorldModel, LatentWorldModel, RLAWorldModel

    mode = args.wm_mode
    logger.info("Loading %s world model ...", mode)
    if mode == "v2a_wm":
        return LatentWorldModel(device=args.device, dynamics_weights_path=args.latent_wm_ckpt,
                                encoder_ckpt=args.encoder_ckpt, history_len=args.history_len,
                                token_dim=args.token_dim, num_latent_tokens=args.num_latent_tokens,
                                action_conditioning=args.action_conditioning)
    if mode == "rla_wm":
        return RLAWorldModel(device=args.device, dynamics_weights_path=args.latent_wm_ckpt,
                             encoder_ckpt=args.encoder_ckpt, history_len=args.history_len,
                             token_dim=args.token_dim, num_latent_tokens=args.num_latent_tokens)
    if mode == "diffusion_wm":
        return DiffusionWorldModel(pretrained_model=args.wm_model, adapter_dir=args.wm_adapter_dir,
                                   decoder_dir=args.wm_decoder_dir, vae_model=args.vae_model,
                                   vae_subfolder=args.vae_subfolder, device=args.device, torch_dtype=torch.float16,
                                   num_inference_steps=args.wm_steps, image_guidance_scale=args.wm_image_guidance,
                                   guidance_scale=args.wm_text_guidance, seed=args.wm_seed)
    raise ValueError(f"no world model for wm_mode {mode!r}")


class Verify2ActBackend:
    """Stateless per call except for a per-session planning-call counter (used to name log folders).

    ``beam_planner`` is None for ``vlm_only``; pass the ``VLMPlanner`` as ``vlm`` and the thresholds as ``settings``."""

    def __init__(self, beam_planner: Optional[BeamSearchPlanner], decoder=None, horizon: int = 4,
                 output_dir: Optional[str] = None, wm_mode: str = "v2a_wm", vlm=None,
                 settings: Optional[Dict[str, Any]] = None):
        if wm_mode not in WM_MODES:
            raise ValueError(f"wm_mode must be one of {WM_MODES}, got {wm_mode!r}")
        if (beam_planner is None) != (wm_mode == "vlm_only"):
            raise ValueError("vlm_only runs without a BeamSearchPlanner; every other mode needs one")
        self.beam_planner = beam_planner
        self.vlm = vlm if vlm is not None else beam_planner.vlm
        self.wm_mode = wm_mode
        if settings is None:
            settings = {"theta_c": beam_planner.temporal_threshold, "theta_p": beam_planner.goal_threshold,
                        "max_replans": beam_planner.max_replans, "max_requery": beam_planner.max_retries}
        # Reported in every ping reply, so the Jetson records the variant and thresholds per session.
        self.settings = {"wm_mode": wm_mode, **settings}
        self.decoder = decoder
        self.horizon = horizon
        self.output_dir = Path(output_dir) if output_dir else None
        self._calls: Dict[str, int] = {}
        self._record: Optional[Dict[str, Any]] = None
        self._instrument()

    def _instrument(self) -> None:
        """Record every plan evaluation and VLM call made during one ``plan()`` request. The sim's result dict keeps
        only the final plan's critic decisions, so it cannot give the session metrics (requeries, rejections, VLM calls)."""
        bp = self.beam_planner
        if bp is not None:
            evaluate = bp._evaluate_trajectory

            def recording_evaluate(*args, **kwargs):
                out = evaluate(*args, **kwargs)
                if self._record is not None:
                    self._record["evaluations"].append(out)
                return out

            bp._evaluate_trajectory = recording_evaluate
        vlm_call = getattr(self.vlm, "_call", None)
        if vlm_call is not None:
            def counting_call(*args, **kwargs):
                if self._record is not None:
                    self._record["vlm_calls"] += 1
                return vlm_call(*args, **kwargs)

            self.vlm._call = counting_call

    @property
    def has_critic(self) -> bool:
        return self.beam_planner is not None and self.beam_planner.critic is not None

    def _summarise_evaluations(self) -> List[Dict[str, Any]]:
        rows = []
        for score, _final, all_scores, steps, step_failed, _failed_step, decisions in self._record["evaluations"]:
            if not self.has_critic:
                # diffusion_wm: the rollout is imagined but never scored (the sim fills in dummy 1.0 scores).
                rows.append({"plan": [hl for hl, _ in steps], "tc": [], "goal": None, "accepted": False,
                             "temporal_rejected": False, "goal_rejected": False, "requeries": 0})
                continue
            head1 = [d for d in decisions if d.startswith("HEAD1")]
            rows.append({
                "plan": [hl for hl, _ in steps],
                "tc": [_json_float(tc) for tc, _ in all_scores],
                "goal": _json_float(score),
                "accepted": (not step_failed) and math.isfinite(score) and score >= self.beam_planner.goal_threshold,
                "temporal_rejected": bool(step_failed) and not head1,   # rollout aborted by the temporal head
                "goal_rejected": any(d.endswith("reflect") for d in head1),
                "requeries": sum("→ requery" in d for d in decisions),
            })
        return rows

    @classmethod
    def from_args(cls, args) -> "Verify2ActBackend":
        """Build the variant ``args.wm_mode`` as ``pipeline/inference_calvin.py::main`` does."""
        from verify2act.pipeline.inference import _build_critic

        device = torch.device(args.device)
        mode = args.wm_mode
        if args.gcp_project:
            # Vertex AI, as in the sim runs. VLMPlanner prefers GEMINI_API_KEY (AI Studio) whenever it is set, and
            # "vertex-ai" is the value gemini_backend uses to select the Vertex endpoint.
            os.environ["GOOGLE_CLOUD_PROJECT"] = args.gcp_project
            os.environ["GEMINI_API_KEY"] = "vertex-ai"

        planner = VLMPlanner(
            RobotPromptManager.from_yaml(args.prompt_config),
            model=args.planner_model,
            max_tokens=args.planner_max_tokens,
            temperature=args.planner_temperature,
            call_delay=args.planner_call_delay,
        )
        settings = {"theta_c": args.theta_c, "theta_p": args.theta_p, "max_replans": args.max_replans,
                    "max_requery": args.max_retries}
        if mode == "vlm_only":
            logger.info("vlm_only: no world model, critic or decoder loaded.")
            return cls(None, horizon=args.horizon, output_dir=args.output_dir, wm_mode=mode, vlm=planner,
                       settings=settings)

        world_model, critic, decoder = build_world_model(args), None, None
        if mode != "diffusion_wm":   # the sim's diffusion mode (ReflectVLM) has no critic
            logger.info("Loading critic ...")
            critic = _build_critic(SimpleNamespace(critic_ckpt=args.critic_ckpt), device)
            decoder = load_feature_decoder(args.wm_decoder_dir, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU memory allocated after loading: %.2f GB", torch.cuda.memory_allocated() / 1e9)

        beam_planner = BeamSearchPlanner(
            vlm_planner=planner,
            world_model=world_model,
            critic=critic,
            beam_width=args.beam_width,
            goal_threshold=args.theta_p,
            plan_expander=expand_subtask_plan,
            temporal_threshold=args.theta_c,
            max_retries=args.max_retries,
            max_replans=args.max_replans,
            wm_mode=SIM_WM_MODE.get(mode, mode),
        )
        backend = cls(beam_planner, decoder=decoder, horizon=args.horizon, output_dir=args.output_dir,
                      wm_mode=mode, settings=settings)
        backend.warmup()
        return backend

    def warmup(self) -> None:
        """One imagine (+ critic) pass on a blank frame (no VLM call), so the critic's lazily loaded DINOv2 and
        the CUDA kernels are ready before the first real request."""
        t0 = time.time()
        self.beam_planner._evaluate_trajectory(plan=["pick and place red block into the bin"],
                                               current_image_np=np.zeros((224, 224, 3), np.uint8),
                                               language_goal="warm up", decoder=None)
        logger.info("Warm-up done in %.1fs", time.time() - t0)

    # ── ops ───────────────────────────────────────────────────────────────────

    def reset(self, session: str) -> None:
        self._calls[session] = 0

    def plan(self, session: str, image_rgb: np.ndarray, goal: str, history: Optional[List[str]] = None,
             obj_labels: Optional[List[str]] = None, horizon: Optional[int] = None,
             feedback: str = "") -> Dict[str, Any]:
        call_idx = self._calls.get(session, 0)
        self._calls[session] = call_idx + 1
        history = list(history or [])
        obj_labels = [f"{c} block" if c in COLORS else c for c in (obj_labels or COLORS)]
        horizon = int(horizon or self.horizon)
        session_dir = self.output_dir / session if self.output_dir else None

        t0 = time.time()
        self._record = {"evaluations": [], "vlm_calls": 0}
        # The operator's note on the previous attempt reaches every propose/reflect prompt of this call (all variants).
        prompt_manager = getattr(self.vlm, "_pm", None)
        if prompt_manager is not None:
            prompt_manager.feedback = feedback or ""
        try:
            if self.beam_planner is None:
                # vlm_only, as the sim's beam_planner=None branch: one propose, accepted as is. A VLM error raises
                # (an error reply) instead of becoming an empty plan.
                vlm_plan = self.vlm.propose(current_image_np=image_rgb, history=history, obj_labels=obj_labels,
                                            horizon=horizon, language_goal=goal)
                res = {"plan": vlm_plan, "plan_accepted": True, "score": float("nan"), "all_scores": [],
                       "failed_step": None, "replan_attempts": 0, "reflection_analyses": [], "critic_decisions": []}
            else:
                res = self.beam_planner.plan(
                    current_image_np=image_rgb,
                    history=history,
                    obj_labels=obj_labels,
                    horizon=horizon,
                    language_goal=goal,
                    timestep=call_idx,
                    output_dir=session_dir,
                    decoder=self.decoder,
                )
            evaluations = self._summarise_evaluations()
            vlm_calls = self._record["vlm_calls"] if hasattr(self.vlm, "_call") else None
        finally:
            self._record = None
            if prompt_manager is not None:
                prompt_manager.feedback = ""
        plan = [step_text(s) for s in res["plan"]]
        plan = [s for s in plan if s]
        if not plan:
            # BeamSearchPlanner swallows VLM errors and falls back to an empty plan. The prompts ask for ["done"] when
            # the goal is met, so an empty plan here means the VLM failed, and it must not reach the Jetson, which
            # reads an empty plan as "nothing left to do".
            raise RuntimeError("VLM produced no plan (API error or unparseable reply; see the server log)")
        # ["done"] goes through the same verification as any plan (no imagination steps: the goal head judges the
        # real frame), but `done` reports only what the VLM said; whether the critic agreed is in `accepted`.
        # On the wire `done` removes the step, so the plan is empty.
        done = DONE in plan
        plan = [s for s in plan if s != DONE]
        out = {
            "plan": plan,
            "accepted": bool(res["plan_accepted"]),
            # diffusion_wm has no critic: the sim's dummy 1.0 scores are not sent.
            "score": _json_float(res["score"]) if self.has_critic else None,
            "all_scores": ([[_json_float(tc), _json_float(prox)] for tc, prox in res["all_scores"]]
                           if self.has_critic else []),
            "failed_step": res["failed_step"],
            "replan_attempts": int(res["replan_attempts"]),
            "reflection_analyses": [str(a) for a in res["reflection_analyses"]],
            "critic_decisions": [str(d) for d in res["critic_decisions"]],
            "invalid_steps": [s for s in plan if not is_valid_subtask(s)],
            "done": done,
            "evaluations": evaluations,
            "stats": {
                "vlm_calls": vlm_calls if vlm_calls is not None else 1 + int(res["replan_attempts"]),
                "plans_evaluated": len(evaluations),
                "requeries": sum(e["requeries"] for e in evaluations),
                "temporal_rejections": sum(e["temporal_rejected"] for e in evaluations),
                "goal_rejections": sum(e["goal_rejected"] for e in evaluations),
            },
            "wm_mode": self.wm_mode,
            "planning_call": call_idx,
            "elapsed_s": round(time.time() - t0, 2),
        }
        if out["invalid_steps"]:
            logger.warning("Plan contains subtasks outside the robot vocabulary: %s", out["invalid_steps"])

        if session_dir:
            call_dir = session_dir / "imagination_logs" / f"planning_call_{call_idx:02d}"
            call_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image_rgb).save(call_dir / "request_image.png")
            with open(call_dir / "request.json", "w") as f:
                json.dump({"goal": goal, "history": history, "feedback": feedback or "", "obj_labels": obj_labels,
                       "horizon": horizon}, f, indent=2)
            with open(call_dir / "response.json", "w") as f:
                json.dump(out, f, indent=2)
        return out
