"""CALVIN Agent Wrapper for Verify2Act.

This module implements the CustomModel interface expected by CALVIN's evaluation script.
It embeds the Verify2Act pipeline (VLM, Latent WM, Critic) as an "Obstacle-Resolving
Cognitive Layer" on top of a baseline low-level continuous policy.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from calvin_agent.models.calvin_base_model import CalvinBaseModel
from verify2act.critic.inference import check_rollout_consistency, CriticDecision, decide_from_proximity
from verify2act.pipeline.inference import preprocess_image_for_critic
from verify2act.pipeline.planner import VLMRefusalError
from calvin_agent.utils.utils import get_last_checkpoint
from calvin_agent.evaluation.utils import get_default_model_and_env

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Per-sequence trace dataclasses  (mirrors EpisodeTrace in inference.py)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CalvinSubtaskRecord:
    """Metrics for one subtask within a sequence."""
    subtask_idx: int
    goal: str = ""
    success: bool = False
    steps_taken: int = 0
    vlm_calls: int = 0
    reflections: int = 0
    critic_accepts: int = 0
    critic_rejects: int = 0
    critic_tp: int = 0   # accepted → subtask succeeded
    critic_fp: int = 0   # accepted → subtask failed


@dataclass
class CalvinEpisodeTrace:
    """Full per-sequence log, analogous to EpisodeTrace in nut assembly."""
    sequence_idx: int = 0
    subtasks_completed: int = 0           # 0-5
    total_subtasks: int = 5
    success: bool = False                 # True iff subtasks_completed == 5
    total_steps: int = 0
    total_vlm_calls: int = 0
    total_reflections: int = 0
    # Critic metrics aggregated over the whole sequence
    critic_accepts: int = 0
    critic_rejects: int = 0
    critic_tp: int = 0
    critic_fp: int = 0
    subtask_records: List[CalvinSubtaskRecord] = field(default_factory=list)


def _save_calvin_trace(trace: CalvinEpisodeTrace, output_dir: Path) -> None:
    """Persist a CalvinEpisodeTrace as JSON (mirrors _save_trace in inference.py)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "episode_trace.json"

    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(asdict(trace), f, indent=2, default=_default)
    logger.info("Calvin episode trace saved to %s", path)

from verify2act.pipeline.calvin_policy_wrapper import LowLevelPolicyFactory


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
        extra_dataset_path: str = None,
        **kwargs,
    ):
        self.vlm_planner = vlm_planner
        self.world_model = world_model
        self.critic = critic
        self.device = device
        self.theta_c = theta_c
        # Separate threshold for Head 1 (goal proximity, DINO-to-CLIP cross-modal cosine sim).
        # Its score range is much lower than temporal consistency, so it needs its own calibration.
        self.theta_p: float = kwargs.pop("theta_p", 0.2)
        self.max_replans = max_replans
        # MC uncertainty gate: predictions with std > this are treated as requery.
        # Default 0.08 matches the observed CALVIN critic std range (0.030-0.065).
        self.critic_unc_threshold: float = kwargs.pop("critic_unc_threshold", 0.08)

        self.low_level_policy = LowLevelPolicyFactory.get_policy(
            policy_type=kwargs.get("low_level_policy_type", "hulc"),
            train_folder=train_folder, 
            dataset_path=dataset_path, 
            device=device,
            extra_dataset_path=extra_dataset_path,
        )

        self.current_subgoal: str = ""
        self.plan_queue: List[str] = []
        self._step_count = 0
        self.subgoal_step_count = 0

        # ── Reflection throttling ─────────────────────────────────────────
        # Prevents the VLM reflection loop from firing on every verification
        # trigger when the imagined outcome is repeatedly rejected.
        self._reflect_count: int = 0          # reflect() calls for current subtask
        self._last_reflect_step: int = -999   # step at which last reflect() fired
        self._last_reflected_plan: List[str] = []  # plan proposed by last reflect()
        # How many steps to wait before allowing the next reflect() call.
        self._reflect_cooldown: int = 30

        # ── Per-sequence trace accumulator ────────────────────────────────
        self._sequence_idx: int = 0
        self._current_trace: CalvinEpisodeTrace = CalvinEpisodeTrace()
        self._current_subtask_rec: CalvinSubtaskRecord = CalvinSubtaskRecord(subtask_idx=0)
        self._subtask_idx: int = 0
        self._pending_critic_accept: bool = False  # True when we've accepted a plan this step

    # ── Trace management helpers ──────────────────────────────────────────

    def start_sequence(self, sequence_idx: int, output_dir: Optional[Path] = None) -> None:
        """Call before each new evaluation sequence to reset the accumulator."""
        self._sequence_idx = sequence_idx
        self.output_dir = output_dir
        self._current_trace = CalvinEpisodeTrace(sequence_idx=sequence_idx)
        self._subtask_idx = 0
        self._current_subtask_rec = CalvinSubtaskRecord(
            subtask_idx=0, goal=""
        )
        self._pending_critic_accept = False

    def _start_subtask(self, goal: str) -> None:
        """Called when a new subtask goal is detected inside step()."""
        self._current_subtask_rec = CalvinSubtaskRecord(
            subtask_idx=self._subtask_idx,
            goal=goal,
        )
        # Reset per-subtask reflection throttle counters.
        self._reflect_count = 0
        self._last_reflect_step = -999
        self._last_reflected_plan = []

    def _close_subtask(self, success: bool) -> None:
        """Commit the current subtask record into the sequence trace."""
        rec = self._current_subtask_rec
        rec.success = success
        rec.steps_taken = self._step_count

        # Resolve pending critic accept → TP or FP based on actual outcome.
        # TP: critic accepted the imagined plan → subtask truly succeeded.
        # FP: critic accepted the imagined plan → subtask actually failed.
        if self._pending_critic_accept:
            if success:
                rec.critic_tp += 1
            else:
                rec.critic_fp += 1

        self._current_trace.subtask_records.append(rec)
        # Roll up into sequence-level totals
        self._current_trace.total_steps += rec.steps_taken
        self._current_trace.total_vlm_calls += rec.vlm_calls
        self._current_trace.total_reflections += rec.reflections
        self._current_trace.critic_accepts += rec.critic_accepts
        self._current_trace.critic_rejects += rec.critic_rejects
        self._current_trace.critic_tp += rec.critic_tp
        self._current_trace.critic_fp += rec.critic_fp
        if success:
            self._current_trace.subtasks_completed += 1
        self._subtask_idx += 1

    def flush_trace(
        self,
        subtasks_completed: int,
        output_dir: Optional[Path] = None,
    ) -> CalvinEpisodeTrace:
        """Finalise and optionally save the current sequence trace.

        Call this after CALVIN's evaluate_sequence() returns so that
        subtasks_completed (the ground-truth count from the task oracle)
        is authoritative.
        """
        trace = self._current_trace
        trace.subtasks_completed = subtasks_completed
        trace.success = subtasks_completed == trace.total_subtasks
        if output_dir is not None:
            seq_dir = output_dir / f"sequence_{trace.sequence_idx:04d}"
            _save_calvin_trace(trace, seq_dir)
        return trace

    def reset(self):
        """Called by CALVIN at the start of each subtask rollout."""
        # Flush the previous subtask record (CALVIN calls reset() per subtask,
        # not per sequence — we track boundaries via goal changes in step()).
        self.current_subgoal = ""
        self.plan_queue = []
        self._step_count = 0
        self.subgoal_step_count = 0
        self._pending_critic_accept = False
        self._reflect_count = 0
        self._last_reflect_step = -999
        self._last_reflected_plan = []
        self._last_verify_dir = None
        if hasattr(self, "low_level_policy"):
            self.low_level_policy.reset()
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
        
        if hasattr(self.low_level_policy, "env") and hasattr(self.low_level_policy.env, "env"):
            raw_obs = self.low_level_policy.env.env.get_obs()
            img_np = raw_obs['rgb_obs']['rgb_static']
        else:
            img_np = obs['rgb_obs']['rgb_static']

        if getattr(self, "_last_verify_dir", None) is not None:
            exec_dir = self._last_verify_dir / "execution" / "horizon_01"
            exec_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(img_np).save(exec_dir / "groundtruth_frame.png")
            with open(exec_dir / "action.txt", "w") as f:
                f.write(self.plan_queue[0] if self.plan_queue else self.current_subgoal)
            self._last_verify_dir = None

        if goal != self.current_subgoal:
            logger.info(f"New CALVIN Goal Received: {goal}")
            self.current_subgoal = goal
            self.plan_queue = [goal]
            self.subgoal_step_count = 0
            self._start_subtask(goal)

        self._step_count = getattr(self, "_step_count", 0) + 1
        self.subgoal_step_count = getattr(self, "subgoal_step_count", 0) + 1
        self._current_subtask_rec.steps_taken = self._step_count

        # 2. Track subgoal transitions
        active_instruction = self.plan_queue[0] if self.plan_queue else self.current_subgoal
        
        max_subgoal_steps = 72
        is_completed = False
        if len(self.plan_queue) > 1:
            if self.subgoal_step_count >= max_subgoal_steps:  # noqa: E501
                logger.info(f"Sub-goal '{active_instruction}' step budget ({max_subgoal_steps}) reached.")
                is_completed = True
            elif self.subgoal_step_count >= 30 and self.subgoal_step_count % 30 == 0:
                # Periodically run VLM verification on live observation.
                # Reduced from every-15 to every-30 to limit API call frequency.
                if hasattr(self.low_level_policy, "env") and hasattr(self.low_level_policy.env, "env"):
                    raw_obs = self.low_level_policy.env.env.get_obs()
                    img_np = raw_obs['rgb_obs']['rgb_static']
                else:
                    img_np = obs['rgb_obs']['rgb_static']
                vlm_verification = self.vlm_planner.verify_goal(img_np, active_instruction)
                if vlm_verification.get("achieved", False):
                    logger.info(f"VLM verified sub-goal '{active_instruction}' as achieved.")
                    is_completed = True
                    
        if is_completed and len(self.plan_queue) > 1:
            logger.info(f"Sub-goal '{active_instruction}' completed (steps={self.subgoal_step_count}). Transitioning to next sub-goal.")
            self.plan_queue.pop(0)
            active_instruction = self.plan_queue[0] if self.plan_queue else self.current_subgoal
            self.subgoal_step_count = 0

        # 3. We only run the Verify2Act imagination phase periodically or at the start of a new sub-goal.
        # For simplicity in this wrapper, let's assume we verify the first step of a new instruction.
        # In a full implementation, you might verify every N steps or when the policy's confidence drops.
        
        # For now, we just pass the instruction to the low-level policy.
        action = self.low_level_policy.step(obs, active_instruction)
        
        # ---
        # Verify2Act Imagination & Reflection logic (The Cognitive Layer)
        # ---
        # Define a frequency to verify (e.g., step 0 and every 10 steps)
        # We assume `obs['rgb_obs']['rgb_static']` gives the numpy image.
        
        # Verify every 30 steps (down from 10) to reduce world-model inference
        # and VLM API calls per subtask.
        should_verify = (self._step_count % 30 == 0) and (self.world_model is not None)

        if should_verify:
            verify_dir = None
            if getattr(self, "output_dir", None) is not None:
                verify_dir = self.output_dir / "imagination_logs" / f"planning_call_{self._step_count:02d}"
                verify_dir.mkdir(parents=True, exist_ok=True)
                from PIL import Image
                init_state_path = verify_dir / "step_0_initial_state.png"
                if not init_state_path.exists():
                    Image.fromarray(img_np).save(init_state_path)
                self._last_verify_dir = verify_dir

            # 2. Propose a trajectory from the low-level policy
            # Note: Dummy policy just returns 10 copies of the single action.
            # A real policy might autoregressively generate a sequence.
            traj = self.low_level_policy.propose_trajectory(obs, active_instruction)
            
            # 3. Imagine the outcome using the World Model
            is_latent_wm = False
            from verify2act.pipeline.world_model import LatentWorldModel
            if isinstance(self.world_model, LatentWorldModel):
                is_latent_wm = True

            if is_latent_wm:
                # Re-anchor history from the real camera frame every verification
                # cycle. Without this, the WM's internal F_t drifts: after the
                # first call initialize_history() is never called again, so
                # F_next = F_t(stale) + delta_F and decoded images look wrong.
                # visualize_wm.py always feeds fresh GT features — we match that
                # by re-seeding the sliding window with the current observation.
                self.world_model.initialize_history(img_np)
                F_next, _ = self.world_model.imagine(None, active_instruction)

                # Decode DINO features → RGB now so we can save + verify regardless of critic gate
                imagined_img_next: Optional[np.ndarray] = None
                if getattr(self, "decoder", None) is not None:
                    imagined_img_next = self.decode_dino_features(F_next, self.decoder)

                if self.critic is not None:
                    # ── Head 2: Temporal consistency ──────────────────────────
                    # Is the imagined transition physically plausible?
                    with torch.no_grad():
                        img_tensor = preprocess_image_for_critic(img_np).to(self.device)
                        F_prev = self.world_model.extractor.extract_dino(img_tensor)
                        emb_prev = self.critic.encode_features(F_prev)
                        emb_next = self.critic.encode_features(F_next)
                        mean_tc, std_tc = self.critic.temporal_sim_with_uncertainty(emb_prev, emb_next)

                    tc_decision = check_rollout_consistency(
                        mean_tc.item(), self.theta_c,
                        uncertainty=std_tc.item(),
                        confidence_threshold=self.critic_unc_threshold,
                    )

                    # ── Head 1: Goal proximity ────────────────────────────────
                    # Does the imagined outcome semantically match the goal?
                    with torch.no_grad():
                        mean_prox, std_prox = self.critic.goal_sim_from_text_with_uncertainty(
                            emb_next, active_instruction
                        )

                    gp_decision = decide_from_proximity(
                        mean_prox.item(), self.theta_p,
                        uncertainty=std_prox.item(),
                        confidence_threshold=self.critic_unc_threshold,
                    )

                    # Combined decision: both heads must pass to accept.
                    # Head 2 (temporal) failure → requery (bad physics).
                    # Head 1 (goal proximity) failure → reflect (plan failed).
                    if tc_decision.action == "requery":
                        decision = tc_decision  # temporal inconsistency — requery
                    elif gp_decision.action == "reflect":
                        decision = gp_decision  # semantically wrong outcome — reflect
                    elif gp_decision.action == "requery":
                        decision = gp_decision  # uncertain proximity — requery
                    else:
                        decision = CriticDecision(action="continue", reason="temporal+proximity ok")

                    # ── Save imagination + critic scores for inspection ───────
                    if verify_dir is not None and imagined_img_next is not None:
                        candidate_str = f"candidate_00" if self._reflect_count == 0 else f"replan_attempt_{self._reflect_count:02d}"
                        horizon_dir = verify_dir / candidate_str / "horizon_01"
                        horizon_dir.mkdir(parents=True, exist_ok=True)
                        from PIL import Image
                        import json as _json
                        Image.fromarray(imagined_img_next).save(horizon_dir / "imagine_frame.png")
                        Image.fromarray(img_np).save(horizon_dir / "real_frame.png")
                        with open(horizon_dir / "action.txt", "w") as f:
                            f.write(active_instruction)
                        with open(horizon_dir / "temporal_critic.json", "w") as f:
                            _json.dump({
                                "temporal_consistency_score": mean_tc.item(),
                                "uncertainty": std_tc.item(),
                                "decision": tc_decision.action,
                                "decision_reason": tc_decision.reason,
                            }, f, indent=2)
                        with open(horizon_dir / "goal_critic.json", "w") as f:
                            _json.dump({
                                "goal_proximity_score": mean_prox.item(),
                                "uncertainty": std_prox.item(),
                                "decision": gp_decision.action,
                                "decision_reason": gp_decision.reason,
                                "combined_decision": decision.action,
                            }, f, indent=2)

                    proximity_ok = (decision.action == "continue")
                    reason = decision.reason
                    all_scores = [(mean_tc.item(), mean_prox.item())]
                else:
                    # No critic: accept by default and let the VLM planner judge during reflection.
                    decision = CriticDecision(action="continue", reason="no critic — accepted")
                    proximity_ok = True
                    reason = "no critic — accepted"
                    all_scores = [(1.0, 1.0)]
            else:
                # RGB / Image World Model path
                imagined_img_next = self.world_model.imagine(img_np, active_instruction) 
                if isinstance(imagined_img_next, tuple):
                    imagined_img_next = imagined_img_next[1] if len(imagined_img_next) > 1 else imagined_img_next[0]
                
                if self.critic is not None:
                    # 4. Check temporal/physical consistency (Critic Head 2)
                    img_224_prev = preprocess_image_for_critic(img_np).to(self.device)
                    img_224_next = preprocess_image_for_critic(imagined_img_next).to(self.device)
                    
                    with torch.no_grad():
                        emb_prev = self.critic.encode(img_224_prev)
                        emb_next = self.critic.encode(img_224_next)
                        mean_tc, std_tc = self.critic.temporal_sim_with_uncertainty(emb_prev, emb_next)
                        
                    decision = check_rollout_consistency(mean_tc.item(), self.theta_c, uncertainty=std_tc.item())

                    # 5. Goal proximity check via language goal (Head 1, CLIP path)
                    with torch.no_grad():
                        emb_next_pe = self.critic.encode(img_224_next)
                        mean_prox, std_prox = self.critic.goal_sim_from_text_with_uncertainty(
                            emb_next_pe, active_instruction
                        )
                    if decision.action != "reflect":
                        if verify_dir is not None:
                            candidate_str = f"candidate_00" if self._reflect_count == 0 else f"replan_attempt_{self._reflect_count:02d}"
                            horizon_dir = verify_dir / candidate_str / "horizon_01"
                            horizon_dir.mkdir(parents=True, exist_ok=True)
                            from PIL import Image
                            Image.fromarray(imagined_img_next).save(horizon_dir / "imagine_frame.png")
                            with open(horizon_dir / "action.txt", "w") as f:
                                f.write(active_instruction)
                            import json
                            with open(horizon_dir / "temporal_critic.json", "w") as f:
                                json.dump({
                                    "temporal_consistency_score": mean_tc.item(),
                                    "uncertainty": std_tc.item(),
                                    "decision": decision.action,
                                    "decision_reason": decision.reason,
                                }, f, indent=2)
                            with open(horizon_dir / "goal_critic.json", "w") as f:
                                json.dump({
                                    "goal_proximity_score": mean_prox.item(),
                                    "uncertainty": std_prox.item(),
                                }, f, indent=2)

                        vlm_verification = self.vlm_planner.verify_goal(imagined_img_next, active_instruction)
                        if not vlm_verification["achieved"]:
                            decision.action = "reflect"
                            decision.reason = f"Goal not achieved: {vlm_verification.get('reason')}"

                    # proximity_ok follows the combined verdict in decision.action:
                    # - Decoder present: VLM is authoritative; raw score is informational.
                    # - No decoder: decision.action was already set above if prox < theta_c.
                    proximity_ok = (decision.action != "reflect")
                    reason = decision.reason if decision.action == "reflect" else "accepted"
                    all_scores = [(mean_tc.item(), mean_prox.item())]
                else:
                    vlm_verification = self.vlm_planner.verify_goal(imagined_img_next, active_instruction)
                    if not vlm_verification["achieved"]:
                        decision = check_rollout_consistency(0.0, self.theta_c)  # dummy failure
                        decision.action = "reflect"
                        decision.reason = f"Goal not achieved (VLM verified): {vlm_verification.get('reason')}"
                        proximity_ok = False
                        reason = decision.reason
                    else:
                        decision = check_rollout_consistency(1.0, self.theta_c)  # dummy success
                        proximity_ok = True
                        reason = "Goal achieved"
                    all_scores = [(1.0, 1.0)]

            # 6. Reflection gate — only "reflect" (semantic failure from Head 1) triggers
            # VLM replanning. "requery" (temporal inconsistency from Head 2) means the
            # imagined transition was physically implausible but we can't re-sample inline,
            # so we log the reject and let execution continue rather than calling the VLM.
            if decision.action == "reflect":
                logger.warning(
                    f"Rejected imagined outcome of '{active_instruction}'. Reason: {reason}"
                )
                # ── Count critic reject ───────────────────────────────────
                self._current_subtask_rec.critic_rejects += 1

                # ── Throttle: skip reflect() if budget exhausted, cooldown active,
                # or the plan hasn't changed since the last reflection. ────
                steps_since_reflect = self._step_count - self._last_reflect_step
                plan_unchanged = (self.plan_queue == self._last_reflected_plan)
                reflect_budget_ok = (self._reflect_count < self.max_replans)
                cooldown_ok = (steps_since_reflect >= self._reflect_cooldown)

                if not reflect_budget_ok:
                    logger.info(
                        "Reflect budget exhausted (%d/%d) for subtask '%s' — skipping VLM call.",
                        self._reflect_count, self.max_replans, active_instruction,
                    )
                elif not cooldown_ok:
                    logger.info(
                        "Reflect cooldown active (%d/%d steps since last) — skipping VLM call.",
                        steps_since_reflect, self._reflect_cooldown,
                    )
                elif plan_unchanged and self._reflect_count > 0:
                    logger.info(
                        "Plan unchanged since last reflection ('%s') — skipping redundant VLM call.",
                        self.plan_queue,
                    )
                else:
                    self._current_subtask_rec.vlm_calls += 1
                    self._current_subtask_rec.reflections += 1
                    self._reflect_count += 1
                    self._last_reflect_step = self._step_count
                    try:
                        reflect_result = self.vlm_planner.reflect(
                            current_image_np=img_np,
                            language_goal=self.current_subgoal,
                            history=self.plan_queue,
                            obj_labels=[],
                            full_plan=self.plan_queue,
                            ctx={
                                "all_scores": all_scores,
                                "imagined_state": imagined_img_next if not is_latent_wm or getattr(self, "decoder", None) is not None else img_np,
                                "failed_step": 0,
                                "failed_action": active_instruction,
                                "failed_highlevel_action": active_instruction,
                                "failure_pattern": reason,
                            },
                        )
                    except VLMRefusalError as refusal_exc:
                        logger.warning(
                            "VLM reflect() was blocked by content policy — keeping current plan. %s",
                            refusal_exc,
                        )
                        reflect_result = {}
                    except Exception as reflect_exc:
                        logger.error(
                            "VLM reflect() failed unexpectedly — keeping current plan. Error: %s",
                            reflect_exc,
                        )
                        reflect_result = {}

                    new_plan = reflect_result.get("revised_plan", self.plan_queue)
                    if new_plan and isinstance(new_plan, list) and len(new_plan) > 0:
                        self._last_reflected_plan = list(new_plan)
                        self.plan_queue = new_plan
                        logger.info(f"VLM Proposed new decomposed plan: {self.plan_queue}")
                        active_instruction = self.plan_queue[0]
                        self.subgoal_step_count = 0
                        action = self.low_level_policy.step(obs, active_instruction)
            else:
                # ── Count critic accept (plan was physically plausible) ───
                self._current_subtask_rec.critic_accepts += 1
                self._pending_critic_accept = True
                # A successful accept resets the stagnation counter so the agent
                # can reflect again if a later step fails.
                self._reflect_count = 0
                self._last_reflected_plan = []
        
        # Convert action to numpy array, as expected by the unwrapped Gym environment
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        if self._step_count <= 2:
            logger.info(
                "Step %d action: %s, rollout_step_counter: %d",
                self._step_count,
                action.tolist() if hasattr(action, "tolist") else list(action),
                self.low_level_policy.model.rollout_step_counter
            )
        return action

    def decode_dino_features(
        self,
        dino_features: torch.Tensor,
        decoder: torch.nn.Module,
    ) -> np.ndarray:
        """Decodes predicted DINO features back to an RGB image."""
        if dino_features.ndim == 2:
            dino_features = dino_features.unsqueeze(0)
        with torch.no_grad():
            rec_img = decoder.decode(dino_features.to(self.device))  # [-1, 1]
            rec_img = (rec_img + 1.0) / 2.0
            rec_img = torch.clamp(rec_img, 0.0, 1.0)
            rec_img = rec_img.squeeze(0).cpu().numpy()  # (3, H, W)
            rec_img = (rec_img * 255.0).astype(np.uint8)
            rgb_img = np.transpose(rec_img, (1, 2, 0))  # (H, W, 3)
            return rgb_img
