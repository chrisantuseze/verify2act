"""Offline tests for the real-robot server: fake VLM / world model / critic, real BeamSearchPlanner.

    python -m pytest verify2act/robot/test_robot_server.py -q
"""

import json
import threading

import numpy as np
import pytest
import torch

from verify2act.pipeline.planner import BeamSearchPlanner
from verify2act.pipeline.world_model import LatentWorldModel
from verify2act.robot.backend import Verify2ActBackend
from verify2act.robot.prompts import RobotPromptManager, expand_subtask_plan, is_valid_subtask, step_text
from verify2act.robot.protocol import decode_image_rgb, encode_image
from verify2act.robot.server import handle_request

GOOD = "pick and place blue block into the bin"
GOOD2 = "pick and place yellow block into the bin"
BAD = "pick and place red block into the bin"   # the fake critic finds this step temporally inconsistent
ACTIONS = [GOOD, GOOD2, BAD]


class FakeWM(LatentWorldModel):
    """Latent WM whose imagined feature value encodes which subtask was imagined."""

    def __init__(self):
        self.history_len = 3
        self._history = None

    def initialize_history(self, start_img_np):
        self._history = torch.full((1, 3, 4, 8), -1.0)
        self._history_mask = torch.tensor([[False, False, True]])

    def imagine(self, current_image_np, action_text):
        F_next = torch.full((1, 4, 8), float(ACTIONS.index(action_text)))
        self._history = torch.cat([self._history[:, 1:], F_next.unsqueeze(1)], dim=1)
        return F_next, 0.0


class FakeCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def encode_features(self, F):
        return F.mean().reshape(1)

    def temporal_sim_with_uncertainty(self, emb_prev, emb_next):
        bad = int(emb_next.item()) == ACTIONS.index(BAD)
        return torch.tensor(0.1 if bad else 0.9), torch.tensor(0.0)

    def goal_sim_from_text_with_uncertainty(self, emb, goal):
        return torch.tensor(0.3), torch.tensor(0.0)


class FakeVLM:
    def __init__(self, candidates, revisions):
        self.candidates, self.revisions = candidates, list(revisions)
        self.reflect_calls = 0

    def propose_candidates(self, **kw):
        return self.candidates

    def reflect(self, **kw):
        self.reflect_calls += 1
        assert kw["ctx"]["imagined_state"].shape[-1] == 3, "reflection should see a decoded RGB image"
        return {"analysis": "red must stay", "revised_plan": self.revisions.pop(0) if self.revisions else [BAD]}


def make_backend(candidates, revisions=(), max_replans=2, output_dir=None):
    vlm = FakeVLM(candidates, revisions)
    bp = BeamSearchPlanner(vlm_planner=vlm, world_model=FakeWM(), critic=FakeCritic(), beam_width=2,
                           goal_threshold=0.05, plan_expander=expand_subtask_plan, temporal_threshold=0.5,
                           max_retries=2, max_replans=max_replans, wm_mode="v2a_wm")
    bp.decode_dino_features = lambda feats, decoder: np.zeros((8, 8, 3), np.uint8)
    return Verify2ActBackend(bp, decoder=object(), horizon=4, output_dir=output_dir), vlm


IMG = np.zeros((48, 64, 3), np.uint8)
GOAL = "Put the blue block and the yellow block into the bin"


def test_good_plan_accepted_without_reflection(tmp_path):
    backend, vlm = make_backend([[GOOD, GOOD2]], output_dir=tmp_path)
    out = backend.plan("s1", IMG, GOAL)
    assert out["accepted"] and out["plan"] == [GOOD, GOOD2]
    assert out["replan_attempts"] == 0 and vlm.reflect_calls == 0
    assert len(out["all_scores"]) == 2 and out["all_scores"][-1][1] == pytest.approx(0.3)
    assert out["invalid_steps"] == [] and not out["done"]
    call_dir = tmp_path / "s1" / "imagination_logs" / "planning_call_00"
    assert json.loads((call_dir / "response.json").read_text())["plan"] == [GOOD, GOOD2]


def test_rejected_plan_is_reflected_and_replanned():
    backend, vlm = make_backend([[BAD, GOOD]], revisions=[[GOOD, GOOD2]])
    out = backend.plan("s1", IMG, GOAL)
    assert out["accepted"] and out["plan"] == [GOOD, GOOD2]
    assert out["replan_attempts"] == 1 and out["reflection_analyses"] == ["red must stay"]


def test_stats_count_every_evaluation():
    backend, _ = make_backend([[BAD], [BAD, GOOD]], revisions=[[GOOD, GOOD2]])
    out = backend.plan("s1", IMG, GOAL)
    ev = out["evaluations"]
    assert [e["plan"] for e in ev] == [[BAD], [BAD, GOOD], [GOOD, GOOD2]]   # 2 candidates + 1 revised plan
    assert [e["temporal_rejected"] for e in ev] == [True, True, False] and ev[-1]["accepted"]
    st = out["stats"]
    assert st["plans_evaluated"] == 3 and st["temporal_rejections"] == 2 and st["goal_rejections"] == 0
    assert st["requeries"] == 2 and st["vlm_calls"] == 2   # fake VLM has no _call: 1 propose + 1 reflect


def test_replan_budget_is_fixed():
    backend, vlm = make_backend([[BAD]])   # sim default: 2 replans
    out = backend.plan("s1", IMG, GOAL)
    assert not out["accepted"]
    assert out["replan_attempts"] == 2 and vlm.reflect_calls == 2


def test_done_is_verified_and_sent_as_empty_plan():
    backend, vlm = make_backend([["done"]])
    out = backend.plan("s1", IMG, GOAL)
    assert out["done"] and out["accepted"] and out["plan"] == [] and out["all_scores"] == []
    assert vlm.reflect_calls == 0


def test_vlm_failure_is_an_error_not_an_empty_plan():
    backend, _ = make_backend([[]], revisions=[[]])   # what BeamSearchPlanner returns after swallowing API errors
    with pytest.raises(RuntimeError, match="VLM produced no plan"):
        backend.plan("s1", IMG, GOAL)
    backend, _ = make_backend([[]], revisions=[[]])
    reply = handle_request(backend, json.dumps({"id": "x", "op": "plan", "image": encode_image(IMG), "goal": GOAL}),
                           threading.Lock())
    assert reply["ok"] is False and "no plan" in reply["error"]


def test_best_candidate_wins_over_failed_one():
    backend, vlm = make_backend([[BAD], [GOOD, GOOD2]])
    out = backend.plan("s1", IMG, GOAL)
    assert out["accepted"] and out["plan"] == [GOOD, GOOD2] and vlm.reflect_calls == 0


def test_request_roundtrip_and_session_counter():
    backend, _ = make_backend([[GOOD]])
    lock = threading.Lock()
    img_b64 = encode_image(IMG)
    assert decode_image_rgb(img_b64).shape == IMG.shape

    assert handle_request(backend, json.dumps({"id": "a", "op": "ping"}), lock) == {
        "id": "a", "ok": True, "wm_mode": "v2a_wm", "theta_c": 0.5, "theta_p": 0.05, "max_replans": 2, "max_requery": 2}
    req = {"id": "b", "op": "plan", "session": "ep0", "image": img_b64, "goal": GOAL,
           "history": ["[FAILED] pick and place blue block into the bin"], "obj_labels": ["blue", "yellow"]}
    r1 = handle_request(backend, json.dumps(req), lock)
    r2 = handle_request(backend, json.dumps(req), lock)
    assert r1["ok"] and r1["plan"] == [GOOD] and r1["wm_mode"] == "v2a_wm" and (r1["planning_call"], r2["planning_call"]) == (0, 1)
    assert handle_request(backend, json.dumps({"id": "c", "op": "reset", "session": "ep0"}), lock)["ok"]
    assert handle_request(backend, json.dumps(req), lock)["planning_call"] == 0

    bad = handle_request(backend, json.dumps({"id": "d", "op": "nope"}), lock)
    assert bad["ok"] is False and "unknown op" in bad["error"]
    assert handle_request(backend, "not json", lock) is None


class FakeProposeVLM:
    """VLMPlanner stand-in for vlm_only: propose() goes through _call, which the backend counts."""

    def __init__(self, plan):
        self.plan = plan

    def _call(self, messages, temperature=None):
        return json.dumps({"plan": self.plan})

    def propose(self, **kw):
        return json.loads(self._call([]))["plan"]


def test_vlm_only_is_one_unverified_propose():
    settings = {"theta_c": 0.5, "theta_p": 0.05, "max_replans": 2, "max_requery": 2}
    backend = Verify2ActBackend(None, wm_mode="vlm_only", vlm=FakeProposeVLM([GOOD, GOOD2]), settings=settings)
    lock = threading.Lock()
    ping = handle_request(backend, json.dumps({"id": "p", "op": "ping"}), lock)
    assert ping == {"id": "p", "ok": True, "wm_mode": "vlm_only", **settings}
    out = handle_request(backend, json.dumps({"id": "q", "op": "plan", "image": encode_image(IMG), "goal": GOAL}), lock)
    assert out["ok"] and out["plan"] == [GOOD, GOOD2] and out["wm_mode"] == "vlm_only"
    assert out["accepted"] is True and out["replan_attempts"] == 0 and out["evaluations"] == []
    assert out["all_scores"] == [] and out["score"] is None and out["failed_step"] is None
    assert out["stats"] == {"vlm_calls": 1, "plans_evaluated": 0, "requeries": 0, "temporal_rejections": 0,
                            "goal_rejections": 0}
    done = Verify2ActBackend(None, wm_mode="vlm_only", vlm=FakeProposeVLM(["done"]), settings=settings)
    out = done.plan("s", IMG, GOAL)
    assert out["done"] and out["plan"] == [] and out["accepted"]
    with pytest.raises(ValueError):
        Verify2ActBackend(None, wm_mode="v2a_wm", vlm=FakeProposeVLM([GOOD]), settings=settings)


class FakeImageWM:
    def imagine(self, current_image_np, action_text):
        return current_image_np


class FakeDiffusionVLM:
    def __init__(self):
        self.reflect_calls = 0

    def propose(self, **kw):
        return [GOOD]

    def reflect(self, **kw):
        self.reflect_calls += 1
        return {"analysis": "looks right", "revised_plan": [GOOD, GOOD2]}


def test_diffusion_wm_reflects_once_without_critic():
    vlm = FakeDiffusionVLM()
    bp = BeamSearchPlanner(vlm_planner=vlm, world_model=FakeImageWM(), critic=None, beam_width=3, goal_threshold=0.05,
                           plan_expander=expand_subtask_plan, temporal_threshold=0.5, max_retries=2, max_replans=2,
                           wm_mode="diffusion")
    backend = Verify2ActBackend(bp, wm_mode="diffusion_wm")
    out = backend.plan("s", IMG, GOAL)
    # The sim's ReflectVLM path: one propose, the rollout is always reflected on once, the revision is accepted.
    assert out["accepted"] and out["plan"] == [GOOD, GOOD2] and vlm.reflect_calls == 1
    assert out["wm_mode"] == "diffusion_wm" and out["score"] is None and out["all_scores"] == []
    assert out["stats"]["plans_evaluated"] == 2 and out["stats"]["temporal_rejections"] == 0
    assert backend.settings["wm_mode"] == "diffusion_wm"


def test_server_args_resolve_per_mode_checkpoints():
    from verify2act.robot.server import parse_args
    rla = parse_args(["--wm-mode", "rla_wm"])
    assert "rla_wm/calvin/wm" in rla.latent_wm_ckpt and rla.wm_decoder_dir.endswith("v2a_wm/calvin/decoder")
    assert rla.output_dir.endswith("real/rla_wm")
    diff = parse_args(["--wm-mode", "diffusion_wm"])
    assert diff.wm_decoder_dir.endswith("diffusion_wm/calvin/decoder/checkpoint-5000")
    assert parse_args(["--latent-wm-ckpt", "x.pt"]).latent_wm_ckpt == "x.pt"


FEEDBACK = "yellow bounced off the bin rim"


def _user_text(msgs):
    return "\n".join(b["text"] for b in msgs[-1]["content"] if b["type"] == "text")


def test_feedback_block_only_when_set():
    pm = RobotPromptManager.from_yaml("verify2act/configs/prompts/dofbot/planner.yaml")
    hist = ["[FAILED] " + GOOD2]
    assert "Operator feedback" not in _user_text(pm.build_propose_messages(IMG, GOAL, hist, ["yellow block"], 4))
    pm.feedback = FEEDBACK
    text = _user_text(pm.build_propose_messages(IMG, GOAL, hist, ["yellow block"], 4, num_candidates=3))
    assert "### Operator feedback on the previous attempt\n" + FEEDBACK in text
    assert text.index("[FAILED] " + GOOD2) < text.index(FEEDBACK) < text.index("### Planning request")
    ctx = {"imagined_state": IMG, "all_scores": [(0.1, 0.0)], "failed_step": 0,
           "failed_action": BAD, "failed_highlevel_action": BAD, "failure_pattern": "temporal"}
    assert FEEDBACK in _user_text(pm.build_reflect_messages(IMG, GOAL, hist, ["red block"], [BAD], ctx))


class PromptRecordingVLM(FakeProposeVLM):
    """vlm_only VLM that builds the real propose prompt, to check what the VLM would see."""

    def __init__(self, plan):
        super().__init__(plan)
        self._pm = RobotPromptManager.from_yaml("verify2act/configs/prompts/dofbot/planner.yaml")
        self.prompts = []

    def propose(self, **kw):
        self.prompts.append(_user_text(self._pm.build_propose_messages(
            kw["current_image_np"], kw["language_goal"], kw["history"], kw["obj_labels"], kw["horizon"])))
        return super().propose(**kw)


def test_feedback_reaches_the_prompt_for_one_call_only(tmp_path):
    vlm = PromptRecordingVLM([GOOD2])
    backend = Verify2ActBackend(None, wm_mode="vlm_only", vlm=vlm, output_dir=tmp_path,
                                settings={"theta_c": 0.5, "theta_p": 0.05, "max_replans": 2, "max_requery": 2})
    lock = threading.Lock()
    req = {"id": "f", "op": "plan", "session": "ep", "image": encode_image(IMG), "goal": GOAL,
           "history": ["[FAILED] " + GOOD2], "feedback": FEEDBACK}
    assert handle_request(backend, json.dumps(req), lock)["ok"]
    assert FEEDBACK in vlm.prompts[-1] and vlm._pm.feedback == ""
    saved = json.loads((tmp_path / "ep" / "imagination_logs" / "planning_call_00" / "request.json").read_text())
    assert saved["feedback"] == FEEDBACK
    del req["feedback"]
    assert handle_request(backend, json.dumps(req), lock)["ok"]
    assert "Operator feedback" not in vlm.prompts[-1]


def test_subtask_vocabulary():
    for s in [GOOD, "pick and place red block on blue block", "pick and place green block to the left of yellow block",
              "pick and place red block to the right of blue block", "done"]:
        assert is_valid_subtask(s), s
    for s in ["pick up the red block", "place red block near blue block", "pick and place purple block into the bin",
              "pick red block", "place red block on blue block", "pick and place red block on red block"]:
        assert not is_valid_subtask(s), s
    assert step_text({"label": " Pick and place Red Block into the bin. "}) == "pick and place red block into the bin"
    assert expand_subtask_plan([GOOD, "done"]) == [(GOOD, GOOD)]


def test_prompt_messages_build():
    pm = RobotPromptManager.from_yaml("verify2act/configs/prompts/dofbot/planner.yaml")
    msgs = pm.build_propose_messages(IMG, GOAL, ["[FAILED] " + GOOD], ["blue block", "yellow block"], 4,
                                     num_candidates=3)
    text = "\n".join(b["text"] for b in msgs[-1]["content"] if b["type"] == "text")
    assert msgs[0]["role"] == "system" and "bin" in msgs[0]["content"]
    assert '"plans"' in text and "[FAILED] " + GOOD in text and "nut" not in text.lower()
    system = msgs[0]["content"]
    for phrase in ("warm = red, yellow", "cool = green, blue", "leave, keep", "except",
                   '["pick and place <c> block to the <left|right> of <b> block"]',
                   '["pick and place <c> block on <b> block"]', 'the plan is ["done"]'):
        assert phrase in system, phrase
    ctx = {"imagined_state": IMG, "all_scores": [(0.1, 0.0)], "failed_step": 0,
           "failed_action": BAD, "failed_highlevel_action": BAD, "failure_pattern": "temporal"}
    msgs = pm.build_reflect_messages(IMG, GOAL, [], ["red block"], [BAD], ctx)
    assert any("revised_plan" in b.get("text", "") for b in msgs[-1]["content"]) and msgs[0]["content"].startswith("I am the replanning")


# Real-robot eval tasks (dofbot-controller verify2act/EVAL_TASKS.md) and the plans the Jetson skills expect.
EVAL_TASK_PLANS = {
    "Put the blue block and the yellow block into the bin":
        ["pick and place blue block into the bin", "pick and place yellow block into the bin"],
    "Clear all cool-colored blocks into the bin and leave the yellow block":
        ["pick and place green block into the bin", "pick and place blue block into the bin"],
    "Clear all warm-colored blocks into the bin and leave the green block":
        ["pick and place red block into the bin", "pick and place yellow block into the bin"],
    "Put the red block, the green block and the blue block into the bin except the yellow block":
        ["pick and place red block into the bin", "pick and place green block into the bin",
         "pick and place blue block into the bin"],
    "Put the red block to the left of the blue block": ["pick and place red block to the left of blue block"],
    "Put the green block to the right of the yellow block": ["pick and place green block to the right of yellow block"],
    "Stack the blue block on top of the yellow block": ["pick and place blue block on yellow block"],
}


@pytest.mark.parametrize("goal,plan", EVAL_TASK_PLANS.items())
def test_eval_task_plans_are_in_vocabulary(goal, plan):
    assert all(is_valid_subtask(s) for s in plan)
    assert len(plan) <= 4   # server default horizon
