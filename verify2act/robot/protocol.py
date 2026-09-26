"""Wire format between the Jetson (robot + episode loop) and the lab-PC Verify2Act server.

Transport: rosbridge topics of type ``std_msgs/String`` carrying JSON. Images are base64 JPEG.
The lab PC connects OUT to the Jetson's rosbridge (``ws://<JETSON_IP>:9090``), so nothing listens
on the lab PC and the Jetson opens no extra ports.

    /v2a/request   Jetson -> lab PC   {"id", "op", ...}
    /v2a/response  lab PC -> Jetson   {"id", "ok", "error"?, ...}

ops
    ping    {}                                   -> {wm_mode, theta_c, theta_p, max_replans, max_requery}
    reset   {session}                            -> {}      new episode: restart the planning-call counter
    plan    {session, image, goal, history?, obj_labels?, horizon?, feedback?}
            -> {plan[str], accepted, score, all_scores[[tc, prox]], failed_step, replan_attempts,
                reflection_analyses[str], critic_decisions[str], invalid_steps[str], done,
                evaluations[{plan, tc, goal, accepted, temporal_rejected, goal_rejected, requeries}],
                stats{vlm_calls, plans_evaluated, requeries, temporal_rejections, goal_rejections},
                wm_mode, planning_call, elapsed_s}

``plan`` runs the same loop as the sim (``BeamSearchPlanner.plan``): the VLM proposes candidate subtask
plans, the latent world model imagines each horizon from the current frame, the critic's temporal head
gates every horizon and its goal head gates the final one, and a rejected plan goes through the
reflect -> replan loop (fixed budget). ``image`` is the robot's CURRENT frame (receding horizon), and
``history`` is the list of subtasks already executed, with failed ones prefixed ``[FAILED] ``.
``feedback`` (re-plan calls only) is the operator's note on the previous attempt; it goes into the propose and reflect
prompts of every variant.
When the VLM judges the goal already satisfied, the reply has ``done: true`` and an empty ``plan``. ``done`` is the VLM's
claim; whether the goal head agreed on the real frame is in ``accepted``. The server runs one variant (``--wm-mode``
v2a_wm | rla_wm | diffusion_wm | vlm_only, see backend.py); diffusion_wm and vlm_only have no critic and always reply ``accepted: true``.

Timeouts (Jetson side): a ``plan`` call makes up to 1 + max_replans VLM calls (plus Gemini rate-limit retries) and up
to beam_width + max_replans imagined rollouts, so it takes tens of seconds to a few minutes. Wait at least
``PLAN_REPLY_TIMEOUT_S`` for its reply; ``ping`` / ``reset`` answer immediately.
"""

import base64

import cv2
import numpy as np

REQUEST_TOPIC = "/v2a/request"
RESPONSE_TOPIC = "/v2a/response"
MSG_TYPE = "std_msgs/String"

PLAN_REPLY_TIMEOUT_S = 600.0   # Jetson-side wait for a `plan` reply
PING_REPLY_TIMEOUT_S = 5.0


def encode_image(img_bgr: np.ndarray, quality: int = 92) -> str:
    """BGR uint8 image -> base64 JPEG (the Jetson side encodes camera frames this way)."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_image_rgb(b64: str) -> np.ndarray:
    """base64 JPEG -> RGB uint8 image (the WM, critic and VLM all expect RGB)."""
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image payload")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
