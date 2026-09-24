# Verify2Act plan server: client integration guide (Jetson side)

Written 2026-09-24 on the lab GPU PC for the Claude session that works on the **Jetson** in `chrisantuseze/dofbot-controller`.
It describes the server this repo now provides (`verify2act/robot/` in `chrisantuseze/verify2act`, branch `main-wm`) and exactly what
the Jetson client must change to get plans from it and execute them. Everything referenced on the Jetson side is on
`dofbot-controller` `origin/main` (commit `2d86d23`).

---

## 1. What changes, in one paragraph

Today the Jetson loop (`verify2act/v2a_pipeline.py`) plans with a local VLM stub, verifies with stub or remote `evaluate_plan`, and
reflects locally. With the plan server, **one `plan` request per timestep does all of that on the lab PC**:
1. Gemini proposes candidate plans.
2. The real latent world model imagines each horizon from the current frame.
3. The real critic's temporal head gates every horizon, and its goal head gates the last one.
4. A rejected plan goes through reflect → replan (budget 2), exactly as in the sim (`BeamSearchPlanner.plan`).

The reply is a verified subtask plan plus the verdict and counters. The Jetson keeps everything physical:
- the episode loop and the session runner;
- homing at episode start, and `locate` before a pick;
- executing skills through `lang_color_grasp.py`;
- transition logging and the human y/n labels.

## 2. Who runs what

| Machine | Process | Notes |
|---|---|---|
| Jetson 192.168.0.8 | roscore, `arm_driver.py`, camera, kinematics, `lang_color_detect.py`, `lang_color_grasp.py` | unchanged |
| Jetson | `roslaunch rosbridge_server rosbridge_websocket.launch` (port 9090) | both the server and the Jetson client connect to it |
| Jetson | `v2a_session.py` / `v2a_pipeline.py` with the changes in §5–§6 | connects to rosbridge at `127.0.0.1` |
| Lab PC 192.168.0.113 | `python -m verify2act.robot.server --jetson-ip 192.168.0.8` | connects OUT to the Jetson's rosbridge; the Jetson opens no ports |

Both ends are ordinary rosbridge clients. rosbridge relays the two `std_msgs/String` topics between them, the same way the old
`remote/v2a_server.py` worked.

### Start-up order
1. Jetson: roscore, then the robot stack, then **rosbridge**.
2. Lab PC:
   ```bash
   conda activate verify2act
   cd ~/Projects/multi-object-manip/verify2act
   python -m verify2act.robot.server --jetson-ip 192.168.0.8
   ```
   Model load plus warm-up takes **about 4 minutes**. The server is ready when it logs
   `Ready: listening on /v2a/request, answering on /v2a/response`.
3. Jetson: start the session (§7). Its `wait_for_server()` pings until the server answers.

If rosbridge restarts, the server logs `Lost the rosbridge connection.` and exits. Restart it after rosbridge is back.

## 3. Wire protocol

Topics (both `std_msgs/String`, JSON in `data`):
- `/v2a/request`: Jetson → lab, `{"id": "<unique>", "op": "...", ...}`
- `/v2a/response`: lab → Jetson, `{"id": "<same id>", "ok": true|false, "error"?: "...", ...}`

This is the same envelope the Jetson's existing `verify2act/remote/proxies.py::ServerLink` already speaks (ids, pending map,
`ok`/`error` handling), so **reuse `ServerLink` unchanged**. Images use the Jetson's existing `remote/protocol.py::encode_image`: a BGR
`cv2` frame → JPEG q92 → base64. The server decodes the image and converts it to RGB itself.

The server handles **one model request at a time** (single GPU). `ping` is answered even while a `plan` is running.

### ops

| op | request fields | reply fields (besides `id`, `ok`) | typical time |
|---|---|---|---|
| `ping` | – | – | < 0.1 s |
| `reset` | `session` | – (restarts that session's planning-call counter) | < 0.1 s |
| `plan` | `session`, `image`, `goal`, `history`, `obj_labels`?, `horizon`? | see below | 20–60 s, up to a few minutes |

`plan` request:
- `session` (str): unique per episode. It names the server's log folder `verify2act/output/real/<session>/`. Use
  `"<session_name>_<episode_id>"`, e.g. `task1a_20260925_101500_ep_003`, so episodes of different sessions don't overwrite each other.
- `image` (str): base64 JPEG of the **current** camera frame (640×480 BGR from `RemoteRobotClient.capture_frame()`).
- `goal` (str): the language goal, e.g. `TASK_PRESETS["task1a"]`.
- `history` (list[str]): subtasks **already executed successfully** this episode, in order, exactly as their `action_text`.
  This is the list `_execute()` appends to.
- `obj_labels` (list[str], optional): block colours in the scene (default: all four). `["red","green","blue","yellow"]` matches the eval scene setup.
- `horizon` (int, optional): max subtasks per plan (server default 4; the longest eval plan is 3).

`plan` reply:

| field | type | meaning |
|---|---|---|
| `plan` | list[str] | subtasks to execute, in the vocabulary of §4. **Empty only when `done` is true.** |
| `accepted` | bool | the critic accepted this plan (temporal head at every horizon and goal head at the end) within the replan budget |
| `done` | bool | the VLM judged the goal already satisfied in the current frame, and the goal head checked it on the real frame |
| `invalid_steps` | list[str] | plan steps outside the vocabulary (should be empty; if not, do not execute) |
| `score` | float/null | goal-head score of the returned plan (null if the rollout was aborted before the goal head) |
| `all_scores` | [[tc, goal]] | per horizon, for the returned plan: temporal score, and goal score on the last horizon |
| `failed_step` | int/null | horizon index where the returned plan was rejected |
| `replan_attempts` | int | reflect → replan cycles used (0–2) |
| `reflection_analyses` | list[str] | the VLM's one-line diagnoses from each reflection |
| `critic_decisions` | list[str] | human-readable critic log for the returned plan |
| `evaluations` | list[obj] | **every** plan evaluated in this call: `{plan, tc[], goal, accepted, temporal_rejected, goal_rejected, requeries}` |
| `stats` | obj | `{vlm_calls, plans_evaluated, requeries, temporal_rejections, goal_rejections}` for this call; use these for the session metrics |
| `planning_call` | int | 0, 1, 2, … within the session (it names the server log folder `planning_call_XX`) |
| `elapsed_s` | float | server time for this call |

Errors: `ok: false` with `error`. The main cases:
- `"RuntimeError: VLM produced no plan ..."`: Gemini failed (API error, quota) or gave an unparseable reply.
- A bad request (missing field, undecodable image).

`ServerLink.call` raises `RuntimeError` for these, and `TimeoutError` if no reply arrives in time. **Never treat an error as an empty
plan.** An empty plan means "done" only when `done` is true.

### Timeouts
- `plan`: wait **600 s** (`PLAN_REPLY_TIMEOUT_S`). A call makes at most 3 Gemini calls (1 propose + 2 reflect), each with a 60 s HTTP
  timeout plus rate-limit back-off. The imagination and critic take < 1 s per plan once warm. `ServerLink`'s default of 60 s is too short.
- `ping` / `reset`: 5 s.
- A reply that arrives after the client has given up is ignored by `ServerLink` (unknown id), so a late reply is harmless.

## 4. Plan vocabulary → skill commands

The server only ever returns these strings (`<c>`, `<b>` ∈ red, green, blue, yellow). They are exactly the strings the stub planner
produced, so the existing `v2a_goal.parse_step()` / `parse_relative()` and `_execute()` handle them.

| subtask text | `parse_step` kind | `Subtask(color, target_placement, kind, base_color)` | `/subtask_cmd` sent by `_execute` |
|---|---|---|---|
| `pick and place <c> block into the bin` | `pick_place` | `(<c>, "left_bin", "pick_place", None)` | `pick_place` |
| `pick <c> block` | `pick` | `(<c>, "hold", "pick", None)` | `locate <ref>` first (if the goal has a reference block), then `pick` |
| `place <c> block on <b> block` | `place_on` | `(<c>, "on_<b>", "place_on", <b>)` | `place_on` |
| `place <c> block to the left of <b> block` | `place_at` | `(<c>, "left_of", "place_at", <b>)` | `place_at` |
| `place <c> block to the right of <b> block` | `place_at` | `(<c>, "right_of", "place_at", <b>)` | `place_at` |

`parse_step` falls back to `"red"` when it finds no colour, so **validate the text before converting** (the client below does).
Homing: keep the single `reset` at episode start in `run()`. `locate`: keep the existing `_ref_block` logic in `_execute()`, since the reference
block still comes from the goal.

## 5. Client code

### 5.1 New file `verify2act/remote/planner_client.py`

```python
"""
verify2act/remote/planner_client.py
===================================
Client for the lab-PC Verify2Act plan server (research repo: verify2act/robot/server.py).
One `plan` round trip = VLM proposal + world-model imagination + critic gating + reflect/replan, all on the server.
"""
import logging
import re
from typing import List, Optional

import numpy as np

from verify2act.remote.protocol import encode_image
from verify2act.remote.proxies import ServerLink
from verify2act.v2a_goal import parse_relative, parse_step
from verify2act.v2a_vlm_planner import COLOR_CODE_MAP, Subtask

logger = logging.getLogger(__name__)

PLAN_REPLY_TIMEOUT_S = 600.0
PING_REPLY_TIMEOUT_S = 5.0

_C = "(red|green|blue|yellow)"
_VALID = [re.compile(p) for p in (
    rf"^pick and place {_C} block into the bin$",
    rf"^pick {_C} block$",
    rf"^place {_C} block on {_C} block$",
    rf"^place {_C} block to the (left|right) of {_C} block$",
)]


def is_valid_subtask(text: str) -> bool:
    return any(p.match(text) for p in _VALID)


def subtask_from_text(text: str) -> Subtask:
    """Server subtask string -> Subtask for _execute(). Raises ValueError outside the vocabulary."""
    text = " ".join(text.strip().lower().split())
    if not is_valid_subtask(text):
        raise ValueError(f"subtask outside the robot vocabulary: {text!r}")
    kind, color, base = parse_step(text)
    if kind == "pick_place":
        target = "left_bin"
    elif kind == "pick":
        target = "hold"
    elif kind == "place_on":
        target = "on_" + base
    else:                                   # place_at
        target = parse_relative(text)[1]    # "left_of" | "right_of"
    return Subtask(action_text=text, color=color, color_id=COLOR_CODE_MAP[color], target_placement=target,
                   kind=kind, base_color=base)


class PlanServerClient:
    """Uses the rosbridge connection the RemoteRobotClient already holds."""

    def __init__(self, ros):
        self.link = ServerLink(ros, timeout=PLAN_REPLY_TIMEOUT_S)

    def wait_for_server(self, attempts: int = 60) -> None:
        self.link.wait_for_server(attempts=attempts)      # pings every ~2 s

    def reset(self, session: str) -> None:
        self.link.call("reset", timeout=PING_REPLY_TIMEOUT_S, session=session)

    def plan(self, session: str, frame_bgr: np.ndarray, goal: str, history: List[str],
             obj_labels: Optional[List[str]] = None, horizon: Optional[int] = None) -> dict:
        payload = {"session": session, "image": encode_image(frame_bgr), "goal": goal, "history": list(history)}
        if obj_labels:
            payload["obj_labels"] = list(obj_labels)
        if horizon:
            payload["horizon"] = int(horizon)
        return self.link.call("plan", timeout=PLAN_REPLY_TIMEOUT_S, **payload)   # RuntimeError / TimeoutError on failure

    def close(self) -> None:
        self.link.close()
```

### 5.2 `verify2act/v2a_pipeline.py`

**Constructor**: add a `plan_server: bool = False` argument. After the robot client is created, and next to the existing `wm_server` block:

```python
        self.planner_client = None
        if plan_server:
            ros = getattr(self.robot_client, "_ros", None)
            if ros is None:
                raise RuntimeError("--plan_server needs the rosbridge connection: pass --jetson_ip <ip> (127.0.0.1 on the Jetson).")
            if wm_server:
                raise RuntimeError("--plan_server replaces --wm_server; pass only one.")
            if simulate_reprompt or simulate_temporal_inconsistency or simulate_uncertainty:
                raise RuntimeError("The simulate_* ablations are stub features and are not supported with --plan_server.")
            from verify2act.remote.planner_client import PlanServerClient
            self.planner_client = PlanServerClient(ros)
            self.planner_client.wait_for_server()
            logger.info("[Pipeline] Planning + verification are served by the lab-PC plan server.")
```
In `close()`, also close `self.planner_client` if it is set.

**Episode**: at the top of `_run(self, goal)`, dispatch to the new loop:
```python
        if self.planner_client is not None:
            return self._run_plan_server(goal)
```
and add:

```python
    def _run_plan_server(self, goal: str) -> bool:
        """Receding horizon with the plan server: observe -> `plan` (propose + imagine + critics + reflect, remote)
        -> execute -> re-observe, until the server reports `done` or the step budget is spent."""
        from verify2act.remote.planner_client import subtask_from_text
        res = self.last_result
        print_banner(f"VERIFY2ACT (plan server): GOAL = '{goal}'")
        s_init = self.observe()
        cv2.imwrite(str(self.output_dir / "s_init.jpg"), s_init)
        obs, history = s_init, []
        session = f"{self.output_dir.parent.name}_{self.episode_id}"
        self.planner_client.reset(session)

        for t in range(self.max_steps):
            res["steps"] = t + 1
            print_banner(f"TIMESTEP {t + 1} / {self.max_steps}")
            try:
                r = self.planner_client.plan(session, obs, goal, history)
            except (RuntimeError, TimeoutError) as e:
                logger.error(f"[Plan server] {e}")
                res["reject_reasons"].append(f"t{t + 1}: server error: {e}")
                return False

            st = r["stats"]
            res["vlm_calls"] += st["vlm_calls"]
            res["replans"] += r["replan_attempts"]
            res["requeries"] += st["requeries"]
            res["temporal_rejections"] += st["temporal_rejections"]
            res["goal_rejections"] += st["goal_rejections"]
            res["critic_rejects"] += sum(not e["accepted"] for e in r["evaluations"])
            for i, e in enumerate(r["evaluations"]):
                print(f"  [eval {i + 1}] {e['plan']}  tc={e['tc']}  goal={e['goal']}  accepted={e['accepted']}")
            for a in r["reflection_analyses"]:
                print(f"  [reflect] {a}")

            if r["done"]:
                print("[Plan server] Goal judged complete (VLM 'done', confirmed by the goal head).")
                res["goal_reached_real"] = True
                return True
            if r["invalid_steps"]:
                logger.error(f"[Plan server] Plan has subtasks outside the vocabulary: {r['invalid_steps']}; not executing.")
                return False
            if r["accepted"]:
                res["critic_accepts"] += 1
                res["verified"] = True
            else:
                res["reject_reasons"].append(f"t{t + 1}: not verified after {r['replan_attempts']} replans")
                if not self.execute_unverified:
                    logger.error("[Verify2Act] Replan budget exhausted without a verified plan; not executing.")
                    return False

            plan = [subtask_from_text(s) for s in r["plan"]]
            res["plan"] = list(r["plan"])
            self._print_plan(plan, r["replan_attempts"])
            if self.dry_run:
                print("\n[DRY RUN]: verification only, robot not moved.")
                return bool(r["accepted"])

            actions = plan[:1] if self.exec_mode == "step_by_step" else plan
            if not self._execute(actions, history, goal_step=t + 1):
                res["goal_reached_real"] = False
                return False
            obs = self.observe()
            cv2.imwrite(str(self.output_dir / f"real_after_step{t + 1}.jpg"), obs)

        return False   # step budget spent without `done`; the human y/n label decides real success
```

Notes on this loop:
- **Termination.** After a full plan executes, the next timestep's `plan` call normally returns `done: true`. Keep `max_steps` ≥ 2
  (default 4). The old local `critic.goal_sim_with_uncertainty` real-frame check is **not used** with the plan server: the server's
  `done` (VLM, confirmed by the goal head) replaces it, and the human label remains the ground truth.
- **Imagined timelines.** The server does not send imagined frames back. They are saved on the lab PC under
  `verify2act/output/real/<session>/imagination_logs/planning_call_XX/` (decoded frames per candidate/horizon, critic JSONs,
  `request_image.png`, `request.json`, `response.json`). `_save_timeline` is not called in this path.
- **Thresholds.** θ_c = 0.5 and θ_p = 0.05 live on the server (the sim values for these checkpoints). The Jetson's `--theta_c/--theta_p`
  flags (θ_p 0.6 was a stub value) are **ignored** with `--plan_server`. Write the server's values into the session meta for the record.
- `--max_replans` and `--max_requery` on the Jetson are likewise ignored. The server uses max_replans = 2 and max_retries = 2.

### 5.3 CLI (`add_loop_args` / `loop_kwargs` in `v2a_pipeline.py`)
```python
    ap.add_argument("--plan_server", action="store_true",
                    help="plan + verify on the lab-PC Verify2Act server (verify2act/robot/server.py); needs --jetson_ip")
```
and add `"plan_server"` to the `keys` list in `loop_kwargs()`. `v2a_session.py` then picks it up automatically, because it forwards
`loop_kwargs(args)`. In the session meta, set `backend` to include `plan_server`, and record `theta_c=0.5, theta_p=0.05, max_replans=2`
as the effective server values.

## 6. Running
```bash
# Jetson, after the robot stack + rosbridge are up and the lab server logs "Ready":
python3 verify2act/v2a_pipeline.py --task task1a --jetson_ip 127.0.0.1 --plan_server --dry_run     # verify only, arm still
python3 verify2act/v2a_pipeline.py --task task1a --jetson_ip 127.0.0.1 --plan_server               # one live episode
python3 verify2act/v2a_session.py  --task task1a --episodes 10 --jetson_ip 127.0.0.1 --plan_server # evaluation session
```
Use `--jetson_ip 127.0.0.1` (not `--local`): the plan client rides on `RemoteRobotClient`'s rosbridge connection.

## 7. Test procedure (the arm moves only in step 4, so be present)
1. **Unit check (no robot):** `subtask_from_text` on each row of §4 gives the listed `Subtask`, and it raises on
   `"pick up the red block"`.
2. **Connectivity:** with the lab server "Ready", `PlanServerClient(ros).wait_for_server()` returns, and `reset("smoke")` succeeds.
3. **Dry run** (`--dry_run --plan_server`, task1a): the log shows one `plan` round trip, the evaluations, and `accepted`/`plan`. The lab PC
   has `verify2act/output/real/<session>/imagination_logs/planning_call_00/`.
4. **Live, task1a:** executes `pick and place blue block into the bin`, then `... yellow ...`, then the next `plan` returns `done`.
5. **Family 2** (task2a): the plan is `pick red block` → `place red block to the left of blue block`, and `_execute` runs `locate blue`
   before the pick.

## 8. Things to know
- **The critic is off the shelf (CALVIN-trained) and is not reliable on real frames yet.** An offline check on a real frame showed
  the goal head at chance for good vs. bad plans, and its score around θ_p varies from run to run. Expect acceptances and rejections to be
  noisy; `accepted: false` can happen for correct plans. Decide per session whether to pass `--execute_unverified`, and record it.
- Gemini runs as `gemini-2.5-flash` on Vertex AI (GCP project `verify2act`), like the sim runs. A Gemini outage shows up as an
  `ok: false` reply, never as an empty plan.
- Each `plan` call re-grounds on the current frame and history. The server keeps no scene state between calls. Only the per-session
  call counter (log naming) persists, and `reset` clears it.
- `remote/v2a_server.py` and the `evaluate_plan` / `goal_check` ops are **not** served by the new server. `--wm_server` is superseded by `--plan_server`.
- Server-side reference: `verify2act/robot/protocol.py` (wire format + timeouts), `server.py` (CLI defaults), `backend.py` (what `plan` runs),
  and `REAL_ROBOT_PLAN.md` (status, checkpoint notes).
