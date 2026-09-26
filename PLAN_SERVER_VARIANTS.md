# Plan server: verification variants + episode semantics (lab-PC changes)

Written 2026-09-26 on the Jetson for the lab-PC Claude session in `chrisantuseze/verify2act` (`verify2act/robot/`).
The real-robot evaluation compares the verification variants **v2a_wm, rla_wm, diffusion_wm and vlm_only** under the same
episode loop as the sim's `run_episode` (`verify2act/pipeline/inference.py`). The Jetson side is done (see the end of this
file). The server currently hard-codes `wm_mode="v2a_wm"` (`backend.py`, `BeamSearchPlanner(...)`) and loads only the latent
WM, so it needs the changes below.

---

## 1. Episode semantics now used on the Jetson (context only: no server change)

These now match `run_episode`:

| | sim | real robot (Jetson) |
|---|---|---|
| timestep | one `beam_planner.plan()` call + executing its plan (all of it in `full_plan`) | one `plan` request + executing the reply |
| goal check | `env_wrapper.is_done()` (ground truth) | the operator answers y/n once per executed plan, never per subtask |
| on "n" | re-plan next timestep | the operator names the failed horizons (→ `[FAILED] <subtask>` in `history`) and adds an optional note (→ `feedback`); re-plan next timestep. On the last timestep "n" ends the episode |
| VLM `done` | ends the timestep only; `is_done()` decides success | same: logged, nothing executed, re-planned next timestep |
| unverified after the replan budget | executed anyway | executed anyway (no flag) |
| skill failure | history gets `[FAILED] <subtask>`, re-plan next timestep | same (the rest of that plan is skipped) |
| `max_steps` | 10 | **2** (one re-plan after an "n"); 1 = no re-plan |

What this means for the server:
- **`history` can contain `[FAILED] <subtask>` entries.** The robot prompts already explain them, and `_history_str` passes
  them through. Only a successful subtask appears without the tag.
- **An episode makes at most `max_steps` = 2 `plan` calls.** It makes 1 when the first plan works.
- A `[FAILED]` entry can now also mean **the skill reported success but the operator saw it fail**, e.g. a block that
  bounced out of the bin or was placed on the wrong side.
- **`done` no longer ends an episode**, and `CLIENT_INTEGRATION.md` §3 "the goal head checked it" was not true: `backend.plan`
  sets `done = DONE in plan` whatever the goal head says. The Jetson now reads `accepted` for that. Keep the field, and fix
  the doc wording.

## 2. Required: `--wm-mode` on the server

```
python -m verify2act.robot.server --jetson-ip 192.168.0.8 --wm-mode {v2a_wm,rla_wm,diffusion_wm,vlm_only}
```
The default is `v2a_wm`. One server process runs one variant. A session runs against one server, so the variant is fixed
per session and recorded there.

- **v2a_wm:** as now.
- **rla_wm / diffusion_wm:** build the world model and decoder the way `inference.py` does for `--wm-mode rla_wm` / `diffusion`
  (~lines 793–815 and 1008–1038). Pass `wm_mode=<mode>` to `BeamSearchPlanner`. The critic, thresholds, prompts, vocabulary
  and replan budget stay identical across variants, so the WM is the only thing that changes.
  Name the diffusion variant on the wire as the sim does, or map `diffusion_wm` ↔ `diffusion`, but use one name consistently.
  Checkpoints: use the CALVIN checkpoints of each WM (the same zero-shot footing as v2a_wm), with flags like
  `--latent-wm-ckpt`.
- **vlm_only:** mirror the sim's `beam_planner=None` branch. Make one `planner.propose(...)` call, with no imagination, no
  critic and no reflection. Reply with:
  - `accepted: true` (sim: `plan_accepted = True`)
  - `replan_attempts: 0`, `evaluations: []`, `all_scores: []`, `score: null`, `failed_step: null`
  - `stats: {vlm_calls: 1, plans_evaluated: 0, requeries: 0, temporal_rejections: 0, goal_rejections: 0}`
  - `done` and `invalid_steps` as now

  It does not need the WM or critic in GPU memory: skip loading them and skip the warm-up.
- Every `plan` reply gets a `"wm_mode"` field, the same value as in `ping` (below).

## 3. Required: the server reports its configuration in `ping`

The Jetson records the variant in each session's `summary.json` / `summary.md` from the `ping` reply. It reads these keys, when
present, and falls back to the old defaults otherwise:

```json
{"id": "...", "ok": true, "wm_mode": "v2a_wm", "theta_c": 0.5, "theta_p": 0.05, "max_replans": 2, "max_requery": 2}
```
`max_requery` = the server's `--max-retries`. For `vlm_only`, send the thresholds anyway; they are unused.

`ping` must stay instant and must not take the model lock (it doesn't today).

## 3b. Required: optional `feedback` field on `plan`

On a re-plan the request can carry `"feedback": "<operator note>"`, e.g. `"yellow bounced off the bin rim"`. It is absent or
empty on the first call. It is the operator's diagnosis of the plan that just ran, and it comes with the `[FAILED]` history
entries of the horizons they marked.

- Add it to `RobotPromptManager.build_propose_messages`. It is the prompt used on a re-plan call, so this is where the VLM
  sees it. A block like this, after the history, when non-empty:
  ```
  ### Operator feedback on the previous attempt
  <feedback>
  (Horizons marked [FAILED] above did not achieve their effect in the real scene and must be redone.)
  ```
  Also pass it into the reflect context (`build_reflect_messages`), so reflections within the same call keep it.
- Pass it through `handle_request` → `backend.plan(..., feedback=r.get("feedback", ""))` → `beam_planner.plan`, or set it on the
  prompt manager for the duration of the call if `BeamSearchPlanner` should stay unchanged. Save it in `request.json`.
- The same for every variant, `vlm_only` included. It is part of the planning input, not of the verifier.
- Until this is in, the server ignores the field (`handle_request` reads named keys only). The `[FAILED]` tags still work.

## 4. Tests / checks on the lab PC

1. `test_robot_server.py`:
   - `ping` returns the five keys;
   - a `vlm_only` backend with a fake VLM gives `accepted=True`, empty `evaluations`, and 1 VLM call;
   - `plan` replies carry `wm_mode`;
   - `feedback` appears in the propose prompt when set, and not at all when empty.
2. Start each variant once. From the Jetson, `--plan_server --dry_run --task task2a` should print `Variant: <mode>` (session) and
   return the one-subtask plan.

## Jetson side (done, `dofbot-controller`)

- `v2a_pipeline.py::_run_plan_server` follows §1:
  - operator y/n once per executed plan; on "n", failed horizons → `[FAILED]` and the note → `feedback`;
  - `--max_steps` defaults to 2;
  - unverified plans always executed;
  - `[FAILED]` history on a skill failure;
  - VLM `done` does not end the episode;
  - a server/Gemini error aborts the episode as infrastructure (`aborted: true`).
- `remote/planner_client.py`: `PlanServerClient.settings` is read from `ping`.
- `v2a_session.py`:
  - meta records `wm_mode`, the thresholds, `execute_unverified=true` and the success-check method;
  - the operator's answer becomes `real_success`;
  - the summary adds `total_unverified_executions`, `total_skill_failures` and `aborted_episodes`.
