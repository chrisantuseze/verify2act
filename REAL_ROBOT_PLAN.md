# Verify2Act on the real DOFBOT: lab-PC server (status + pickup doc)

Updated 2026-09-23 on the lab GPU PC (192.168.0.113, RTX 5060 8 GB), branch `main-wm`. Read it with `BACKEND_HANDOFF.md`
(written from the Jetson side). **This supersedes the first version of this doc**: the lab PC does NOT run the episode loop
or drive the arm. There is no env wrapper and no `run_episode`.

## Agreed design

- **Jetson (192.168.0.8)** keeps the episode loop and skill execution (`lang_color_detect.py` / `lang_color_grasp.py`), and runs rosbridge on :9090.
- **Lab PC** is a model server. `verify2act/robot/server.py` connects OUT to the Jetson's rosbridge with roslibpy and answers requests.
- Per planning call, the Jetson sends the **current** frame + language goal + executed-subtask history. The lab PC runs the
  **sim loop unchanged**, `BeamSearchPlanner.plan` (`pipeline/planner.py:1131`):
  1. Gemini proposes `beam_width` candidate subtask plans.
  2. The latent WM is seeded from the real frame's DINO latent and imagines horizon k from horizon k-1 plus subtask k.
  3. The **temporal head** compares consecutive states at every horizon: real→imag1, imag1→imag2, ...
     A low score triggers a requery: re-imagine that step, up to `max_retries`, then reflect.
  4. The **goal head** compares the final imagination with the language goal (text).
  5. A rejected plan goes through reflect→replan with a **fixed budget `max_replans=2`** (the sim default). The VLM sees the decoded imagined scene.
- The reply is the verified plan plus the verdict, and the Jetson executes it. Receding horizon: the Jetson calls again from the new frame.
- `dofbot-controller` is **not used on this machine**. It was only read, via `git show origin/main:...` in `~/Projects/dofbot-controller`,
  to copy its rosbridge request/response pattern (`verify2act/remote/protocol.py`, `proxies.py`, `v2a_server.py`).
- Fine-tuning on real data is out of scope for now.

## Built (untracked, not committed; the user reviews first)

```
verify2act/robot/__init__.py
verify2act/robot/protocol.py        topics, base64-JPEG helpers, op documentation
verify2act/robot/prompts.py         RobotPromptManager (subtask vocabulary), plan expander, vocabulary validator
verify2act/robot/backend.py         loads Gemini planner + LatentWorldModel + critic + FeatureDecoder; plan()/reset()
verify2act/robot/server.py          roslibpy server + CLI (+ --offline-image mode)
verify2act/robot/test_robot_server.py   17 offline tests (fake VLM/WM/critic, real BeamSearchPlanner)
verify2act/robot/CLIENT_INTEGRATION.md  guide for the Jetson session: protocol, client module, v2a_pipeline changes, tests
verify2act/configs/prompts/dofbot/{planner.yaml, system/propose.yaml, system/reflect.yaml}
```
No edits were made to `verify2act/pipeline/`. `roslibpy` was pip-installed into the `verify2act` conda env.

### Wire protocol (`std_msgs/String` JSON on rosbridge)
- `/v2a/request` (Jetson→lab): `{"id", "op", ...}`
- `/v2a/response` (lab→Jetson): `{"id", "ok", "error"?, ...}`

| op | request | reply |
|---|---|---|
| `ping` | – | `wm_mode, theta_c, theta_p, max_replans, max_requery` (never takes the model lock) |
| `reset` | `session` | `{}` (restarts that session's planning-call counter) |
| `plan` | `session, image` (b64 JPEG of a BGR cv2 frame), `goal, history?` (executed subtask strings, as the Jetson loop records them; `[FAILED] ` = skill failed or the operator saw it fail), `obj_labels?` (e.g. `["red","blue"]`, default all 4), `horizon?`, `feedback?` (operator note on the previous attempt, re-plan calls only; goes into every propose/reflect prompt of the call) | `plan` [subtask str], `accepted`, `score` (goal head, null if not reached), `all_scores` [[tc, prox]], `failed_step`, `replan_attempts`, `reflection_analyses`, `critic_decisions`, `invalid_steps`, `done`, `planning_call`, `elapsed_s` |

When the VLM answers `["done"]`, the goal head judges the real frame (there are no imagination steps). The reply is then `done: true` with an **empty** `plan`. `done` is the VLM's claim only; the goal head's verdict is in `accepted`. Since 09-26 the Jetson does not end the episode on `done`. The operator answers y/n once per executed plan, like `is_done()` in the sim, and unverified plans are executed anyway, as in `run_episode`. On "n" the operator marks the failed horizons (→ `[FAILED]` in `history`) and can add a note (→ `feedback`), and the Jetson re-plans. `--max_steps` is 2, so an episode makes at most 2 `plan` calls.

### Variants (`--wm-mode`, 2026-09-26, `PLAN_SERVER_VARIANTS.md`)
One server process = one variant, built like `pipeline/inference_calvin.py`. Every `plan` reply carries `wm_mode`. Logs go to
`verify2act/output/real/<wm_mode>/<session>/`.
- `v2a_wm` (default): as before.
- `rla_wm`: `RLAWorldModel`, `rla_wm/calvin/wm/ckpt/latent_dynamics_best.pt`, with the same v2a encoder, decoder and critic (as the sim's CALVIN rla_wm run).
- `diffusion_wm` (the sim's `diffusion`, ReflectVLM): InstructPix2Pix + `diffusion_wm/calvin/wm/best/unet_lora` + VAE decoder
  `diffusion_wm/calvin/decoder/checkpoint-5000`. **No critic**, as in the sim: 1 propose → imagine → always 1 reflection → accepted.
  `score` / `all_scores` are null / empty (the sim's dummy 1.0 scores are not sent).
- `vlm_only`: 1 `propose`, `accepted: true`, no WM/critic loaded, no warm-up.
- The thresholds are the same for all variants (θ_c 0.5, θ_p 0.05). Note: the sim's CALVIN `rla_wm` command used θ_p 0.2. Pass `--theta-p 0.2` to match it.

**Jetson-side reply timeout: `protocol.PLAN_REPLY_TIMEOUT_S = 600 s`** for `plan` (5 s for `ping`). Measured after warm-up, one 3-step
imagine + critic pass takes about 0.2–0.4 s, so a `plan` call is dominated by Gemini: at most 1 + max_replans = 3 VLM calls, each
with a 60 s HTTP timeout, plus rate-limit backoff. That is about 3 min normally, and 600 s leaves room for backoffs. The Jetson's
`ServerLink` currently defaults to 60 s, so that value has to change in the Jetson session.

Subtask vocabulary (validated by `is_valid_subtask`; `invalid_steps` lists any the VLM got wrong). **Revised 2026-09-25**
(`PLAN_SERVER_VOCAB_CHANGE.md`): one subtask = one WM horizon = one complete pick-and-place that ends with nothing held,
because the arm cannot hold a block through a `plan` call:
`pick and place <c> block into the bin` · `pick and place <c> block on <b> block` ·
`pick and place <c> block to the left of <b> block` · `pick and place <c> block to the right of <b> block` · `done`,
with `<c>` ≠ `<b>`. The retired `pick <c> block` / `place <c> block ...` are rejected. The Jetson runs each compound
subtask as `locate <b>` → `pick <c>` → `place_on|place_at` back to back, and handles `reset`/homing itself.

### Defaults (`server.py --help`)
- Checkpoints (CALVIN **wider**):
  - `v2a_wm/calvin/wm/ckpt/latent_dynamics_best_weights.pt` (renamed from `wm_wider` on 09-24; the old narrow model now lives under `rla_wm/`)
  - `v2a_wm/calvin/encoder/ckpt/delta_encoder_best.pt` (renamed from `encoder_wider`)
  - `calvin/decoder`
  - `contrastive/calvin/best_contrastive_critic.pt`
  - history_len 3, token_dim 128, 32 tokens, cross_attn. These match the CALVIN v2a_wm command in `commands/v2a_wm.sh`.
- beam_width 3, horizon 4, max_retries 2, **max_replans 2**, planner gemini-2.5-pro (8192 tokens), logs in `verify2act/output/real/<session>/`.
- θ_c 0.5 / θ_p 0.05 are the values every sim v2a_wm run in `commands/v2a_wm.sh` uses (the user confirmed: use the sim's). The handoff's stub value θ_p 0.6 would reject everything with this critic.

### Verified so far
- `python -m pytest verify2act/robot/test_robot_server.py -q` → 22 passed (09-26: + vlm_only, diffusion_wm, ping settings, per-mode args, feedback).
- 09-26, on `lear` (192.168.0.116, no GPU): the `rla_wm` CALVIN checkpoint loads into `RLAWorldModel` (strict) on CPU. `diffusion_wm` and a
  GPU run of each variant are still unchecked; do them on the lab GPU PC (.113). Covers: accept path; reflect→replan; budget exhausted after
  exactly 2 reflections; `done` → verified empty plan; best candidate chosen; request round trip; session counter; vocabulary;
  prompt building (checks the eval-task rules are in the system prompt); the expected plan for each eval task is in vocabulary.
- Real checkpoints load, and startup runs a warm-up pass that loads the critic's lazy DINOv2-L (load + warm-up ≈ 110 s). After that,
  a 3-step evaluation takes 0.2–0.4 s; peak VRAM is 4.2 GB. No Gemini call has been made yet.

### VLM prompts (`configs/prompts/dofbot/system/`) are aligned with the eval set
These follow `dofbot-controller` `origin/main:verify2act/EVAL_TASKS.md` and its stub planner `v2a_vlm_planner.py`:
- **Family 1, bin clearing (task1a–d):** one `pick and place <c> block into the bin` per block. Warm = red/yellow, cool = green/blue.
  Never move a leave/keep/except block. Plan only blocks still on the table.
- **Family 2, rearrangement (task2a/b):** the single subtask `pick and place <c> block to the <side> of <b> block`.
- **Family 3, stacking (task3a, deferred):** the single subtask `pick and place <c> block on <b> block`.
- Goal already met → `["done"]`.
- The prompt's few-shot goals deliberately use **different colours** from the eval goals, so the eval tasks are not given away.

**Client side:** everything the Jetson session needs is in `verify2act/robot/CLIENT_INTEGRATION.md`:
- the protocol and reply fields;
- a drop-in `remote/planner_client.py`;
- the `_run_plan_server` loop and the `--plan_server` flag;
- the vocabulary → `Subtask` mapping and the test procedure.

Its client code was checked against the Jetson's `origin/main` modules (extracted with `git archive` into the scratchpad), including
a fake-rosbridge round trip. The `plan` reply also carries `evaluations` (every plan evaluated in the call) and `stats`
(vlm_calls, plans_evaluated, requeries, temporal/goal rejections), because the sim result dict keeps only the final plan's decisions.

## Next steps

1. **Offline with Gemini** (no robot): save one real Jetson frame, then
   `python -m verify2act.robot.server --offline-image frame.jpg --goal "..." --gcp-project verify2act`.
   Check the plan wording, the scores on real images, and the `output/real/offline/` logs.
2. **Live connection**: on the Jetson, start rosbridge. On the lab PC, `python -m verify2act.robot.server --jetson-ip 192.168.0.8`.
   From the Jetson, `ping`, then `plan`.
3. **Jetson-side changes (done in the Jetson session, not here)**: have its loop call `plan` instead of its local VLM stub / `evaluate_plan`.
   Raise the `ServerLink` wait for `plan` to 600 s (`PLAN_REPLY_TIMEOUT_S`).
   Send `reset` at episode start. Stop when `done` is true. Execute only valid steps.
4. Session metrics (success rate, critic precision, replans, VLM calls, ...) are logged Jetson-side from the `plan` replies.

## Open items to raise with the user
- `pipeline/planner.py` (~L84) prints `GEMINI_API_KEY` at startup. The user said to leave it as is.
- The CALVIN critic lazily loads its own DINOv2-L in addition to the WM's extractor (two copies, ~4 GB peak). It fits in 8 GB; sharing
  the backbone is possible later.
- task3a (stacking) stays deferred (handoff §5).

## Off-the-shelf checkpoint check on the real frame (2026-09-24, `verify2act/robot/s_init.jpg`, no VLM calls)
CALVIN-wider and Nut-Assembly WM + critic were compared on the real frame, plus colour-mask-inpainted copies with blocks removed
(crude "goal achieved / wrong blocks moved" proxies):
- **Goal head on real frames is at chance for both.**
  - task1a margin (achieved frame minus best non-achieved frame): CALVIN -0.022, Nut +0.011.
  - task1c margin: CALVIN +0.001, Nut -0.006.
- **Imagined rollouts: correct and wrong plans get about the same goal score, and every one is below θ_p 0.05.**
  - CALVIN: -0.02 to -0.36.
  - Nut: -0.06 to -0.13.
  - So every plan would be rejected, and each planning call would use the full replan budget.
  - The Nut WM gives identical temporal scores whatever the action text (it ignores the language).
  - The CALVIN WM decodes the scene as the CALVIN desk.
  - The Nut decoder keeps the real layout (sheet, block positions), but maps the colours into the robosuite palette.
- Recommendation given: CALVIN (coloured blocks + natural-language goals are the closest match). Off the shelf, neither critic
  separates good plans from bad on real frames, so expect every plan to come back as not accepted until at least the critic heads are adapted.
- Server note: the Nut WM now also has `wm_causal/ckpt/latent_dynamics_best_weights.pt`, so the server can load it through
  `--latent-wm-ckpt` / `--encoder-ckpt` / `--wm-decoder-dir` / `--critic-ckpt` with no code change.
