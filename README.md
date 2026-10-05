# Verify2Act

A VLM proposes subtask plans, a latent world model imagines them, and a critic verifies them before the robot acts.
This file is the single project doc: setup, the real-robot plan server, the real-robot evaluation, the DOFBOT digital
twin, the twin model results and the RA-L video. Branch `main-wm`.

1. [Setup](#1-setup)
2. [Machines](#2-machines)
3. [Real-robot plan server](#3-real-robot-plan-server)
4. [Real-robot evaluation](#4-real-robot-evaluation)
5. [Digital twin](#5-digital-twin)
6. [Twin models: what was trained and how it scores](#6-twin-models-what-was-trained-and-how-it-scores)
7. [RA-L video](#7-ra-l-video)
8. [Known limits and gotchas](#8-known-limits-and-gotchas)

---

## 1. Setup

```bash
conda create -n verify2act python=3.10 && conda activate verify2act
pip install -r requirements.txt               # torch, accelerate, transformers, openai-clip, mujoco, pyyaml, ...
pip install "opencv-python>=4.8" scipy        # cv2 (+ aruco, for twin/textures.py) and scipy (twin/fit_camera.py)
pip install roslibpy google-auth              # plan server, Vertex AI

cd robosuite && pip install -e . && cd ..     # or: conda env create -f Points2Plans/conda_env.yml

# calvin
git submodule update --init calvin
pip install -e calvin/calvin_env
pip install -e calvin/calvin_models --no-deps
pip install pytorch-lightning gym pyhash

# third_party
pip install -e third_party/MoDE_Diffusion_Policy
```

Activate the env before running `mjpython ...`. Twin rendering is headless through EGL (`MUJOCO_GL=egl`, NVIDIA
driver); `MUJOCO_GL=osmesa` works without it (CPU, several times slower, needs `libosmesa6`).

### Gemini on Vertex AI (per machine)

The planner calls `gemini-2.5-flash` through Vertex AI on the GCP project `verify2act`. The org blocks service-account
keys, so each machine authenticates as the user with Application Default Credentials:

```bash
curl https://sdk.cloud.google.com | bash -s -- --disable-prompts && exec -l $SHELL
gcloud auth application-default login --no-launch-browser    # open the URL locally, paste the code back
gcloud auth application-default set-quota-project verify2act
python3 -c "import google.auth; c, _ = google.auth.default(); print(getattr(c, 'quota_project_id', 'NOT SET'))"   # verify2act
```

`pipeline/planner.py` picks the backend in this order: `GEMINI_API_KEY` set → AI Studio (free tier, rate-limited);
`~/.config/gcloud/application_default_credentials.json` present → Vertex AI (`pipeline/gemini_backend.py`); neither →
`ValueError: GEMINI_API_KEY environment variable is not set`.

| symptom | fix |
|---|---|
| `ValueError: GEMINI_API_KEY ... not set`, `DefaultCredentialsError` | ADC file missing: re-run the login and quota-project steps |
| `quota_project_id: NOT SET` | re-run `set-quota-project` |
| HTTP 403 on the Vertex call | check `aiplatform.googleapis.com` is enabled for the project |
| no "Vertex AI User" role in the console | it is now called **Agent Platform User** (`aiplatform.user`) |
| `Project: None` in the check | normal for user ADC; the code falls back to `quota_project_id` |

## 2. Machines

| machine | role |
|---|---|
| Jetson, 192.168.0.8 | the robot: roscore, arm driver, camera, skills, the episode loop (`dofbot-controller` repo), rosbridge on :9090 |
| lab PC, 192.168.0.113 (RTX 5060, 8 GB) | model server (`verify2act/robot/server.py`), twin rendering, paper and video builds |
| csg2 (3× T4 15 GB, 64 cores), repo at `/home/scratch1/cheze/verify2act` | twin data generation and all training; checkpoints live here |

`verify2act/output/` is gitignored: checkpoints, datasets, logs and eval JSONs are not in git. `dofbot-controller` is
read for reference only on the lab PC; it is never run or edited there.

## 3. Real-robot plan server

The Jetson keeps the episode loop and skill execution. The lab PC is a model server: `verify2act/robot/server.py`
connects **out** to the Jetson's rosbridge with roslibpy and answers requests, so the Jetson opens no ports. Per
planning call the Jetson sends the current frame, the language goal and the history of executed subtasks, and the lab PC
runs the sim loop unchanged (`BeamSearchPlanner.plan`):

1. Gemini proposes `beam_width` candidate subtask plans.
2. The latent WM is seeded from the frame's DINO latent and imagines horizon k from horizon k-1 plus subtask k.
3. The temporal head compares consecutive states at every horizon. A low score re-imagines that step, up to
   `max_retries` times.
4. The goal head scores the final imagination against the language goal.
5. A rejected plan goes through reflect → replan, with a fixed budget `max_replans` = 2. The VLM sees the decoded
   imagined scene.

The reply is the plan plus the verdict. The server keeps no scene state between calls; only the per-session call
counter persists, and `reset` clears it.

Code: `verify2act/robot/{protocol,prompts,backend,server}.py`, tests in `test_robot_server.py`, prompts in
`verify2act/configs/prompts/dofbot/`. The Jetson-side client (`remote/planner_client.py`, `_run_plan_server` in
`v2a_pipeline.py`) lives in `dofbot-controller`, which is authoritative for it.

### Wire protocol

Two `std_msgs/String` topics carrying JSON: `/v2a/request` (Jetson → lab, `{"id", "op", ...}`) and `/v2a/response`
(lab → Jetson, `{"id", "ok", "error"?, ...}`). Images are base64 JPEG of a BGR cv2 frame. One model request runs at a
time; `ping` is answered even during a `plan` and never takes the model lock.

| op | request | reply |
|---|---|---|
| `ping` | – | `wm_mode, theta_c, theta_p, max_replans, max_requery` |
| `reset` | `session` | – |
| `plan` | `session, image, goal, history`, optional `obj_labels` (default all four colours), `horizon` (default 4), `feedback` | below |

`plan` reply fields:

| field | meaning |
|---|---|
| `plan` | subtasks to execute. Empty only when `done` is true |
| `accepted` | the critic accepted the plan within the replan budget. Always true for `vlm_only`, and for `diffusion_wm` after its one reflection |
| `done` | the VLM answered `["done"]`. It is the VLM's claim only; the goal head's verdict on the real frame is in `accepted` |
| `invalid_steps` | steps outside the vocabulary; do not execute if non-empty |
| `score`, `all_scores` | goal-head score of the returned plan, and `[tc, goal]` per horizon (null / empty without a critic) |
| `failed_step`, `replan_attempts`, `reflection_analyses`, `critic_decisions` | where and why the returned plan was rejected |
| `evaluations` | every plan evaluated in the call: `{plan, tc[], goal, accepted, temporal_rejected, goal_rejected, requeries}` |
| `stats` | `{vlm_calls, plans_evaluated, requeries, temporal_rejections, goal_rejections}` |
| `wm_mode`, `planning_call`, `elapsed_s`, `real2sim` | variant, call index in the session, server time, real2sim fit rms and time |

A Gemini failure or an unparseable reply comes back as `ok: false` with `RuntimeError: VLM produced no plan ...`,
never as an empty plan. Timeouts on the Jetson: 600 s for `plan` (`PLAN_REPLY_TIMEOUT_S`), 5 s for `ping` / `reset`.

### Subtask vocabulary

One subtask = one WM horizon = one complete pick-and-place that ends with nothing held, because the arm cannot hold a
block through a planning call. `<c>` ≠ `<b>`, colours are red, green, blue, yellow:

- `pick and place <c> block into the bin`
- `pick and place <c> block on <b> block`
- `pick and place <c> block to the left of <b> block` / `... to the right of <b> block`
- `done`

The Jetson runs the last three as `locate <b>` → `pick <c>` → `place_on | place_at`, back to back.

### Variants (`--wm-mode`)

One server process runs one variant. Prompts, vocabulary and replan budget are the same for all of them, so the
verifier is the only thing that changes.

| `wm_mode` | world model | critic | per `plan` call |
|---|---|---|---|
| `v2a_wm` (default) | Verify2Act latent WM | yes | candidates → imagine → critic → reflect / replan (≤ 2) |
| `rla_wm` | RLA-WM latent WM | yes | same |
| `diffusion_wm` | InstructPix2Pix + LoRA (ReflectVLM; the sim's `diffusion`) | no | 1 propose → imagine → always 1 reflection → accepted |
| `vlm_only` | none | no | 1 propose, accepted as is; no WM or critic loaded |

`--preset calvin` (default) loads the CALVIN checkpoints with θ_c 0.5, θ_p 0.05, max_retries 2. `--preset twin` loads
the twin checkpoints with θ_c 0.5, θ_p 0.5 (the goal head returns P(goal)), max_retries 3, beam width 3, horizon 4
(`server.py::PRESETS`). With `--real2sim` the WM and critic start from a twin re-render of the camera frame, while the
VLM still sees the camera frame.

### Episode semantics (Jetson side, matching the sim's `run_episode`)

- A timestep is one `plan` request plus executing the reply. `--max_steps` is 2, so an episode makes at most 2 `plan`
  calls (one re-plan).
- The operator answers y/n once per executed plan, like `is_done()` in the sim. That answer is `real_success`.
- On "n", the operator names the failed horizons, which become `[FAILED] <subtask>` entries in `history`, and can add a
  note, which is sent as `feedback` and goes into every propose and reflect prompt of the next call.
- `[FAILED]` can mean the skill failed or that the operator saw it fail (e.g. a block bounced out of the bin).
- A VLM `done` does not end the episode. Plans still unverified after the replan budget are executed anyway.
- A server or Gemini error aborts the episode as infrastructure (`aborted: true`).

### Eval tasks

| task | goal (verbatim) | intended plan |
|---|---|---|
| task1a | Put the blue block and the yellow block into the bin | 2 × `into the bin` |
| task1b | Clear all cool-colored blocks into the bin and leave the yellow block | green, blue into the bin |
| task1c | Clear all warm-colored blocks into the bin and leave the green block | red, yellow into the bin |
| task1d | Put the red block, the green block and the blue block into the bin except the yellow block | 3 × `into the bin` |
| task2a | Put the red block to the left of the blue block | `pick and place red block to the left of blue block` |
| task2b | Put the green block to the right of the yellow block | `pick and place green block to the right of yellow block` |
| task3a | Stack the blue block on top of the yellow block | `pick and place blue block on yellow block` |

Warm = red / yellow, cool = green / blue. The prompts' few-shot goals use different colours from the eval goals.

## 4. Real-robot evaluation

Four variants, one server process each, all on the twin models. Every WM variant runs with `--real2sim`.

| variant | `--wm-mode` | world model | verifier | log dir under `verify2act/output/real/twin/` |
|---|---|---|---|---|
| V2A (ours) | `v2a_wm` | wm3 (`v2a_wm/dofbot_twin/wm3`) | twin temporal head + goal head v3 | `v2a_wm_r2s/` |
| RLA-WM | `rla_wm` | RLA-WM fine-tuned on twin data | twin temporal head + its own goal head (`goal_head/rla_twin`) | `rla_wm_r2s/` |
| Diffusion | `diffusion_wm` | InstructPix2Pix LoRA fine-tuned on twin data | none | `diffusion_wm_r2s/` |
| VLM only | `vlm_only` | – | none | `vlm_only/` |

### Before starting (lab PC)

```bash
conda activate verify2act && cd ~/Projects/multi-object-manip/verify2act
git pull origin main-wm
python -m pytest verify2act/robot/test_robot_server.py -q
```

Checkpoints (about 3.3 GB, from csg2):

```bash
CSG2=cheze@csg2
R=/home/scratch1/cheze/verify2act/./
rsync -avR --progress \
  $CSG2:$R/verify2act/output/v2a_wm/dofbot_twin/wm3/ckpt/latent_dynamics_best_weights.pt \
  $CSG2:$R/verify2act/output/v2a_wm/dofbot_twin/encoder/ckpt/delta_encoder_best.pt \
  $CSG2:$R/verify2act/output/v2a_wm/dofbot_twin/decoder/latent_decoder_best.pt \
  $CSG2:$R/verify2act/output/v2a_wm/dofbot_twin/decoder/config.json \
  $CSG2:$R/verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt \
  $CSG2:$R/verify2act/output/goal_head/v3/goal_head_last.pt \
  $CSG2:$R/verify2act/output/goal_head/rla_twin/goal_head_last.pt \
  $CSG2:$R/verify2act/output/rla_wm/dofbot_twin/wm/ckpt/latent_dynamics_best.pt \
  $CSG2:$R/verify2act/output/diffusion_wm/dofbot_twin/wm/best \
  ./
```

The diffusion variant also downloads `timbrooks/instruct-pix2pix` and the SD-1.5 VAE from Hugging Face on first start.
`--real2sim` needs MuJoCo with EGL and the twin textures in `verify2act/twin/assets/` (in git).

### Offline smoke test per variant (no robot, one Gemini call each)

```bash
export MUJOCO_GL=egl
F=frame.png; G="Put the red block to the left of the blue block"
python -m verify2act.robot.server --preset twin --real2sim --wm-mode v2a_wm       --offline-image $F --goal "$G"
python -m verify2act.robot.server --preset twin --real2sim --wm-mode rla_wm       --offline-image $F --goal "$G"
python -m verify2act.robot.server --preset twin --real2sim --wm-mode diffusion_wm --offline-image $F --goal "$G"
python -m verify2act.robot.server --preset twin            --wm-mode vlm_only     --offline-image $F --goal "$G"
```

Check: a plan in vocabulary, `real2sim.rms` < 2.5 in the reply, and under `.../<mode>_r2s/offline/` a
`real2sim_render.png` that looks like the frame and an `imagine_frame.png` that moved the right block.

### Live, one variant at a time

Jetson, once: roscore, robot stack, then `roslaunch rosbridge_server rosbridge_websocket.launch`.

Lab PC, one of (wait for `Ready: listening on /v2a/request`, 2-4 min):

```bash
export MUJOCO_GL=egl
python -m verify2act.robot.server --jetson-ip 192.168.0.8 --preset twin --real2sim --wm-mode v2a_wm
python -m verify2act.robot.server --jetson-ip 192.168.0.8 --preset twin --real2sim --wm-mode rla_wm
python -m verify2act.robot.server --jetson-ip 192.168.0.8 --preset twin --real2sim --wm-mode diffusion_wm
python -m verify2act.robot.server --jetson-ip 192.168.0.8 --preset twin            --wm-mode vlm_only
```

Jetson (`dofbot-controller`), with that server up:

```bash
python3 verify2act/v2a_pipeline.py --task task2a --jetson_ip 127.0.0.1 --plan_server --dry_run   # prints "Variant: <mode>", arm still
python3 verify2act/v2a_session.py  --task <task> --episodes <N> --jetson_ip 127.0.0.1 --plan_server
```

Use `--jetson_ip 127.0.0.1`, not `--local`: the plan client rides on the rosbridge connection. Switching variant = stop
the server, start the next one, restart the Jetson session. If rosbridge restarts, the server exits and has to be
restarted.

### Start layouts and run order

`verify2act/real_eval_layouts/<task>/<task>-L<k>.png`, with the states in `layouts.json` (generated by
`verify2act/twin/make_eval_layouts.py --n 12 --main 10`). Each sheet shows a to-scale top view with each block's centre
in cm from the sheet's far edge and left edge as seen from the robot, its rotation, which end carries the tag, and the
expected arm-camera view. All four blocks are on the sheet, fully in view from every fitted home pose, the goal does
not hold yet, and the intended plan works in the twin. For task2a/2b, odd layouts start with the block on the wrong
side, even ones on the right side but too far away.

L1-L10 are the eval layouts (70 episodes per variant). L11-L12 are spares: if a layout is unreachable on the real arm,
replace it by the next unused spare of that task for every variant. Put the layout id in the session notes.

Round k = layout Lk of every task. Within a round, run one variant over all of the round's layouts, then the next one
over the same layouts, re-placing the blocks from the sheet each time. Tasks within a round: task2a, task2b, task1a,
task1c, task1b, task1d, task3a. The variant order rotates so no variant is always first:

| rounds | variant order |
|---|---|
| 1, 5, 9 | v2a_wm → vlm_only → rla_wm → diffusion_wm |
| 2, 6, 10 | vlm_only → rla_wm → diffusion_wm → v2a_wm |
| 3, 7 | rla_wm → diffusion_wm → v2a_wm → vlm_only |
| 4, 8 | diffusion_wm → v2a_wm → vlm_only → rla_wm |

Stop only at the end of a round. Dropping a task: drop it from every round and say which in the paper.

### Results

From `verify2act/output/paper/real_numbers.json` (figures: `make_real_figs.py`, which reads everything under
`output/real/results`):

| variant | success | macro rate |
|---|---|---|
| V2A | 42/55 | 0.77 |
| RLA-WM | 34/55 | 0.62 |
| VLM-only | 16/30 | 0.53 |
| Diffusion | 14/30 | 0.47 |

The counts are unequal: V2A and RLA-WM ran 10 layouts per task (5 for task2b), the other two ran 5, and task1c was
dropped. Either show n or restrict all four to the shared layouts L1-L5. V2A's failures are execution failures (13),
not plan failures (0).

## 5. Digital twin

Goal: fine-tune the world model and critic on synthetic data that looks like the real arm-camera frames, without
collecting real data. Code in `verify2act/twin/`, config `verify2act/configs/twin/dofbot_twin.yaml`.

### Facts the design rests on

- The camera is on the arm and planning frames are taken at the home pose, so the arm is never in view. The home pose
  is not perfectly repeatable, so the twin jitters the camera per episode and per frame.
- The bin is never in frame: "into the bin" = the block disappears from the table.
- Blocks are 30×30×60 mm, lying on a 60 mm side, long axis roughly away from the robot, with a printed tag on one end
  face (random ArUco in the twin). They never end up upright; a stacked block sits on the base block.
- The scene is 4 blocks on a white US Letter sheet (portrait) on a wooden table, 640×480.
- World frame: origin at the sheet centre, x away from the robot (up in the image), y = image left. "Left of" = +y.

Because every frame the models see is a static "arm at home" scene, the twin is **kinematic**: no arm, no grasping. An
oracle teleports the block to its target pose with placement noise, MuJoCo settles it, and the calibrated camera
renders it. That gives the `(s_t, subtask, s_t+1)` transitions the WM and critic train on.

### Components

| file | what it does |
|---|---|
| `config.py` | dataclasses + YAML loading (supports `base:` inheritance) |
| `textures.py` | albedo textures from real frames: wood, sheet, a colour patch per block, tag faces |
| `scene.py` | MJCF builder and `DofbotTwin`: `reset`, `apply(subtask)`, `render`, state, predicates, visibility |
| `tasks.py` | goal sampler, goal predicates, oracle planner, precondition-conflict scenes |
| `augment.py` | webcam look: white balance, gamma, blur, sensor noise, vignetting, JPEG round trip |
| `generate.py` | writes the dataset in the robosuite layout the trainers already read |
| `fit_camera.py` | fits home camera poses and one fovy to real frames from the known block and sheet sizes |
| `calibrate.py`, `overlay.py` | checkerboard calibration (alternative), and a twin-over-real blend to check the camera |
| `real2sim.py` | `Real2Sim`: fits the blocks in a camera frame and returns a twin render (12-16 s, CPU) |
| `goal_labels.py`, `train_goal_head.py`, `gen_imagined.py` | state-based goal labels, the spatial goal head, WM-imagined training sets |
| `eval_plans.py`, `eval_real2sim.py`, `eval_conflict.py`, `eval_temporal.py`, `diag_*.py` | offline evals behind section 6 |
| `make_eval_layouts.py` | the fixed start layouts for the real eval |
| `test_twin.py` | offline tests: `MUJOCO_GL=egl python -m pytest verify2act/twin/test_twin.py -q` |

Camera fit: fovy 36.9°, about 0.27 m above the table, pitch 53-61°; 10 home poses in the YAML (`camera.poses`), one
sampled per episode.

### Dataset

One row per subtask. Layout: `episodes/ep_NNNNNN/frame_*.jpg`, `goal.jpg` (the final state re-rendered with another
camera and lighting draw), `states.json`, `transitions.jsonl` (`image_t`, `image_t1`, `action_text`, `lang_goal`,
`family`, `task`, `episode_success`), `metadata.json`.

Goal-directed episodes (80%) by family (`episodes.family_weights`): `eval` 30% (the eval tasks verbatim), `bin` 30%,
`side` 17%, `stack` 15%, `compound` 8%. The other 20% are random walks of 1-3 valid subtasks with no language goal.
20k episodes ≈ 35k transitions ≈ 3.5 GB. The DINOv2-L feature cache both trainers share is ~38 GB for that.

| YAML key | what | now |
|---|---|---|
| `episodes.random_walk_frac` | episodes without a language goal | 0.2 |
| `placement.p_tag_toward_camera` | blocks showing their tag end | 0.4 |
| `placement.yaw_std / yaw_max` | block yaw spread (deg) | 20 / 50 |
| `placement.p_fail` | real-robot style misplacements (0 = the WM learns the intended effect) | 0 |
| `camera.poses` | fitted home poses; refit with `fit_camera.py --write` after new real logs | 10 |

### Commands (csg2)

```bash
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=verify2act/data/twin/dofbot_v1; CACHE=verify2act/data/twin/dino_features

python -m verify2act.twin.textures --frames 'verify2act/output/real/*/imagination_logs/*/request_image.png'
python -m verify2act.twin.fit_camera --frames 'verify2act/output/real/*/imagination_logs/*/request_image.png' --write

python -m verify2act.twin.generate --out $OUT --num-episodes 200 --workers 8 --seed 0          # smoke test first
python -m verify2act.twin.generate --out $OUT --num-episodes 20000 --workers 32 --chunk 50 --seed 0
# append later without touching existing episodes: --start-episode 20000 --num-episodes 10000 (same --out and seed)

# WM (build the DINO cache first, sharded: a cold cache kills a multi-GPU run)
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/latent_wm/train_dynamics.py \
  --dataset-type robosuite --dataset-dir $OUT --cache-dir $CACHE \
  --output-dir verify2act/output/v2a_wm/dofbot_twin/wm \
  --encoder-ckpt verify2act/output/v2a_wm/dofbot_twin/encoder/ckpt/delta_encoder_best.pt \
  --token-dim 128 --num-latent-tokens 32 --history-len 3 --causal-masking \
  --num-epochs 50 --batch-size 16 --lr 1e-4 --checkpoint-freq 10 \
  --resume-from <weights-only .pt>

# critic
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=bf16 \
  verify2act/critic/train_contrastive.py \
  --dataset-dir $OUT --dataset-type dofbot --cached-dino-dir $CACHE \
  --output-dir verify2act/output/contrastive/dofbot_twin \
  --epochs 25 --batch-size 16 --learning-rate 1e-4 --lambda1 0.5 --lambda2 0.7 --kl-weight 5e-4 \
  --init-from verify2act/output/contrastive/calvin/best_contrastive_critic.pt
```

Keep `--history-len 3 --token-dim 128 --num-latent-tokens 32` and cross-attention conditioning: the server loads the
model with these. Pass weights-only files to `--resume-from` / `--init-from`; a full checkpoint restores the source
run's epoch, optimizer and best score, so nothing trains.

## 6. Twin models: what was trained and how it scores

The served V2A model is **wm3 + goal head v3 + the twin temporal head**, with the twin delta autoencoder and decoder.

### Why the first fine-tune was at chance, and the fixes

1. The critic mean-pools DINOv2 patch tokens, so it cannot see spatial relations. Probe for "is A left of B" on twin
   frames: 0.495 on mean-pooled tokens vs 0.998 for a small transformer on the 16×16 grid. Per-task goal AUROC on true
   frames was 0.75 / 0.61 for task2a / 2b while the pooled val AUROC read 0.9997.
   **Fix:** `critic/goal_head.py::SpatialGoalHead`, a 4-layer transformer over CLIP text tokens and the 256 patch
   tokens that returns P(goal met) (`--goal-head-ckpt`, use θ_p ≈ 0.5). task2a / 2b went to 0.993 / 0.973.
2. The text side could not express argument order: cos("red left of blue", "blue left of red") = 0.994 with pooled
   CLIP. The goal head uses per-token text features and is trained with flipped-side and swapped-argument negatives.
3. The CALVIN delta autoencoder does not transfer to the twin (relative reconstruction error 1.01), so no dynamics on
   that latent space could work. **Fix:** a twin delta AE fine-tuned from the CALVIN weights; goal-head AUROC on AE
   reconstructions went from 0.50 to 0.90+. `--encoder-ckpt` must point at it.
4. On real frames the WM often removes the moved block without placing it. **Fixes:** train the goal head on
   WM-imagined states (v2, v3), take the best of 3 WM samples, and bridge with real2sim so the WM starts from a twin
   render.

### Model lineage

| model | what | note |
|---|---|---|
| wm_old | first twin WM, CALVIN AE frozen | imaginations are copies of the input |
| wm2 | twin AE, 30 epochs, best val 0.2743 | |
| wm3 | wm2 + precondition-conflict data (`dofbot_v1c`), 6 epochs, val 0.2637 | **served** |
| wm4 | wm3 continued, 10 epochs | no better; differences are within WM sampling noise (~0.03) |
| goal head v1 → v2 → v3 | true frames → + wm2 imaginations → + wm3 imaginations and conflicts | **v3 served** |

### Offline scores (top1 / pairwise unless noted; chance top1 ≈ 0.06)

| eval | old WM + pooled critic | V2A (wm3 + goal head v3) | RLA-WM + its goal head |
|---|---|---|---|
| twin held-out plans (57 frames) | 0.18 / 0.72 | 0.965 / 0.998 | 0.579 / 0.948 |
| real scenes, real2sim render (42 cases, k=1) | – | 0.976 / 0.999 | 0.714 / 0.944 |
| real scenes, real image (k=1) | 0.06 / 0.62 | 0.71 / 0.92 | 0.24 / 0.79 |
| conflicts n=150, pairwise / naive_rej / clearing_acc | – | 0.907 / 0.82 / 0.81 | 0.840 / 0.85 / 0.59 |

The old-system column is from the earlier 50-case real set, so it is not directly comparable to the 42-case runs.
Goal head v3 on the true outcome frames gives naive_rej 0.93 and clearing_acc 1.00, so the conflict gap is in the WM.
JSONs: `verify2act/output/real_eval/`.

Baselines: RLA-WM got 10 twin epochs from CALVIN weights (val 0.2975, still falling), V2A's wm3 descends from 30 + 6;
say so in the paper. The diffusion LoRA (best at step 1000 of 3000) draws the twin scene but rarely carries out the
action: pairwise 0.45, the moved block usually vanishes or changes colour.

Temporal gate (θ_c 0.5, `eval_temporal.py`): passes 97-100% of true, imagined and real transitions and rejects 99% of
different-scene swaps. It does not separate right from wrong actions, and the uncertainty gate never fires.

## 7. RA-L video

Requirements: one video (mp4, H.264 + AAC), in one zip of at most 50 MB with `ReadMe.txt` and `Summary.txt`; aim for
1-3 minutes, title and authors at the start, credits at the end, English voice-over or captions, and nothing that
reads as extra pages. Sources: the RA-L information for authors and the IEEE RAS video submission guidelines.

State: a first cut exists. `verify2act/output/real_video/video_build/build.py` renders the cards and assembles
`verify2act_video.mp4` (2:02, 16.8 MB, anonymous, metadata stripped); `verify2act_video_submission.zip` adds the two
text files. Trim points, speeds and captions are in `build.py::main`. `ffmpeg` lives in the conda env `ffmpeg`.

- The footage (`output/real_video/`) was filmed **without the sheet and without `--real2sim`**, so the live V2A logs
  from those takes are not usable: imaginations are smears and correct plans scored near 0 and ran unverified.
- The planning panels come from `real_video/replay_nosheet/`: `nosheet.py` fixes the camera at the calibrated home pose
  and uses whole-frame colour masks, and `replay.py` re-verifies the plans the VLM proposed live, with no VLM call. The
  offline replay is stated in `Summary.txt`; the user chose no on-screen note.
- Keep video sessions in `output/real_video/`, never `output/real/`: `make_real_figs.py` would count them in the
  paper's numbers.

| use | take | replay result |
|---|---|---|
| hero | task1b take 2 (`IMG_8670.MOV`) | three red plans rejected (0.14, 0.00, 0.21), replan green + blue accepted 0.82, real success |
| breadth | task2a take 1 (`IMG_8686.MOV`), task3a take 3 (`IMG_8691.MOV`) | accepted 0.98 / 1.00, real success |
| failure | task3a take 2 (`IMG_8690.MOV`) | plan accepted, execution failed |
| contrast | VLM-only task1b take 2 (`IMG_8675.MOV`) | live log valid: bins red |

Why task1b is the story: the VLM sometimes proposes binning the red block (warm, must stay), the goal head scores those
candidates 0.08-0.29, and the replan drops red. The VLM-only failures on task1b and task2a in the eval are grasp
failures with the same plan V2A executed, so a side-by-side on those would credit the verifier with grasp luck.

## 8. Known limits and gotchas

Say in the paper:
- real2sim fits blocks on the table only. Frames with a stack or with blocks in the bin may fit poorly; when rms > 2.5
  px the server plans on the real frame. Without a sheet it fails silently (no blocks found, rms 0.0).
- The temporal head is blind to layout errors; plan selection rests on the goal head.
- Diffusion has no critic, as in the sim, and its reflections work from poor imaginations.
- RLA-WM trained for fewer epochs than V2A's WM.

Open:
- "Left of" = image left = world +y assumes the robot's left matches the camera's left at the home pose.
- MuJoCo shadows are harder-edged than the lab's diffuse light.
- The critic loads its own DINOv2-L in addition to the WM's extractor (two copies, ~4 GB peak).
- The `compound` family came out at 2.3% of episodes against a ~6.4% target (not investigated).

csg2:
- The user is capped at 1024 threads. Export `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` (1 for
  generation, 4 alongside training), or new Python processes die at `import numpy`. Check with `ps -L -u cheze | wc -l`.
- `pkill -f` / `pgrep -f <pattern>` matches the calling shell's own command line. Use a bracketed pattern
  (`grep "train_decode[r]"`) or a PID.
- Run scripts as `python -m verify2act...`; by path they need `PYTHONPATH=$PWD`.
- The planner returns `-inf` for a candidate if a step fails the temporal gate or if `max_retries` is 0.
- The lab PC's 8 GB GPU runs WM training only at batch 1 without a DINO cache; train on csg2.
