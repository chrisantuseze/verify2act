# DOFBOT digital twin (MuJoCo) for sim-to-real WM + critic training

Started 2026-09-25 on the lab GPU PC (branch `main-wm`). Goal: fine-tune the **world model and the critic** on synthetic
data that looks like the real DOFBOT camera frames, without collecting real data. The encoder and the decoder stay the
CALVIN-wider checkpoints the server already loads (`REAL_ROBOT_PLAN.md`).

## Facts the design rests on (from the user and from the real frames)

- The camera is **on the arm**. Planning frames are taken at the home/observation pose, so the arm is never in view.
  The home pose is not perfectly repeatable (two frames of one run show a slightly different framing), so the twin
  jitters the camera pose per episode and per frame.
- The **bin is never in the frame**. "Into the bin" = the block disappears from the table.
- Blocks never end up upright or tipped. A stacked block **sits on the base block**.
- Blocks are **30×30×60 mm**, lying on a 60 mm side (user). In the real frames the long axis points roughly away from
  the robot, with yaw up to about ±40°. The blocks have a printed tag on one 30×30 end face. The user said random
  tags are fine, so the twin uses a random ArUco marker per block and random 180° flips.
- The sheet is **US Letter in portrait**. The robot base stands on its near part.
- Scene: 4 blocks (red, green, blue, yellow) on a white sheet on a wooden table, 640×480 webcam, arm homed.
- One subtask = one WM horizon = one complete pick-and-place that ends with nothing held
  (`pick and place <c> block into the bin | on <b> block | to the left/right of <b> block`). The WM action text is the
  subtask string itself (`robot/prompts.py: expand_subtask_plan`).

## Design decision: a kinematic twin (no arm in the sim)

Every frame the models see is a static "arm at home" scene, so the twin does not model the DOFBOT, controllers or
grasping. A subtask is executed by an oracle that teleports the block to its target pose (with realistic placement
noise), lets MuJoCo physics settle it (stacks), and renders from the calibrated arm camera. This gives exactly the
`(s_t, subtask, s_t+1)` transitions the WM and critic train on.

## Components (`verify2act/twin/`)

| file | what it does |
|---|---|
| `configs/twin/dofbot_twin.yaml` | block dims, paper/table size, camera (pos, look-at, fovy or calibrated intrinsics), jitter and noise ranges, texture paths |
| `config.py` | dataclasses + YAML loading |
| `textures.py` | builds albedo textures from real frames: wood strip, paper colour, a colour patch per block (HSV masks); optional rectified face photos (4 clicked corners per face) incl. the tag faces |
| `scene.py` | MJCF builder (table plane, sheet, 4 textured boxes with free joints, one camera) and `DofbotTwin`: `reset`, `apply(subtask)`, `render`, `get_state/set_state`, predicates, visibility check |
| `tasks.py` | goal sampler (eval families + paraphrases), goal predicates, oracle planner, random valid subtasks |
| `augment.py` | photometric realism: white balance, brightness/contrast/saturation, gamma, blur, sensor noise, vignetting, JPEG round trip |
| `generate.py` | writes the dataset in the existing robosuite layout (`episodes/ep_*/frame_*.png`, `goal.png`, `transitions.jsonl`, `metadata.json`), plus `lang_goal`/`family` per row |
| `overlay.py` | renders the twin at the configured camera and blends it with a real frame, to check and hand-tune the camera |
| `fit_camera.py` | fits the home camera poses (and one fovy) to logged real frames from the known block/sheet sizes → YAML |
| `calibrate.py` | OpenCV checkerboard calibration at the home pose → intrinsics + camera pose → YAML |
| `test_twin.py` | offline tests (scene, subtasks, predicates, oracle plans for every eval goal, dataset readable by the loaders) |

### Appearance ("cut and paste the real object on the sim blocks")
MuJoCo `cube` textures give each box face its own image (`fileright/left/up/down/front/back` = ±X, ±Y, ±Z). v1 uses real
colour patches cropped from logged frames; v2 uses rectified per-face photos (and the tag faces). Material `rgba` tints
the texture at run time, so per-episode colour jitter needs no recompile. Lights (direction, intensity, shadows) are
randomised at run time too.

### Subtask semantics (world frame: origin at the sheet centre on the table, x away from the robot = up in the image, y = image left)
- `into the bin`: block parked out of view (behind the camera), marked absent.
- `on <b>`: centre on `b`'s centre + N(0, ~5 mm), yaw = `b` yaw + small noise, then physics settles it on `b`.
- `left/right of <b>`: centre at `b` ± y by 6–9 cm (placement ~7 cm centre to centre), small x and yaw noise, on the
  table, no overlap with other blocks, stays in view.
- Validity: the moved block and the base block must have nothing on top of them; `<c>` ≠ `<b>`.
- Goal predicates follow `EVAL_TASKS.md`: named blocks gone and others present; correct side at 0.6–3.5 block widths and
  level; top block's centre inside the base footprint.

### Episodes
- Initial scene: 2–4 blocks at random non-overlapping poses on the sheet, all fully in view, long axis ≈ x ± 20°, plus an
  optional unrecorded prelude of 0–2 random valid subtasks (some blocks already gone, some stacks).
- Goal-directed episodes (~80%): sample a goal (family 1 explicit / warm-cool / leave / except, family 2, family 3,
  a few compound goals), run the oracle plan. `episode_success = True`, `lang_goal` = the goal text.
- Random-walk episodes (~20%): 1–3 random valid subtasks, no language goal (used for temporal consistency only).
- `goal.png` is a second render of the final state with a different camera jitter and photometric draw, so the goal
  head's anchor and positive are not pixel-identical.
- Real-robot failure modes are optional (`p_fail`, default 0): the WM should learn the intended effect of a subtask.

### Training (WM + critic only)
- WM: `train_dynamics.py --dataset-type robosuite --dataset-dir <twin> --encoder-ckpt <calvin wider encoder>`, resumed from
  the CALVIN-wider WM checkpoint, `--history-len 3 --token-dim 128 --num-latent-tokens 32 --causal-masking`
  (cross_attn default, as the server's v2a_wm).
- Critic: `train_contrastive.py --dataset-type dofbot` (new): the language goal comes from each row, and the goal-head
  negative is a non-goal frame **from the same episode** (hard negative), falling back to another episode.
- Both loaders resize the full 640×480 frame to a square, as the server does with the Jetson frame.

### Validation on real data
- Real transitions already exist in the server logs: `planning_call_00/request_image.png` → executed plan →
  `planning_call_01/request_image.png` (`verify2act/output/real/*/imagination_logs/`).
- Compare CALVIN vs twin checkpoints on them: goal-head margins (achieved vs not), temporal scores for correct vs wrong
  subtasks, WM rollout feature error.
- Sim-to-real gap metric: DINO feature distance between a real frame and its twin re-render (block poses from colour
  detection).
- Then point `server.py` at the new checkpoints.

## Phases / status (2026-09-25)

**Running it:** `TWIN_DATASET.md` is the step-by-step guide for generating the dataset and fine-tuning on csg1
(3× 16 GB). The eval tasks (1a–1d, 2a/2b, 3a) are a goal family of their own (30%, verbatim, uniform over the tasks).
The tag end faces the camera for 40% of blocks (`placement.p_tag_toward_camera`). Frames are saved as JPEG.

0. **Geometry — done (2026-09-25).**
   - Block and sheet sizes come from the user.
   - The camera was fitted from the logged real frames with `fit_camera.py`; no checkerboard was needed.
   - Method: per frame, least squares on the block silhouettes' image moments plus a chamfer term on the sheet
     edges, with one fovy shared by all frames.
   - Result: fovy 36.9°, camera about 0.27 m above the table, pitch 53–61°.
   - The 10 good home poses are in the YAML (`camera.poses`); each episode samples one.
   - `offline` is a different home pose, and the frame with a block cut off at the image edge is excluded.
   - Fit overlays and `fits.json`: `verify2act/output/twin/camera_fit/`. `calibrate.py` (checkerboard) stays as an
     alternative.
1. **Scene + oracle + tasks — done.** `scene.py`, `tasks.py`; 15 tests pass
   (`MUJOCO_GL=egl python -m pytest verify2act/twin/test_twin.py -q`).
2. **Appearance — done (v1).**
   - `textures.py` built wood/sheet/block textures from the 11 logged real frames, plus the tag faces
     (`verify2act/twin/assets/`).
   - `augment.py` adds the webcam look.
   - Real-vs-twin re-creations of logged frames (fitted camera and block poses) and example episodes are in
     `verify2act/output/twin/examples/`.
   - Remaining look gap: MuJoCo shadows are harder than the lab's diffuse light.
3. **Generator — done.** About 1.2 s per episode per worker (EGL). Family mix of goal-directed episodes:
   bin 45%, side 25%, stack 20%, compound 10%. Random-walk episodes: 20%.
   - Side goals only pick a (block, reference, side) that has room, as the eval setup guarantees ≥ 10 cm free.
   - Per-frame camera jitter (1.5 mm / 2 mm / 0.2°) moves the image by up to ~15 px between frames, similar to
     the real home-pose repeatability.
4. **Trainer hooks — done, smoke-tested on a 60-episode set.**
   - `train_contrastive.py --dataset-type dofbot` (`data_loader.build_dofbot_contrastive_datasets`).
   - `--init-from` = weights-only init: the CALVIN critic checkpoint is a full checkpoint, and `--resume-from`
     would restore its epoch and best AUROC.
   - WM: `--dataset-type robosuite` reads the twin layout as is. `--resume-from latent_dynamics_best_weights.pt`
     (weights only) starts at epoch 0 with a fresh best val loss.
   - The 8 GB lab GPU runs WM training only at batch 1 without a DINO cache. Train on the HPC.
5. **Train, validate on the logged real transitions, then serve** — next, after phase 0.
6. Optional closed-loop sim eval in the twin.

## Commands

```bash
# textures from real frames (re-run after new logs or face photos)
python -m verify2act.twin.textures --frames 'verify2act/output/real/*/imagination_logs/*/request_image.png'

# camera
MUJOCO_GL=egl python -m verify2act.twin.fit_camera --frames 'verify2act/output/real/*/imagination_logs/*/request_image.png' --write
python -m verify2act.twin.calibrate --images 'calib/*.jpg' --home calib/home.jpg --board 9 6 --square 0.024   # alternative
MUJOCO_GL=egl python -m verify2act.twin.overlay --frame <real.png> --out overlay.png [--pitch 65 --dist 0.40 --fovy 34]

# data
MUJOCO_GL=egl python -m verify2act.twin.generate --out verify2act/output/twin/dofbot_v1 --num-episodes 20000 --workers 8

# critic (fine-tune the CALVIN critic)
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=bf16 \
  verify2act/critic/train_contrastive.py \
  --dataset-dir verify2act/output/twin/dofbot_v1 --dataset-type dofbot \
  --output-dir verify2act/output/contrastive/dofbot_twin \
  --epochs 25 --batch-size 16 --learning-rate 1e-4 --lambda1 0.5 --lambda2 0.7 --kl-weight 5e-4 \
  --cached-dino-dir verify2act/output/twin/dino_features \
  --init-from verify2act/output/contrastive/calvin/best_contrastive_critic.pt

# WM (fine-tune the CALVIN-wider WM; encoder frozen, unchanged)
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/latent_wm/train_dynamics.py \
  --dataset-type robosuite --dataset-dir verify2act/output/twin/dofbot_v1 \
  --output-dir verify2act/output/v2a_wm/dofbot_twin/wm \
  --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder/ckpt/encoder_only_best.pt \
  --token-dim 128 --num-latent-tokens 32 --history-len 3 --causal-masking \
  --num-epochs 50 --batch-size 32 --lr 1e-4 --checkpoint-freq 10 \
  --resume-from verify2act/output/v2a_wm/calvin/wm/ckpt/latent_dynamics_best_weights.pt
```

## Open items

- "Left of" = image left = world +y. This assumes the robot's left matches the camera's left at the home pose, which
  matches the eval doc (the bin is to the robot's left, off camera). Confirm on the robot.
- MuJoCo shadows are hard-edged; the lab light is diffuse. Kept faint via low directional diffuse / high ambient.
- The renderer assumes a centred principal point and no lens distortion (`calibrate.py` reports both).
- The server's `rla_wm` / `diffusion_wm` variants are not part of this (WM + critic of `v2a_wm` only).
