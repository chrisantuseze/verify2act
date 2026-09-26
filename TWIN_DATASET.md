# Twin dataset: generate and fine-tune on csg1 (3× 16 GB)

How to generate the DOFBOT digital-twin dataset and fine-tune the **world model and critic** on
`csg1.cs.okstate.edu`. Design and status: `DIGITAL_TWIN_PLAN.md`. Code: `verify2act/twin/`. Config:
`verify2act/configs/twin/dofbot_twin.yaml`.

## What the dataset contains

- Episodes rendered in a MuJoCo twin of the real scene:
  - 30×30×60 mm blocks on a US Letter sheet;
  - the arm camera at one of the 10 home poses fitted to the logged real frames;
  - real-image textures, with random ArUco tags on one end face (toward the camera for 40% of blocks);
  - webcam-style photometric noise.
- One row per subtask (one WM horizon), using the robot vocabulary:
  `pick and place <c> block into the bin | on <b> block | to the left/right of <b> block`.
- Episode mix (`episodes.family_weights` in the YAML):

| family | share of goal-directed episodes | goals |
|---|---|---|
| `eval` | 30% | the eval tasks **verbatim**, uniform over task1a, 1b, 1c, 1d, 2a, 2b, 3a; all four blocks, goal not yet met |
| `bin` | 30% | family 1 paraphrases: explicit colours, warm/cool, leave/keep/except, all |
| `side` | 17% | family 2 paraphrases (any pair, left/right, only where there is room) |
| `stack` | 15% | family 3 paraphrases (any pair) |
| `compound` | 8% | two goals in order ("…, then …") on different blocks |

  A further 20% of episodes are random walks: 1–3 random valid subtasks with no language goal. They are used only for
  the critic's temporal head and for WM transitions.

| task | goal (verbatim) | oracle plan |
|---|---|---|
| task1a | Put the blue block and the yellow block into the bin | 2 × `into the bin` |
| task1b | Clear all cool-colored blocks into the bin and leave the yellow block | green, blue into the bin |
| task1c | Clear all warm-colored blocks into the bin and leave the green block | red, yellow into the bin |
| task1d | Put the red block, the green block and the blue block into the bin except the yellow block | 3 × `into the bin` |
| task2a | Put the red block to the left of the blue block | `pick and place red block to the left of blue block` |
| task2b | Put the green block to the right of the yellow block | `pick and place green block to the right of yellow block` |
| task3a | Stack the blue block on top of the yellow block | `pick and place blue block on yellow block` |

- Layout: the robosuite layout both trainers already read.
  - `episodes/ep_NNNNNN/frame_*.jpg`
  - `goal.jpg`: the final state re-rendered with another camera and lighting draw.
  - `states.json`
  - `transitions.jsonl`: `image_t`, `image_t1`, `action_text`, `lang_goal`, `family`, `task`, `episode_success`, …
  - `metadata.json`: config, counts per family and task, git commit.
- Size and speed (measured on the lab PC):
  - ~1.75 transitions per episode;
  - ~185 KB of JPEG per episode;
  - ~0.08 s per episode with 8 workers.
  - So 20k episodes ≈ 35k transitions ≈ 3.7 GB, in ~30–40 min on 8 CPU workers.

## 1. Get the code and the files git does not carry

The twin code, the configs and this file are **not committed yet**. Commit and push them from the lab PC. The
textures are PNGs, which `.gitignore` excludes, so they need `-f`:

```bash
# lab PC
git add verify2act/twin verify2act/configs/twin DIGITAL_TWIN_PLAN.md TWIN_DATASET.md \
        verify2act/data_loader.py verify2act/critic/train_contrastive.py
git add -f verify2act/twin/assets/*.png          # 520 KB of textures built from the real frames
git commit -m "..." && git push
```

The checkpoints the fine-tuning starts from live under `verify2act/output/`, which is gitignored. Copy them:

```bash
# lab PC -> csg1 (adjust the remote repo path)
R=csg1.cs.okstate.edu:~/verify2act
for f in verify2act/output/v2a_wm/calvin/encoder/ckpt/encoder_only_best.pt \
         verify2act/output/v2a_wm/calvin/wm/ckpt/latent_dynamics_best_weights.pt \
         verify2act/output/contrastive/calvin/best_contrastive_critic.pt; do
  rsync -avR "$f" "$R/"
done
rsync -avR verify2act/output/v2a_wm/calvin/decoder "$R/"   # optional: only for visualize_wm.py
```

If you skip `git add -f` for the textures, rsync `verify2act/twin/assets/` instead. Or rebuild them on csg1 from the
real frames with `python -m verify2act.twin.textures --frames '<real frames>'`; that needs `verify2act/output/real/`
copied over too. `generate.py` refuses to run without the textures rather than render flat colours.

## 2. Environment on csg1

```bash
git clone https://github.com/chrisantuseze/verify2act.git && cd verify2act && git checkout main-wm
git submodule update --init calvin            # not needed by the twin; harmless
conda create -n verify2act python=3.10 -y && conda activate verify2act
pip install -r requirements.txt               # torch, accelerate, transformers, openai-clip, mujoco, pyyaml, ...
pip install "opencv-python>=4.8" scipy        # cv2 (+ aruco, only for textures.py) and scipy (fit_camera.py)
```

Rendering is headless through EGL (`MUJOCO_GL=egl`, NVIDIA driver). Check it once:

```bash
MUJOCO_GL=egl python -m pytest verify2act/twin/test_twin.py -q      # expect 24 passed
```

If EGL is missing, `MUJOCO_GL=osmesa` works (CPU rendering, several times slower; needs `libosmesa6`).

## 3. Generate

```bash
export MUJOCO_GL=egl
OUT=verify2act/output/twin/dofbot_v1
python -m verify2act.twin.generate --out $OUT --num-episodes 200 --workers 8 --seed 0     # smoke test first
python -m verify2act.twin.generate --out $OUT --num-episodes 20000 --workers $(( $(nproc) - 2 )) --chunk 50 --seed 0
```

- `--seed` plus the episode index fix every episode, so a run is reproducible. To add more episodes later without
  touching the existing ones, use `--start-episode 20000 --num-episodes 10000` (same `--out`, same seed): it appends
  to `transitions.jsonl`.
- Each worker process has its own EGL context and uses the GPU only lightly. It can share the GPUs with nothing else
  running.
- At the end, the run prints episodes per family and per eval task (also in `metadata.json`). The seven eval tasks
  should be roughly equal.

Quick look before training:

```bash
python - <<'EOF'
import json, collections
rows = [json.loads(l) for l in open("verify2act/output/twin/dofbot_v1/transitions.jsonl")]
print(len(rows), "transitions")
print(collections.Counter(r["task"] for r in rows if r["timestep"] == 0 and r["task"]))
EOF
```

Open a few `episodes/ep_*/frame_*.jpg`. Example strips (real-vs-twin, one per eval task) are in
`verify2act/output/twin/examples/` on the lab PC.

## 4. DINOv2 feature cache (shared by both trainers)

Both trainers cache DINOv2-L patch features: 256×1024 fp16, ~0.5 MB per image. This covers `image_t`, `image_t1` and
`goal`, about 3.7 images per episode, so **20k episodes ≈ 37 GB**. Check the disk first. Point both trainers at the
same directory. The WM trainer builds the cache on its first run; the critic then reuses it.

```bash
CACHE=verify2act/output/twin/dino_features
```

## 5. Fine-tune the world model (3 GPUs)

The encoder is frozen and unchanged (CALVIN-wider). The run starts from the CALVIN-wider WM **weights-only** file, so
it begins at epoch 0 with a fresh best val loss. Don't pass a full checkpoint to `--resume-from`: that restores
CALVIN's epoch, optimizer and best score.

```bash
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 \
  verify2act/latent_wm/train_dynamics.py \
  --dataset-type robosuite --dataset-dir $OUT \
  --output-dir verify2act/output/v2a_wm/dofbot_twin/wm \
  --cache-dir $CACHE \
  --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder/ckpt/encoder_only_best.pt \
  --token-dim 128 --num-latent-tokens 32 --history-len 3 --causal-masking \
  --num-epochs 50 --batch-size 16 --lr 1e-4 --checkpoint-freq 10 \
  --resume-from verify2act/output/v2a_wm/calvin/wm/ckpt/latent_dynamics_best_weights.pt
```

- `--action-conditioning` defaults to `cross_attn`, as the server's `v2a_wm` expects. Keep `--history-len 3`,
  `--token-dim 128` and `--num-latent-tokens 32`: the server loads the model with these.
- `--batch-size` is per GPU. 16 is a safe start on 16 GB with the cache; try 32 if memory allows. Without the cache,
  DINOv2-L runs online, and even batch 4 did not fit in 8 GB.
- The best checkpoint is `verify2act/output/v2a_wm/dofbot_twin/wm/ckpt/latent_dynamics_best_weights.pt`.

## 6. Fine-tune the critic (3 GPUs)

`--dataset-type dofbot`:
- every row carries its episode's language goal;
- the goal-head negative is a not-yet-achieved frame from the **same** episode (hard negative), 70% of the time.

`--init-from` loads the CALVIN critic's weights only (fresh epochs and best AUROC).

```bash
accelerate launch --num_processes=3 --num_machines=1 --dynamo_backend=no --mixed_precision=bf16 \
  verify2act/critic/train_contrastive.py \
  --dataset-dir $OUT --dataset-type dofbot \
  --output-dir verify2act/output/contrastive/dofbot_twin \
  --cached-dino-dir $CACHE \
  --epochs 25 --batch-size 16 --learning-rate 1e-4 \
  --lambda1 0.5 --lambda2 0.7 --kl-weight 5e-4 \
  --init-from verify2act/output/contrastive/calvin/best_contrastive_critic.pt
```

The best checkpoint is `verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt`. It is selected by the
mean of the goal and temporal val AUROC, on twin val episodes.

The WM and critic runs are independent. With 3 GPUs, run them one after the other, each on all 3. Or run them at the
same time with `CUDA_VISIBLE_DEVICES` and `--num_processes` split (e.g. 2 + 1).

## 7. Check, then serve

1. **WM rollouts on twin val episodes:**

   ```bash
   python verify2act/latent_wm/visualize_wm.py --dataset-type robosuite --dataset-dir $OUT \
     --wm-ckpt verify2act/output/v2a_wm/dofbot_twin/wm/ckpt/latent_dynamics_best_weights.pt \
     --encoder-ckpt verify2act/output/v2a_wm/calvin/encoder/ckpt/delta_encoder_best.pt \
     --decoder-ckpt verify2act/output/v2a_wm/calvin/decoder/latent_decoder_best.pt \
     --history-len 3 --num-samples 10 --token-dim 128 --num-latent-tokens 32 --causal-masking
   ```

2. **Real frames (the point of all this):** copy both checkpoints back to the lab PC. Compare them with the CALVIN
   ones on the logged real planning calls (`verify2act/output/real/*/imagination_logs/`). The goal head should now
   separate achieved from not-achieved real frames; it was at chance with CALVIN (`REAL_ROBOT_PLAN.md`). The
   thresholds θ_c / θ_p may need recalibrating: `verify2act/critic/calibrate_thresholds.py`.

3. **Serve:**

   ```bash
   python -m verify2act.robot.server --wm-mode v2a_wm \
     --latent-wm-ckpt verify2act/output/v2a_wm/dofbot_twin/wm/ckpt/latent_dynamics_best_weights.pt \
     --critic-ckpt verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt
   ```

   The encoder and decoder defaults stay the CALVIN-wider ones.

## Knobs (YAML)

| key | what | now |
|---|---|---|
| `episodes.family_weights` | goal-family mix | eval 0.3, bin 0.3, side 0.17, stack 0.15, compound 0.08 |
| `episodes.random_walk_frac` | episodes without a language goal | 0.2 |
| `placement.p_tag_toward_camera` | blocks showing their tag end | 0.4 |
| `placement.yaw_std / yaw_max` | block yaw spread (deg) | 20 / 50 |
| `placement.p_fail` | real-robot style misplacements (0 = the WM learns the intended effect) | 0 |
| `camera.poses` | fitted home poses; refit with `fit_camera.py --write` after new real logs | 10 poses |
| `render.lights`, `augment` | lighting and webcam-look ranges | see YAML |
