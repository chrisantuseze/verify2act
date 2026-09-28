# Twin dataset session log (csg2, 2026-09-25 → 09-26)

Branch `main-wm`, repo `/home/scratch1/cheze/verify2act`. Host is **csg2** (3× Tesla T4 15 GB, 64 cores, 92 GB RAM),
not csg1 as `TWIN_DATASET.md` says. Follows `TWIN_DATASET.md` (procedure) and `DIGITAL_TWIN_PLAN.md` (design).

## Status at time of writing (2026-09-26 ~11:40 CDT)

| step | state | where |
|---|---|---|
| twin dataset (20k episodes) | done | `verify2act/data/twin/dofbot_v1` (35,373 transitions, 3.5 GB) |
| DINOv2 feature cache | done | `verify2act/data/twin/dino_features` (75,373 files, 38 GB) |
| WM fine-tune | done, early-stopped at epoch 44/50 | `verify2act/output/v2a_wm/dofbot_twin/wm_old/ckpt/latent_dynamics_best_weights.pt` |
| critic fine-tune | done (25 epochs) | `verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt` |
| decoder fine-tune | **RUNNING**, epoch ~9/30 at 11:38, ~11 min/epoch, best val 0.1131 (ep 8) | `verify2act/output/v2a_wm/dofbot_twin/decoder/latent_decoder_best.pt`, log `verify2act/output/twin_decoder_train.log` |
| real-frame comparison | done, see findings | `verify2act/output/real_eval/*.json` |

The decoder job was started with `nohup accelerate launch ...` (3 GPUs). **After the restart, check it is still alive
(`ps -eo pid,etime,cmd | grep "train_decode[r]"`) and how far it got.** If it died, restart from
`latent_decoder_best.pt` weights-only (see gotchas: a full checkpoint restores epoch/best-val).
ETA if it survived: ~epoch 30 at roughly 15:45 CDT. `latest`/`ep5` checkpoints every 5 epochs; best is saved on val improvement.
Early stop is fine: the best file is usable at any time.

## What was done

1. **Generation**: `python -m verify2act.twin.generate --out verify2act/data/twin/dofbot_v1 --num-episodes 20000 --workers 32 --chunk 50 --seed 0`
   (5 min). Smoke test (200 eps) looked right (textures, ArUco tags, mix). Family counts: eval 4758, bin 5110, side 2759, stack 2591,
   random 4326, **compound 456 (2.3%, target ~6.4%; probably rejection sampling; not investigated)**.
2. **DINOv2 cache** built with 3 sharded processes (one per GPU).
3. **WM**: 3-GPU accelerate, fp16, batch 16/GPU, from CALVIN-wider weights-only. Val loss best ≈0.416 around epochs 20-24, then
   overfit (train 0.33, val 0.46); early stopping fired at epoch 44.
4. **Critic**: 3-GPU, bf16, `--dataset-type dofbot`, `--init-from` CALVIN. Val mean AUROC 1.0 (goal 0.9997, temporal 1.0), ~1.5 min/epoch.
5. **Decoder**: `train_decoder.py`, twin `image_t1` frames, L1 + LPIPS, 30 epochs, batch 16/GPU, weights-only init from the CALVIN decoder
   (`.../dofbot_twin/decoder/init_calvin_weights.pt`). The CALVIN decoder gave brown noise on twin frames (`visualize_wm.py` samples in
   `verify2act/output/visualizations/dofbot_twin/`), which is why it is being fine-tuned.
6. **Real-frame / diagnostic evaluation** (new scripts, all untracked):
   - `verify2act/twin/eval_real.py`: re-scores the initial real frames in `verify2act/output/real/*/imagination_logs/` (6 frames, mostly
     task2a "red left of blue") for every (WM, critic) pair over ~40 enumerated candidate plans; "pairwise" = P(correct plan's goal
     score > wrong plan's). `--twin-dir` runs it on twin frames instead.
   - `verify2act/twin/diag_critic_wm.py`: separates critic from WM on twin task2a episodes.
   - `verify2act/twin/diag_critic_tasks.py`: goal-head AUROC on ground-truth twin frames per task.

## Findings

- **Real frames: every pair is at chance** (pairwise 0.45-0.54; CALVIN+CALVIN 0.50, twin+twin 0.54). Tiny sample (6 frames, 5 scenes);
  labels are mine (correct = red left of blue, or blue right of red).
- **Twin frames (n=20)**: twin+twin only 0.64 (CALVIN+CALVIN 0.64). So the gap is not only domain shift.
- **Twin critic on ground-truth frames, by task** (achieved final vs start): task1a/1b/1c 0.97/0.95/0.98, task1d 1.00, task3a 0.97,
  **task2a 0.75, task2b 0.61**. The 0.9997 val AUROC pooled all tasks and hid this: left/right relations are near chance.
  Nearly all the real robot logs are task2a.
- **WM (task2a, twin, 60 episodes)**: correct action scores above a wrong action 78% of the time (CALVIN 66%), but the imagined latent is
  as close to the true final frame for a wrong action as for the right one (cosine 0.834 vs 0.830): it mostly copies the scene.
  Imagined-frame critic AUROC 0.65 vs 0.72 on ground truth (so the WM costs little there; the critic is the main limit).
- CALVIN pair: imagined frames score systematically lower than real ones (AUROC 0.009): an offset, not discrimination.
- Twin critic shifts real-frame goal scores to +0.4..0.56 (above θ_p 0.05), so it would stop rejecting every plan, but with no discrimination.
  Twin WM raises the temporal score 0.64 → 0.89.

## Proposed next steps (not started; user has not approved)

1. **Critic**: more left/right signal: hard negatives that differ only in left/right (swap the pair order / mirrored placement), upweight
   `side` episodes (now 17%), maybe more `side` data, then retrain on the cached features (1-2 h). Re-run `diag_critic_tasks.py`
   (target: task2a/2b ≫ 0.75/0.61).
2. **WM**: probably needs a loss that weights changed regions (frame-copy already scores well on flow matching). Bigger change; critic first.
3. After the decoder finishes: run `visualize_wm.py` again with `--decoder-ckpt verify2act/output/v2a_wm/dofbot_twin/decoder/latent_decoder_best.pt`
   (add `--output-dir`), look at imagined frames, and check the decoder on real frames (`output/real/.../request_image.png`). If weak on
   real, add real frames to the decoder data.
4. Optionally investigate the low `compound` share; recalibrate θ_c / θ_p (`verify2act/critic/calibrate_thresholds.py`); serve with
   `python -m verify2act.robot.server --wm-mode v2a_wm --latent-wm-ckpt <twin wm> --critic-ckpt <twin critic> --wm-decoder-dir <twin decoder dir>`.

## Uncommitted changes in the repo

- `M verify2act/critic/cache_utils.py`: `ensure_cache_complete(..., shard=0, num_shards=1)` (parallel cache filling; default behaviour unchanged).
- `?? verify2act/twin/eval_real.py`, `diag_critic_wm.py`, `diag_critic_tasks.py`, this file.
- Installed into the user env: `openai-clip` (from git), `lpips` (both were in `requirements.txt`, missing here).
- `verify2act/output/` is gitignored (checkpoints, dataset, logs, eval JSONs live only on csg2).

## Gotchas learned (csg2)

- **`RLIMIT_NPROC` is 1024.** Each spawned worker starting 64 OpenBLAS threads made most workers crash. Always
  `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for generation and training.
- `train_dynamics.py` builds the DINOv2 cache on rank 0 only; ranks 1-2 wait at a barrier with a 10-min NCCL timeout, so a cold cache
  kills a 3-GPU run. Build the cache first (sharded, see `cache_utils.py`), then launch training.
- Full checkpoints passed to `--resume-from` restore epoch/best-val (CALVIN decoder is at epoch 100, so nothing would train). Use a
  weights-only state dict (e.g. `init_calvin_weights.pt`).
- **`pkill -f` / `pgrep -f <pattern>` matches the calling shell's own command line** (killed the shell twice and made a wait loop hang
  for 2.5 h). Use a bracketed pattern (`grep "train_decode[r]"`) or a PID.
- Scripts run by path need `PYTHONPATH=$PWD`; use `python -m verify2act...` instead.
- The planner returns `-inf` for a candidate if a step fails the temporal gate or `max_retries=0` (the attempt loop is
  `range(max_retries)`); `eval_real.py` uses `temporal_threshold=-1e9, max_retries=1`. The critic's uncertainty gate can still return `-inf`.
- EGL rendering works here (`MUJOCO_GL=egl`, 24 twin tests pass). Generation with 32 workers: ~5 min for 20k episodes.
- Logs: `verify2act/output/{twin_gen,twin_wm_train,twin_critic_train,twin_decoder_train}.log`, `cache_shard{0,1,2}.log`.

---

# Session 2 (2026-09-26 ~11:45 → , autonomous research pass)

## Root causes found

1. **Critic mean-pools DINOv2 patch tokens** (`critic/model.py` `encode_features`). Pair-conditioned probe on twin
   frames, "is A left of B": MLP on mean-pooled tokens **0.495** (chance) vs a small transformer on the 16x16 grid
   **0.998** (`verify2act/twin/diag_probe2.py`; linear probes `diag_probe.py` are weak for both, relation is nonlinear).
2. **Language side cannot express argument order**: pooled CLIP + `clip_goal_proj`, cos("red left of blue",
   "blue left of red") = 0.994 (left vs right word: 0.78). And the text loss is only an MSE pull toward achieved
   frames (`train_contrastive.py` `loss_lang`), nothing pushes a goal away from frames where it is false.
3. **WM: the CALVIN delta autoencoder does not transfer to the twin.** On twin deltas: rel. recon error 1.01,
   cosine on the 10 most-changed patches 0.03, magnitude 20%. Goal head on F_t + AE(F_t1-F_t) = AUROC **0.50** (true
   F_t1: 0.97), so no dynamics on that latent space can work (old twin WM: 0.50, P(correct > wrong action) 0.47).
   (`verify2act/twin/diag_wm_chain.py`.) The previous WM fine-tune only trained dynamics, with the CALVIN AE frozen.
4. `eval_real.py` labels were noisy: real frames 6/7 (run 021122/021206 call_00) already satisfy "red left of blue",
   so most plans are correct there, but only 2 were labelled correct.

## Fixes built

- `verify2act/twin/goal_labels.py`: state-based predicates (match `scene.py` exactly: 964/964 episode outcomes) +
  goal sampler per frame (eval tasks verbatim, true side/stack relations with flipped-side and swapped-argument hard
  negatives, bin goals of all styles).
- `verify2act/critic/goal_head.py`: `SpatialGoalHead` (4-layer transformer over [CLS, CLIP per-token text features,
  256 patch tokens]) -> P(goal satisfied); `GoalScorer` wrapper. Attached to the critic as `critic.goal_scorer`
  (`--goal-head-ckpt` in `pipeline/inference.py` and `robot/server.py`); `ProbEmbedding` now carries `tokens`;
  `goal_sim_from_text*` route to it (returns prob, std 0). Planner code unchanged. Use `--theta-p ~0.5` with it.
- `verify2act/twin/train_goal_head.py`: trains it on the cached features (55k frames in RAM, ~29 GB, ~6 min/epoch
  on one T4). Real-frame check: the 11 real request frames, hand labels for "red left of blue" (pos 5,6,7,8,10;
  neg 0,2,3,9) + their horizontal mirrors (flip swaps left/right) -> `verify2act/output/goal_head/real_feats.pt`.
- Twin delta autoencoder: `train_encoder.py` fine-tuned from CALVIN weights (weights-only init
  `dofbot_twin/encoder/init_calvin_weights.pt`), stopped at epoch ~27 (val MSE 0.84 -> 0.557):
  `verify2act/output/v2a_wm/dofbot_twin/encoder/ckpt/delta_encoder_best.pt`. Top-10-patch cosine 0.03 -> 0.84;
  goal-head AUROC on AE reconstructions 0.50 -> 0.90 (epoch 12 snapshot).
- `verify2act/twin/eval_plans.py`: plan ranking on held-out twin episodes with **simulated** labels (every valid
  subtask executed in the twin, goal predicate on the result): top1 / pairwise / acc@threshold per (WM, critic).

## Results

Goal head v1 (`verify2act/output/goal_head/v1/goal_head_last.pt`, 10 epochs), start vs final frame AUROC on held-out
twin episodes: task1a-d 1.00, **task2a 0.993 (was 0.75)**, **task2b 0.973 (was 0.61)**, task3a 1.00; sampled-goal val
AUROC 0.989. Real (tiny): all 5 positives > their mirrors, P(pos) 0.77-0.95, P(neg) ~0, false eval goals ~0.
Remaining task2b errors are borderline predicate cases (green already right of yellow but >1 block length offset in
depth, or 11.4 cm > 10.5 cm away): the head is lenient on the alignment tolerance, not blind to left/right.

Decoder fine-tune stopped at epoch 11 (best val 0.1076, `latent_decoder_best.pt`) to free GPUs; resume from `ep10`.

### WM retrain (wm2) and plan-level results

- wm2: `train_dynamics.py` with the twin AE (`--encoder-ckpt dofbot_twin/encoder/ckpt/delta_encoder_best.pt`),
  `--history-len 3 --causal-masking` (needed to load the old weights), init = old twin WM weights, 3 GPUs,
  ~8 min/epoch -> `verify2act/output/v2a_wm/dofbot_twin/wm2`. Log `verify2act/output/twin_wm2_train.log`.
  (First two launches died: missing --history-len/--causal-masking; then OOM from sharing GPU 0 with the goal head.)
- Chain diagnostic, wm2 after 1 epoch (goal head v1, n=79 held-out single-step episodes): AE ceiling 0.957,
  imagined correct action AUROC 0.76 (mean P 0.44), wrong action 0.52, **P(correct > wrong) 0.85** (old WM 0.47).
- Plan ranking (`eval_plans.py`, 39 held-out frames, ~28 valid plans each, simulated labels), top1 / pairwise:
    old WM + pooled 0.18 / 0.72 · old WM + goal head 0.10 / 0.55 · wm2(ep1) + pooled 0.08 / 0.72 ·
    **wm2(ep1) + goal head 0.54 / 0.85** (task2a 0.57, task2b 0.82, task3a 0.29). Both fixes are needed.
- Decoder (ep11) reconstructs real frames faithfully (`verify2act/twin/viz_imagine.py`,
  `output/visualizations/dofbot_twin/imagine_oldwm.png`): old WM imaginations are exact copies of the input.

### Real frames: sim-to-real gap in the WM, and a real-to-sim bridge

- wm2 on real frames (8 WM samples per plan, goal head v1): the correct action reaches P(goal) 0.81-0.97 in its best
  sample but mean 0.1-0.17 (twin frames: mean ~0.65): on real inputs the WM usually *removes* the moved block
  without placing it. More flow steps (20 vs 5) make it worse, so it is not under-integration.
- `fit_camera.py` already fits camera + block (x, y, yaw) per real frame from colour masks. Ran it on the 11 real
  frames (`verify2act/output/twin/camera_fit/fits.json`, rms <= 1.02 px except frame 10: 1.62, skipped) and on their
  mirrors (`output/real_mirror/`, `camera_fit_mirror/`). Twin re-renders of the fitted scenes match closely
  (`output/real_eval/real2sim_renders/`). Limitation: blocks assumed on the table (no stacks).
- `verify2act/twin/eval_real2sim.py`: 20 real scenes x {task2a, task2b, task3a} not yet met = 50 cases, labels by
  simulating every valid plan from the fitted state (strict predicates; 5.5% of plans correct -> chance top1 ~0.06).
  Results (wm2 = epoch-5 snapshot, goal head v1):
    old WM + pooled critic, real image        top1 0.06  pairwise 0.62   (= the system that ran on the robot)
    wm2 + goal head, real image               top1 0.28  pairwise 0.69   (2a 0.33, 2b 0.44, 3a 0.10)
    wm2 + goal head, twin re-render (r2s)     top1 0.48  pairwise 0.76   (2a 0.75, 2b 0.50, 3a 0.30)
- Imagined training set for the goal head: `gen_imagined.py` with wm2 ep5, 6000 training-episode frames x 3 valid
  subtasks, labels from simulated next states -> `verify2act/data/twin/imagined_wm2ep5.pt` (13,275 samples).

### Server wiring

- `robot/backend.py` built the critic with `SimpleNamespace(critic_ckpt=...)` only, so a goal head would have been
  ignored by the server: now passes `goal_head_ckpt` (`--goal-head-ckpt`).
- `verify2act/twin/real2sim.py` (`Real2Sim`): FrameFit with the config camera as init and fixed fovy, 6 restarts,
  12-16 s per frame on CPU, returns a twin render at the real resolution (None if fit rms > 2.5 px).
  `--real2sim` on `robot/server.py`: the planner (VLM + WM + critic) gets the render; `request_image.png` still logs
  the real frame, the render goes to `real2sim_render.png`, fit rms/time in the response (`real2sim`).
  `test_robot_server.py`: 22 passed.

### Goal head v2: trained on WM-imagined states

- `train_goal_head.py --init v1/goal_head_last.pt --imagined verify2act/data/twin/imagined_wm2ep5.pt --imagined-repeat 2 --ae-ckpt <twin AE>
  --ae-frac 0.3 --batch 40 --lr 1e-4` (-> `verify2act/output/goal_head/v2`; ~25-35 min/epoch while sharing a GPU).
  After 1 epoch (`v2/snap_ep0.pt`): true-frame task2a 0.997, task2b 0.998; AE-reconstructed 0.992 / 0.963; real
  checks unchanged.
- With wm2 ep5:
    twin plans (eval_plans, 39 frames):  top1 0.74 (v1 0.82), pairwise 0.954 (v1 0.926)
    real scenes, real image:             top1 0.58 (v1 0.28), pairwise 0.894 (v1 0.689)  2a 0.25 / 2b 0.61 / 3a 0.75
    real scenes, real2sim render:        top1 0.84 (v1 0.48), pairwise 0.949 (v1 0.759)  2a 0.92 / 2b 0.67 / 3a 0.95
  Training the critic on imagined states is what makes it robust to the WM's artifacts on real inputs.

## Final numbers (wm2 epoch-13 snapshot `wm2/wm2_ep13_weights.pt` + goal head v2 `goal_head/v2/goal_head_last.pt`)

Chain (79 held-out twin single-step episodes, `diag_wm_chain.py`): true 0.999, AE ceiling 0.996, **WM-imagined correct
action 0.994** (mean P 0.94), wrong action 0.59 (mean P 0.08), **P(correct > wrong) 1.00** (session start: 0.50 / 0.47).

| plan ranking, top1 / pairwise                         | old WM + pooled | wm2 + goal head v2 |
|-------------------------------------------------------|-----------------|--------------------|
| twin held-out (59 frames, `eval_plans.py --n 60`)     | 0.18 / 0.72     | **0.90 / 0.98** (acc@0.5 0.94) |
| real scenes, real image (50 cases, `eval_real2sim`)   | 0.06 / 0.62     | **0.70 / 0.90**    |
| real scenes, real2sim re-render                       | -               | **0.86 / 0.98**    |

Per task, real image: 2a 0.67, 2b 0.50, 3a 0.90; real2sim: 2a 0.75, 2b 0.78, 3a 1.00. Chance top1 ~0.06.
With 3 WM samples per plan (planner `max_retries=3`, score = best sample; `--max-retries 3` on the server):
real image **0.80 / 0.90** (2a 0.83, 2b 0.56, 3a 1.00), real2sim 0.86 / 0.97.
JSONs: `verify2act/output/real_eval/final_*.json`.

### How to serve it

    python -m verify2act.robot.server --wm-mode v2a_wm \
      --latent-wm-ckpt verify2act/output/v2a_wm/dofbot_twin/wm2/ckpt/latent_dynamics_best_weights.pt \
      --encoder-ckpt verify2act/output/v2a_wm/dofbot_twin/encoder/ckpt/delta_encoder_best.pt \
      --critic-ckpt verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt \
      --goal-head-ckpt verify2act/output/goal_head/v2/goal_head_last.pt \
      --wm-decoder-dir verify2act/output/v2a_wm/dofbot_twin/decoder --theta-p 0.5 --max-retries 3 [--real2sim]

`--encoder-ckpt` is required: the server default is the CALVIN delta AE, which cannot represent twin/real changes.
The temporal head (theta_c) is still the old pooled one; not re-validated with wm2.

### Open issues / next steps

1. Real robot runs with the command above (with and without --real2sim), to replace the 50-case offline proxy.
   A larger labelled real set: log a few dozen request frames per task; `fit_camera` + simulation labels them.
2. Real-image WM gap (0.70 vs 0.86 via real2sim): WM often drops the moved block on real inputs. Options: photometric /
   texture randomisation in DINO space during WM training, or generate twin data from the fitted real cameras.
3. real2sim assumes blocks on the table (no stacks) and takes 12-16 s per frame (6 restarts); fewer restarts or a
   warm start from the previous call would cut this.
4. Remaining misses: red/green confusion on some real scenes for task2b; borderline predicate cases (alignment
   tolerance, stacked block "left of") where the head is lenient.
5. Temporal head (theta_c) and uncertainty gate were not re-examined; the goal head returns std 0.
6. wm2 kept training after the snapshot (val 0.2975 at ep13 -> 0.2861 at ep16, still improving; early stopping,
   patience 6, max 30 epochs, log `output/twin_wm2_train.log`). Re-run the final evals on its final best:
   `eval_plans --n 60`, `eval_real2sim` (with and without `--samples 3`), `diag_wm_chain`. The imagined training set
   for the goal head came from the ep5 snapshot; regenerating it from the final WM and fine-tuning v2 may help.
7. Decoder fine-tune stopped at epoch 11 (fine on real frames); `compound` share still 2.3%.

## Final numbers, fully trained wm2 (30 epochs, best val 0.2743 at epoch 26, `wm2/ckpt/latent_dynamics_best_weights.pt`)
## + goal head v2 (`goal_head/v2/goal_head_last.pt`). JSONs: `output/real_eval/final2_*.json`.

- Chain (79 twin episodes): imagined correct action AUROC 0.996 (mean P 0.96), wrong 0.59 (P 0.07), P(correct > wrong) 1.00.
- Twin plans (59 frames): top1 **0.915**, pairwise 0.992, acc@0.5 0.945 (2a 0.79, 2b 0.93, 3a 1.00).
- Real scenes (50 cases), top1 / pairwise:
    real image, 1 sample     0.78 / 0.91   (2a 0.83, 2b 0.61, 3a 0.90)
    real image, 3 samples    0.78 / 0.935  (2a 0.75, 2b 0.61, 3a 0.95)
    real2sim,   1 sample     0.86 / 0.972  (2a 0.75, 2b 0.78, 3a 1.00)
    real2sim,   3 samples    **0.92 / 0.969**  (2a 0.83, 2b 0.89, 3a 1.00)
  (old system: 0.06 / 0.62.) The ep13-snapshot numbers above are superseded; item 6 of "Open issues" is done.
