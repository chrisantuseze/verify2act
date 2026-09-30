# Real-robot evaluation: commands (2026-09-30)

Four variants, one server process each, all on the **twin** models (`--preset twin`). Every WM variant runs with
`--real2sim`: the WM and critic start from a twin re-render of the camera frame, **the VLM sees the camera frame** (as in
`vlm_only`), so the variants differ only in how plans are verified. `vlm_only` has no WM, so real2sim does not apply.

| variant | `--wm-mode` | world model | verifier | server log dir (lab PC) |
|---|---|---|---|---|
| V2A (ours) | `v2a_wm` | wm3 (`v2a_wm/dofbot_twin/wm3`) | twin temporal head + goal head v3 (trained on wm3 imaginations) | `verify2act/output/real/twin/v2a_wm_r2s/` |
| RLA-WM | `rla_wm` | RLA-WM fine-tuned on twin data (`rla_wm/dofbot_twin/wm`) | twin temporal head + its own goal head (`goal_head/rla_twin`, trained on RLA imaginations) | `.../real/twin/rla_wm_r2s/` |
| Diffusion (ReflectVLM) | `diffusion_wm` | InstructPix2Pix LoRA fine-tuned on twin data (`diffusion_wm/dofbot_twin/wm/best`) | none: VLM reflects once on the imagined image, as in the sim | `.../real/twin/diffusion_wm_r2s/` |
| VLM only | `vlm_only` | – | none: one propose, executed as is | `.../real/twin/vlm_only/` |

Preset values (`verify2act/robot/server.py::PRESETS["twin"]`): θ_c 0.5, θ_p 0.5 (the goal head returns P(goal)),
max_retries 3 (best of 3 WM samples), max_replans 2, beam width 3, horizon 4, gemini-2.5-flash on Vertex (`verify2act`).
θ_c 0.5 was checked on 2026-09-29 (`verify2act/twin/eval_temporal.py`, see `SESSION_LOG_TWIN.md`).

## 0. Before starting (lab PC, 192.168.0.113)

Code (branch `main-wm`; push from csg2 first if not yet pushed: `git push origin main-wm`):
```bash
conda activate verify2act
cd ~/Projects/multi-object-manip/verify2act
git pull origin main-wm
python -m pytest verify2act/robot/test_robot_server.py -q          # 24 passed
```

Checkpoints (gitignored; they live on csg2). Run on the lab PC, replacing `CSG2` with your ssh login for csg2:
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
  $CSG2:$R/verify2act/output/diffusion_wm/calvin/decoder/checkpoint-5000 \
  ./
```
About 3.5 GB. `--real2sim` also needs MuJoCo with EGL on the lab PC (`python -c "import mujoco"`), and the twin textures in
`verify2act/twin/assets/` (now in git).

## 1. Offline smoke test per variant (no robot; one Gemini call each)

Use any real request frame, e.g. one copied from an earlier run (`imagination_logs/planning_call_00/request_image.png`):
```bash
export MUJOCO_GL=egl
F=frame.png; G="Put the red block to the left of the blue block"
python -m verify2act.robot.server --preset twin --real2sim --wm-mode v2a_wm       --offline-image $F --goal "$G"
python -m verify2act.robot.server --preset twin --real2sim --wm-mode rla_wm       --offline-image $F --goal "$G"
python -m verify2act.robot.server --preset twin --real2sim --wm-mode diffusion_wm --offline-image $F --goal "$G"
python -m verify2act.robot.server --preset twin            --wm-mode vlm_only     --offline-image $F --goal "$G"
```
Check: a plan in vocabulary, `real2sim.rms` < 2.5 in the reply, and under `verify2act/output/real/twin/<mode>_r2s/offline/`
`real2sim_render.png` looks like the frame and the decoded imagination (`imagine_frame.png`) moved the right block.

## 2. Live: one variant at a time

Jetson (192.168.0.8), once: roscore, robot stack, then `roslaunch rosbridge_server rosbridge_websocket.launch`.

Lab PC, one of (wait for `Ready: listening on /v2a/request`, ~2-4 min):
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
Tasks: `task1a task1b task1c task1d` (bin clearing), `task2a task2b` (left/right of), `task3a` (stack). The session meta
records `wm_mode`, the thresholds and `execute_unverified=true`; the operator's y/n per episode is `real_success`.
Switching variant = stop the server (Ctrl-C), start the next one; the Jetson session just needs a restart.

### Suggested protocol under time pressure
- Same scene layouts for every variant: set up layout k, run it on all four variants before moving to layout k+1
  (or at least fix the list of layouts per task and reuse it), so variants see the same starts.
- Priority if time runs short: task2a, task2b, task1a, task1c (most discriminative offline), then task1b/1d, task3a.
  Variant order within a layout: v2a_wm, vlm_only, rla_wm, diffusion_wm (the two needed for the main claim first).
- 3 episodes per task per variant = 84 episodes at 7 tasks; about 3-5 min each.

## 3. Known limits (say them in the paper)
- real2sim fits blocks on the table only. Frames with a stack (task3a after success) or blocks in the bin may fit poorly;
  when rms > 2.5 px the server plans on the real frame (reply `real2sim.rms`, log line `poor fit, planning on the real
  frame`). Fit time 12-16 s per call (CPU).
- The temporal head (θ_c) mean-pools DINO patches: it rejects scene-level jumps (different scene: 99% rejected) but is
  blind to layout errors; plan selection rests on the goal head.
- Diffusion has no critic (as in the sim). Its LoRA was fine-tuned on twin data for 3000 steps from the CALVIN LoRA
  (the CALVIN run used 16k); see `SESSION_LOG_TWIN.md` for its offline check on real2sim renders.
- RLA-WM trained for fewer epochs than V2A's WM (overnight, time-boxed); its goal head was trained on its own
  imaginations with the same recipe as V2A's, starting from the true-frame-only head v1.
