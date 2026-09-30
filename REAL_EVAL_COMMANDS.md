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
  ./
```
About 3.3 GB. The diffusion variant also downloads `timbrooks/instruct-pix2pix` and the SD-1.5 VAE from Hugging Face on first start. `--real2sim` also needs MuJoCo with EGL on the lab PC (`python -c "import mujoco"`), and the twin textures in
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

Protocol, layouts and run order: section 3.

## 3. Start layouts (same for every variant)

`real_eval_layouts/<task>-L<k>.png` (generated by `verify2act/twin/make_eval_layouts.py`, states in
`real_eval_layouts/layouts.json`): per layout, a to-scale top view of the sheet with each block's centre in cm from the
sheet's **far edge** and **left edge** (as seen from the robot), its rotation and which end carries the tag, and the
expected arm-camera view to compare with the live camera before starting. All four blocks are on the sheet in every
layout; each is fully in view from every fitted home pose, the goal does not hold yet, and the intended plan works in the
twin. For task2a/2b, odd layouts start with the block on the wrong side, even ones on the right side but too far away.

**3 EVAL layouts per task (L1-L3) = 21 episodes per variant, 84 in total.** L4-L5 are spares: use one if an EVAL layout
turns out unreachable on the real arm, and use the same spare for every variant. Put the layout id in the session notes.

Run each layout on all four variants before moving on (re-place the blocks from the sheet each time). The variant order
rotates so no variant always goes first. In task priority order:

| layout | variant order |
|---|---|
| task2a-L1 | v2a_wm → vlm_only → rla_wm → diffusion_wm |
| task2a-L2 | vlm_only → rla_wm → diffusion_wm → v2a_wm |
| task2a-L3 | rla_wm → diffusion_wm → v2a_wm → vlm_only |
| task2b-L1 | diffusion_wm → v2a_wm → vlm_only → rla_wm |
| task2b-L2 | v2a_wm → vlm_only → rla_wm → diffusion_wm |
| task2b-L3 | vlm_only → rla_wm → diffusion_wm → v2a_wm |
| task1a-L1 | rla_wm → diffusion_wm → v2a_wm → vlm_only |
| task1a-L2 | diffusion_wm → v2a_wm → vlm_only → rla_wm |
| task1a-L3 | v2a_wm → vlm_only → rla_wm → diffusion_wm |
| task1c-L1 | vlm_only → rla_wm → diffusion_wm → v2a_wm |
| task1c-L2 | rla_wm → diffusion_wm → v2a_wm → vlm_only |
| task1c-L3 | diffusion_wm → v2a_wm → vlm_only → rla_wm |
| task1b-L1 | v2a_wm → vlm_only → rla_wm → diffusion_wm |
| task1b-L2 | vlm_only → rla_wm → diffusion_wm → v2a_wm |
| task1b-L3 | rla_wm → diffusion_wm → v2a_wm → vlm_only |
| task1d-L1 | diffusion_wm → v2a_wm → vlm_only → rla_wm |
| task1d-L2 | v2a_wm → vlm_only → rla_wm → diffusion_wm |
| task1d-L3 | vlm_only → rla_wm → diffusion_wm → v2a_wm |
| task3a-L1 | rla_wm → diffusion_wm → v2a_wm → vlm_only |
| task3a-L2 | diffusion_wm → v2a_wm → vlm_only → rla_wm |
| task3a-L3 | v2a_wm → vlm_only → rla_wm → diffusion_wm |

If time runs out, stop at a layout boundary so every variant has the same layouts. Switching the server between variants
for every layout costs ~3 min per switch; to save time you can run one variant over a block of layouts (e.g. all task2a
and task2b layouts), then the next variant over the same block, as long as each block is finished by all four.

## 4. Known limits (say them in the paper)
- real2sim fits blocks on the table only. Frames with a stack (task3a after success) or blocks in the bin may fit poorly;
  when rms > 2.5 px the server plans on the real frame (reply `real2sim.rms`, log line `poor fit, planning on the real
  frame`). Fit time 12-16 s per call (CPU).
- The temporal head (θ_c) mean-pools DINO patches: it rejects scene-level jumps (different scene: 99% rejected) but is
  blind to layout errors; plan selection rests on the goal head.
- Diffusion has no critic (as in the sim). Its LoRA was fine-tuned on twin data from the CALVIN LoRA (best at step 1000
  of 3000; the CALVIN run used 16k). Offline on the real2sim renders it draws the twin scene but rarely carries out the
  action (blocks vanish or change colour): expect its reflections to work from poor imaginations (`SESSION_LOG_TWIN.md`).
- RLA-WM trained for fewer epochs than V2A's WM (overnight, time-boxed); its goal head was trained on its own
  imaginations with the same recipe as V2A's, starting from the true-frame-only head v1.
