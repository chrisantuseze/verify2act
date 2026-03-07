# World Model Data Collection Pipeline

This document describes the end-to-end data collection process, pipeline design, and output format used to produce training data for the diffusion world model (InstructPix2Pix LoRA finetune + decoder finetune) and the latent-space critic classifier.

---

## 1. Pipeline Overview

The data collection runs in two stages:

**Stage 1 — Episode Collection:** Roll out expert and noisy policies in robosuite, saving RGB frames (512×512, `agentview` camera), simulator states, and action prompts at every timestep. Goal images are extracted from the last frame of successful episodes.

**Stage 2 — Reachability Labeling:** A post-processing script restores saved simulator states and runs the expert policy from each state to determine whether task success is reachable within a fixed horizon H, producing binary labels for the critic.

---

## 2. Episode Collection Process

### 2.1 Environment Setup

- Environment: `ClutteredNutAssembly` or `NutAssembly` (robosuite)
- Robot: Panda
- Camera: `agentview` — fixed overhead-angled view, 512×512 resolution, RGB only (no depth/segmentation)
- `use_camera_obs=True`, `has_offscreen_renderer=True`

### 2.2 Policy Modes

| Mode | Description | Dataset share |
|---|---|---|
| `expert` | Unmodified heuristic policy (deterministic, high success rate) | 40% |
| `noisy_0.02` | Expert + Gaussian noise σ=0.02 on action vector | 20% |
| `noisy_0.05` | Expert + Gaussian noise σ=0.05 | 20% |
| `noisy_0.10` | Expert + Gaussian noise σ=0.10 | 20% |

Noisy policies produce off-path states and frequent failures, which are critical for world model diversity and critic negative examples.

### 2.3 Per-Timestep Recording

At every timestep `t`:

1. **Render RGB frame:** `agentview` at 512×512 → save as `frame_{t:05d}.png`
2. **Save simulator state:** `env.sim.get_state()` → `numpy.savez_compressed` as `state_{t:05d}.npz` (qpos + qvel)
3. **Extract action metadata from policy:** skill type (`pick` / `place` / `insert`), target object name, Cartesian target position
4. **Build action prompt:** using a fixed template → `"pick round nut. position: (0.12, -0.05, 0.92)."`
5. **Buffer transition:** `{image_t, image_t1, action_text, action_params, state_t, state_t1}`

### 2.4 Action Prompt Templates

Three skill types, fixed format:

```
"pick [object]. position: (x, y, z)."
"place [object]. position: (x, y, z)."
"insert [object]. position: (x, y, z)."
```

Position is the Cartesian end-effector target in world frame (metres), extracted from the heuristic policy's internal target at each step. Object names match robosuite names: `round nut`, `square nut`, etc.

### 2.5 Goal Image

At episode end, if the episode succeeded, the last rendered frame is saved as `goal.png`. All transitions in that episode reference this file. For failed episodes, the goal image is borrowed from the nearest successful episode (tracked via a `goal_image_source` field).

### 2.6 Episode Flow

```
episode_start():
    env.reset() → s0
    render agentview 512×512 → frame_00000.png
    save sim state → state_00000.npz

for t in 0..T-1:
    policy.step() → action_array, skill_type, cartesian_target
    build_action_prompt(skill_type, object_name, target) → action_text
    env.step(action_array) → obs, reward, done, info
    render agentview 512×512 → frame_{t+1:05d}.png
    save sim state → state_{t+1:05d}.npz
    buffer transition: {frame_t, frame_t+1, action_text, params, state_t, state_t+1}

episode_end():
    if success:
        copy last frame → goal.png
    else:
        goal.png = reference from nearest successful episode
    flush all buffered transitions → append to transitions.jsonl
    set label_reachable = -1 (unlabeled, computed in Stage 2)
```

---

## 3. Reachability Labeling (Stage 2 — Post-Processing)

Run separately after all episodes are collected, via `compute_labels.py`.

### 3.1 Procedure Per Transition

1. Load `state_{t+1}.npz` → call `env.sim.set_state(qpos, qvel)` and `env.sim.forward()` to restore simulator
2. Run expert policy from restored state for up to H timesteps
3. Check `env._check_success()` at each step
4. If success within H steps → `label_reachable = 1`, else `label_reachable = 0`
5. Write label into `labels.jsonl` keyed by `(episode_id, timestep)`

### 3.2 Horizon Budget (H)

Fixed per task: 300 for ClutteredNutAssembly, 200 for NutAssembly. Matches the planner's nominal horizon.

### 3.3 Class Balance

Target: ≥30% positive labels. If imbalanced, collect more expert trajectories or oversample positives during critic training.

---

## 4. Output Format

### 4.1 JSONL Transition Schema

One JSON line per transition in `transitions.jsonl`:

```jsonc
{
  "episode_id": "ep_00001",
  "timestep": 42,
  "image_t":   "episodes/ep_00001/frame_00042.png",
  "image_t1":  "episodes/ep_00001/frame_00043.png",
  "goal_image": "episodes/ep_00001/goal.png",
  "goal_image_source": "self",
  "action_text": "pick round nut. position: (0.12, -0.05, 0.92).",
  "action_params": {
    "skill": "pick",
    "object": "round nut",
    "cartesian_target": [0.12, -0.05, 0.92]
  },
  "state_t":  "episodes/ep_00001/state_00042.npz",
  "state_t1": "episodes/ep_00001/state_00043.npz",
  "policy_type": "expert",
  "episode_success": true,
  "label_reachable": -1
}
```

### 4.2 Directory Layout

```
data_capture_wm/
├── batch_collect.py          # Main collection entry point
├── episode_recorder.py       # Per-step: save RGB, sim state, build prompt
├── policy_wrappers.py        # Policy adapters + noisy wrapper
├── prompt_utils.py           # Action prompt template builder
├── compute_labels.py         # Stage 2: reachability labeling
├── data_loader.py            # PyTorch Dataset over transitions.jsonl
└── dataset/
    └── nut_assembly/
        ├── transitions.jsonl
        ├── labels.jsonl
        ├── metadata.json
        └── episodes/
            ├── ep_00001/
            │   ├── frame_00000.png
            │   ├── frame_00001.png
            │   ├── ...
            │   ├── goal.png
            │   ├── state_00000.npz
            │   ├── state_00001.npz
            │   └── meta.json
            └── ep_00002/
                └── ...
```

---

## 5. Collection Targets

| Task | Minimum transitions | Target transitions |
|---|---|---|
| ClutteredNutAssembly | 5 000 | 20 000 |
| NutAssembly | 3 000 | 10 000 |

Images are saved as lossless PNG. Simulator states as `numpy.savez_compressed`. All paths in `transitions.jsonl` are relative to the dataset root.

---

## 6. CLI Commands

```bash
# Run from robosuite/data_capture_wm/

# Expert episodes
xvfb-run -a python batch_collect.py \
    --env ClutteredNutAssembly \
    --num-episodes 200 \
    --policy-mode expert \
    --output-dir dataset/nut_assembly \
    --camera agentview \
    --image-size 512 \
    --seed 42

# Noisy episodes (σ=0.05)
xvfb-run -a python batch_collect.py \
    --env ClutteredNutAssembly \
    --num-episodes 300 \
    --policy-mode noisy \
    --noise-sigma 0.05 \
    --output-dir dataset/nut_assembly \
    --camera agentview \
    --image-size 512 \
    --seed 0

# Post-processing: reachability labels

## You need to run compute_labels.py to rollout from saved sim states and call env._check_success(), which writes labels.jsonl. That's the only missing step before the critic can train.


python compute_labels.py \
    --dataset-dir dataset/nut_assembly \
    --env ClutteredNutAssembly \
    --horizon 300 \
    --output labels.jsonl
```
