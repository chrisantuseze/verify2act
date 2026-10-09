# Verify2Act

Code for **Verify2Act: Critic-Guided Latent World Models for Verifying Language-Conditioned Manipulation Plans**.

Chrisantus Eze, Christopher Crick · Oklahoma State University

[Project page](https://verify2act.github.io)

![Verify2Act overview](https://verify2act.github.io/static/images/architectural_diagram.png)

Verify2Act checks a manipulation plan before the robot executes it:

1. a **vision-language model** proposes a few candidate multi-step plans from the camera image and the language goal;
2. a **latent world model (V2A-WM)** imagines each plan in DINOv2 feature space;
3. a **dual-head critic** checks every imagined step for temporal consistency and scores the imagined outcome against
   the goal. Rejected plans go back to the VLM for re-planning; only an accepted plan is executed.

V2A-WM is a flow-matching world model with cross-attention action grounding and a causal temporal history window.
For the real robot, the world model and critic are trained only on data from a digital twin of the workspace.

## Repository layout

| path | what it is |
|---|---|
| `verify2act/latent_wm/` | V2A-WM: delta encoder, latent dynamics, decoder, and their training scripts |
| `verify2act/critic/` | dual-head critic (temporal consistency and goal), training and threshold calibration |
| `verify2act/pipeline/` | planner (VLM backends), beam search, reflection, and the simulation inference loops |
| `verify2act/twin/` | MuJoCo digital twin of the DOFBOT workspace: data generation, camera fitting, real2sim, offline evals |
| `verify2act/robot/` | real-robot plan server (talks to the robot over rosbridge) |
| `verify2act/configs/` | prompts and twin configuration |
| `verify2act/real_eval_layouts/` | fixed start layouts for the real-robot evaluation |
| `verify2act/rla_wm_baseline/`, `dino_wm_baseline/`, `diffusion_wm/` | baseline world models used in the comparison |
| `commands/` | the training and evaluation commands for each world-model variant |
| `robosuite/`, `calvin/`, `rla-wm/`, `dino_wm/`, `reflect-vlm/`, `Points2Plans/`, `third_party/` | simulators, benchmarks and baseline code this project builds on |
| `docs/project_notes.md` | detailed project record: plan server protocol, real-robot evaluation, twin design, model lineage |

## Setup

```bash
conda create -n verify2act python=3.10 && conda activate verify2act
pip install -r requirements.txt
pip install "opencv-python>=4.8" scipy        # twin textures and camera fitting
pip install roslibpy google-auth              # real-robot plan server, Vertex AI

cd robosuite && pip install -e . && cd ..

# CALVIN
git submodule update --init calvin
pip install -e calvin/calvin_env
pip install -e calvin/calvin_models --no-deps
pip install pytorch-lightning gym pyhash

pip install -e third_party/MoDE_Diffusion_Policy
```

Twin rendering is headless through EGL (`MUJOCO_GL=egl`); `MUJOCO_GL=osmesa` works on CPU.

The planner supports OpenAI (`gpt-*`) and Gemini (`gemini-*`) models; set the matching API key, or use Vertex AI
application-default credentials for Gemini. Details are in [`docs/project_notes.md`](docs/project_notes.md#1-setup).

Checkpoints, datasets and logs live under `verify2act/output/` and `verify2act/data/`, which are not in the repository.

## Training

The full commands for every variant are in [`commands/`](commands) (`v2a_wm.sh`, `rla_wm.sh`, `dino_wm.sh`,
`diffusion_wm.sh`). The main pieces:

```bash
# world model
accelerate launch verify2act/latent_wm/train_dynamics.py \
  --dataset-type robosuite --dataset-dir <dataset> --cache-dir <dino cache> \
  --output-dir verify2act/output/v2a_wm/<name>/wm \
  --encoder-ckpt verify2act/output/v2a_wm/<name>/encoder/ckpt/delta_encoder_best.pt \
  --token-dim 128 --num-latent-tokens 32 --history-len 3 --causal-masking

# critic
accelerate launch verify2act/critic/train_contrastive.py \
  --dataset-dir <dataset> --dataset-type <robosuite|calvin|dofbot> \
  --output-dir verify2act/output/contrastive/<name>
```

## Simulation evaluation

```bash
# cluttered nut assembly (robosuite)
xvfb-run -a python verify2act/pipeline/inference.py --wm-mode v2a_wm \
  --critic-ckpt <critic.pt> --latent-wm-ckpt <dynamics.pt> --encoder-ckpt <encoder.pt> --wm-decoder-dir <decoder dir> \
  --num-episodes 100 --theta-c 0.5 --theta-p 0.05 --history-len 3 --action-conditioning cross_attn

# CALVIN ABCD -> D
python3 verify2act/pipeline/inference_calvin.py --wm-mode v2a_wm \
  --critic-ckpt <critic.pt> --latent-wm-ckpt <dynamics.pt> --encoder-ckpt <encoder.pt> --wm-decoder-dir <decoder dir> \
  --dataset-path calvin/dataset/task_ABCD_D_filtered --num-sequences 100
```

`--wm-mode` selects the verifier: `v2a_wm` (ours), `rla_wm`, `diffusion_wm` or `vlm_only`.

## Digital twin

```bash
export MUJOCO_GL=egl
python -m verify2act.twin.generate --out <dataset> --num-episodes 200 --workers 8 --seed 0     # smoke test
python -m verify2act.twin.generate --out <dataset> --num-episodes 20000 --workers 32 --seed 0
python -m pytest verify2act/twin/test_twin.py -q
```

The twin is kinematic: an oracle places each block with noise, MuJoCo settles it, and the calibrated arm camera
renders the result, which gives the `(state, subtask, next state)` transitions the world model and critic train on.

## Real robot

The robot's onboard computer runs the episode loop and the pick-and-place skills. The lab PC runs this plan server,
which connects to the robot's rosbridge and answers planning requests:

```bash
export MUJOCO_GL=egl
python -m verify2act.robot.server --jetson-ip <ROBOT_IP> --preset twin --real2sim --wm-mode v2a_wm

# without a robot: one planning call on a saved frame
python -m verify2act.robot.server --preset twin --real2sim --wm-mode v2a_wm \
  --offline-image frame.png --goal "Put the red block to the left of the blue block"
```

The wire protocol, subtask vocabulary, evaluation procedure and known limits are documented in
[`docs/project_notes.md`](docs/project_notes.md).

## Citation

```bibtex
@misc{eze2026verify2act,
  title  = {Verify2Act: Critic-Guided Latent World Models for Verifying
            Language-Conditioned Manipulation Plans},
  author = {Eze, Chrisantus and Crick, Christopher},
  year   = {2026},
  note   = {Under review}
}
```
