import argparse
import os
import sys
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Project root setup
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from verify2act.latent_wm.decoder import FeatureDecoder
from verify2act.latent_wm.train_dynamics import FeatureExtractor
from verify2act.pipeline.world_model import LatentWorldModel, RLAWorldModel, DINOWorldModel, DiffusionWorldModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_history_len_from_config(ckpt_path: str, default_val: int) -> int:
    """Read history_len from config.json next to the checkpoint or in its parent/grandparent directory."""
    if not ckpt_path:
        return default_val
    p = Path(ckpt_path)
    # config.json might be in the grandparent directory of the checkpoint
    config_path = p.parent.parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            val = cfg.get("history_len", default_val)
            print(f"  [CONFIG] Resolved history_len={val} from {config_path}")
            return val
        except Exception as e:
            print(f"  [CONFIG] Error reading config at {config_path}: {e}")
    return default_val


def decode_dino_features(dino_features: torch.Tensor, decoder: FeatureDecoder, device: torch.device) -> Image.Image:
    """Decodes DINO features back to a PIL image using the FeatureDecoder."""
    if dino_features.ndim == 2:
        dino_features = dino_features.unsqueeze(0)

    with torch.no_grad():
        rec_img = decoder.decode(dino_features.to(device))  # (B, 3, H, W) in [-1, 1]
        rec_img = (rec_img + 1.0) / 2.0
        rec_img = torch.clamp(rec_img, 0.0, 1.0)
        rec_img = rec_img.squeeze(0).cpu().numpy()          # (3, H, W)
        rec_img = (rec_img * 255.0).astype(np.uint8)
        rec_img = np.transpose(rec_img, (1, 2, 0))          # (H, W, 3)
        return Image.fromarray(rec_img)


def make_method_dirs(base: str, methods: list[str]) -> dict[str, Path]:
    """Create one subdirectory per method and return the mapping."""
    dirs = {}
    for m in methods:
        p = Path(base) / m
        p.mkdir(parents=True, exist_ok=True)
        dirs[m] = p
    return dirs


def frame_name(ep_id: str, step_idx: int) -> str:
    return f"ep_{ep_id}_step{step_idx:02d}.png"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Decide which methods we are running so we can pre-create directories
    active_methods = ["ground_truth"]

    # 1. Load FeatureDecoder (shared by all latent models)
    print(f"Loading FeatureDecoder from {args.decoder_dir} ...")
    decoder = FeatureDecoder(dino_channels=1024).to(device)
    decoder.eval()
    dec_path = Path(args.decoder_dir) / "latent_decoder_best.pt"
    if dec_path.exists():
        state_dict = torch.load(dec_path, map_location=device)
        # Handle wrapped vs bare state-dict
        if "decoder.input_proj.0.weight" not in state_dict and "input_proj.0.weight" in state_dict:
            state_dict = {f"decoder.{k}": v for k, v in state_dict.items()}
        decoder.load_state_dict(state_dict)
        print("FeatureDecoder loaded successfully.")
    else:
        print(f"[WARN] FeatureDecoder checkpoint not found at {dec_path}. Reconstructions will be random.")

    # 2. Instantiate world models
    print("Loading World Models ...")

    # Resolve history lengths dynamically to avoid mismatch issues
    v2a_hist_len = get_history_len_from_config(args.v2a_ckpt, args.history_len)
    dino_hist_len = get_history_len_from_config(args.dino_ckpt, args.history_len)
    rla_hist_len = get_history_len_from_config(args.rla_ckpt, args.history_len)

    # v2a-wm (Ours)
    v2a_wm = None
    if args.v2a_ckpt:
        print(f" -> Loading v2a-wm (history_len={v2a_hist_len}) ...")
        try:
            v2a_wm = LatentWorldModel(
                device=args.device,
                dynamics_weights_path=args.v2a_ckpt,
                encoder_ckpt=args.v2a_encoder_ckpt,
                history_len=v2a_hist_len,
            )
            active_methods.append("v2a_wm")
        except Exception as e:
            print(f"   [ERROR] v2a-wm: {e}")

    # dino-wm (Baseline)
    dino_wm = None
    if args.dino_ckpt:
        print(f" -> Loading dino-wm (history_len={dino_hist_len}) ...")
        try:
            dino_wm = DINOWorldModel(
                device=args.device,
                dynamics_weights_path=args.dino_ckpt,
                history_len=dino_hist_len,
            )
            active_methods.append("dino_wm")
        except Exception as e:
            print(f"   [ERROR] dino-wm: {e}")

    # rla-wm (Baseline)
    rla_wm = None
    if args.rla_ckpt:
        print(f" -> Loading rla-wm (history_len={rla_hist_len}) ...")
        try:
            rla_wm = RLAWorldModel(
                device=args.device,
                dynamics_weights_path=args.rla_ckpt,
                encoder_ckpt=args.v2a_encoder_ckpt,  # Stage-1 DeltaEncoder is shared
                history_len=rla_hist_len,
            )
            active_methods.append("rla_wm")
        except Exception as e:
            print(f"   [ERROR] rla-wm: {e}")

    # diffusion (Baseline)
    diffusion_wm = None
    if not args.no_diffusion:
        print(" -> Loading diffusion ...")
        try:
            diffusion_wm = DiffusionWorldModel(
                pretrained_model="timbrooks/instruct-pix2pix",
                adapter_dir=args.diffusion_adapter,
                decoder_dir=args.diffusion_decoder,
                device=args.device,
                torch_dtype=torch.float16,
            )
            active_methods.append("diffusion")
        except Exception as e:
            print(f"   [ERROR] diffusion: {e}")
    else:
        print(" -> Skipping diffusion (--no-diffusion)")

    # 3. Pre-create output directories
    method_dirs = make_method_dirs(args.output_dir, active_methods)
    print(f"\nOutput directories:")
    for m, p in method_dirs.items():
        print(f"  {m}: {p}")

    # 4. Load dataset transitions
    dataset_path = Path(args.dataset_dir)
    transitions_file = dataset_path / "transitions.jsonl"
    if not transitions_file.exists():
        raise FileNotFoundError(f"transitions.jsonl not found at {transitions_file}")

    print(f"\nReading transitions from {transitions_file} ...")
    with open(transitions_file) as f:
        rows = [json.loads(line) for line in f]

    # Group and sort by episode
    episodes: dict[str, list] = {}
    for r in rows:
        ep_id = r["episode_id"]
        episodes.setdefault(ep_id, []).append(r)
    for ep_id in episodes:
        episodes[ep_id].sort(key=lambda x: int(x["timestep"]))

    if args.episode_id:
        if args.episode_id not in episodes:
            raise ValueError(f"Requested episode '{args.episode_id}' not found in {transitions_file}")
        print(f"Filtering to run exclusively on requested episode: '{args.episode_id}'")
        ep_keys = [args.episode_id]
    else:
        print(f"Found {len(episodes)} episodes. Processing first {args.num_samples}.")
        ep_keys = list(episodes.keys())[:args.num_samples]

    # Feature extractor (only used for DINO feature caching if needed)
    extractor = FeatureExtractor(device, dino_channels=1024)

    # 5. Per-episode rollout
    for sample_idx, ep_id in enumerate(ep_keys):
        print(f"\n=== Episode {ep_id}  ({sample_idx + 1}/{len(ep_keys)}) ===")
        ep_rows = episodes[ep_id]

        # --- Initial / context frame ---
        start_row = ep_rows[0]
        start_img_path = dataset_path / start_row["image_t"]
        if not start_img_path.exists():
            print(f"  [SKIP] Start image not found: {start_img_path}")
            continue

        start_img_np = np.array(Image.open(start_img_path).convert("RGB").resize((224, 224)))

        # Save the ground-truth initial frame (step -1 / context)
        gt_init = Image.fromarray(start_img_np)
        gt_init_path = method_dirs["ground_truth"] / f"ep_{ep_id}_step_init.png"
        gt_init.save(gt_init_path)
        print(f"  Saved initial frame -> {gt_init_path}")

        # Initialize model histories
        if v2a_wm:   v2a_wm.initialize_history(start_img_np)
        if dino_wm:  dino_wm.initialize_history(start_img_np)
        if rla_wm:   rla_wm.initialize_history(start_img_np)

        current_img_diffusion = np.array(Image.open(start_img_path).convert("RGB").resize((512, 512)))

        # --- Autoregressive rollout ---
        max_steps = min(args.horizon, len(ep_rows))
        for step_idx, row in enumerate(ep_rows[:max_steps]):
            action_text = row["action_text"]
            gt_t1_path  = dataset_path / row["image_t1"]
            if not gt_t1_path.exists():
                print(f"  [BREAK] GT image not found at step {step_idx} (path: {gt_t1_path})")
                break

            print(f"  Step {step_idx:02d}  action='{action_text}'")

            # ---- Ground truth ----
            gt_t1_img = Image.open(gt_t1_path).convert("RGB").resize((224, 224))
            gt_save = method_dirs["ground_truth"] / frame_name(ep_id, step_idx)
            gt_t1_img.save(gt_save)

            # ---- v2a-wm ----
            if v2a_wm:
                try:
                    F_next, _ = v2a_wm.imagine(None, action_text)
                    img = decode_dino_features(F_next, decoder, device).resize((224, 224))
                    img.save(method_dirs["v2a_wm"] / frame_name(ep_id, step_idx))
                except Exception as e:
                    print(f"    [ERROR] v2a-wm step {step_idx}: {e}")

            # ---- dino-wm ----
            if dino_wm:
                try:
                    F_next, _ = dino_wm.imagine(None, action_text)
                    img = decode_dino_features(F_next, decoder, device).resize((224, 224))
                    img.save(method_dirs["dino_wm"] / frame_name(ep_id, step_idx))
                except Exception as e:
                    print(f"    [ERROR] dino-wm step {step_idx}: {e}")

            # ---- rla-wm ----
            if rla_wm:
                try:
                    F_next, _ = rla_wm.imagine(None, action_text)
                    img = decode_dino_features(F_next, decoder, device).resize((224, 224))
                    img.save(method_dirs["rla_wm"] / frame_name(ep_id, step_idx))
                except Exception as e:
                    print(f"    [ERROR] rla-wm step {step_idx}: {e}")

            # ---- diffusion ----
            if diffusion_wm:
                try:
                    pred_np = diffusion_wm.imagine(current_img_diffusion, action_text)
                    img = Image.fromarray(pred_np).resize((224, 224))
                    img.save(method_dirs["diffusion"] / frame_name(ep_id, step_idx))
                    current_img_diffusion = pred_np  # advance autoregressively
                except Exception as e:
                    print(f"    [ERROR] diffusion step {step_idx}: {e}")

        print(f"  Episode {ep_id} done.")

    print(f"\nAll done. Results saved under: {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save per-method imagination rollouts in separate directories")

    # --- Checkpoints ---
    parser.add_argument("--v2a-ckpt",         type=str, default="verify2act/output/v2a_wm/nut_assembly/wm_history_1_sparsity_01/ckpt/latent_dynamics_best.pt")
    parser.add_argument("--v2a-encoder-ckpt", type=str, default="verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/delta_encoder_best.pt")
    parser.add_argument("--dino-ckpt",        type=str, default="verify2act/output/dino_wm/nut_assembly/wm/ckpt/latent_dynamics_best.pt")
    parser.add_argument("--rla-ckpt",         type=str, default="verify2act/output/rla_wm/nut_assembly/wm/ckpt/latent_dynamics_best.pt")
    parser.add_argument("--decoder-dir",      type=str, default="verify2act/output/v2a_wm/nut_assembly/decoder")

    # --- Diffusion ---
    parser.add_argument("--diffusion-adapter",  type=str, default="verify2act/output/diffusion_wm/nut_assembly/wm/best/unet_lora")
    parser.add_argument("--diffusion-decoder",  type=str, default="verify2act/output/diffusion_wm/nut_assembly/decoder/checkpoint-5000")
    parser.add_argument("--no-diffusion", action="store_true", default=False,
                        help="Skip diffusion baseline (avoids downloading large InstructPix2Pix weights)")

    # --- Data & output ---
    parser.add_argument("--dataset-dir",  type=str, default="robosuite/data_capture/dataset/nut_assembly_merged")
    parser.add_argument("--output-dir",   type=str, default="verify2act/output/comparison_visuals")
    parser.add_argument("--num-samples",  type=int, default=3,  help="Number of episodes to visualise")
    parser.add_argument("--horizon",      type=int, default=10, help="Steps per episode to roll out")
    parser.add_argument("--episode-id",   type=str, default=None, help="Process exactly this episode ID")

    # --- Misc ---
    parser.add_argument("--history-len",      type=int,  default=1)
    parser.add_argument("--device",           type=str,  default="cuda")
    parser.add_argument("--causal-masking",   action="store_true", default=False)

    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    args = parser.parse_args()

    if not args.causal_masking:
        args.history_len = 1

    run_comparison(args)
