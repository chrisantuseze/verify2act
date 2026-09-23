import argparse
import os
import sys
import json
from pathlib import Path
from typing import Optional, List
import torch
import numpy as np
from PIL import Image

# Project root setup
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from verify2act.latent_wm.decoder import FeatureDecoder
from verify2act.latent_wm.train_dynamics import FeatureExtractor
from verify2act.pipeline.world_model import LatentWorldModel, RLAWorldModel, DiffusionWorldModel, DINOWorldModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_img_as_pil(path: Path) -> Image.Image:
    if str(path).endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        img_np = data["rgb_static"]
        return Image.fromarray(img_np).convert("RGB")
    return Image.open(path).convert("RGB")


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


def compose_figure(
    output_dir: str,
    episode_id: str,
    method_dirs: dict,
    horizon: int,
    save_path: str,
    action_texts: Optional[list[str]] = None,
) -> None:
    """
    Assembles saved per-frame PNGs into a publication-ready panel figure.
    Rows = methods (ground_truth, v2a_wm, rla_wm, diffusion)
    Columns = timesteps 0..horizon-1
    Only includes methods that have saved frames for this episode.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    row_labels = {
        "ground_truth": "Ground Truth",
        "v2a_wm":       "V2A-WM (Ours)",
        "rla_wm":       "RLA-WM",
        "diffusion":    "Diffusion-WM",
        "dino_wm":      "DINO-WM",
    }
    # Only include rows that exist and have at least one frame
    active_rows = []
    for method_key, label in row_labels.items():
        if method_key not in method_dirs:
            continue
        frame_path = method_dirs[method_key] / frame_name(episode_id, 0)
        if frame_path.exists():
            active_rows.append((method_key, label))

    if not active_rows:
        print("[compose_figure] No frames found — skipping figure composition.")
        return

    n_rows = len(active_rows)
    n_cols = horizon
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols, 2.4 * n_rows),
        gridspec_kw={"hspace": 0.06, "wspace": 0.04},
    )
    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (method_key, label) in enumerate(active_rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            fp = method_dirs[method_key] / frame_name(episode_id, col_idx)
            if fp.exists():
                img = mpimg.imread(str(fp))
                ax.imshow(img)
            else:
                ax.set_facecolor("#222")
                ax.text(0.5, 0.5, "N/A", color="white",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.axis("off")
            # Column headers (step index + action text) on the top row
            if row_idx == 0:
                header = f"$t={col_idx}$"
                if action_texts and col_idx < len(action_texts):
                    import textwrap
                    act_str = action_texts[col_idx]
                    wrap_width = max(16, min(24, int(90 / n_cols)))
                    wrapped_act = textwrap.fill(act_str, width=wrap_width)
                    header += f"\n\"{wrapped_act}\""
                ax.set_title(header, fontsize=7.5, pad=5, multialignment="center")

    # Row labels: position in left margin at the exact vertical center
    # of each row's subplot bounding box.
    for row_idx, (method_key, label) in enumerate(active_rows):
        pos = axes[row_idx, 0].get_position()
        y_centre = pos.y0 + pos.height / 2.0
        fig.text(
            0.01, y_centre, label,
            ha="left", va="center",
            fontsize=10,
            fontweight="bold" if method_key == "v2a_wm" else "normal",
            rotation=90,
        )

    # Tighten left margin so labels don't clip
    fig.subplots_adjust(left=0.08)

    fig.suptitle(
        # f"Latent Imagination Comparison — episode {episode_id}",
        f"Latent Imagination Comparison",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[compose_figure] Saved panel figure -> {save_path}")
    plt.close(fig)


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
        ckpt = torch.load(dec_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

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
                token_dim=128,
                num_latent_tokens=32,
            )
            active_methods.append("v2a_wm")
        except Exception as e:
            print(f"   [ERROR] v2a-wm: {e}")

    # dino-wm (Baseline) — no causal history, like RLA-WM
    dino_wm = None
    if args.dino_ckpt:
        print(f" -> Loading dino-wm (history_len=1, no causal masking) ...")
        try:
            dino_wm = DINOWorldModel(
                device=args.device,
                dynamics_weights_path=args.dino_ckpt,
                history_len=1,  # DINO-WM uses single-frame history, same as RLA-WM
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
                token_dim=128,
                num_latent_tokens=32,
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
    lang_ann_path = dataset_path / "lang_annotations" / "auto_lang_ann.npy"
    if not lang_ann_path.exists():
        lang_ann_path = dataset_path / "auto_lang_ann.npy"

    episodes: dict[str, list] = {}
    if transitions_file.exists():
        print(f"\nReading transitions from {transitions_file} ...")
        with open(transitions_file) as f:
            rows = [json.loads(line) for line in f]
        for r in rows:
            ep_id = r["episode_id"]
            episodes.setdefault(ep_id, []).append(r)
        for ep_id in episodes:
            episodes[ep_id].sort(key=lambda x: int(x["timestep"]))
    elif lang_ann_path.exists():
        print(f"\nParsing CALVIN transitions from {lang_ann_path} ...")
        data = np.load(lang_ann_path, allow_pickle=True).item()
        indices = data["info"]["indx"]
        texts = data["language"]["ann"]
        sorted_pairs = sorted(zip(indices, texts), key=lambda x: x[0][0])
        
        ep_id = 0
        current_ep = []
        for (s, e), text in sorted_pairs:
            row = {
                "episode_id": f"ep_{ep_id:05d}",
                "timestep": len(current_ep),
                "image_t": f"episode_{s:07d}.npz",
                "image_t1": f"episode_{e:07d}.npz",
                "action_text": text,
            }
            if not current_ep:
                current_ep.append(row)
            else:
                prev_e = int(current_ep[-1]["image_t1"].replace("episode_", "").replace(".npz", ""))
                if abs(s - prev_e) <= 5:
                    current_ep.append(row)
                else:
                    episodes[f"ep_{ep_id:05d}"] = current_ep
                    ep_id += 1
                    current_ep = [row]
        if current_ep:
            episodes[f"ep_{ep_id:05d}"] = current_ep
    else:
        raise FileNotFoundError(f"Neither transitions.jsonl nor auto_lang_ann.npy found at {dataset_path}")

    if args.episode_id:
        target_ep = args.episode_id if args.episode_id.startswith("ep_") else f"ep_{args.episode_id}"
        if target_ep not in episodes:
            raise ValueError(f"Requested episode '{args.episode_id}' not found in {transitions_file}")
        print(f"Filtering to run exclusively on requested episode: '{target_ep}'")
        ep_keys = [target_ep]
    else:
        # Filter by minimum step count if requested — useful for picking episodes
        # with enough timesteps to fill a multi-column visualization horizon.
        if args.min_steps > 0:
            filtered = [k for k, rows in episodes.items() if len(rows) >= args.min_steps]
            print(f"Found {len(episodes)} episodes total; {len(filtered)} have >= {args.min_steps} steps.")
            ep_keys = filtered[:args.num_samples]
        else:
            print(f"Found {len(episodes)} episodes. Processing first {args.num_samples}.")
            ep_keys = list(episodes.keys())[:args.num_samples]
        if not ep_keys:
            raise ValueError(f"No episodes found with >= {args.min_steps} steps. Lower --min-steps.")

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

        start_img_np = np.array(load_img_as_pil(start_img_path).resize((224, 224)))

        # Save the ground-truth initial frame (step -1 / context)
        gt_init = Image.fromarray(start_img_np)
        gt_init_path = method_dirs["ground_truth"] / f"ep_{ep_id}_step_init.png"
        gt_init.save(gt_init_path)
        print(f"  Saved initial frame -> {gt_init_path}")

        # Initialize model histories
        if v2a_wm:   v2a_wm.initialize_history(start_img_np)
        if dino_wm:  dino_wm.initialize_history(start_img_np)
        if rla_wm:   rla_wm.initialize_history(start_img_np)

        current_img_diffusion = np.array(load_img_as_pil(start_img_path).resize((512, 512)))

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
            gt_t1_img = load_img_as_pil(gt_t1_path).resize((224, 224))
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

            # ---- dino-wm (last, no causal history) ----
            if dino_wm:
                try:
                    F_next, _ = dino_wm.imagine(None, action_text)
                    img = decode_dino_features(F_next, decoder, device).resize((224, 224))
                    img.save(method_dirs["dino_wm"] / frame_name(ep_id, step_idx))
                except Exception as e:
                    print(f"    [ERROR] dino-wm step {step_idx}: {e}")

        print(f"  Episode {ep_id} done.")

        # --- Compose figure for this episode if requested ---
        if args.compose_figure:
            compose_ep = args.compose_episode or ep_id
            if compose_ep and not compose_ep.startswith("ep_"):
                compose_ep = f"ep_{compose_ep}"
            if ep_id == compose_ep:
                ep_name_clean = ep_id if ep_id.startswith("ep_") else f"ep_{ep_id}"
                out_path = str(Path(args.output_dir) / f"imagination_panel_{ep_name_clean}.png")
                action_texts = [r["action_text"] for r in ep_rows[:max_steps]]
                compose_figure(
                    output_dir=args.output_dir,
                    episode_id=ep_id,
                    method_dirs=method_dirs,
                    horizon=min(args.horizon, len(episodes[ep_id])),
                    save_path=out_path,
                    action_texts=action_texts,
                )

    print(f"\nAll done. Results saved under: {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save per-method imagination rollouts in separate directories")

    # --- Checkpoints ---
    parser.add_argument("--v2a-ckpt",         type=str, default="verify2act/output/v2a_wm/nut_assembly/wm_causal/ckpt/latent_dynamics_best_weights.pt")
    parser.add_argument("--v2a-encoder-ckpt", type=str, default="verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/delta_encoder_best.pt")
    parser.add_argument("--dino-ckpt",        type=str, default=None, help="Path to DINO-WM baseline checkpoint (BaselineDINOWM weights). Set to skip.")
    parser.add_argument("--rla-ckpt",         type=str, default="verify2act/output/rla_wm/nut_assembly/wm/ckpt/latent_dynamics_best.pt")
    parser.add_argument("--decoder-dir",      type=str, default="verify2act/output/v2a_wm/nut_assembly/decoder")

    # --- Diffusion ---
    parser.add_argument("--diffusion-adapter",  type=str, default="verify2act/output/diffusion_wm/nut_assembly/wm/best/unet_lora")
    parser.add_argument("--diffusion-decoder",  type=str, default="verify2act/output/diffusion_wm/nut_assembly/decoder/checkpoint-500")
    parser.add_argument("--no-diffusion", action="store_true", default=False,
                        help="Skip diffusion baseline (avoids downloading large InstructPix2Pix weights)")

    # --- Data & output ---
    parser.add_argument("--dataset-type", type=str, default="robosuite", choices=["robosuite", "calvin"],
                        help="Dataset format: robosuite or calvin")
    parser.add_argument("--dataset-dir",  type=str, default="robosuite/data_capture/dataset/nut_assembly_merged")
    parser.add_argument("--output-dir",   type=str, default="verify2act/output/comparison_visuals")
    parser.add_argument("--num-samples",  type=int, default=3,  help="Number of episodes to visualise")
    parser.add_argument("--horizon",      type=int, default=10, help="Steps per episode to roll out")
    parser.add_argument("--episode-id",   type=str, default=None, help="Process exactly this episode ID")
    parser.add_argument("--min-steps",    type=int, default=0,
                        help="Only consider episodes with at least this many timesteps (useful for rich visualizations)")

    # --- Figure composition ---
    parser.add_argument("--compose-figure",  action="store_true", default=False,
                        help="After rollout, compose a publication-ready panel PNG (rows=methods, cols=timesteps)")
    parser.add_argument("--compose-episode", type=str, default=None,
                        help="Episode ID to compose the figure for (defaults to first processed episode)")

    # --- Misc ---
    parser.add_argument("--history-len",      type=int,  default=3)
    parser.add_argument("--device",           type=str,  default="cuda")
    parser.add_argument("--causal-masking",   action="store_true", default=True)

    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    args = parser.parse_args()

    # Auto-switch defaults if dataset_type is calvin or 'calvin' is in dataset_dir path
    if args.dataset_type == "calvin" or "calvin" in args.dataset_dir:
        if args.v2a_ckpt == "verify2act/output/v2a_wm/nut_assembly/wm_causal/ckpt/latent_dynamics_best_weights.pt":
            args.v2a_ckpt = "verify2act/output/v2a_wm/calvin/wm_wider/ckpt/latent_dynamics_best_weights.pt"
        if args.v2a_encoder_ckpt == "verify2act/output/v2a_wm/nut_assembly/encoder/ckpt/delta_encoder_best.pt":
            args.v2a_encoder_ckpt = "verify2act/output/v2a_wm/calvin/encoder_wider/ckpt/delta_encoder_best.pt"
        if args.dino_ckpt is None:
            args.dino_ckpt = "verify2act/output/dino_wm/calvin/wm/ckpt/latent_dynamics_best.pt" if Path("verify2act/output/dino_wm/calvin/wm/ckpt/latent_dynamics_best.pt").exists() else None
        if args.rla_ckpt == "verify2act/output/rla_wm/nut_assembly/wm/ckpt/latent_dynamics_best.pt":
            args.rla_ckpt = "verify2act/output/rla_wm/calvin/wm/ckpt/latent_dynamics_best.pt"
        if args.decoder_dir == "verify2act/output/v2a_wm/nut_assembly/decoder":
            args.decoder_dir = "verify2act/output/v2a_wm/calvin/decoder"
        if args.diffusion_adapter == "verify2act/output/diffusion_wm/nut_assembly/wm/best/unet_lora":
            args.diffusion_adapter = "verify2act/output/diffusion_wm/calvin/wm/best/unet_lora"
        if args.diffusion_decoder == "verify2act/output/diffusion_wm/nut_assembly/decoder/checkpoint-500":
            args.diffusion_decoder = "verify2act/output/diffusion_wm/calvin/decoder/checkpoint-500"
    else:
        # Robosuite: auto-detect dino_wm checkpoint
        if args.dino_ckpt is None:
            _nut_dino = "verify2act/output/dino_wm/nut_assembly/wm/ckpt/latent_dynamics_best.pt"
            args.dino_ckpt = _nut_dino if Path(_nut_dino).exists() else None

    if not args.causal_masking:
        args.history_len = 1

    run_comparison(args)
