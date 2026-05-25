import argparse
import os
import torch
from pathlib import Path
from torchvision.utils import save_image
import torch.nn.functional as F
import sys

latent_wm_path = Path(__file__).resolve().parent.parent / "latent_wm"
if str(latent_wm_path) not in sys.path:
    sys.path.append(str(latent_wm_path))

from train_dynamics import LatentDynamicsDataset, FeatureExtractor
from decoder import FeatureDecoder
from delta_encoder import DeltaDecoder

from dynamics import BaselineRLAWM

def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset
    print(f"Loading dataset from {args.dataset_dir}...")
    if args.dataset_type == "calvin":
        import sys
        from pathlib import Path
        root_dir = Path(__file__).resolve().parent.parent.parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
        from verify2act.data_loader_calvin import build_calvin_datasets
        train_dataset, val_dataset = build_calvin_datasets(
            dataset_dir=args.dataset_dir,
            val_frac=0.1,
            image_size=args.image_size,
            history_len=args.history_len,
            seed=42,
            use_cache=False,
            cached_dino_dir=None
        )
        dataset = val_dataset  # Use validation set for visualization
    else:
        dataset = LatentDynamicsDataset(
            dataset_dir=args.dataset_dir,
            transitions_file=args.transitions_file,
            history_len=args.history_len,
            image_size=args.image_size,
            use_cache=False,
        )
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 2. Load Feature Extractor
    extractor = FeatureExtractor(device)

    # 3. Load World Model
    print(f"Loading Baseline RLA-WM from {args.wm_ckpt}...")
    wm = BaselineRLAWM(
        dino_channels=args.dino_channels,
        clip_channels=512,
        history_len=args.history_len,
        num_patches=256,
    ).to(device)
    
    if os.path.exists(args.wm_ckpt):
        wm.load_state_dict(torch.load(args.wm_ckpt, map_location=device))
        print("WM weights loaded successfully.")
    else:
        print(f"Warning: WM checkpoint not found at {args.wm_ckpt}. Using random weights.")
    wm.eval()

    # 4a. Load Stage 1 DeltaDecoder (maps compact latent tokens → ΔF)
    print(f"Loading DeltaDecoder from {args.encoder_ckpt}...")
    delta_decoder = DeltaDecoder(
        token_dim=64,
        dino_channels=args.dino_channels,
        num_patches=256,
    ).to(device)
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        try:
            ckpt = torch.load(args.encoder_ckpt, map_location=device)
            dec_sd = ckpt.get("decoder", ckpt)  # support full or decoder-only ckpt
            delta_decoder.load_state_dict(dec_sd)
            print("DeltaDecoder weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load DeltaDecoder weights: {e}. ΔF reconstruction will be random.")
    else:
        print("Warning: No encoder_ckpt provided — DeltaDecoder will use random weights.")
    delta_decoder.eval()

    # 4b. Load Visualizer (Decoder)
    print(f"Loading Visualizer from {args.decoder_ckpt}...")
    visualizer = FeatureDecoder(dino_channels=args.dino_channels, model_channels=256).to(device)
    
    if args.decoder_ckpt and os.path.exists(args.decoder_ckpt):
        try:
            ckpt = torch.load(args.decoder_ckpt, map_location=device)
            if "model" in ckpt:
                visualizer.decoder.load_state_dict(ckpt["model"])
            elif "state_dict" in ckpt:
                visualizer.decoder.load_state_dict(ckpt["state_dict"])
            else:
                visualizer.decoder.load_state_dict(ckpt)
            print("Visualizer weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load decoder weights: {e}. Output images may look like noise.")
    else:
        print(f"Warning: Decoder checkpoint not found at {args.decoder_ckpt}. Output images will look like noise.")
    visualizer.eval()

    task_name = "calvin" if args.dataset_type == "calvin" else "nut_assembly"
    args.output_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # 5. Inference & Visualization Loop
    num_samples = min(args.num_samples, len(dataset))
    
    with torch.no_grad():
        for i in range(num_samples):
            print(f"Processing sample {i+1}/{num_samples}...")
            sample = dataset[i]
            if args.dataset_type == "calvin":
                history_imgs = sample["history_imgs"]
                target_img = sample["image_t1"]
                action_text = sample["action_text"]
            else:
                history_imgs, target_img, action_text, _ = sample
            
            history_imgs = history_imgs.unsqueeze(0).to(device)
            target_img = target_img.unsqueeze(0).to(device)     
            
            print(f"  Action: '{action_text}'")

            F_history = extractor.extract_dino(history_imgs)
            F_target = extractor.extract_dino(target_img)    
            A_clip = extractor.extract_clip([action_text])   

            # Predict next step features using ODE solver (Euler, 5 steps)
            pred_latent = wm.step(F_history, A_clip, num_steps=5)

            # Decode compact tokens to DINO difference (feature residual)
            delta_F_pred = delta_decoder(pred_latent)          # (1, 256, dino_channels)

            # Reconstruct predicted next state by adding residual to current visual state
            F_t = F_history[:, -1, :, :]                      # (1, 256, dino_channels)
            F_pred = F_t + delta_F_pred                        # (1, 256, dino_channels)

            # Decode features back to images
            pred_rgb = visualizer.decode(F_pred)             
            target_rgb = visualizer.decode(F_target)         

            pred_rgb = (pred_rgb + 1) / 2.0
            target_rgb = (target_rgb + 1) / 2.0
            pred_rgb = pred_rgb.clamp(0, 1)
            target_rgb = target_rgb.clamp(0, 1)

            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            target_img_denorm = (target_img * std + mean).clamp(0, 1)

            current_img = history_imgs[:, -1]
            current_img_denorm = (current_img * std + mean).clamp(0, 1)

            if pred_rgb.shape[-2:] != target_img_denorm.shape[-2:]:
                pred_rgb = F.interpolate(pred_rgb, size=target_img_denorm.shape[-2:], mode='bilinear')
                target_rgb = F.interpolate(target_rgb, size=target_img_denorm.shape[-2:], mode='bilinear')

            comparison = torch.cat([current_img_denorm, target_rgb, pred_rgb, target_img_denorm], dim=-1)
            
            save_path = os.path.join(args.output_dir, f"sample_{i:03d}.png")
            save_image(comparison, save_path)
            print(f"  Saved visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Baseline World Model Predictions")
    parser.add_argument("--dataset-dir", type=str, default="robosuite/data_capture_wm/dataset/nut_assembly_merged")
    parser.add_argument("--dataset-type", type=str, default="robosuite", choices=["robosuite", "calvin"])
    parser.add_argument("--transitions-file", type=str, default="transitions.jsonl")
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--wm-ckpt", type=str, default="verify2act/output/rla_wm/latent_dynamics_best.pt", help="Path to trained World Model checkpoint")
    parser.add_argument("--dino-channels", type=int, default=1024, help="DINO feature dimension (768 for ViT-B, 1024 for ViT-L)")
    parser.add_argument("--encoder-ckpt", type=str, default="", help="Path to encoder+decoder checkpoint from train_encoder.py (for DeltaDecoder)")
    parser.add_argument("--decoder-ckpt", type=str, default="", help="Path to trained rla-wm decoder checkpoint (e.g. runs/xxx.pt)")
    parser.add_argument("--output-dir", type=str, default="verify2act/output/rla_wm/visualizations")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    
    args = parser.parse_args()
    visualize(args)
