import argparse
import os
import torch
from pathlib import Path
from torchvision.utils import save_image
import torch.nn.functional as F

from dynamics import LatentDynamicsModel
from decoder import FeatureDecoder
from train_dynamics import LatentDynamicsDataset, FeatureExtractor

def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset
    print(f"Loading dataset from {args.dataset_dir}...")
    dataset = LatentDynamicsDataset(
        dataset_dir=args.dataset_dir,
        transitions_file=args.transitions_file,
        history_len=args.history_len,
        image_size=args.image_size
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 2. Load Feature Extractor
    extractor = FeatureExtractor(device)

    # 3. Load World Model
    print(f"Loading Latent Dynamics Model from {args.wm_ckpt}...")
    wm = LatentDynamicsModel(
        dino_channels=768,
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

    # 4. Load Visualizer (Decoder)
    print(f"Loading Visualizer from {args.decoder_ckpt}...")
    decoder = FeatureDecoder(dino_channels=768, model_channels=256).to(device)
    
    if args.decoder_ckpt and os.path.exists(args.decoder_ckpt):
        try:
            ckpt = torch.load(args.decoder_ckpt, map_location=device)
            # rla-wm generic trainer saves model state under 'model' or 'state_dict'
            if "model" in ckpt:
                decoder.decoder.load_state_dict(ckpt["model"])
            elif "state_dict" in ckpt:
                decoder.decoder.load_state_dict(ckpt["state_dict"])
            else:
                # Direct state dict
                decoder.decoder.load_state_dict(ckpt)
            print("Visualizer weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load decoder weights: {e}. Output images may look like noise.")
    else:
        print(f"Warning: Decoder checkpoint not found at {args.decoder_ckpt}. Output images will look like noise.")
    decoder.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # 5. Inference & Visualization Loop
    num_samples = min(args.num_samples, len(dataset))
    
    with torch.no_grad():
        for i in range(num_samples):
            print(f"Processing sample {i+1}/{num_samples}...")
            history_imgs, target_img, action_text, history_mask = dataset[i]
            
            # Add batch dimension and move to device
            history_imgs = history_imgs.unsqueeze(0).to(device) # (1, H, 3, 224, 224)
            target_img = target_img.unsqueeze(0).to(device)     # (1, 3, 224, 224)
            history_mask = history_mask.unsqueeze(0).to(device) if args.causal_masking else None
            
            print(f"  Action: '{action_text}'")

            # Extract features
            F_history = extractor.extract_dino(history_imgs) # (1, H, 256, 768)
            F_target = extractor.extract_dino(target_img)    # (1, 256, 768)
            A_clip = extractor.extract_clip([action_text])   # (1, seq_len, 512)

            # Predict next step features using ODE solver (Euler, 5 steps)
            F_pred = wm.step(F_history, A_clip, num_steps=5, history_mask=history_mask) # (1, 256, 768)

            # Decode features back to images
            pred_rgb = decoder.decode(F_pred)             # (1, 3, H, W)
            target_rgb = decoder.decode(F_target)         # (1, 3, H, W)

            # Typical decoders output [-1, 1], map to [0, 1]
            pred_rgb = (pred_rgb + 1) / 2.0
            target_rgb = (target_rgb + 1) / 2.0
            pred_rgb = pred_rgb.clamp(0, 1)
            target_rgb = target_rgb.clamp(0, 1)

            # Denormalize ground truth target_img for comparison
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            target_img_denorm = (target_img * std + mean).clamp(0, 1)

            # Also denormalize the last history image (current state)
            current_img = history_imgs[:, -1]
            current_img_denorm = (current_img * std + mean).clamp(0, 1)

            # Resize decoded images to match original (if they don't already)
            if pred_rgb.shape[-2:] != target_img_denorm.shape[-2:]:
                pred_rgb = F.interpolate(pred_rgb, size=target_img_denorm.shape[-2:], mode='bilinear')
                target_rgb = F.interpolate(target_rgb, size=target_img_denorm.shape[-2:], mode='bilinear')

            # Create a comparison grid: [Current State] [Decoded GT F_target] [Decoded Pred F_pred] [GT Target Image]
            comparison = torch.cat([current_img_denorm, target_rgb, pred_rgb, target_img_denorm], dim=-1)
            
            save_path = os.path.join(args.output_dir, f"sample_{i:03d}.png")
            save_image(comparison, save_path)
            print(f"  Saved visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize World Model Predictions")
    parser.add_argument("--dataset-dir", type=str, default="robosuite/data_capture_wm/dataset/nut_assembly_merged")
    parser.add_argument("--transitions-file", type=str, default="transitions.jsonl")
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--wm-ckpt", type=str, default="verify2act/output/latent_wm/latent_dynamics_best.pt", help="Path to trained World Model checkpoint")
    parser.add_argument("--decoder-ckpt", type=str, default="", help="Path to trained rla-wm decoder checkpoint (e.g. runs/xxx.pt)")
    parser.add_argument("--output-dir", type=str, default="verify2act/output/latent_wm/visualizations")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument(
        "--causal-masking", action="store_true", default=False,
        help="Use Transformer-native causal attention masking + learnable [START] tokens for "
             "early-episode history padding. Omit to use the legacy repeat-first-frame baseline."
    )
    
    # Ensure huggingface cache doesn't blow up
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    
    args = parser.parse_args()
    visualize(args)
