#!/usr/bin/env python3
"""
Quick visual demo for Verify2Act world-model generation.

Usage patterns:
1) Direct image + prompt:
   python verify2act/world_model/demo_wm.py \
     --image-path path/to/frame.png \
     --prompt "pick round nut. position: (0.1, -0.05, 0.83)." \
     --adapter-dir verify2act/output/wm/best/unet_lora

2) Pull sample from dataset transitions_subskill.jsonl:
   python verify2act/world_model/demo_wm.py \
     --dataset-dir robosuite/data_capture_wm/dataset/nut_assembly_merged \
     --transition-index 0 \
     --adapter-dir verify2act/output/wm/best/unet_lora
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `verify2act` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from peft import PeftModel
from PIL import Image

try:
    from verify2act.utils import load_vae_decoder, load_vae_encoder
except ImportError:
    from utils import load_vae_decoder, load_vae_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Visual demo for Verify2Act world model")

    parser.add_argument("--pretrained-model", type=str, default="timbrooks/instruct-pix2pix")
    parser.add_argument(
        "--vae-model",
        type=str,
        default="",
        help="Optional VAE source; if empty uses --pretrained-model.",
    )
    parser.add_argument(
        "--vae-subfolder",
        type=str,
        default="auto",
        help="VAE subfolder to load (e.g. 'vae', 'vae_ema', 'root'). Use 'auto' to resolve automatically.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load VAE only from local cache/files; do not reach HuggingFace Hub.",
    )
    parser.add_argument("--adapter-dir", type=str, default=None, help="Path to LoRA adapter directory")
    parser.add_argument(
        "--decoder-dir",
        type=str,
        default=None,
        help=(
            "Path to finetuned VAE decoder. Accepts either:\n"
            "  (a) a full diffusers-format VAE directory (e.g. output/decoder/final/vae)\n"
            "  (b) a Phase B checkpoint directory containing decoder_state_dict.pt\n"
            "      (e.g. output/decoder/checkpoint-epoch5)\n"
            "If omitted, the original pretrained decoder is used."
        ),
    )

    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)

    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--transition-index", type=int, default=0)
    parser.add_argument(
        "--transitions-file",
        type=str,
        default="transitions.jsonl",
        help="JSONL filename inside dataset-dir (e.g. 'transitions.jsonl' or 'transitions_subskill.jsonl').",
    )

    parser.add_argument("--output-path", type=str, default="verify2act/output/wm_demo/generated.png")
    parser.add_argument("--meta-path", type=str, default="verify2act/output/wm_demo/run_meta.json")

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--guidance-scale", type=float, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")

    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def load_transition_sample(dataset_dir: Path, index: int, transitions_file: str = "transitions.jsonl"):
    transitions_path = dataset_dir / transitions_file
    if not transitions_path.exists():
        raise FileNotFoundError(f"Missing transitions file: {transitions_path}")

    with open(transitions_path, "r") as handle:
        for i, line in enumerate(handle):
            if i == index:
                row = json.loads(line)
                image_path = dataset_dir / row["image_t"]
                gt_path = dataset_dir / row["image_t1"] if "image_t1" in row else None
                prompt = row["action_text"]
                return image_path, gt_path, prompt, row

    raise IndexError(f"transition-index={index} is out of range for {transitions_path}")


def ensure_inputs(args):
    if args.image_path and args.prompt:
        return Path(args.image_path), None, args.prompt, {"source": "direct_args"}

    if args.dataset_dir:
        image_path, gt_path, prompt, row = load_transition_sample(
            Path(args.dataset_dir), args.transition_index, args.transitions_file
        )
        return image_path, gt_path, prompt, row

    raise ValueError(
        "Provide either (--image-path and --prompt) or (--dataset-dir with optional --transition-index)."
    )


def save_collage(image_in: Image.Image, gt_image: Image.Image | None, image_out: Image.Image, output_path: Path):
    """Save a side-by-side collage of input | ground truth | generated."""
    from PIL import ImageDraw, ImageFont

    w, h = image_in.size
    labels = ["Input", "Ground Truth", "Generated"]
    panels = [image_in, gt_image if gt_image is not None else Image.new("RGB", (w, h), (30, 30, 30)), image_out]

    label_height = 24
    collage = Image.new("RGB", (w * 3, h + label_height), (20, 20, 20))
    draw = ImageDraw.Draw(collage)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for idx, (panel, label) in enumerate(zip(panels, labels)):
        panel_resized = panel.resize((w, h))
        collage.paste(panel_resized, (idx * w, label_height))
        text_x = idx * w + w // 2
        draw.text((text_x, 4), label, fill=(255, 255, 255), font=font, anchor="mt")

    if gt_image is None:
        draw.text((w + w // 2, label_height + h // 2), "N/A", fill=(180, 180, 180), font=font, anchor="mm")

    collage.save(output_path)
    print(f"Saved collage:         {output_path}")


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    torch_dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)

    image_path, gt_path, prompt, sample_meta = ensure_inputs(args)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    print("Loading InstructPix2Pix pipeline...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
    ).to(device)

    vae_model = args.vae_model if args.vae_model else args.pretrained_model
    vae, resolved_subfolder = load_vae_encoder(
        model_name_or_path=vae_model,
        device=device,
        torch_dtype=torch_dtype,
        subfolder=args.vae_subfolder,
        local_files_only=args.local_files_only,
    )
    pipe.vae = vae
    print(f"Using VAE encoder from model={vae_model} (subfolder={resolved_subfolder})")

    # ── Optionally replace the decoder with a finetuned one (Phase B) ─────────
    if args.decoder_dir:
        pipe.vae = load_vae_decoder(
            decoder_dir=args.decoder_dir,
            vae=pipe.vae,
            device=device,
            torch_dtype=torch_dtype,
        )
        print(f"Using finetuned VAE decoder from: {args.decoder_dir}")

    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
        print(f"Loading LoRA adapter from: {adapter_dir}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, str(adapter_dir)).to(device)

    image_in = Image.open(image_path).convert("RGB")
    original_size = image_in.size
    image_resized = image_in.resize((args.resolution, args.resolution))

    generator = torch.Generator(device.type).manual_seed(args.seed)

    print("Running generation...")
    with torch.inference_mode():
        image_out = pipe(
            prompt,
            image=image_resized,
            num_inference_steps=args.num_inference_steps,
            image_guidance_scale=args.image_guidance_scale,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

    image_out = image_out.resize(original_size)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_out.save(output_path)

    gt_image = None
    if gt_path is not None and Path(gt_path).exists():
        gt_image = Image.open(gt_path).convert("RGB").resize(original_size)
    collage_path = output_path.parent / (output_path.stem + "_collage.png")
    save_collage(image_in.resize(original_size), gt_image, image_out, collage_path)

    meta = {
        "pretrained_model": args.pretrained_model,
        "vae_model": vae_model,
        "vae_subfolder": resolved_subfolder,
        "adapter_dir": args.adapter_dir,
        "decoder_dir": args.decoder_dir,
        "input_image": str(image_path),
        "gt_image": str(gt_path) if gt_path is not None else None,
        "prompt": prompt,
        "output_image": str(output_path),
        "collage_image": str(collage_path),
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "image_guidance_scale": args.image_guidance_scale,
        "guidance_scale": args.guidance_scale,
        "resolution": args.resolution,
        "device": args.device,
        "dtype": args.dtype,
        "sample_meta": sample_meta,
    }

    meta_path = Path(args.meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)

    print("Done.")
    print(f"Saved generated image: {output_path}")
    print(f"Saved run metadata:    {meta_path}")


if __name__ == "__main__":
    main()
