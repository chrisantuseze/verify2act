#!/usr/bin/env python3
"""
Quick visual demo for Verify2Act world-model generation.

Usage patterns:
1) Direct image + prompt:
   python verify2act/demo_wm.py \
     --image-path path/to/frame.png \
     --prompt "pick round nut. position: (0.1, -0.05, 0.83)." \
     --adapter-dir verify2act/output/wm/final/unet_lora

2) Pull sample from dataset transitions.jsonl:
   python verify2act/demo_wm.py \
     --dataset-dir robosuite/data_capture_wm/dataset/nut_assembly \
     --transition-index 0 \
     --adapter-dir verify2act/output/wm/final/unet_lora
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from peft import PeftModel
from PIL import Image

try:
    from verify2act.utils import load_vae_encoder
except ImportError:
    from utils import load_vae_encoder


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

    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)

    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--transition-index", type=int, default=0)

    parser.add_argument("--output-path", type=str, default="verify2act/output/wm_demo/generated.png")
    parser.add_argument("--meta-path", type=str, default="verify2act/output/wm_demo/run_meta.json")

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
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


def load_transition_sample(dataset_dir: Path, index: int):
    transitions_path = dataset_dir / "transitions.jsonl"
    if not transitions_path.exists():
        raise FileNotFoundError(f"Missing transitions file: {transitions_path}")

    with open(transitions_path, "r") as handle:
        for i, line in enumerate(handle):
            if i == index:
                row = json.loads(line)
                image_path = dataset_dir / row["image_t"]
                prompt = row["action_text"]
                return image_path, prompt, row

    raise IndexError(f"transition-index={index} is out of range for {transitions_path}")


def ensure_inputs(args):
    if args.image_path and args.prompt:
        return Path(args.image_path), args.prompt, {"source": "direct_args"}

    if args.dataset_dir:
        image_path, prompt, row = load_transition_sample(Path(args.dataset_dir), args.transition_index)
        return image_path, prompt, row

    raise ValueError(
        "Provide either (--image-path and --prompt) or (--dataset-dir with optional --transition-index)."
    )


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    torch_dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)

    image_path, prompt, sample_meta = ensure_inputs(args)
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

    meta = {
        "pretrained_model": args.pretrained_model,
        "vae_model": vae_model,
        "vae_subfolder": resolved_subfolder,
        "adapter_dir": args.adapter_dir,
        "input_image": str(image_path),
        "prompt": prompt,
        "output_image": str(output_path),
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
