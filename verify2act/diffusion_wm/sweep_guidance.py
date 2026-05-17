#!/usr/bin/env python3
"""
Guidance-scale sweep for the Verify2Act world model.

Loads the pipeline once and generates images across a grid of
image_guidance_scale and guidance_scale values, saving each to a
separate PNG for visual comparison.

Usage:
    python verify2act/world_model/sweep_guidance.py \
      --dataset-dir robosuite/data_capture_wm/dataset/nut_assembly_merged \
      --transition-index 2 \
      --adapter-dir verify2act/output/wm_v2/best/unet_lora \
      --output-dir verify2act/output/wm_guidance_sweep
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from peft import PeftModel
from PIL import Image

try:
    from verify2act.utils import load_vae_encoder
except ImportError:
    from utils import load_vae_encoder


def load_transition_sample(dataset_dir: Path, index: int, transitions_file: str):
    path = dataset_dir / transitions_file
    with open(path) as f:
        for i, line in enumerate(f):
            if i == index:
                row = json.loads(line)
                return dataset_dir / row["image_t"], row["action_text"], row
    raise IndexError(f"transition-index={index} out of range for {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-model", default="timbrooks/instruct-pix2pix")
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--transition-index", type=int, default=0)
    p.add_argument("--transitions-file", default="transitions.jsonl")
    p.add_argument("--image-path", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--output-dir", default="verify2act/output/wm_guidance_sweep")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    # Sweep grids — comma-separated floats
    p.add_argument("--image-guidance-scales", default="1.5,2.5,2.8,3.0,4.0",
                   help="Comma-separated image_guidance_scale values to sweep.")
    p.add_argument("--guidance-scales", default="7.5",
                   help="Comma-separated guidance_scale values to sweep.")
    return p.parse_args()


def main():
    args = parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    igs_values = [float(x) for x in args.image_guidance_scales.split(",")]
    gs_values  = [float(x) for x in args.guidance_scales.split(",")]

    if args.image_path and args.prompt:
        image_path = Path(args.image_path)
        prompt = args.prompt
    elif args.dataset_dir:
        image_path, prompt, _ = load_transition_sample(
            Path(args.dataset_dir), args.transition_index, args.transitions_file
        )
    else:
        raise ValueError("Provide --image-path + --prompt, or --dataset-dir.")

    print(f"Input image : {image_path}")
    print(f"Prompt      : {prompt}")
    print(f"IGS sweep   : {igs_values}")
    print(f"GS  sweep   : {gs_values}")

    print("\nLoading pipeline (once)...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
    ).to(device)

    vae, _ = load_vae_encoder(
        model_name_or_path=args.pretrained_model,
        device=device,
        torch_dtype=torch_dtype,
    )
    pipe.vae = vae

    adapter_dir = Path(args.adapter_dir)
    print(f"Loading LoRA adapter from: {adapter_dir}")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(adapter_dir)).to(device)

    image_in = Image.open(image_path).convert("RGB").resize((args.resolution, args.resolution))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the input image for easy side-by-side comparison
    image_in.save(output_dir / "input.png")

    total = len(igs_values) * len(gs_values)
    done = 0
    for gs in gs_values:
        for igs in igs_values:
            generator = torch.Generator(device.type).manual_seed(args.seed)
            with torch.inference_mode():
                out = pipe(
                    prompt,
                    image=image_in,
                    num_inference_steps=args.num_inference_steps,
                    image_guidance_scale=igs,
                    guidance_scale=gs,
                    generator=generator,
                ).images[0]

            fname = f"igs{igs:.1f}_gs{gs:.1f}.png"
            out.save(output_dir / fname)
            done += 1
            print(f"[{done}/{total}] igs={igs}  gs={gs}  → {fname}")

    print(f"\nAll outputs saved to: {output_dir}")
    print(f"Input saved to:       {output_dir / 'input.png'}")


if __name__ == "__main__":
    main()
