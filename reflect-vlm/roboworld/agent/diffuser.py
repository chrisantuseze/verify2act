import PIL
from PIL import Image
import json
import os
import torch
import random
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import StableDiffusionInstructPix2PixPipeline
from torchvision import transforms
import numpy as np

class DiffusionSim(object):

    def __init__(self, unet_dir=None, vae_dir=None, pretrained_model='timbrooks/instruct-pix2pix'):
        super(DiffusionSim, self).__init__()
        self.unet_dir = unet_dir
        self.vae_dir = vae_dir
        print(f"Loading pretrained diffusion from {pretrained_model}")
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16).to("cuda")
        if unet_dir is not None and os.path.exists(unet_dir):
            print(f"Loading unet from {unet_dir}")
            self.unet = UNet2DConditionModel.from_pretrained(unet_dir, subfolder="unet_ema", torch_dtype=torch.float16, local_files_only=True).to('cuda')
            self.pipeline.unet = self.unet
        if vae_dir is not None and os.path.exists(vae_dir):
            print(f"Loading vae from {vae_dir}")
            self.vae = AutoencoderKL.from_pretrained(vae_dir, subfolder="vae_ema", torch_dtype=torch.float16, local_files_only=True).to('cuda')
            self.pipeline.vae = self.vae
        self.pipeline.set_progress_bar_config(disable=True)

        self.generator = torch.Generator("cuda").manual_seed(0)
        self.num_inference_steps = 50
        self.image_guidance_scale = 1.5
        self.guidance_scale = 10


    def forward(self, curr_image: np.ndarray, act_text: str) -> np.ndarray:
        curr_image = Image.fromarray(curr_image)
        image_size = curr_image.size
        curr_image = curr_image.resize((512, 512))
        next_image = self.pipeline(act_text, 
            image=curr_image, 
            num_inference_steps=self.num_inference_steps, 
            image_guidance_scale=self.image_guidance_scale, 
            guidance_scale=self.guidance_scale,
            generator=self.generator,
        ).images[0]
        next_image = next_image.resize(image_size)
        return np.array(next_image)


# model_id = "model_sd15_datav13_aug_failure_aug002/checkpoint-3000" # <- replace this 
# unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet_ema", torch_dtype=torch.float16).to('cuda')

# ckpt_dir = "/mnt/lustre/hanjiaming/petrelfs/code/diffusers/examples/autoencoder_kl/autoencoder_sd15_train-decoder_datav13/checkpoint-18000"
# vae = AutoencoderKL.from_pretrained(ckpt_dir, subfolder="vae_ema", torch_dtype=torch.float16).to('cuda')
