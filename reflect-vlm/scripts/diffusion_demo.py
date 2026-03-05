from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline


def generate(
   action,
   cur_image_path,
   pipe,
   num_inference_steps=50,
   image_guidance_scale=1.5,
   guidance_scale=10
):
   cur_image = Image.open(cur_image_path).convert('RGB')
   img_shape = cur_image.size
   cur_image = cur_image.resize((512, 512))

   gen_image = pipe(action, 
      image=cur_image, 
      num_inference_steps=num_inference_steps, 
      image_guidance_scale=image_guidance_scale, 
      guidance_scale=guidance_scale,
      generator=generator,
   ).images[0]
   gen_image = gen_image.resize(img_shape)
   return gen_image


if __name__ == '__main__':

   generator = torch.Generator("cuda").manual_seed(0)
   model_path='yunhaif/ReflectVLM-diffusion'
   pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")

   cur_image_path = "assets/images/diffusion_examples/0.png"
   action = "pick up orange"
   gen_image = generate(action, cur_image_path, pipe)
   gen_image.save(f"assets/images/diffusion_examples/gen-0-{action}.png")

   cur_image_path = "assets/images/diffusion_examples/1.png"
   action = "pick up orange"
   gen_image = generate(action, cur_image_path, pipe)
   gen_image.save(f"assets/images/diffusion_examples/gen-1-{action}.png")
