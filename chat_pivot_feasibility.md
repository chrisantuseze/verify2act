
Pivoting to an image-based diffusion world model (like ReflectVLM's approach) is actually a very strategic move, especially given the brittleness we just fought with the 3D point cloud coordinate frames. Image-to-image translation abstracts away the strict geometric matrices and ordering constraints, treating state transitions purely as visual patterns.

Here is a realistic breakdown of your hardware and timeline constraints:

1. Can you fine-tune InstructPix2Pix on a 24GB GPU?
Yes, absolutely. InstructPix2Pix is built on top of Stable Diffusion 1.5.

A full fine-tune is tight on 24GB, but utilizing LoRA (Low-Rank Adaptation) combined with gradient checkpointing and 8-bit Adam optimizer reduces the VRAM requirement to roughly 12–16 GB for a standard 512x512 resolution at a batch size of 1 or 2.
Hugging Face's diffusers library has a dedicated script (train_instruct_pix2pix.py) that supports LoRA and xFormers out of the box, meaning you won't need to write the training loop from scratch.
2. Can you finish the entire pipeline (including Critic) in 2 months?
Yes, but it will be a tight sprint. Because you are on a time crunch, you must aggressively cut corners on things that aren't the core contribution.

Here is how you realistically partition those 8 weeks:

Weeks 1–2 (Data Collection): Convert your Robosuite rollouts into image triplets: [Image_t] + "Move cubeA onto table" -> [Image_t+1]. You don't need point clouds anymore, just rendered camera frames and text prompts.
Weeks 3–4 (World Model Finetuning): Use the diffusers LoRA script to fine-tune InstructPix2Pix. You can likely complete a training run in 1-2 days on a 24GB GPU, giving you time to iterate on hyperparameters.
Weeks 5–6 (The Critic): Do not train a standard critic from scratch. To save time, either:
Fine-tune a lightweight VLM (like LLaVA-1.5-7B or Qwen2-VL-7B) using LoRA to classify (Image, Action) -> Success/Fail. Both can easily be fine-tuned on 24GB.
Train a simple ViT or ResNet that takes the generated Image_t+1 and outputs a scalar feasibility score (much faster to train than a generative model).
Weeks 7–8 (Integration & Evaluation): Hook the diffusion world model and the critic into your planner (like the MCTS or reflection loop you had in Points2Plans) and run the final success metrics.
My Recommendation
If you make this pivot:

Dump the relational_dynamics point cloud architecture completely.
Grab the diffusers library immediately and run a test LoRA training script over a dummy dataset of 100 images tonight.
Your pipeline becomes: Current State Image --[Action Text]--> InstructPix2Pix --[Predicted Image]--> Critic --[Score]--> Action Selection.
It's a highly practical pivot that aligns with the current cutting-edge (like ReflectVLM or SuSIE), and relies on established HuggingFace tooling rather than custom PyTorch 3D matrix math.