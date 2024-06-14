# Author: Sijin Yu

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
import torch
import torch.nn.functional as F
from MindVideo import MindVideoPipeline
from dataclasses import dataclass

from torch.optim.lr_scheduler import LambdaLR
from diffusers import DDPMScheduler
import torch.optim as optim

import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.MindVideo.models.unet import UNet3DConditionModel
from src.MindVideo.utils.dataset import create_Wen_dataset
from src.MindVideo.models.fmri_encoder import fMRIEncoder
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from einops import rearrange

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 2
    eval_batch_size = 2  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "logs"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 3407
    checkpoint_path = "/content/drive/MyDrive/neurips-release/pretrains/sub1" # os.path.join(self.root_path, 'results/generation/25-08-2022-08:02:55/checkpoint.pth')
    wen_path = "/content/drive/MyDrive/neurips-release/wen2017"
    patch_size = 16
    wen_subs = ['subject1']
    batch_size = train_batch_size
    half_precision = True

def train_unet_loop(config, unet, vae, fmri_encoder, noise_scheduler, optimizer, train_dataloader, lr_scheduler, device, dtype):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    unet, vae, fmri_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, vae, fmri_encoder, optimizer, train_dataloader, lr_scheduler
    )
    
    vae.eval()
    fmri_encoder.eval()
    global_step = 0

    # Now you train the unet
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["image"].to(device, dtype=dtype)
            fmri = batch["fmri"].to(device, dtype=dtype)
            uncon_fmri = batch["uncon_fmri"].to(device, dtype=dtype)
            # clean_latents = vae.encode(clean_images).sample
            # clean_latents = clean_latents * 0.18215
            # create latents from image
            clean_latents = encode_video(vae, clean_images, dtype, device)
            # Sample noise to add to the latents
            noise = torch.randn(clean_latents.shape, device=clean_latents.device, dtype=dtype)
            bs = clean_latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_latents.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)
            
            fmri_embeddings = _encode_fmri(fmri_encoder, fmri, fmri_encoder.device, 1, True, uncon_fmri)
            
            with accelerator.accumulate(unet):
                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, return_dict=False, encoder_hidden_states=fmri_embeddings)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the unet
        if accelerator.is_main_process:
            pipeline = MindVideoPipeline(vae, fmri_encoder, unet, noise_scheduler)

            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            #     evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)

@torch.no_grad()                    
def _encode_fmri(fmri_encoder, fmri, device, num_videos_per_fmri, do_classifier_free_guidance, negative_prompt):
    dtype = fmri_encoder.dtype
    fmri_embeddings = fmri_encoder(fmri.to(device, dtype=dtype))
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, embed_dim = fmri_embeddings.shape
    fmri_embeddings = fmri_embeddings.repeat(1, num_videos_per_fmri, 1)
    fmri_embeddings = fmri_embeddings.view(bs_embed * num_videos_per_fmri, seq_len, -1)

    # support classification free guidance
    if do_classifier_free_guidance:
        # uncond_input = torch.zeros_like(fmri).to(device, dtype=dtype)
        uncond_input = negative_prompt.to(device, dtype=dtype)
        uncond_embeddings = fmri_encoder(
            uncond_input
        )
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_fmri, 1)
        uncond_embeddings = uncond_embeddings.view(bs_embed * num_videos_per_fmri, seq_len, -1)
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        fmri_embeddings = torch.cat([uncond_embeddings, fmri_embeddings])

    return fmri_embeddings

def encode_video(vae, video, dtype, device):
    print(f'Encoding video with shape: {video.shape} at {dtype}')
    video = video.to(device=device, dtype=dtype)

    # Ensure the video is in the correct range and format
    video = (video - 0.5) * 2  # Scale video to [-1, 1]

    # Reshape video to match the expected input shape for the encoder
    batch_size, frames, height, width, channels = video.shape
    video = rearrange(video, "b f h w c -> (b f) c h w")

    # Encode the video to get the latents
    latents = vae.encode(video).latent_dist.sample()

    # Reshape latents to the original latent shape
    latents = rearrange(latents, "(b f) c h w -> b c f h w", b=batch_size, f=frames)

    # Scale the latents to match the scale used during decoding
    latents = latents * 0.18215

    return latents

if __name__ == '__main__':
    config = TrainingConfig()

    train_set, test_set = create_Wen_dataset(path=config.wen_path, patch_size=config.patch_size, 
                                                 fmri_transform=torch.FloatTensor, subjects=config.wen_subs, window_size=2)
    num_voxels = test_set.num_voxels
    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if config.half_precision else torch.float32
    unet = UNet3DConditionModel.from_pretrained_2d(config.checkpoint_path, subfolder="unet").to(device, dtype=dtype)
    fmri_encoder = fMRIEncoder.from_pretrained(config.checkpoint_path, subfolder='fmri_encoder', num_voxels=num_voxels).to(device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(config.checkpoint_path, subfolder="vae").to(device, dtype=dtype)

    # Define the noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Define the optimizer
    optimizer = optim.AdamW(unet.parameters(), lr=1e-4)

    # Define the learning rate scheduler with warmup
    def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        return LambdaLR(optimizer, lr_lambda)

    # Assuming num_training_steps is the total number of steps in your training loop
    num_training_steps = config.num_epochs * len(train_dataloader)
    num_warmup_steps = config.lr_warmup_steps

    lr_scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
    train_unet_loop(config, 
                    unet, 
                    vae, 
                    fmri_encoder, 
                    noise_scheduler, 
                    optimizer, 
                    train_dataloader, 
                    lr_scheduler,
                    device, 
                    dtype)