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

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
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

def train_unet_loop(config, unet, vae, fmri_encoder, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
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
            clean_images = batch["image"]
            fmri = batch["fmri"]
            uncon_fmri = batch["uncon_fmri"]
            clean_latents = vae.encode(clean_images).sample
            clean_latents = clean_latents * 0.18215
            # Sample noise to add to the latents
            noise = torch.randn(clean_latents.shape, device=clean_latents.device)
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