import os, sys
import wandb
import argparse
import torch
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from MindVideo import MindVideoPipeline
from einops import rearrange

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.MindVideo.models.unet import UNet3DConditionModel
from src.MindVideo.utils.dataset import create_Wen_dataset
from src.MindVideo.models.fmri_encoder import fMRIEncoder
from configs.config import Config_Generative_Model

os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."

class wandb_logger:
    def __init__(self, config):
        wandb.init( project='mind-vis',
                    group="augmented_stable_diffusion_train",
                    anonymous="allow",
                    config=config,
                    reinit=True)

        self.config = config
        self.step = None
    
    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

def get_args_parser():
    parser = argparse.ArgumentParser('Training Augmented Stable Diffusion', add_help=False)

    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--mask_ratio', type=float)

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--include_nonavg_test', type=bool)   
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)
                        
    return parser

def add_noise(inputs, noise_type='gaussian', noise_params=None):
    """
    Add noise to input data before feeding it into the model.

    Args:
        inputs (torch.Tensor): Input data tensor.
        noise_params (dict): Parameters specific to the noise type.

    Returns:
        torch.Tensor: Noisy input data tensor.
    """
    if noise_type == 'gaussian':
        std = noise_params.get('std', 0.1)
        noise = torch.randn_like(inputs) * std
        noisy_inputs = inputs + noise
    else:
        raise NotImplementedError(f"Noise type '{noise_type}' not implemented.")

    return noisy_inputs

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

def main(config):
    if config.dataset == "Wen":
        train_set, test_set = create_Wen_dataset(path=config.wen_path, patch_size=config.patch_size, 
                                                 fmri_transform=torch.FloatTensor, subjects=config.wen_subs)
        num_voxels = test_set.num_voxels
    else:
        raise NotImplementedError("Dataset not implemented.")

    dataloader_wen = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load models
    dtype = torch.float16 if config.half_precision else torch.float32
    unet = UNet3DConditionModel.from_pretrained_2d(config.checkpoint_path, subfolder="unet").to(device, dtype=dtype)
    fmri_encoder = fMRIEncoder.from_pretrained(config.checkpoint_path, subfolder='fmri_encoder', num_voxels=num_voxels).to(device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(config.checkpoint_path, subfolder="vae").to(device, dtype=dtype)

    vae.eval()
    fmri_encoder.eval()
    unet.train()  

    for batch in dataloader_wen:
        inputs = batch['image']
        targets = batch['image'].to(device)  
        fmri = batch["fmri"]
        uncon_fmri = batch["uncon_fmri"]
        
        # create latents from image
        latents = encode_video(vae, inputs, dtype, device)

        # Add noise to latents
        noisy_latents = add_noise(latents, noise_type='gaussian', noise_params={'std': 0.1})

        # Sample a random timestep for each image
        timestep = torch.randint(1, 100, size=(dataloader_wen.batch_size,), device=device, dtype=torch.float32)

        fmri_embeddings = _encode_fmri(fmri_encoder, fmri, fmri_encoder.device, 1, True, uncon_fmri)

        # Forward pass
        outputs = unet(noisy_latents, timestep, encoder_hidden_states=fmri_embeddings)

        # Compute loss (e.g., MSE loss between outputs and targets)
        loss = torch.nn.functional.mse_loss(outputs, targets)

        # Backpropagation

        # Log training metrics
        # ...

    # Evaluate model and save checkpoints periodically
    if (epoch + 1) % eval_interval == 0:
        # Evaluate model
        # Save checkpoint
        torch.save(model.state_dict(), f'checkpoint_{epoch}.pt')

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_Generative_Model()
    config = update_config(args, config)
    main(config)