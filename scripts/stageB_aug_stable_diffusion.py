import os, sys
import wandb
import argparse
import torch
from torch.utils.data import DataLoader
from MindVideo import MindVideoPipeline

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.MindVideo.models.unet import UNet3DConditionModel
from src.MindVideo.utils.dataset import create_Wen_dataset
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

def add_noise_to_inputs(inputs, noise_type='gaussian', noise_params=None):
    """
    Add noise to input data before feeding it into the model.

    Args:
        inputs (torch.Tensor): Input data tensor.
        noise_type (str): Type of noise to add ('gaussian', 'dropout', etc.).
        noise_params (dict): Parameters specific to the noise type.

    Returns:
        torch.Tensor: Noisy input data tensor.
    """
    if noise_type == 'gaussian':
        std = noise_params.get('std', 0.1)
        noise = torch.randn_like(inputs) * std
        noisy_inputs = inputs + noise
    elif noise_type == 'dropout':
        p = noise_params.get('p', 0.1)
        mask = torch.bernoulli(torch.full_like(inputs, 1 - p))
        noisy_inputs = inputs * mask
    else:
        raise NotImplementedError(f"Noise type '{noise_type}' not implemented.")

    return noisy_inputs

def main(config):
    if config.dataset == "Wen":
        train_set, test_set = create_Wen_dataset(path=config.wen_path, patch_size=config.patch_size, 
                                                 fmri_transform=torch.FloatTensor, subjects=config.wen_subs)
    else:
        raise NotImplementedError("Dataset not implemented.")

    dataloader_wen = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load models
    unet = UNet3DConditionModel.from_pretrained_2d(checkpoint_path, subfolder="unet").to(device, dtype=dtype)
    fmri_encoder = fMRIEncoder.from_pretrained(checkpoint_path, subfolder='fmri_encoder', num_voxels=num_voxels).to(device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae").to(device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    unet.train()  

    for batch in dataloader_wen:
        inputs = batch['image'].to(device)  
        targets = batch['image'].to(device)  

        # create latents from image
        # TODO: vae encode image

        # Add noise to latents
        # TODO: add noise to latents
        # sample = add_noise_to_inputs(inputs, noise_type='gaussian', noise_params={'std': 0.1})

        # Sample a random timestep for each image
        timestep = torch.randint(1, 100, size=(dataloader_wen.batch_size,), device=device, dtype=torch.float32)

        encoder_hidden_states = _encode_fmri(fmri, device, num_videos_per_fmri, do_classifier_free_guidance, negative_prompt)

        # Forward pass
        outputs = unet(sample, timestep, encoder_hidden_states)

        # Compute loss (e.g., MSE loss between outputs and targets)
        loss = torch.nn.functional.mse_loss(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

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