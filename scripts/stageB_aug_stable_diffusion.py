import os, sys
import wandb
import argparse
import torch
from torch.utils.data import DataLoader

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

def main(config):
    # create dataset and dataloader
    if config.dataset == "Wen":
        train_set, test_set = create_Wen_dataset(path=config.wen_path, patch_size=config.patch_size, 
                fmri_transform=torch.FloatTensor, subjects=config.wen_subs)
    else:
        raise NotImplementedError 

    dataloader_wen = DataLoader(train_set, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = UNet3DConditionModel()
    model.to(device)  

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(total_steps):
        for batch in dataloader_wen:
            inputs = batch['frames']  # Assuming 'frames' are the video frames
            targets = batch['targets']  # Assuming 'targets' are the ground truth frames

            # Forward pass
            outputs = model(inputs)

            # Compute loss (e.g., MSE loss between outputs and targets)
            loss = ...

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