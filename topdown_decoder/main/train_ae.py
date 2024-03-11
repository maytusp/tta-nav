import logging
import os

import hydra
import pytorch_lightning as pl
import torchvision.transforms as T
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
import torch
from models.ae import AE
from util import configure_device, get_dataset

logger = logging.getLogger(__name__)

# print("Environment Vars:", os.environ)
@hydra.main(config_path="configs")

def train(config):
    # Get config and setup
    config = config.dataset.vae
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)

    # Dataset
    root = config.data.root
    d_type = config.data.name
    image_size = config.data.image_size
    dataset = get_dataset(d_type, root, image_size, norm=False, flip=config.data.hflip)
    N = len(dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)

    # Model
    vae = AE(
        input_res=image_size,
        dec_block_str=config.model.dec_block_config,
        dec_channel_str=config.model.dec_channel_config,
        lr=config.training.lr,
        alpha=config.training.alpha,
    )

    # Load PointNav model and freeze
    checkpoint_path = '/mnt/iusers01/fatpou01/compsci01/n70579mp/ttanav_ae/pretrained_models/se_resneXt50_pointnav.pt'
    checkpoint =  torch.load(checkpoint_path)
    vae.enc.load_state_dict(checkpoint, strict=False)
    print(f"Loaded Visual Encoder from {checkpoint_path}")
    
    for param in vae.enc.parameters():
        param.requires_grad = False
    vae.enc.eval()
    print("Freeze Visual Encoder")


    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path is not None:
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"vae-{config.training.chkpt_prefix}"
        + "-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    device = config.training.device
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        from pytorch_lightning.plugins import DDPPlugin

        train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if config.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(vae, train_dataloader=loader)


if __name__ == "__main__":
    train()
