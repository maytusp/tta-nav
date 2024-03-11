# This part is for testing Autoencoder (No ELBO Loss)
# The encoder is from Point-Goal Navigation ResNeXt-50 trained with 2.5 Billions Frames

import os

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ae import AE
from util import configure_device, get_dataset, save_as_images
from torch import tensor
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.regression import MeanSquaredError

@click.group()
def cli():
    pass

np.random.seed(0)
torch.manual_seed(0)
# TODO: Upgrade the commands in this script to use hydra config
# and support Multi-GPU inference
@cli.command()
@click.argument("chkpt-path")
@click.argument("root")
@click.option("--device", default="gpu:0")
@click.option("--adapt", default=None)
@click.option("--corruption-type", default=None)
@click.option("--severity", default=0)
@click.option("--dataset", default="celebamaskhq")
@click.option("--image-size", default=128)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default=os.getcwd())
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
def reconstruct(
    chkpt_path,
    root,
    adapt,
    corruption_type=None,
    severity=0,
    device="gpu:0",
    dataset="celebamaskhq",
    image_size=128,
    num_samples=-1,
    save_path=os.getcwd(),
    write_mode="image",
):
    if "gpu" in device:
        dev, _ = configure_device(device)
        if "[0]" in dev:
            dev = dev.replace("[0]","0")
    else:
        dev = configure_device(device)
    if num_samples == 0:
        raise ValueError(f"`--num-samples` can take value=-1 or > 0")

    # Dataset
    return_clean = True
    dataset = get_dataset(dataset, root, image_size, norm=False, flip=False, 
                        corruption_type=corruption_type, severity=severity,
                        return_orig=return_clean)

    # Loader
    loader = DataLoader(
        dataset,
        1,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )
    dec_block_config_str = "1x1,1u1,1t4,4x2,4u1,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
    dec_channel_config_str = "256:64,128:64,64:64,32:128,16:128,8:256,4:512,1:512"


    # vae = AE(256,
    #     dec_block_config_str,
    #     dec_channel_config_str,
    # )
    # print(vae)
    vae = AE.load_from_checkpoint(chkpt_path, input_res=256, dec_block_str=dec_block_config_str, dec_channel_str=dec_channel_config_str).to(dev)
    vae.eval()

    sample_list = []
    img_list = []
    clean_img_list = []
    count = 0
    decay_factor=0.94
    min_mom=0.005
    mom_pre=0.1
    mean_squared_error = torch.nn.MSELoss()
    mean_absolute_error = torch.nn.L1Loss()
    adaptation_step=0
    running_mse = 0.0
    running_mae = 0.0
    for step, batch in tqdm(enumerate(loader)):
        vae.eval()
        clean_img, trans_img = batch
        clean_img, trans_img = clean_img.to(dev), trans_img.to(dev)
        if adapt == "dua":
            # print(vae.enc)
            encoder = vae.enc
            encoder, mom_pre, decay_factor, min_mom = _adapt(encoder, mom_pre, decay_factor, min_mom)

        with torch.no_grad():
            recons = vae.forward_recons(trans_img)
            vae.eval()
            mse_loss = mean_squared_error(clean_img, recons)
            mae_loss = mean_absolute_error(clean_img, recons)
            running_mse += mse_loss.item()
            running_mae += mae_loss.item()

        if count + recons.size(0) >= num_samples and num_samples != -1:
            # img_list.append(trans_img[:num_samples, :, :, :].cpu())
            # sample_list.append(recons[:num_samples, :, :, :].cpu())
            # clean_img_list.append(clean_img[:num_samples, :, :, :].cpu())
            break

        # # Not transferring to CPU leads to memory overflow in GPU!
        # sample_list.append(recons.cpu())
        # img_list.append(trans_img.cpu())
        # clean_img_list.append(clean_img.cpu())
        count += recons.size(0)

    # cat_img = torch.cat(img_list, dim=0)
    # cat_sample = torch.cat(sample_list, dim=0)
    # cat_clean_img = torch.cat(clean_img_list, dim=0)
    total_samples = count + recons.size(0)
    print(f"adapt: {adapt} corruption: {corruption_type}, MSE: {running_mse/total_samples}, MAE: {running_mae/total_samples}")    
    # Save the image and reconstructions as numpy arrays
    # if not(os.path.exists(save_path)):
    #     os.makedirs(save_path, exist_ok=True)
    # if write_mode == "image":
    #     save_as_images(
    #         cat_sample,
    #         file_name=os.path.join(save_path, "recon"),
    #         denorm=False,
    #     )
    #     save_as_images(
    #         cat_img,
    #         file_name=os.path.join(save_path, "orig"),
    #         denorm=False,
    #     )
    #     save_as_images(
    #         cat_clean_img,
    #         file_name=os.path.join(save_path, "orig_clean"),
    #         denorm=False,
    #     )
    # else:
    #     np.save(os.path.join(save_path, "images.npy"), cat_img.numpy())
    #     np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())

def _adapt(encoder,
            mom_pre,
            decay_factor,
            min_mom):
    mom_new = (mom_pre * decay_factor)
    min_momentum_constant = min_mom
    for _, m in enumerate(encoder.modules()):
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.train()
            m.momentum = mom_new + min_momentum_constant
    mom_pre = mom_new
    return encoder, mom_pre, decay_factor, min_mom

if __name__ == "__main__":
    cli()
