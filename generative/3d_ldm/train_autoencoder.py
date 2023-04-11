# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, ThreadDataLoader, partition_dataset
from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from visualize_image import visualize_one_slice_in_3d_image


def setup_ddp(rank, world_size):
    print(f"Running DDP diffusion example on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device


def prepare_dataloader(args, batch_size, rank=0, world_size=1, cache=1.0):
    # %% [markdown]
    #     # ## Setup Decathlon Dataset and training and validation data loaders
    #     #
    #     # In this tutorial, we will use the 3D T1 weighted brain images from the [2016 and 2017 Brain Tumor Segmentation (BraTS) challenges](https://www.med.upenn.edu/sbia/brats2017/data.html). This dataset can be easily downloaded using the [DecathlonDataset](https://docs.monai.io/en/stable/apps.html#monai.apps.DecathlonDataset) from MONAI (`task="Task01_BrainTumour"`). To load the training and validation images, we are using the `data_transform` transformations that are responsible for the following:
    #     #
    #     # 1. `LoadImaged`:  Loads the brain images from files.
    #     # 2. `Lambdad`: Choose channel 1 of the image, which is the T1-weighted image.
    #     # 3. `AddChanneld`: Add the channel dimension of the input data.
    #     # 4. `ScaleIntensityd`: Apply a min-max scaling in the intensity values of each image to be in the `[0, 1]` range.
    #     # 5. `CenterSpatialCropd`: Crop the background of the images using a roi of size `[160, 200, 155]`.
    #     # 6. `Resized`: Resize the images to a volume with size `[32, 40, 32]`.
    #     #
    #     # For the data loader, we are using mini-batches of 8 images, which consumes about 21GB of GPU memory during training. Please, reduce this value to run on smaller GPUs.

    ddp_bool = world_size > 1
    channel = args.channel  # 0 = Flair, 1 = T1
    assert channel in [0, 1, 2, 3], "Choose a valid channel"
    roi_size = (176, 192, 144)
    size_divisible = 2 ** (len(args.Autoencoder["num_channels"]) + len(args.DiffusionModel["num_channels"]) - 2)
    resized_roi_size = [
        int(np.floor(roi_size[i] / args.spacing[i] / size_divisible)) * size_divisible for i in range(3)
    ]
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            AddChanneld(keys=["image"]),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            CenterSpatialCropd(keys=["image"], roi_size=roi_size),
            Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear")),
            CenterSpatialCropd(keys=["image"], roi_size=resized_roi_size),
            ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        ]
    )
    train_ds = DecathlonDataset(
        root_dir=args.data_base_dir,
        task="Task01_BrainTumour",
        section="training",  # validation
        cache_rate=cache,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=8,
        download=False,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
        transform=train_transforms,
    )
    val_ds = DecathlonDataset(
        root_dir=args.data_base_dir,
        task="Task01_BrainTumour",
        section="validation",  # validation
        cache_rate=cache,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=8,
        download=False,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
        transform=train_transforms,
    )
    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(not ddp_bool), num_workers=0, pin_memory=False, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, sampler=val_sampler
    )
    if rank == 0:
        print(f'Image shape {train_ds[0]["image"].shape}')
    return train_loader, val_loader


def define_autoencoder(args, device):
    autoencoder = AutoencoderKL(
        spatial_dims=args.spatial_dims,
        in_channels=1,
        out_channels=1,
        num_channels=args.Autoencoder["num_channels"],
        latent_channels=args.latent_channels,
        num_res_blocks=args.Autoencoder["num_res_blocks"],
        norm_num_groups=16,
        attention_levels=args.Autoencoder["attention_levels"],
        with_encoder_nonlocal_attn=False,  # current attention block causes stride warning when using ddp
        with_decoder_nonlocal_attn=False,
    )
    autoencoder.to(device)
    return autoencoder


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_48g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    ddp_bool = args.gpus > 1
    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = 0

    torch.cuda.set_device(device)
    print(f"Using {device}")

    print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    # %% [markdown]
    #     # ## Setup data directory
    #     #
    #     # You can specify a directory with the MONAI_DATA_DIRECTORY environment variable.
    #     #
    #     # This allows you to save results and reuse downloads.
    #     #
    #     # If not specified a temporary directory will be used.

    # %%
    directory = os.environ.get("MONAI_DATA_DIRECTORY")

    # %% [markdown]
    #     # ## Set deterministic training for reproducibility

    # %%
    set_determinism(42)

    # %%
    train_loader, val_loader = prepare_dataloader(args, args.Autoencoder["batch_size"], rank, world_size, cache=1.0)

    # ## Autoencoder KL
    #
    # ### Define Autoencoder KL network
    #
    # In this section, we will define an autoencoder with KL-regularization for the LDM. The autoencoder's primary purpose is to transform input images into a latent representation that the diffusion model will subsequently learn. By doing so, we can decrease the computational resources required to train the diffusion component, making this approach suitable for learning high-resolution medical images.
    #

    # +
    autoencoder = define_autoencoder(args, device)

    discriminator_norm = "INSTANCE"
    if ddp_bool and discriminator_norm == "BATCH":
        raise ValueError("When using DDP, discriminator does not support BatchNorm.")
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        num_channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    )
    discriminator.to(device)

    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
    trained_d_path = os.path.join(args.model_dir, "discriminator.pt")

    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location))
            print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")
        except:
            print(f"Rank {rank}: Train autoencoder from scratch.")

        try:
            discriminator.load_state_dict(torch.load(trained_d_path, map_location=map_location))
            print(f"Rank {rank}: Load trained discriminator from {trained_d_path}")
        except:
            print(f"Rank {rank}: Train discriminator from scratch.")

    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[device], output_device=rank, find_unused_parameters=True)
    # -

    # ### Defining Losses
    #
    # We will also specify the perceptual and adversarial losses, including the involved networks, and the optimizers to use during the training process.

    # +
    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)

    def KL_loss(z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]

    adv_weight = 0.01
    perceptual_weight = args.Autoencoder["perceptual_weight"]
    # kl_weight: important hyper-parameter. If too large, decoder cannot recon good results from latent space. If too small, latent space will not be regularized enough for the diffusion model
    kl_weight = args.Autoencoder["kl_weight"]

    # -
    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.Autoencoder["lr"] * world_size)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.Autoencoder["lr"] * world_size)

    # initialize tensorboard writer
    if rank == 0:
        Path(args.tfevent_path + "autoencoder").mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(args.tfevent_path + "autoencoder")

    # ### Train model

    # +
    n_epochs = args.n_epochs
    autoencoder_warm_up_n_epochs = 5
    val_interval = args.val_interval
    val_recon_epoch_loss_list = []
    intermediary_images = []
    n_example_images = 4
    best_val_recon_epoch_loss = 100.0
    total_step = 0

    for epoch in range(n_epochs):
        # train
        autoencoder.train()
        discriminator.train()
        if ddp_bool:
            # if ddp, distribute data across n gpus
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)

            # train Generator part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)

            recons_loss = l1_loss(reconstruction, images)
            kl_loss = KL_loss(z_mu, z_sigma)
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_g + adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # train Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                tensorboard_writer.add_scalar("train_recon_loss_iter", recons_loss, total_step)
                tensorboard_writer.add_scalar("train_kl_loss_iter", kl_loss, total_step)
                tensorboard_writer.add_scalar("train_perceptual_loss_iter", p_loss, total_step)
                if epoch > autoencoder_warm_up_n_epochs:
                    tensorboard_writer.add_scalar("train_adv_loss_iter", generator_loss, total_step)
                    tensorboard_writer.add_scalar("train_fake_loss_iter", loss_d_fake, total_step)
                    tensorboard_writer.add_scalar("train_real_loss_iter", loss_d_real, total_step)

        # validation
        if (epoch + 1) % val_interval == 0:
            autoencoder.eval()
            val_recon_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)  # choose only one of Brats channels
                with torch.no_grad():
                    reconstruction, z_mu, z_sigma = autoencoder(images)
                    recons_loss = l1_loss(reconstruction.float(), images.float())

                val_recon_epoch_loss += recons_loss.item()

            val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
            if rank == 0:
                # save best model
                print(f"Epoch {epoch} val_recon_loss: {val_recon_epoch_loss}")
                val_recon_epoch_loss_list.append(val_recon_epoch_loss)
                if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                    best_val_recon_epoch_loss = val_recon_epoch_loss
                    if ddp_bool:
                        torch.save(autoencoder.module.state_dict(), trained_g_path)
                        torch.save(discriminator.module.state_dict(), trained_d_path)
                    else:
                        torch.save(autoencoder.state_dict(), trained_g_path)
                        torch.save(discriminator.state_dict(), trained_d_path)
                    print("Got best val recon loss.")
                    print("Save trained autoencoder to", trained_g_path)
                    print("Save trained discriminator to", trained_d_path)

                # write val loss for each epoch into tensorboard
                tensorboard_writer.add_scalar("val_recon_loss", val_recon_epoch_loss, epoch)
                for axis in range(3):
                    tensorboard_writer.add_image(
                        "val_img_" + str(axis),
                        visualize_one_slice_in_3d_image(images[0, 0, ...], axis).transpose([2, 1, 0]),
                        epoch + 1,
                    )
                    tensorboard_writer.add_image(
                        "val_recon_" + str(axis),
                        visualize_one_slice_in_3d_image(reconstruction[0, 0, ...], axis).transpose([2, 1, 0]),
                        epoch + 1,
                    )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
