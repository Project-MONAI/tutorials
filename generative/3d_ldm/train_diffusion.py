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
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train_autoencoder import define_autoencoder, prepare_dataloader, setup_ddp
from visualize_image import visualize_one_slice_in_3d_image


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

    # %%
    train_loader, val_loader = prepare_dataloader(args, args.DiffusionModel["batch_size"], rank, world_size, cache=1.0)

    # initialize tensorboard writer
    if rank == 0:
        tensorboard_writer = SummaryWriter(args.tfevent_path + "diffusion")

    # ## Load Autoencoder KL
    #
    # ### Define Autoencoder KL network
    #
    # In this section, we will define an autoencoder with KL-regularization for the LDM. The autoencoder's primary purpose is to transform input images into a latent representation that the diffusion model will subsequently learn. By doing so, we can decrease the computational resources required to train the diffusion component, making this approach suitable for learning high-resolution medical images.
    #

    # +
    autoencoder = define_autoencoder(args, device)

    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location))
    print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")

    # +
    # ### Scaling factor
    #
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    #
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
    #
    with torch.no_grad():
        with autocast(enabled=True):
            check_data = first(train_loader)
            z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))
            if rank == 0:
                print(f"Latent feature shape {z.shape}")
                for axis in range(3):
                    tensorboard_writer.add_image(
                        "train_img_" + str(axis),
                        visualize_one_slice_in_3d_image(check_data["image"][0, 0, ...], axis).transpose([2, 1, 0]),
                        1,
                    )
                print(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1 / torch.std(z)
    # -

    # -

    # ## Diffusion Model
    #
    # ### Define diffusion model and scheduler
    #
    # In this section, we will define the diffusion model that will learn data distribution of the latent representation of the autoencoder. Together with the diffusion model, we define a beta scheduler responsible for defining the amount of noise tahat is added across the diffusion's model Markov chain.

    # +
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=args.latent_channels,
        out_channels=args.latent_channels,
        num_res_blocks=args.DiffusionModel["num_res_blocks"],
        num_channels=args.DiffusionModel["num_channels"],
        attention_levels=args.DiffusionModel["attention_levels"],
        num_head_channels=args.DiffusionModel["num_head_channels"],
    )
    unet.to(device)

    trained_diffusion_path = os.path.join(args.model_dir, "diffusion_unet.pt")

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            unet.load_state_dict(torch.load(trained_diffusion_path, map_location=map_location))
            print(f"Rank {rank}: Load trained diffusion model from", trained_diffusion_path)
        except:
            print(f"Rank {rank}: Train diffusion model from scratch.")

    scheduler = DDPMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        beta_schedule="scaled_linear",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
    )

    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)
        unet = DDP(unet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # We define the inferer using the scale factor:

    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=args.DiffusionModel["lr"] * world_size)

    # ### Train model

    # +
    n_epochs = args.n_epochs
    val_interval = args.val_interval
    diffusion_warm_up_n_epochs = 1000
    autoencoder.eval()
    scaler = GradScaler()
    total_step = 0
    best_val_recon_epoch_loss = 100.0

    for epoch in range(n_epochs):
        unet.train()
        epoch_loss = 0
        if ddp_bool:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # Generate random noise
                noise_shape = [images.shape[0]] + list(z.shape[1:])
                noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                # Get model prediction
                if ddp_bool:
                    inferer_autoencoder = autoencoder.module
                else:
                    inferer_autoencoder = autoencoder
                noise_pred = inferer(
                    inputs=images,
                    autoencoder_model=inferer_autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                tensorboard_writer.add_scalar("train_diffusion_loss_iter", loss, total_step)

        # validation
        if (epoch + 1) % val_interval == 0:
            autoencoder.eval()
            unet.eval()
            val_recon_epoch_loss = 0
            with torch.no_grad():
                with autocast(enabled=True):
                    # compute val loss
                    for step, batch in enumerate(val_loader):
                        images = batch["image"].to(device)
                        noise_shape = [images.shape[0]] + list(z.shape[1:])
                        noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()

                        # Get model prediction
                        if ddp_bool:
                            inferer_autoencoder = autoencoder.module
                        else:
                            inferer_autoencoder = autoencoder
                        noise_pred = inferer(
                            inputs=images,
                            autoencoder_model=inferer_autoencoder,
                            diffusion_model=unet,
                            noise=noise,
                            timesteps=timesteps,
                        )
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())
                        val_recon_epoch_loss += val_loss.item()
                    val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)

                    # write val loss and save best model
                    if rank == 0:
                        tensorboard_writer.add_scalar("val_diffusion_loss", val_recon_epoch_loss, epoch)
                        print(f"Epoch {epoch} val_diffusion_loss: {val_recon_epoch_loss}")
                        if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                            best_val_recon_epoch_loss = val_recon_epoch_loss
                            if ddp_bool:
                                torch.save(unet.module.state_dict(), trained_diffusion_path)
                            else:
                                torch.save(unet.state_dict(), trained_diffusion_path)
                            print("Got best val noise pred loss.")
                            print("Save trained latent diffusion model to", trained_diffusion_path)

                        # visualize synthesized image
                        if (epoch + 1) % (50 * val_interval) == 0:  # time cost of synthesizing images is large
                            synthetic_images = inferer.sample(
                                input_noise=noise[0:1, ...],
                                autoencoder_model=inferer_autoencoder,
                                diffusion_model=unet,
                                scheduler=scheduler,
                            )
                            for axis in range(3):
                                tensorboard_writer.add_image(
                                    "val_diff_synimg_" + str(axis),
                                    visualize_one_slice_in_3d_image(synthetic_images[0, 0, ...], axis).transpose(
                                        [2, 1, 0]
                                    ),
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
