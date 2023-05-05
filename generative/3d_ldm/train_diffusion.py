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
from pathlib import Path

import os
import sys

import torch
import torch.nn.functional as F
from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils import define_instance, prepare_dataloader, setup_ddp
from visualize_image import visualize_one_slice_in_3d_image


def main():
    parser = argparse.ArgumentParser(description="PyTorch Latent Diffusion Model Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    # Step 0: configuration
    ddp_bool = args.gpus > 1  # whether to use distributed data parallel
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

    set_determinism(42)

    # Step 1: set data loader
    size_divisible = 2 ** (len(args.autoencoder_def["num_channels"]) + len(args.diffusion_def["num_channels"]) - 2)
    train_loader, val_loader = prepare_dataloader(
        args,
        args.diffusion_train["batch_size"],
        args.diffusion_train["patch_size"],
        randcrop=False,
        rank=rank,
        world_size=world_size,
        cache=1.0,
        size_divisible=size_divisible,
        amp=True,
    )

    # initialize tensorboard writer
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(args.tfevent_path, "diffusion")
        tensorboard_writer = SummaryWriter(tensorboard_path)

    # Step 2: Define Autoencoder KL network and diffusion model
    # Load Autoencoder KL network
    autoencoder = define_instance(args, "autoencoder_def").to(device)

    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location))
    print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")

    # Compute Scaling factor
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
    # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
    # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
    # and the results will not differ from those obtained when it is not used._

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

    # Define Diffusion Model
    unet = define_instance(args, "diffusion_def").to(device)

    trained_diffusion_path = os.path.join(args.model_dir, "diffusion_unet.pt")
    trained_diffusion_path_last = os.path.join(args.model_dir, "diffusion_unet_last.pt")

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

    # Step 3: training config
    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=args.diffusion_train["lr"] * world_size)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_diff, milestones=[100, 1000], gamma=0.1)

    # Step 4: training
    n_epochs = args.diffusion_train["n_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    autoencoder.eval()
    scaler = GradScaler()
    total_step = 0
    best_val_recon_epoch_loss = 100.0

    for epoch in range(n_epochs):
        unet.train()
        epoch_loss = 0
        lr_scheduler.step()
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
        if epoch % val_interval == 0:
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
                        # save last model
                        if ddp_bool:
                            torch.save(unet.module.state_dict(), trained_diffusion_path_last)
                        else:
                            torch.save(unet.state_dict(), trained_diffusion_path_last)

                        # save best model
                        if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                            best_val_recon_epoch_loss = val_recon_epoch_loss
                            if ddp_bool:
                                torch.save(unet.module.state_dict(), trained_diffusion_path)
                            else:
                                torch.save(unet.state_dict(), trained_diffusion_path)
                            print("Got best val noise pred loss.")
                            print("Save trained latent diffusion model to", trained_diffusion_path)

                        # visualize synthesized image
                        if (epoch) % (50 * val_interval) == 0:  # time cost of synthesizing images is large
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
                                    epoch,
                                )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
