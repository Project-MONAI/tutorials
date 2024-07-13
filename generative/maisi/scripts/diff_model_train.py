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
import os
import logging
from datetime import datetime
from pathlib import Path

import monai
from monai.data import ThreadDataLoader, partition_dataset
from monai.transforms import Compose
from monai.utils import first

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance


def load_filenames(data_list_path: str) -> list:
    """
    Load filenames from the JSON data list.

    Args:
        data_list_path (str): Path to the JSON data list file.

    Returns:
        list: List of filenames.
    """
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_train = json_data["training"]
    return [_item["image"].replace(".nii.gz", "_emb.nii.gz") for _item in filenames_train]


def prepare_data(train_files: list, device: torch.device, cache_rate: float, num_workers: int = 2) -> ThreadDataLoader:
    """
    Prepare training data.

    Args:
        train_files (list): List of training files.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.

    Returns:
        ThreadDataLoader: Data loader for training.
    """
    train_transforms = Compose(
        [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Lambdad(
                keys="top_region_index",
                func=lambda x: torch.FloatTensor(json.load(open(x))["top_region_index"]),
            ),
            monai.transforms.Lambdad(
                keys="bottom_region_index",
                func=lambda x: torch.FloatTensor(json.load(open(x))["bottom_region_index"]),
            ),
            monai.transforms.Lambdad(
                keys="spacing",
                func=lambda x: torch.FloatTensor(json.load(open(x))["spacing"]),
            ),
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
        ]
    )

    train_ds = monai.data.CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    num_images_per_batch = args.diffusion_unet_train["batch_size"]
    return ThreadDataLoader(train_ds, num_workers=6, batch_size=num_images_per_batch, shuffle=True)


def load_unet(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> torch.nn.Module:
    """
    Load the UNet model.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load the model on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.nn.Module: Loaded UNet model.
    """
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    if torch.cuda.device_count() > 1:
        unet = DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)

    if args.existing_ckpt_filepath is None:
        logger.info("Training from scratch.")
    else:
        checkpoint_unet = torch.load(f"{args.existing_ckpt_filepath}", map_location=device)
        if torch.cuda.device_count() > 1:
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Pretrained checkpoint {args.existing_ckpt_filepath} loaded.")

    return unet


def calculate_scale_factor(train_loader: ThreadDataLoader, device: torch.device, logger: logging.Logger) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset.

    Args:
        train_loader (ThreadDataLoader): Data loader for training.
        device (torch.device): Device to use for calculation.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.Tensor: Calculated scaling factor.
    """
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    logger.info(f"Scaling factor set to {scale_factor}.")

    dist.barrier()
    dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    logger.info(f"Rank {local_rank}: scale_factor -> {scale_factor}.")
    return scale_factor


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Created optimizer.
    """
    return torch.optim.Adam(params=model.parameters(), lr=lr)


def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.PolynomialLR:
    """
    Create learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        total_steps (int): Total number of training steps.

    Returns:
        torch.optim.lr_scheduler.PolynomialLR: Created learning rate scheduler.
    """
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)


def train_one_epoch(
    epoch: int,
    unet: torch.nn.Module,
    train_loader: ThreadDataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    loss_pt: torch.nn.L1Loss,
    scaler: GradScaler,
    scale_factor: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    num_images_per_batch: int,
    num_train_timesteps: int,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
) -> torch.Tensor:
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        train_loader (ThreadDataLoader): Data loader for training.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler.PolynomialLR): Learning rate scheduler.
        loss_pt (torch.nn.L1Loss): Loss function.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        scale_factor (torch.Tensor): Scaling factor.
        noise_scheduler (torch.nn.Module): Noise scheduler.
        num_images_per_batch (int): Number of images per batch.
        num_train_timesteps (int): Number of training timesteps.
        device (torch.device): Device to use for training.
        logger (logging.Logger): Logger for logging information.
        local_rank (int): Local rank for distributed training.

    Returns:
        torch.Tensor: Training loss for the epoch.
    """
    if local_rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, lr {current_lr}.")

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    unet.train()
    for train_data in train_loader:
        current_lr = optimizer.param_groups[0]["lr"]

        _iter += 1
        images = train_data["image"].to(device)
        images = images * scale_factor

        top_region_index_tensor = train_data["top_region_index"].to(device)
        bottom_region_index_tensor = train_data["bottom_region_index"].to(device)
        spacing_tensor = train_data["spacing"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            noise = torch.randn(
                (
                    num_images_per_batch,
                    4,
                    images.size(-3),
                    images.size(-2),
                    images.size(-1),
                ),
                device=device,
            )

            timesteps = torch.randint(
                0,
                num_train_timesteps,
                (images.shape[0],),
                device=images.device,
            ).long()

            noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

            noise_pred = unet(
                x=noisy_latent,
                timesteps=timesteps,
                top_region_index_tensor=top_region_index_tensor,
                bottom_region_index_tensor=bottom_region_index_tensor,
                spacing_tensor=spacing_tensor,
            )

            loss = loss_pt(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        if local_rank == 0:
            logger.info(
                f"[{str(datetime.now())[:19]}] epoch {epoch + 1}, iter {_iter}/{len(train_loader)}, loss: {loss.item():.4f}, lr: {current_lr:.12f}."
            )

    if torch.cuda.device_count() > 1:
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

    return loss_torch


def save_checkpoint(
    epoch: int,
    unet: torch.nn.Module,
    loss_torch_epoch: float,
    num_train_timesteps: int,
    scale_factor: torch.Tensor,
    ckpt_folder: str,
    args: argparse.Namespace,
) -> None:
    """
    Save checkpoint.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        loss_torch_epoch (float): Training loss for the epoch.
        num_train_timesteps (int): Number of training timesteps.
        scale_factor (torch.Tensor): Scaling factor.
        ckpt_folder (str): Checkpoint folder path.
        args (argparse.Namespace): Configuration arguments.
    """
    unet_state_dict = unet.module.state_dict() if torch.cuda.device_count() > 1 else unet.state_dict()
    torch.save(
        {
            "epoch": epoch + 1,
            "loss": loss_torch_epoch,
            "num_train_timesteps": num_train_timesteps,
            "scale_factor": scale_factor,
            "unet_state_dict": unet_state_dict,
        },
        f"{ckpt_folder}/{args.model_filename}",
    )


def diff_model_train(env_config_path: str, model_config_path: str, model_def_path: str) -> None:
    """
    Main function to train a diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    logger = setup_logging()
    local_rank, world_size, device = initialize_distributed()

    logger.info(f"Using {device} of {world_size}")

    if local_rank == 0:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        logger.info(f"[config] data_root -> {args.embedding_base_dir}.")
        logger.info(f"[config] data_list -> {args.json_data_list}.")
        logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
        logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")
        logger.info(f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}.")

        Path(args.model_dir).mkdir(parents=True, exist_ok=True)


    filenames_train = load_filenames(args.json_data_list)
    if local_rank == 0:
        logger.info(f"num_files_train: {len(filenames_train)}")

    train_files = []
    for _i in range(len(filenames_train)):
        str_img = os.path.join(args.embedding_base_dir, filenames_train[_i])
        if not os.path.exists(str_img):
            continue

        str_info = os.path.join(args.embedding_base_dir, filenames_train[_i]) + ".json"
        train_files.append(
            {
                "image": str_img,
                "top_region_index": str_info,
                "bottom_region_index": str_info,
                "spacing": str_info,
            }
        )

    train_files = partition_dataset(
        data=train_files,
        shuffle=True,
        num_partitions=dist.get_world_size(),
        even_divisible=True,
    )[local_rank]

    train_loader = prepare_data(train_files, device, args.diffusion_unet_train["cache_rate"])

    unet = load_unet(args, device, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")

    scale_factor = calculate_scale_factor(train_loader, device, logger)
    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])

    total_steps = (args.diffusion_unet_train["n_epochs"] * len(train_loader.dataset)) / args.diffusion_unet_train["batch_size"]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = torch.nn.L1Loss()
    scaler = GradScaler()

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")

    for epoch in range(args.diffusion_unet_train["n_epochs"]):
        loss_torch = train_one_epoch(
            epoch, unet, train_loader, optimizer, lr_scheduler, loss_pt, scaler,
            scale_factor, noise_scheduler, args.diffusion_unet_train["batch_size"],
            args.noise_scheduler["num_train_timesteps"], device, logger, local_rank
        )

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or local_rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.info(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}.")

            save_checkpoint(epoch, unet, loss_torch_epoch, args.noise_scheduler["num_train_timesteps"], scale_factor, args.model_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_maisi.json",
        help="Path to model configuration file",
    )

    args = parser.parse_args()
    diff_model_train(args.env_config, args.model_config, args.model_def)
