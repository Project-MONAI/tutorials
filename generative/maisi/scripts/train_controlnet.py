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
from argparse import Namespace
import json
import logging
from pathlib import Path
import time
from datetime import timedelta
from typing import Any
import os
import sys
import copy
import torch
import torch.distributed as dist
import torch.nn.functional as F
import logging
from monai.utils import RankFilter
from monai.data import DataLoader, CacheDataset, partition_dataset
from monai.networks.utils import copy_model_state
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Compose,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
)
from monai.bundle import ConfigParser
from utils import binarize_labels
import logging


def setup_ddp(rank: int, world_size: int) -> torch.device:
    """Initialize the distributed process group.

    Args:
        rank (int): rank of the current process.
        world_size (int): number of processes participating in the job.
    """
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return device


def define_instance(args: Namespace, instance_def_key: str) -> Any:
    """Get the parsed instance based on provided attributes.

    Args:
        args (Namespace): a object for storing attributes.
        instance_def_key (str): key associated to target instance.

    Returns:
        Any: parsed instance.
    """
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def add_data_dir2path(list_files: list, data_dir: str, fold: int = None) -> tuple[list, list]:
    """Read a list of data dictionary.

    Args:
        list_files (list): input data to load and transform to generate dataset for model.
        data_dir (str): directory of files.
        fold (int, optional): fold index for cross validation. Defaults to None.

    Returns:
        tuple[list, list]: A tuple of two arrays (training, validation).
    """
    new_list_files = copy.deepcopy(list_files)
    if fold is not None:
        new_list_files_train = []
        new_list_files_val = []
    for d in new_list_files:
        d["image"] = os.path.join(data_dir, d["image"])

        if "label" in d:
            d["label"] = os.path.join(data_dir, d["label"])

        if fold is not None:
            if d["fold"] == fold:
                new_list_files_val.append(copy.deepcopy(d))
            else:
                new_list_files_train.append(copy.deepcopy(d))

    if fold is not None:
        return new_list_files_train, new_list_files_val
    else:
        return new_list_files, []


def prepare_maisi_controlnet_json_dataloader(
    json_data_list: list | str,
    data_base_dir: list | str,
    batch_size: int = 1,
    fold: int = 0,
    cache_rate: float = 0.0,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """Prepare dataloaders for training and validation.

    Args:
        json_data_list (list | str): the name of JSON files listing the data.
        data_base_dir (list | str): directory of files.
        batch_size (int, optional): how many samples per batch to load . Defaults to 1.
        fold (int, optional): fold index for cross validation. Defaults to 0.
        cache_rate (float, optional): percentage of cached data in total. Defaults to 0.0.
        rank (int, optional): rank of the current process. Defaults to 0.
        world_size (int, optional): number of processes participating in the job. Defaults to 1.

    Returns:
        tuple[DataLoader, DataLoader]:  A tuple of two dataloaders (training, validation).
    """
    ddp_bool = world_size > 1
    if isinstance(json_data_list, list):
        assert isinstance(data_base_dir, list)
        list_train = []
        list_valid = []
        for data_list, data_root in zip(json_data_list, data_base_dir):
            with open(data_list, "r") as f:
                json_data = json.load(f)["training"]
            train, val = add_data_dir2path(json_data, data_root, fold)
            list_train += train
            list_valid += val
    else:
        with open(json_data_list, "r") as f:
            json_data = json.load(f)["training"]
        list_train, list_valid = add_data_dir2path(json_data, data_base_dir, fold)

    common_transform = [
        LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
        Orientationd(keys=["label"], axcodes="RAS"),
        EnsureTyped(keys=["label"], dtype=torch.uint8, track_meta=True),
        Lambdad(keys="top_region_index", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys="bottom_region_index", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
        Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
        Lambdad(keys="spacing", func=lambda x: x * 1e2),
    ]
    train_transforms, val_transforms = Compose(common_transform), Compose(common_transform)

    train_loader = None

    if ddp_bool:
        list_train = partition_dataset(
            data=list_train,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=True,
        )[rank]
    train_ds = CacheDataset(data=list_train, transform=train_transforms, cache_rate=cache_rate, num_workers=8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    if ddp_bool:
        list_valid = partition_dataset(
            data=list_valid,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=False,
        )[rank]
    val_ds = CacheDataset(
        data=list_valid,
        transform=val_transforms,
        cache_rate=cache_rate,
        num_workers=8,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="maisi.controlnet.training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment_maisi_controlnet_train.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_maisi_controlnet_train.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    # Step 0: configuration
    logger = logging.getLogger("maisi.controlnet.training")
    # whether to use distributed data parallel
    ddp_bool = args.gpus > 1
    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World_size: {world_size}")

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    # initialize tensorboard writer
    if rank == 0:
        tensorboard_path = os.path.join(args.tfevent_path, args.exp_name)
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)

    # Step 1: set data loader
    train_loader, _ = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
    )

    # Step 2: define diffusion model and controlnet
    # define diffusion Model
    unet = define_instance(args, "difusion_unet_def").to(device)
    # load trained diffusion model
    diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device)
    unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"])
    # load scale factor
    scale_factor = diffusion_model_ckpt["scale_factor"]
    logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")
    logger.info(f"loaded scale_factor from diffusion model ckpt -> {scale_factor}.")
    # define ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    # copy weights from the DM to the controlnet
    copy_model_state(controlnet, unet.state_dict())
    # load trained controlnet model if it is provided
    if args.trained_controlnet_path is not None:
        controlnet.load_state_dict(
            torch.load(args.trained_controlnet_path, map_location=device)["controlnet_state_dict"]
        )
        logger.info(f"load trained controlnet model from {args.trained_controlnet_path}")
    else:
        logger.info("train controlnet model from scratch.")
    # we freeze the parameters of the diffusion model.
    for p in unet.parameters():
        p.requires_grad = False

    noise_scheduler = define_instance(args, "noise_scheduler")

    if ddp_bool:
        controlnet = DDP(controlnet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # Step 3: training config
    weighted_loss = args.controlnet_train["weighted_loss"]
    weighted_loss_label = args.controlnet_train["weighted_loss_label"]
    optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=args.controlnet_train["lr"])
    total_steps = (args.controlnet_train["n_epochs"] * len(train_loader.dataset)) / args.controlnet_train["batch_size"]
    logger.info(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    # Step 4: training
    n_epochs = args.controlnet_train["n_epochs"]
    scaler = GradScaler()
    total_step = 0
    best_loss = 1e4

    if weighted_loss > 0:
        logger.info(f"apply weighted loss = {weighted_loss} on labels: {weighted_loss_label}")

    controlnet.train()
    unet.eval()
    prev_time = time.time()
    for epoch in range(n_epochs):
        epoch_loss_ = 0
        for step, batch in enumerate(train_loader):
            # get image embedding and label mask and scale image embedding by the provided scale_factor
            inputs = batch["image"].to(device) * scale_factor
            labels = batch["label"].to(device)
            # get corresponding conditions
            top_region_index_tensor = batch["top_region_index"].to(device)
            bottom_region_index_tensor = batch["bottom_region_index"].to(device)
            spacing_tensor = batch["spacing"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # generate random noise
                noise_shape = list(inputs.shape)
                noise = torch.randn(noise_shape, dtype=inputs.dtype).to(device)

                # use binary encoding to encode segmentation mask
                controlnet_cond = binarize_labels(labels.as_tensor().to(torch.uint8)).float()

                # create timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (inputs.shape[0],), device=device
                ).long()

                # create noisy latent
                noisy_latent = noise_scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)

                # get controlnet output
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=noisy_latent, timesteps=timesteps, controlnet_cond=controlnet_cond
                )
                # get noise prediction from diffusion unet
                noise_pred = unet(
                    x=noisy_latent,
                    timesteps=timesteps,
                    top_region_index_tensor=top_region_index_tensor,
                    bottom_region_index_tensor=bottom_region_index_tensor,
                    spacing_tensor=spacing_tensor,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

            if weighted_loss > 1.0:
                weights = torch.ones_like(inputs).to(inputs.device)
                roi = torch.zeros([noise_shape[0]] + [1] + noise_shape[2:]).to(inputs.device)
                interpolate_label = F.interpolate(labels, size=inputs.shape[2:], mode="nearest")
                # assign larger weights for ROI (tumor)
                for label in weighted_loss_label:
                    roi[interpolate_label == label] = 1
                weights[roi.repeat(1, inputs.shape[1], 1, 1, 1) == 1] = weighted_loss
                loss = (F.l1_loss(noise_pred.float(), noise.float(), reduction="none") * weights).mean()
            else:
                loss = F.l1_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_step += 1

            if rank == 0:
                # write train loss for each batch into tensorboard
                tensorboard_writer.add_scalar(
                    "train/train_controlnet_loss_iter", loss.detach().cpu().item(), total_step
                )
                batches_done = step + 1
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                logger.info(
                    "\r[Epoch %d/%d] [Batch %d/%d] [LR: %.8f] [loss: %.4f] ETA: %s "
                    % (
                        epoch + 1,
                        n_epochs,
                        step + 1,
                        len(train_loader),
                        lr_scheduler.get_last_lr()[0],
                        loss.detach().cpu().item(),
                        time_left,
                    )
                )
            epoch_loss_ += loss.detach()

        epoch_loss = epoch_loss_ / (step + 1)

        if ddp_bool:
            dist.barrier()
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.AVG)

        if rank == 0:
            tensorboard_writer.add_scalar("train/train_controlnet_loss_epoch", epoch_loss.cpu().item(), total_step)
            # save controlnet only on master GPU (rank 0)
            controlnet_state_dict = controlnet.module.state_dict() if world_size > 1 else controlnet.state_dict()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "controlnet_state_dict": controlnet_state_dict,
                },
                f"{args.model_dir}/{args.exp_name}_current.pt",
            )

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                logger.info(f"best loss -> {best_loss}.")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "loss": best_loss,
                        "controlnet_state_dict": controlnet_state_dict,
                    },
                    f"{args.model_dir}/{args.exp_name}_best.pt",
                )

        torch.cuda.empty_cache()
    if ddp_bool:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
