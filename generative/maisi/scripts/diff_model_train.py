# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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
import yaml

from datetime import datetime, timedelta
from pathlib import Path

import monai
from monai.data import (
    ThreadDataLoader,
    partition_dataset,
)
from monai.transforms import Compose
from monai.utils import first

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast

from utils import define_instance


def diff_model_train(env_config_path: str, model_config_path: str) -> None:
    """
    Main function to train a diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
    """
    args = argparse.Namespace()

    # Load environment configuration
    with open(env_config_path, "r") as f:
        env_config = json.load(f)
    
    for k, v in env_config.items():
        setattr(args, k, v)

    # Load model configuration
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    for k, v in model_config.items():
        setattr(args, k, v)

    ckpt_folder = args.model_dir
    data_root = args.embedding_base_dir
    data_list = args.json_data_list
    existing_ckpt_filepath = args.existing_ckpt_filepath

    lr = args.diffusion_unet_train["lr"]
    num_epochs = args.diffusion_unet_train["n_epochs"]
    num_train_timesteps = args.noise_scheduler["num_train_timesteps"]

    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=7200))

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(dist.get_world_size())
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    print(f"Using {device} of {world_size}")

    if local_rank == 0:
        print(f"[config] ckpt_folder -> {ckpt_folder}.")
        print(f"[config] data_root -> {data_root}.")
        print(f"[config] data_list -> {data_list}.")
        print(f"[config] lr -> {lr}.")
        print(f"[config] num_epochs -> {num_epochs}.")
        print(f"[config] num_train_timesteps -> {num_train_timesteps}.")

        Path(ckpt_folder).mkdir(parents=True, exist_ok=True)

    with open(args.json_data_list, "r") as file:
        json_data = json.load(file)
    filenames_train = json_data["training"]
    filenames_train = [
        _item["image"].replace(".nii.gz", "_emb.nii.gz") for _item in filenames_train
    ]

    if local_rank == 0:
        print(f"num_files_train: {len(filenames_train)}")

    # Training data preparation
    train_files = []
    for _i in range(len(filenames_train)):
        str_img = os.path.join(data_root, filenames_train[_i])
        if not os.path.exists(str_img):
            continue

        str_info = os.path.join(data_root, filenames_train[_i]) + ".json"
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

    cache_rate = args.diffusion_unet_train["cache_rate"]
    print(f"cache_rate: {cache_rate}")
    train_ds = monai.data.CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=2,
    )

    num_images_per_batch = args.diffusion_unet_train["batch_size"]
    print(f"num_images_per_batch -> {num_images_per_batch}.")
    train_loader = ThreadDataLoader(train_ds, num_workers=6, batch_size=num_images_per_batch, shuffle=True)

    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        unet = DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)

    if existing_ckpt_filepath == None:
        print("training from scratch.")
    else:
        checkpoint_unet = torch.load(f"{existing_ckpt_filepath}", map_location=device)
        if torch.cuda.device_count() > 1:
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        print(f"pretrained checkpoint {existing_ckpt_filepath} loaded.")

    noise_scheduler = define_instance(args, "noise_scheduler")

    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}.")

    dist.barrier()
    dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    print(f"Rank {local_rank}: scale_factor -> {scale_factor}.")

    optimizer = torch.optim.Adam(params=unet.parameters(), lr=lr)
    print(f"optimizer -> Adam; lr -> {lr}.")

    total_steps = (num_epochs * len(train_loader.dataset)) / num_images_per_batch
    print(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    loss_pt = torch.nn.L1Loss()
    best_loss = 1e4

    scaler = GradScaler()

    torch.set_float32_matmul_precision("highest")
    print("torch.set_float32_matmul_precision -> highest.")

    for epoch in range(num_epochs):
        if local_rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch + 1}/{num_epochs}, lr {current_lr}.")

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
                    num_images_per_batch,
                    4,
                    images.size(-3),
                    images.size(-2),
                    images.size(-1),
                ).to(device)

                timesteps = torch.randint(
                    0,
                    num_train_timesteps,
                    (images.shape[0],),
                    device=images.device,
                ).long()

                noisy_latent = noise_scheduler.add_noise(
                    original_samples=images,
                    noise=noise,
                    timesteps=timesteps
                )

                noise_pred = unet(
                    x=noisy_latent,
                    timesteps=timesteps,
                    top_region_index_tensor=top_region_index_tensor,
                    bottom_region_index_tensor=bottom_region_index_tensor,
                    spacing_tensor=spacing_tensor
                )

                loss = loss_pt(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()

            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0

            if local_rank == 0:
                print(
                    f"[{str(datetime.now())[:19]}] epoch {epoch + 1}, iter {_iter}/{len(train_loader)}, loss: {loss.item():.4f}, lr: {current_lr:.12f}."
                )

        if torch.cuda.device_count() > 1:
            dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or local_rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            print(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}.")

        if local_rank == 0:
            unet_state_dict = unet.module.state_dict()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument("--env_config", type=str, required=True, help="Path to environment configuration file")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model configuration file")

    args = parser.parse_args()

    diff_model_train(args.env_config, args.model_config)
