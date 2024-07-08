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

import fire
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

from inferer import DiffusionInferer, LatentDiffusionInferer
from custom_network_diffusion import CustomDiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler


def main(
    ckpt_folder="./models",
    ckpt_prefix="unet3d",
    data_root="./data",
    data_list="./dataset_image.json",
    data_list_order="sorted",
    ignore_prev_loss=False,
    output_size=512,
    pretrained_ckpt_filepath="scratch",
    lr=1e-4,
    num_epochs=100,
    num_training_data=1024,
    num_train_timesteps=1e3,
    scheduler_method="ddpm",
):
    """
    Main function to train a diffusion model.

    Args:
        ckpt_folder (str): Directory to save checkpoints.
        ckpt_prefix (str): Prefix for checkpoint filenames.
        data_root (str): Root directory of the dataset.
        data_list (str): Path to the dataset list file.
        data_list_order (str): Order of the data list ('sorted' or 'random').
        ignore_prev_loss (bool): Flag to ignore previous loss.
        output_size (int): Output size of the images.
        pretrained_ckpt_filepath (str): Path to the pretrained checkpoint file.
        lr (float): Learning rate for the optimizer.
        num_epochs (int): Number of training epochs.
        num_training_data (int): Number of training data samples.
        num_train_timesteps (int): Number of training timesteps for the scheduler.
        scheduler_method (str): Scheduler method ('ddpm' or 'ddim').
    """
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=7200))

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(dist.get_world_size())
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    print(f"Using {device} of {world_size}")

    if local_rank == 0:
        print(f"[config] ckpt_folder -> {ckpt_folder}.")
        print(f"[config] ckpt_prefix -> {ckpt_prefix}.")
        print(f"[config] data_root -> {data_root}.")
        print(f"[config] data_list -> {data_list}.")
        print(f"[config] data_list_order -> {data_list_order}.")
        print(f"[config] ignore_prev_loss -> {ignore_prev_loss}.")
        print(f"[config] lr -> {lr}.")
        print(f"[config] num_epochs -> {num_epochs}.")
        print(f"[config] num_training_data -> {num_training_data}.")
        print(f"[config] num_train_timesteps -> {num_train_timesteps}.")
        print(f"[config] output_size -> {output_size}.")
        print(f"[config] pretrained_ckpt_filepath -> {pretrained_ckpt_filepath}.")
        print(f"[config] scheduler_method -> {scheduler_method}.")

        Path(ckpt_folder).mkdir(parents=True, exist_ok=True)

    with open(data_list, "r") as file:
        lines = file.readlines()
    filenames = [_item.strip() for _item in lines]
    filenames.sort()
    num_files = len(filenames)
    filenames_train = filenames

    if local_rank == 0:
        print(f"num_files: {num_files}")
        print(f"num_files_train: {len(filenames_train)}")

    # Training data preparation
    files = []
    for _i in range(len(filenames_train)):
        str_img = os.path.join(data_root, filenames_train[_i])
        if not os.path.exists(str_img):
            continue

        str_info = os.path.join(data_root, filenames_train[_i].replace("_emb.nii.gz", "_image.nii.gz.json"))
        files.append(
            {
                "image": str_img,
                "top_region_index": str_info,
                "bottom_region_index": str_info,
                "spacing": str_info,
            }
        )

    if isinstance(num_training_data, str) and num_training_data.lower() == "all":
        train_files = files
        train_files = partition_dataset(
            data=train_files,
            shuffle=True,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
        )[local_rank]
    else:
        if data_list_order.lower() == "sorted":
            with open("analyze_data_z.yaml") as file:
                _analyze_data = yaml.load(file, Loader=yaml.FullLoader)
            case_names = _analyze_data.keys()
            sorted_case_names = sorted(case_names, key=_analyze_data.get)

            case_names_subset = sorted_case_names[-num_training_data:]
            print(f"case_names_subset {len(case_names_subset)}")

            train_files = []
            for case_name in case_names_subset:
                for _ii in range(len(files)):
                    if case_name + ".nii.gz" in files[_ii]["image"]:
                        train_files.append(files[_ii])
                        break
            print(f"train_files {len(train_files)}")
        elif data_list_order.lower() == "random":
            train_files = files[:num_training_data]

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

    cache_rate = 0.0
    print(f"cache_rate: {cache_rate}")
    train_ds = monai.data.CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=2,
    )

    num_images_per_batch = 1
    print(f"num_images_per_batch -> {num_images_per_batch}.")
    train_loader = ThreadDataLoader(train_ds, num_workers=6, batch_size=num_images_per_batch, shuffle=True)

    unet = CustomDiffusionModelUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        num_channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_head_channels=(0, 0, 32, 32),
        num_res_blocks=2,
        use_flash_attention=True,
        input_top_region_index=True,
        input_bottom_region_index=True,
        input_spacing=True,
    )
    unet.to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        unet = DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)

    if pretrained_ckpt_filepath == "scratch":
        print("training from scratch.")
    else:
        checkpoint_unet = torch.load(f"{pretrained_ckpt_filepath}", map_location=device)
        if torch.cuda.device_count() > 1:
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        print(f"pretrained checkpoint {pretrained_ckpt_filepath} loaded.")

    if scheduler_method == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            schedule="scaled_linear_beta",
            beta_start=0.0015,
            beta_end=0.0195,
            clip_sample=False,
        )
    elif scheduler_method == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            schedule="scaled_linear_beta",
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )
    print(f"scheduler_method -> {scheduler_method}.")

    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}.")

    dist.barrier()
    dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    print(f"Rank {local_rank}: scale_factor -> {scale_factor}.")

    if False:
        scale_factor = 1.0
        inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    else:
        inferer = DiffusionInferer(scheduler)

    optimizer = torch.optim.Adam(params=unet.parameters(), lr=lr)
    print(f"optimizer -> Adam; lr -> {lr}.")

    total_steps = (num_epochs * len(train_loader.dataset)) / num_images_per_batch
    print(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    loss_pt = torch.nn.L1Loss()

    if ignore_prev_loss:
        best_loss = 1e4
    else:
        try:
            best_loss = min(1e4, checkpoint_unet["loss"])
        except:
            best_loss = 1e4
    print(f"current loss -> {best_loss}.")
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
                    inferer.scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=images.device,
                ).long()

                noise_pred = inferer(
                    inputs=images,
                    diffusion_model=unet,
                    noise=noise,
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
            unet_state_dict = unet.module.state_dict() if world_size > 1 else unet.state_dict()

            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": loss_torch_epoch,
                    "num_train_timesteps": num_train_timesteps,
                    "scale_factor": scale_factor,
                    "scheduler_method": scheduler_method,
                    "output_size": output_size,
                    "unet_state_dict": unet_state_dict,
                },
                f"{ckpt_folder}/{ckpt_prefix}_current.pt",
            )

            if loss_torch_epoch < best_loss:
                best_loss = loss_torch_epoch if loss_torch_epoch < best_loss else best_loss
                print(f"best loss -> {best_loss}.")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "loss": best_loss,
                        "num_train_timesteps": num_train_timesteps,
                        "scale_factor": scale_factor,
                        "scheduler_method": scheduler_method,
                        "output_size": output_size,
                        "unet_state_dict": unet_state_dict,
                    },
                    f"{ckpt_folder}/{ckpt_prefix}_best.pt",
                )


if __name__ == "__main__":
    fire.Fire(main)
