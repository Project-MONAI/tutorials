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

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from monai.apps import DecathlonDataset
from monai.bundle import ConfigParser
from monai.data import DataLoader
from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    ScaleIntensityRangePercentilesd,
    SplitDimd,
    SqueezeDimd,
)


def setup_ddp(rank, world_size):
    print(f"Running DDP diffusion example on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=36000),
        rank=rank,
        world_size=world_size,
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device


def prepare_brats2d_dataloader(
    args,
    batch_size,
    patch_size,
    amp=False,
    sample_axis=2,
    randcrop=True,
    rank=0,
    world_size=1,
    cache=1.0,
    download=False,
    size_divisible=4,
    num_center_slice=80,
):
    ddp_bool = world_size > 1
    channel = args.channel  # 0 = Flair, 1 = T1
    assert channel in [0, 1, 2, 3], "Choose a valid channel"

    if sample_axis == 0:
        # sagittal
        train_patch_size = [1] + patch_size
        val_patch_size = [num_center_slice] + patch_size
        size_divisible_3d = [1, size_divisible, size_divisible]
    elif sample_axis == 1:
        # coronal
        train_patch_size = [patch_size[0], 1, patch_size[1]]
        val_patch_size = [patch_size[0], num_center_slice, patch_size[1]]
        size_divisible_3d = [size_divisible, 1, size_divisible]
    elif sample_axis == 2:
        # axial
        train_patch_size = patch_size + [1]
        val_patch_size = patch_size + [num_center_slice]
        size_divisible_3d = [size_divisible, size_divisible, 1]
    else:
        raise ValueError("sample_axis has to be in [0,1,2]")

    if randcrop:
        train_crop_transform = RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=train_patch_size,
            random_size=False,
            num_samples=batch_size,
        )
    else:
        train_crop_transform = CenterSpatialCropd(keys=["image"], roi_size=val_patch_size)

    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            AddChanneld(keys=["image"]),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            CenterSpatialCropd(keys=["image"], roi_size=val_patch_size),
            DivisiblePadd(keys=["image"], k=size_divisible_3d),
            ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=100.0, b_min=-1, b_max=1),
            train_crop_transform,
            SqueezeDimd(keys="image", dim=1 + sample_axis),
            EnsureTyped(keys="image", dtype=compute_dtype),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            AddChanneld(keys=["image"]),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            CenterSpatialCropd(keys=["image"], roi_size=val_patch_size),
            DivisiblePadd(keys=["image"], k=size_divisible_3d),
            ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=100.0, b_min=-1, b_max=1),
            SplitDimd(keys=["image"], dim=1 + sample_axis, keepdim=False, list_output=True),
            EnsureTyped(keys="image", dtype=compute_dtype),
        ]
    )
    os.makedirs(args.data_base_dir, exist_ok=True)
    train_ds = DecathlonDataset(
        root_dir=args.data_base_dir,
        task="Task01_BrainTumour",
        section="training",  # validation
        cache_rate=cache,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=8,
        download=download,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
        transform=train_transforms,
    )
    val_ds = DecathlonDataset(
        root_dir=args.data_base_dir,
        task="Task01_BrainTumour",
        section="validation",  # validation
        cache_rate=cache,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=8,
        download=download,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
        transform=val_transforms,
    )
    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=(not ddp_bool),
        num_workers=0,
        pin_memory=False,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=val_sampler,
    )
    if rank == 0:
        print(f'Image shape {train_ds[0][0]["image"].shape}')
    return train_loader, val_loader


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]
