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

import copy
import json
import os
from argparse import Namespace
from datetime import timedelta
from typing import Any, Sequence

import torch
import torch.distributed as dist
from monai.apps.generation.maisi.utils.morphological_ops import dilate, erode
from monai.bundle import ConfigParser
from monai.data import CacheDataset, DataLoader, partition_dataset
from monai.transforms import Compose, EnsureTyped, Lambdad, LoadImaged, Orientationd
from torch import Tensor


def erode_one_img(mask_t: Tensor, filter_size: int | Sequence[int] = 3, pad_value: float = 1.0) -> Tensor:
    """
    Erode 2D/3D binary mask with data type as torch tensor.

    Args:
        mask_t: input 2D/3D binary mask, [M,N] or [M,N,P] torch tensor.
        filter_size: erosion filter size, has to be odd numbers, default to be 3.
        pad_value: the filled value for padding. We need to pad the input before filtering
                   to keep the output with the same size as input. Usually use default value
                   and not changed.

    Return:
        Tensor: eroded mask, same shape as input.
    """
    return (
        erode(
            mask_t.float()
            .unsqueeze(0)
            .unsqueeze(
                0,
            ),
            filter_size,
            pad_value=pad_value,
        )
        .squeeze(0)
        .squeeze(0)
    )


def dilate_one_img(mask_t: Tensor, filter_size: int | Sequence[int] = 3, pad_value: float = 0.0) -> Tensor:
    """
    Dilate 2D/3D binary mask with data type as torch tensor.

    Args:
        mask_t: input 2D/3D binary mask, [M,N] or [M,N,P] torch tensor.
        filter_size: dilation filter size, has to be odd numbers, default to be 3.
        pad_value: the filled value for padding. We need to pad the input before filtering
                   to keep the output with the same size as input. Usually use default value
                   and not changed.

    Return:
        Tensor: dilated mask, same shape as input.
    """
    return (
        dilate(
            mask_t.float()
            .unsqueeze(0)
            .unsqueeze(
                0,
            ),
            filter_size,
            pad_value=pad_value,
        )
        .squeeze(0)
        .squeeze(0)
    )


def binarize_labels(x: Tensor, bits: int = 8) -> Tensor:
    """
    Apply binary encoding to integer segmentation mask.

    Args:
        x (Tensor): the input tensor with shape (B, 1, H, W, D).
        bits (int, optional): the num of channel to represent the data. Defaults to 8.

    Returns:
        Tensor: encoded mask
    """
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().squeeze(1).permute(0, 4, 1, 2, 3)


def setup_ddp(rank: int, world_size: int) -> torch.device:
    """
    Initialize the distributed process group.

    Args:
        rank (int): rank of the current process.
        world_size (int): number of processes participating in the job.

     Returns:
        torch.device: device of the current process.
    """
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return device


def define_instance(args: Namespace, instance_def_key: str) -> Any:
    """
    Get the parsed instance based on provided attributes.

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
    """
    Read a list of data dictionary.

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
    """
    Prepare dataloaders for training and validation.

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
    use_ddp = world_size > 1
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
        Lambdad(keys=["top_region_index", "bottom_region_index", "spacing"], func=lambda x: x * 1e2),
    ]
    train_transforms, val_transforms = Compose(common_transform), Compose(common_transform)

    train_loader = None

    if use_ddp:
        list_train = partition_dataset(
            data=list_train,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=True,
        )[rank]
    train_ds = CacheDataset(data=list_train, transform=train_transforms, cache_rate=cache_rate, num_workers=8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    if use_ddp:
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
