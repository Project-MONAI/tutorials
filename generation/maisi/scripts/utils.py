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
import logging
import math
import os
from argparse import Namespace
from datetime import timedelta
from typing import Any, Sequence

import numpy as np
import skimage
import torch
import torch.distributed as dist
from monai.bundle import ConfigParser
from monai.config import DtypeLike, NdarrayOrTensor
from monai.data import CacheDataset, DataLoader, partition_dataset
from monai.transforms import Compose, EnsureTyped, Lambdad, LoadImaged, Orientationd
from monai.transforms.utils_morphological_ops import dilate, erode
from monai.utils import TransformBackends, convert_data_type, convert_to_dst_type, get_equivalent_dtype
from scipy import stats
from torch import Tensor


def remap_labels(mask, label_dict_remap_json):
    """
    Remap labels in the mask according to the provided label dictionary.

    This function reads a JSON file containing label mapping information and applies
    the mapping to the input mask.

    Args:
        mask (Tensor): The input mask tensor to be remapped.
        label_dict_remap_json (str): Path to the JSON file containing the label mapping dictionary.

    Returns:
        Tensor: The remapped mask tensor.
    """
    with open(label_dict_remap_json, "r") as f:
        mapping_dict = json.load(f)
    mapper = MapLabelValue(
        orig_labels=[pair[0] for pair in mapping_dict.values()],
        target_labels=[pair[1] for pair in mapping_dict.values()],
        dtype=torch.uint8,
    )
    return mapper(mask[0, ...])[None, ...].to(mask.device)


def get_index_arr(img):
    """
    Generate an index array for the given image.

    This function creates a 3D array of indices corresponding to the dimensions of the input image.

    Args:
        img (ndarray): The input image array.

    Returns:
        ndarray: A 3D array containing the indices for each dimension of the input image.
    """
    return np.moveaxis(
        np.moveaxis(
            np.stack(np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]))), 0, 3
        ),
        0,
        1,
    )


def supress_non_largest_components(img, target_label, default_val=0):
    """
    Suppress all components except the largest one(s) for specified target labels.

    This function identifies the largest component(s) for each target label and
    suppresses all other smaller components.

    Args:
        img (ndarray): The input image array.
        target_label (list): List of label values to process.
        default_val (int, optional): Value to assign to suppressed voxels. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - ndarray: Modified image with non-largest components suppressed.
            - int: Number of voxels that were changed.
    """
    index_arr = get_index_arr(img)
    img_mod = copy.deepcopy(img)
    new_background = np.zeros(img.shape, dtype=np.bool_)
    for label in target_label:
        label_cc = skimage.measure.label(img == label, connectivity=3)
        uv, uc = np.unique(label_cc, return_counts=True)
        dominant_vals = uv[np.argsort(uc)[::-1][:2]]
        if len(dominant_vals) >= 2:  # Case: no predictions
            new_background = np.logical_or(
                new_background,
                np.logical_not(np.logical_or(label_cc == dominant_vals[0], label_cc == dominant_vals[1])),
            )

    for voxel in index_arr[new_background]:
        img_mod[tuple(voxel)] = default_val
    diff = np.sum((img - img_mod) > 0)

    return img_mod, diff


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
    Convert input tensor to binary representation.

    This function takes an input tensor and converts it to a binary representation
    using the specified number of bits.

    Args:
        x (Tensor): Input tensor with shape (B, 1, H, W, D).
        bits (int, optional): Number of bits to use for binary representation. Defaults to 8.

    Returns:
        Tensor: Binary representation of the input tensor with shape (B, bits, H, W, D).
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
    Define and instantiate an object based on the provided arguments and instance definition key.

    This function uses a ConfigParser to parse the arguments and instantiate an object
    defined by the instance_def_key.

    Args:
        args: An object containing the arguments to be parsed.
        instance_def_key (str): The key used to retrieve the instance definition from the parsed content.

    Returns:
        The instantiated object as defined by the instance_def_key in the parsed configuration.
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
        Lambdad(keys="top_region_index", func=lambda x: torch.FloatTensor(x), allow_missing_keys=True),
        Lambdad(keys="bottom_region_index", func=lambda x: torch.FloatTensor(x), allow_missing_keys=True),
        Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
        Lambdad(
            keys=["top_region_index", "bottom_region_index", "spacing"], func=lambda x: x * 1e2, allow_missing_keys=True
        ),
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


def organ_fill_by_closing(data, target_label, device, close_times=2, filter_size=3, pad_value=0.0):
    """
    Fill holes in an organ mask using morphological closing operations.

    This function performs a series of dilation and erosion operations to fill holes
    in the organ mask identified by the target label.

    Args:
        data (ndarray): The input data containing organ labels.
        target_label (int): The label of the organ to be processed.
        device (str): The device to perform the operations on (e.g., 'cuda:0').
        close_times (int, optional): Number of times to perform the closing operation. Defaults to 2.
        filter_size (int, optional): Size of the filter for dilation and erosion. Defaults to 3.
        pad_value (float, optional): Value used for padding in dilation and erosion. Defaults to 0.0.

    Returns:
        ndarray: Boolean mask of the filled organ.
    """
    mask = (data == target_label).astype(np.uint8)
    mask = torch.from_numpy(mask).to(device)
    for _ in range(close_times):
        mask = dilate_one_img(mask, filter_size=filter_size, pad_value=pad_value)
        mask = erode_one_img(mask, filter_size=filter_size, pad_value=pad_value)
    return mask.cpu().numpy().astype(np.bool_)


def organ_fill_by_removed_mask(data, target_label, remove_mask, device):
    """
    Fill an organ mask in regions where it was previously removed.

    Args:
        data (ndarray): The input data containing organ labels.
        target_label (int): The label of the organ to be processed.
        remove_mask (ndarray): Boolean mask indicating regions where the organ was removed.
        device (str): The device to perform the operations on (e.g., 'cuda:0').

    Returns:
        ndarray: Boolean mask of the filled organ in previously removed regions.
    """
    mask = (data == target_label).astype(np.uint8)
    mask = dilate_one_img(torch.from_numpy(mask).to(device), filter_size=3, pad_value=0.0)
    mask = dilate_one_img(mask, filter_size=3, pad_value=0.0)
    roi_oragn_mask = dilate_one_img(mask, filter_size=3, pad_value=0.0).cpu().numpy()
    return (roi_oragn_mask * remove_mask).astype(np.bool_)


def get_body_region_index_from_mask(input_mask):
    """
    Determine the top and bottom body region indices from an input mask.

    Args:
        input_mask (Tensor): Input mask tensor containing body region labels.

    Returns:
        tuple: Two lists representing the top and bottom region indices.
    """
    region_indices = {}
    # head and neck
    region_indices["region_0"] = [22, 120]
    # thorax
    region_indices["region_1"] = [28, 29, 30, 31, 32]
    # abdomen
    region_indices["region_2"] = [1, 2, 3, 4, 5, 14]
    # pelvis and lower
    region_indices["region_3"] = [93, 94]

    nda = input_mask.cpu().numpy().squeeze()
    unique_elements = np.lib.arraysetops.unique(nda)
    unique_elements = list(unique_elements)
    # print(f"nda: {nda.shape} {unique_elements}.")
    overlap_array = np.zeros(len(region_indices), dtype=np.uint8)
    for _j in range(len(region_indices)):
        overlap = any(element in region_indices[f"region_{_j}"] for element in unique_elements)
        overlap_array[_j] = np.uint8(overlap)
    overlap_array_indices = np.nonzero(overlap_array)[0]
    top_region_index = np.eye(len(region_indices), dtype=np.uint8)[np.amin(overlap_array_indices), ...]
    top_region_index = list(top_region_index)
    top_region_index = [int(_k) for _k in top_region_index]
    bottom_region_index = np.eye(len(region_indices), dtype=np.uint8)[np.amax(overlap_array_indices), ...]
    bottom_region_index = list(bottom_region_index)
    bottom_region_index = [int(_k) for _k in bottom_region_index]
    # print(f"{top_region_index} {bottom_region_index}")
    return top_region_index, bottom_region_index


def general_mask_generation_post_process(volume_t, target_tumor_label=None, device="cuda:0"):
    """
    Perform post-processing on a generated mask volume.

    This function applies various refinement steps to improve the quality of the generated mask,
    including body mask refinement, tumor prediction refinement, and organ-specific processing.

    Args:
        volume_t (ndarray): Input volume containing organ and tumor labels.
        target_tumor_label (int, optional): Label of the target tumor. Defaults to None.
        device (str, optional): Device to perform operations on. Defaults to "cuda:0".

    Returns:
        ndarray: Post-processed volume with refined organ and tumor labels.
    """
    # assume volume_t is np array with shape (H,W,D)
    hepatic_vessel = volume_t == 25
    airway = volume_t == 132

    # ------------ refine body mask pred
    body_region_mask = (
        erode_one_img(torch.from_numpy((volume_t > 0)).to(device), filter_size=3, pad_value=0.0).cpu().numpy()
    )
    body_region_mask, _ = supress_non_largest_components(body_region_mask, [1])
    body_region_mask = (
        dilate_one_img(torch.from_numpy(body_region_mask).to(device), filter_size=3, pad_value=0.0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    volume_t = volume_t * body_region_mask

    # ------------ refine tumor pred
    tumor_organ_dict = {23: 28, 24: 4, 26: 1, 27: 62, 128: 200}
    for t in [23, 24, 26, 27, 128]:
        if t != target_tumor_label:
            volume_t[volume_t == t] = tumor_organ_dict[t]
        else:
            volume_t[organ_fill_by_closing(volume_t, target_label=t, device=device)] = t
            volume_t[organ_fill_by_closing(volume_t, target_label=t, device=device)] = t
    # we only keep the largest connected componet for tumors except hepatic tumor and bone lesion
    if target_tumor_label != 26 and target_tumor_label != 128:
        volume_t, _ = supress_non_largest_components(volume_t, [target_tumor_label], default_val=200)
    target_tumor = volume_t == target_tumor_label

    # ------------ remove undesired organ pred
    # general post-process non-largest components suppression
    # process 4 ROI organs + spleen + 2 kidney + 5 lung lobes + duodenum + inferior vena cava
    oran_list = [1, 4, 10, 12, 3, 28, 29, 30, 31, 32, 5, 14, 13, 6, 7, 8, 9, 10]
    if target_tumor_label != 128:
        oran_list += list(range(33, 60))  # + list(range(63,87))
    data, _ = supress_non_largest_components(volume_t, oran_list, default_val=200)  # 200 is body region
    organ_remove_mask = (volume_t - data).astype(np.bool_)
    # process intestinal system (stomach 12, duodenum 13, small bowel 19, colon 62)
    intestinal_mask_ = (
        (data == 12).astype(np.uint8)
        + (data == 13).astype(np.uint8)
        + (data == 19).astype(np.uint8)
        + (data == 62).astype(np.uint8)
    )
    intestinal_mask, _ = supress_non_largest_components(intestinal_mask_, [1], default_val=0)
    # process small bowel 19
    small_bowel_remove_mask = (data == 19).astype(np.uint8) - (data == 19).astype(np.uint8) * intestinal_mask
    # process colon 62
    colon_remove_mask = (data == 62).astype(np.uint8) - (data == 62).astype(np.uint8) * intestinal_mask
    intestinal_remove_mask = (small_bowel_remove_mask + colon_remove_mask).astype(np.bool_)
    data[intestinal_remove_mask] = 200

    # ------------ full correponding organ in removed regions
    for organ_label in oran_list:
        data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label

    if target_tumor_label == 23 and np.sum(target_tumor) > 0:
        # speical process for cases with lung tumor
        dia_lung_tumor_mask = (
            dilate_one_img(torch.from_numpy((data == 23)).to(device), filter_size=3, pad_value=0.0).cpu().numpy()
        )
        tmp = (
            (data * (dia_lung_tumor_mask.astype(np.uint8) - (data == 23).astype(np.uint8))).astype(np.float32).flatten()
        )
        tmp[tmp == 0] = float("nan")
        mode = int(stats.mode(tmp.flatten(), nan_policy="omit")[0])
        if mode in [28, 29, 30, 31, 32]:
            dia_lung_tumor_mask = (
                dilate_one_img(torch.from_numpy(dia_lung_tumor_mask).to(device), filter_size=3, pad_value=0.0)
                .cpu()
                .numpy()
            )
            lung_remove_mask = dia_lung_tumor_mask.astype(np.uint8) - (data == 23).astype(np.uint8).astype(np.uint8)
            data[organ_fill_by_removed_mask(data, target_label=mode, remove_mask=lung_remove_mask, device=device)] = (
                mode
            )
        dia_lung_tumor_mask = (
            dilate_one_img(torch.from_numpy(dia_lung_tumor_mask).to(device), filter_size=3, pad_value=0.0).cpu().numpy()
        )
        data[
            organ_fill_by_removed_mask(
                data, target_label=23, remove_mask=dia_lung_tumor_mask * organ_remove_mask, device=device
            )
        ] = 23
        for organ_label in [28, 29, 30, 31, 32]:
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label

    if target_tumor_label == 26 and np.sum(target_tumor) > 0:
        # speical process for cases with hepatic tumor
        # process liver 1
        data[organ_fill_by_removed_mask(data, target_label=1, remove_mask=intestinal_remove_mask, device=device)] = 1
        data[organ_fill_by_removed_mask(data, target_label=1, remove_mask=intestinal_remove_mask, device=device)] = 1
        # process spleen 2
        data[organ_fill_by_removed_mask(data, target_label=3, remove_mask=organ_remove_mask, device=device)] = 3
        data[organ_fill_by_removed_mask(data, target_label=3, remove_mask=organ_remove_mask, device=device)] = 3
        dia_tumor_mask = (
            dilate_one_img(torch.from_numpy((data == target_tumor_label)).to(device), filter_size=3, pad_value=0.0)
            .cpu()
            .numpy()
        )
        dia_tumor_mask = (
            dilate_one_img(torch.from_numpy(dia_tumor_mask).to(device), filter_size=3, pad_value=0.0).cpu().numpy()
        )
        data[
            organ_fill_by_removed_mask(
                data, target_label=target_tumor_label, remove_mask=dia_tumor_mask * organ_remove_mask, device=device
            )
        ] = target_tumor_label
        # refine hepatic tumor
        hepatic_tumor_vessel_liver_mask_ = (
            (data == 26).astype(np.uint8) + (data == 25).astype(np.uint8) + (data == 1).astype(np.uint8)
        )
        hepatic_tumor_vessel_liver_mask_ = (hepatic_tumor_vessel_liver_mask_ > 1).astype(np.uint8)
        hepatic_tumor_vessel_liver_mask, _ = supress_non_largest_components(
            hepatic_tumor_vessel_liver_mask_, [1], default_val=0
        )
        removed_region = (hepatic_tumor_vessel_liver_mask_ - hepatic_tumor_vessel_liver_mask).astype(np.bool_)
        data[removed_region] = 200
        target_tumor = (target_tumor * hepatic_tumor_vessel_liver_mask).astype(np.bool_)
        # refine liver
        data[organ_fill_by_closing(data, target_label=1, device=device)] = 1
        data[organ_fill_by_closing(data, target_label=1, device=device)] = 1
        data[organ_fill_by_closing(data, target_label=1, device=device)] = 1

    if target_tumor_label == 27 and np.sum(target_tumor) > 0:
        # speical process for cases with colon tumor
        dia_tumor_mask = (
            dilate_one_img(torch.from_numpy((data == target_tumor_label)).to(device), filter_size=3, pad_value=0.0)
            .cpu()
            .numpy()
        )
        dia_tumor_mask = (
            dilate_one_img(torch.from_numpy(dia_tumor_mask).to(device), filter_size=3, pad_value=0.0).cpu().numpy()
        )
        data[
            organ_fill_by_removed_mask(
                data, target_label=target_tumor_label, remove_mask=dia_tumor_mask * organ_remove_mask, device=device
            )
        ] = target_tumor_label

    if target_tumor_label == 129 and np.sum(target_tumor) > 0:
        # speical process for cases with kidney tumor
        for organ_label in [5, 14]:
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label
    # TODO: current model does not support hepatic vessel by size control.
    # we treat it as liver for better visiaulization
    print(
        "Current model does not support hepatic vessel by size control, "
        "so we treat generated hepatic vessel as part of liver for better visiaulization."
    )
    data[hepatic_vessel] = 1
    data[airway] = 132
    if target_tumor_label is not None:
        data[target_tumor] = target_tumor_label

    return data


class MapLabelValue:
    """
    Utility to map label values to another set of values.
    For example, map [3, 2, 1] to [0, 1, 2], [1, 2, 3] -> [0.5, 1.5, 2.5], ["label3", "label2", "label1"] -> [0, 1, 2],
    [3.5, 2.5, 1.5] -> ["label0", "label1", "label2"], etc.
    The label data must be numpy array or array-like data and the output data will be numpy array.

    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, orig_labels: Sequence, target_labels: Sequence, dtype: DtypeLike = np.float32) -> None:
        """
        Args:
            orig_labels: original labels that map to others.
            target_labels: expected label values, 1: 1 map to the `orig_labels`.
            dtype: convert the output data to dtype, default to float32.
                if dtype is from PyTorch, the transform will use the pytorch backend, else with numpy backend.

        """
        if len(orig_labels) != len(target_labels):
            raise ValueError("orig_labels and target_labels must have the same length.")

        self.orig_labels = orig_labels
        self.target_labels = target_labels
        self.pair = tuple((o, t) for o, t in zip(self.orig_labels, self.target_labels) if o != t)
        type_dtype = type(dtype)
        if getattr(type_dtype, "__module__", "") == "torch":
            self.use_numpy = False
            self.dtype = get_equivalent_dtype(dtype, data_type=torch.Tensor)
        else:
            self.use_numpy = True
            self.dtype = get_equivalent_dtype(dtype, data_type=np.ndarray)

    def __call__(self, img: NdarrayOrTensor):
        """
        Apply the label mapping to the input image.

        Args:
            img (NdarrayOrTensor): Input image to be remapped.

        Returns:
            NdarrayOrTensor: Remapped image.
        """
        if self.use_numpy:
            img_np, *_ = convert_data_type(img, np.ndarray)
            _out_shape = img_np.shape
            img_flat = img_np.flatten()
            try:
                out_flat = img_flat.astype(self.dtype)
            except ValueError:
                # can't copy unchanged labels as the expected dtype is not supported, must map all the label values
                out_flat = np.zeros(shape=img_flat.shape, dtype=self.dtype)
            for o, t in self.pair:
                out_flat[img_flat == o] = t
            out_t = out_flat.reshape(_out_shape)
        else:
            img_t, *_ = convert_data_type(img, torch.Tensor)
            out_t = img_t.detach().clone().to(self.dtype)  # type: ignore
            for o, t in self.pair:
                out_t[img_t == o] = t
        out, *_ = convert_to_dst_type(src=out_t, dst=img, dtype=self.dtype)
        return out


def KL_loss(z_mu, z_sigma):
    """
    Compute the Kullback-Leibler (KL) divergence loss for a variational autoencoder (VAE).

    The KL divergence measures how one probability distribution diverges from a second, expected probability distribution.
    In the context of VAEs, this loss term ensures that the learned latent space distribution is close to a standard normal distribution.

    Args:
        z_mu (torch.Tensor): Mean of the latent variable distribution, shape [N,C,H,W,D] or [N,C,H,W].
        z_sigma (torch.Tensor): Standard deviation of the latent variable distribution, same shape as 'z_mu'.

    Returns:
        torch.Tensor: The computed KL divergence loss, averaged over the batch.
    """
    eps = 1e-10
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


def dynamic_infer(inferer, model, images):
    """
    Perform dynamic inference using a model and an inferer, typically a monai SlidingWindowInferer.

    This function determines whether to use the model directly or to use the provided inferer
    (such as a sliding window inferer) based on the size of the input images.

    Args:
        inferer: An inference object, typically a monai SlidingWindowInferer, which handles patch-based inference.
        model (torch.nn.Module): The model used for inference.
        images (torch.Tensor): The input images for inference, shape [N,C,H,W,D] or [N,C,H,W].

    Returns:
        torch.Tensor: The output from the model or the inferer, depending on the input size.
    """
    if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
        return model(images)
    else:
        # Extract the spatial dimensions from the images tensor (H, W, D)
        spatial_dims = images.shape[2:]
        orig_roi = inferer.roi_size

        # Check that roi has the same number of dimensions as spatial_dims
        if len(orig_roi) != len(spatial_dims):
            raise ValueError(f"ROI length ({len(orig_roi)}) does not match spatial dimensions ({len(spatial_dims)}).")

        # Iterate and adjust each ROI dimension
        adjusted_roi = [min(roi_dim, img_dim) for roi_dim, img_dim in zip(orig_roi, spatial_dims)]
        inferer.roi_size = adjusted_roi
        output = inferer(network=model, inputs=images)
        inferer.roi_size = orig_roi
        return output
