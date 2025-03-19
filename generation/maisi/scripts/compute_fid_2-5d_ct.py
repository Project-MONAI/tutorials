# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

"""
Compute 2.5D FID using distributed GPU processing.

SHELL Usage Example:
-------------------
    #!/bin/bash

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
    NUM_GPUS=7

    torchrun --nproc_per_node=${NUM_GPUS} compute_fid_2-5d_ct.py \
        --model_name "radimagenet_resnet50" \
        --real_dataset_root "path/to/datasetA" \
        --real_filelist "path/to/filelistA.txt" \
        --real_features_dir "datasetA" \
        --synth_dataset_root "path/to/datasetB" \
        --synth_filelist "path/to/filelistB.txt" \
        --synth_features_dir "datasetB" \
        --enable_center_slices_ratio 0.4 \
        --enable_padding True \
        --enable_center_cropping True \
        --enable_resampling_spacing "1.0x1.0x1.0" \
        --ignore_existing True \
        --num_images 100 \
        --output_root "./features/features-512x512x512" \
        --target_shape "512x512x512"

This script loads two datasets (real vs. synthetic) in 3D medical format (NIfTI)
and extracts feature maps via a 2.5D approach. It then computes the Frechet
Inception Distance (FID) across three orthogonal planes. Data parallelism
is implemented using torch.distributed with an NCCL backend.

Function Arguments (main):
--------------------------
    real_dataset_root (str):
        Root folder for the real dataset.

    real_filelist (str):
        Text file listing 3D images for the real dataset.

    real_features_dir (str):
        Subdirectory (under `output_root`) in which to store feature files
        extracted from the real dataset.

    synth_dataset_root (str):
        Root folder for the synthetic dataset.

    synth_filelist (str):
        Text file listing 3D images for the synthetic dataset.

    synth_features_dir (str):
        Subdirectory (under `output_root`) in which to store feature files
        extracted from the synthetic dataset.

    enable_center_slices_ratio (float or None):
        - If not None, only slices around the specified center ratio will be used
          (analogous to "enable_center_slices=True" with that ratio).
        - If None, no center-slice selection is performed
          (analogous to "enable_center_slices=False").

    enable_padding (bool):
        Whether to pad images to `target_shape`.

    enable_center_cropping (bool):
        Whether to center-crop images to `target_shape`.

    enable_resampling_spacing (str or None):
        - If not None, resample images to the specified voxel spacing (e.g. "1.0x1.0x1.0")
          (analogous to "enable_resampling=True" with that spacing).
        - If None, resampling is skipped
          (analogous to "enable_resampling=False").

    ignore_existing (bool):
        If True, ignore any existing .pt feature files and force re-extraction.

    model_name (str):
        Model identifier. Typically "radimagenet_resnet50" or "squeezenet1_1".

    num_images (int):
        Max number of images to process from each dataset (truncate if more are present).

    output_root (str):
        Folder where extracted .pt feature files, logs, and results are saved.

    target_shape (str):
        Target shape as "XxYxZ" for padding, cropping, or resampling operations.
"""


from __future__ import annotations

import os
import sys
import torch
import fire
import monai
import re
import torch.distributed as dist
import torch.nn.functional as F

from datetime import timedelta
from pathlib import Path
from monai.metrics.fid import FIDMetric
from monai.transforms import Compose

import logging

# ------------------------------------------------------------------------------
# Create logger
# ------------------------------------------------------------------------------
logger = logging.getLogger("fid_2-5d_ct")
if not logger.handlers:
    # Configure logger only if it has no handlers (avoid reconfiguring in multi-rank scenarios)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger.setLevel(logging.INFO)


def drop_empty_slice(slices, empty_threshold: float):
    """
    Decide which 2D slices to keep by checking if their maximum intensity
    is below a certain threshold.

    Args:
        slices (tuple or list of Tensors): Each element is (B, C, H, W).
        empty_threshold (float): If the slice's maximum value is below this threshold,
            it is considered "empty".

    Returns:
        list[bool]: A list of booleans indicating for each slice whether to keep it.
    """
    outputs = []
    n_drop = 0
    for s in slices:
        largest_unique = torch.max(torch.unique(s))
        if largest_unique < empty_threshold:
            outputs.append(False)
            n_drop += 1
        else:
            outputs.append(True)

    logger.info(f"Empty slice drop rate {round((n_drop/len(slices))*100,1)}%")
    return outputs


def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Subtract per-channel means (ImageNet-like: [0.406, 0.456, 0.485])
    from the input 4D or 5D tensor. Expects channels in the first dimension
    after the batch dimension: (B, C, H, W) or (B, C, H, W, D).
    """
    mean = [0.406, 0.456, 0.485]
    x[:, 0, ...] -= mean[0]
    x[:, 1, ...] -= mean[1]
    x[:, 2, ...] -= mean[2]
    return x


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """
    Average out the spatial dimensions of a tensor, preserving or removing them
    according to `keepdim`. This is used to produce a 1D feature vector
    out of a feature map.

    Args:
        x (torch.Tensor): Input tensor (B, C, H, W, ...) or (B, C, H, W).
        keepdim (bool): Whether to keep dimension or not after averaging.

    Returns:
        torch.Tensor: Tensor with reduced spatial dimensions.
    """
    dim = len(x.shape)
    # 2D -> no average
    if dim == 2:
        return x
    # 3D -> average over last dim
    if dim == 3:
        return x.mean([2], keepdim=keepdim)
    # 4D -> average over H,W
    if dim == 4:
        return x.mean([2, 3], keepdim=keepdim)
    # 5D -> average over H,W,D
    if dim == 5:
        return x.mean([2, 3, 4], keepdim=keepdim)
    return x


def medicalnet_intensity_normalisation(volume: torch.Tensor) -> torch.Tensor:
    """
    Intensity normalization approach from MedicalNet:
    (volume - mean) / (std + 1e-5) across spatial dims.
    Expects (B, C, H, W) or (B, C, H, W, D).
    """
    dim = len(volume.shape)
    if dim == 4:
        mean = volume.mean([2, 3], keepdim=True)
        std = volume.std([2, 3], keepdim=True)
    elif dim == 5:
        mean = volume.mean([2, 3, 4], keepdim=True)
        std = volume.std([2, 3, 4], keepdim=True)
    else:
        return volume
    return (volume - mean) / (std + 1e-5)


def radimagenet_intensity_normalisation(volume: torch.Tensor, norm2d: bool = False) -> torch.Tensor:
    """
    Intensity normalization for radimagenet_resnet. Optionally normalizes each 2D slice individually.

    Args:
        volume (torch.Tensor): Input (B, C, H, W) or (B, C, H, W, D).
        norm2d (bool): If True, normalizes each (H,W) slice to [0,1], then subtracts the ImageNet mean.
    """
    logger.info(f"norm2d: {norm2d}")
    dim = len(volume.shape)
    # If norm2d is True, only meaningful for 4D data (B, C, H, W):
    if dim == 4 and norm2d:
        max2d, _ = torch.max(volume, dim=2, keepdim=True)
        max2d, _ = torch.max(max2d, dim=3, keepdim=True)
        min2d, _ = torch.min(volume, dim=2, keepdim=True)
        min2d, _ = torch.min(min2d, dim=3, keepdim=True)
        # Scale each slice to 0..1
        volume = (volume - min2d) / (max2d - min2d + 1e-10)
        # Subtract channel mean
        return subtract_mean(volume)
    elif dim == 4:
        # 4D but no per-slice normalization
        max3d = torch.max(volume)
        min3d = torch.min(volume)
        volume = (volume - min3d) / (max3d - min3d + 1e-10)
        return subtract_mean(volume)
    # Fallback for e.g. 5D data is simply a min-max over entire volume
    if dim == 5:
        maxval = torch.max(volume)
        minval = torch.min(volume)
        volume = (volume - minval) / (maxval - minval + 1e-10)
        return subtract_mean(volume)
    return volume


def get_features_2p5d(
    image: torch.Tensor,
    feature_network: torch.nn.Module,
    center_slices: bool = False,
    center_slices_ratio: float = 1.0,
    sample_every_k: int = 1,
    xy_only: bool = True,
    drop_empty: bool = False,
    empty_threshold: float = -700,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Extract 2.5D features from a 3D image by slicing it along XY, YZ, ZX planes.

    Args:
        image (torch.Tensor): Input 5D tensor in shape (B, C, H, W, D).
        feature_network (torch.nn.Module): Model that processes 2D slices (C,H,W).
        center_slices (bool): Whether to slice only the center portion of each axis.
        center_slices_ratio (float): Ratio of slices to keep in the center if `center_slices` is True.
        sample_every_k (int): Downsampling factor along each axis when slicing.
        xy_only (bool): If True, return only the XY-plane features.
        drop_empty (bool): Drop slices that are deemed "empty" below `empty_threshold`.
        empty_threshold (float): Threshold to decide emptiness of slices.

    Returns:
        tuple of torch.Tensor or None: (XY_features, YZ_features, ZX_features).
    """
    logger.info(f"center_slices: {center_slices}, ratio: {center_slices_ratio}")

    # If there's only 1 channel, replicate to 3 channels
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1, 1)

    # Convert from 'RGB'→(R,G,B) to (B,G,R)
    image = image[:, [2, 1, 0], ...]

    B, C, H, W, D = image.size()
    with torch.no_grad():
        # ---------------------- XY-plane slicing along D ----------------------
        if center_slices:
            start_d = int((1.0 - center_slices_ratio) / 2.0 * D)
            end_d = int((1.0 + center_slices_ratio) / 2.0 * D)
            slices = torch.unbind(image[:, :, :, :, start_d:end_d:sample_every_k], dim=-1)
        else:
            slices = torch.unbind(image, dim=-1)

        if drop_empty:
            mapping_index = drop_empty_slice(slices, empty_threshold)
        else:
            mapping_index = [True for _ in range(len(slices))]

        images_2d = torch.cat(slices, dim=0)
        images_2d = radimagenet_intensity_normalisation(images_2d)
        images_2d = images_2d[mapping_index]

        feature_image_xy = feature_network.forward(images_2d)
        feature_image_xy = spatial_average(feature_image_xy, keepdim=False)
        if xy_only:
            return feature_image_xy, None, None

        # ---------------------- YZ-plane slicing along H ----------------------
        if center_slices:
            start_h = int((1.0 - center_slices_ratio) / 2.0 * H)
            end_h = int((1.0 + center_slices_ratio) / 2.0 * H)
            slices = torch.unbind(image[:, :, start_h:end_h:sample_every_k, :, :], dim=2)
        else:
            slices = torch.unbind(image, dim=2)

        if drop_empty:
            mapping_index = drop_empty_slice(slices, empty_threshold)
        else:
            mapping_index = [True for _ in range(len(slices))]

        images_2d = torch.cat(slices, dim=0)
        images_2d = radimagenet_intensity_normalisation(images_2d)
        images_2d = images_2d[mapping_index]

        feature_image_yz = feature_network.forward(images_2d)
        feature_image_yz = spatial_average(feature_image_yz, keepdim=False)

        # ---------------------- ZX-plane slicing along W ----------------------
        if center_slices:
            start_w = int((1.0 - center_slices_ratio) / 2.0 * W)
            end_w = int((1.0 + center_slices_ratio) / 2.0 * W)
            slices = torch.unbind(image[:, :, :, start_w:end_w:sample_every_k, :], dim=3)
        else:
            slices = torch.unbind(image, dim=3)

        if drop_empty:
            mapping_index = drop_empty_slice(slices, empty_threshold)
        else:
            mapping_index = [True for _ in range(len(slices))]

        images_2d = torch.cat(slices, dim=0)
        images_2d = radimagenet_intensity_normalisation(images_2d)
        images_2d = images_2d[mapping_index]

        feature_image_zx = feature_network.forward(images_2d)
        feature_image_zx = spatial_average(feature_image_zx, keepdim=False)

    return feature_image_xy, feature_image_yz, feature_image_zx


def pad_to_max_size(tensor: torch.Tensor, max_size: int, padding_value: float = 0.0) -> torch.Tensor:
    """
    Zero-pad a 2D feature map or other tensor along the first dimension to match a specified size.

    Args:
        tensor (torch.Tensor): The feature tensor to pad.
        max_size (int): Desired size along the first dimension.
        padding_value (float): Value to fill during padding.

    Returns:
        torch.Tensor: Padded tensor matching `max_size` along dim=0.
    """
    pad_size = [0, 0] * (len(tensor.shape) - 1) + [0, max_size - tensor.shape[0]]
    return F.pad(tensor, pad_size, "constant", padding_value)


def main(
    real_dataset_root: str = "path/to/datasetA",
    real_filelist: str = "path/to/filelistA.txt",
    real_features_dir: str = "datasetA",
    synth_dataset_root: str = "path/to/datasetB",
    synth_filelist: str = "path/to/filelistB.txt",
    synth_features_dir: str = "datasetB",
    enable_center_slices_ratio: float = None,
    enable_padding: bool = True,
    enable_center_cropping: bool = True,
    enable_resampling_spacing: str = None,
    ignore_existing: bool = False,
    model_name: str = "radimagenet_resnet50",
    num_images: int = 100,
    output_root: str = "./features/features-512x512x512",
    target_shape: str = "512x512x512",
):
    """
    Compute 2.5D FID using distributed GPU processing.

    This function loads two datasets (real vs. synthetic) in 3D medical format (NIfTI)
    and extracts feature maps via a 2.5D approach, then computes the Frechet Inception
    Distance (FID) across three orthogonal planes. Data parallelism is implemented
    using torch.distributed with an NCCL backend.

    Args:
        real_dataset_root (str):
            Root folder for the real dataset.
        real_filelist (str):
            Path to a text file listing 3D images (e.g., NIfTI files) for the real dataset.
            Each line in this file should contain a relative path (or filename) to a NIfTI file.
            For example, your "real_filelist.txt" could look like:
                case001.nii.gz
                case002.nii.gz
                case003.nii.gz
                ...
            These entries will be appended to `real_dataset_root`.
        real_features_dir (str):
            Name of the directory under `output_root` in which to store
            extracted features for the real dataset.

        synth_dataset_root (str):
            Root folder for the synthetic dataset.
        synth_filelist (str):
            Path to a text file listing 3D images (e.g., NIfTI files) for the synthetic dataset.
            The format is the same as the real dataset file list, for example:
                synth_case001.nii.gz
                synth_case002.nii.gz
                synth_case003.nii.gz
                ...
            These entries will be appended to `synth_dataset_root`.
        synth_features_dir (str):
            Name of the directory under `output_root` in which to store
            extracted features for the synthetic dataset.

        enable_center_slices_ratio (float or None):
            - If not None, only slices around the specified center ratio are used.
              (similar to "enable_center_slices=True" with that ratio in an earlier script).
            - If None, no center-slice selection is performed
              (similar to "enable_center_slices=False").

        enable_padding (bool):
            Whether to pad images to `target_shape`.

        enable_center_cropping (bool):
            Whether to center-crop images to `target_shape`.

        enable_resampling_spacing (str or None):
            - If not None, resample images to this voxel spacing (e.g. "1.0x1.0x1.0")
              (similar to "enable_resampling=True" with that spacing).
            - If None, skip resampling (similar to "enable_resampling=False").

        ignore_existing (bool):
            If True, ignore any existing .pt feature files and force re-computation.

        model_name (str):
            Model identifier. Typically "radimagenet_resnet50" or "squeezenet1_1".

        num_images (int):
            Maximum number of images to load from each dataset (truncate if more are present).

        output_root (str):
            Parent folder where extracted .pt files and logs will be saved.

        target_shape (str):
            Target shape, e.g. "512x512x512", for padding, cropping, or resampling operations.

    Returns:
        None
    """
    # -------------------------------------------------------------------------
    # Initialize Process Group (Distributed)
    # -------------------------------------------------------------------------
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=7200))

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(dist.get_world_size())
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    logger.info(f"[INFO] Running process on {device} of total {world_size} ranks.")

    # Convert potential string bools to actual bools (if using Fire or similar)
    if not isinstance(enable_padding, bool):
        enable_padding = enable_padding.lower() == "true"
    if not isinstance(enable_center_cropping, bool):
        enable_center_cropping = enable_center_cropping.lower() == "true"
    if not isinstance(ignore_existing, bool):
        ignore_existing = ignore_existing.lower() == "true"

    # Merge logic for center slices
    enable_center_slices = enable_center_slices_ratio is not None

    # Merge logic for resampling
    enable_resampling = enable_resampling_spacing is not None

    # Print out some flags on rank 0
    if local_rank == 0:
        logger.info(f"Real dataset root: {real_dataset_root}")
        logger.info(f"Synth dataset root: {synth_dataset_root}")
        logger.info(f"enable_center_slices_ratio: {enable_center_slices_ratio}")
        logger.info(f"enable_center_slices: {enable_center_slices}")
        logger.info(f"enable_padding: {enable_padding}")
        logger.info(f"enable_center_cropping: {enable_center_cropping}")
        logger.info(f"enable_resampling_spacing: {enable_resampling_spacing}")
        logger.info(f"enable_resampling: {enable_resampling}")
        logger.info(f"ignore_existing: {ignore_existing}")

    # -------------------------------------------------------------------------
    # Load feature extraction model
    # -------------------------------------------------------------------------
    if model_name == "radimagenet_resnet50":
        feature_network = torch.hub.load(
            "Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True, trust_repo=True
        )
        suffix = "radimagenet_resnet50"
    else:
        import torchvision

        feature_network = torchvision.models.squeezenet1_1(pretrained=True)
        suffix = "squeezenet1_1"

    feature_network.to(device)
    feature_network.eval()

    # -------------------------------------------------------------------------
    # Parse shape/spacings
    # -------------------------------------------------------------------------
    t_shape = [int(x) for x in target_shape.split("x")]
    target_shape_tuple = tuple(t_shape)

    # If not None, parse the resampling spacing
    if enable_resampling:
        rs_spacing = [float(x) for x in enable_resampling_spacing.split("x")]
        rs_spacing_tuple = tuple(rs_spacing)
        if local_rank == 0:
            logger.info(f"Resampling spacing: {rs_spacing_tuple}")
    else:
        rs_spacing_tuple = (1.0, 1.0, 1.0)

    # Use the ratio if provided, otherwise 1.0
    center_slices_ratio_final = enable_center_slices_ratio if enable_center_slices else 1.0
    if local_rank == 0:
        logger.info(f"center_slices_ratio: {center_slices_ratio_final}")

    # -------------------------------------------------------------------------
    # Prepare Real Dataset
    # -------------------------------------------------------------------------
    output_root_real = os.path.join(output_root, real_features_dir)
    with open(real_filelist, "r") as rf:
        real_lines = [l.strip() for l in rf.readlines()]
    real_lines.sort()
    real_lines = real_lines[:num_images]

    real_filenames = [{"image": os.path.join(real_dataset_root, f)} for f in real_lines]
    real_filenames = monai.data.partition_dataset(
        data=real_filenames, shuffle=False, num_partitions=world_size, even_divisible=False
    )[local_rank]

    # -------------------------------------------------------------------------
    # Prepare Synthetic Dataset
    # -------------------------------------------------------------------------
    output_root_synth = os.path.join(output_root, synth_features_dir)
    with open(synth_filelist, "r") as sf:
        synth_lines = [l.strip() for l in sf.readlines()]
    synth_lines.sort()
    synth_lines = synth_lines[:num_images]

    synth_filenames = [{"image": os.path.join(synth_dataset_root, f)} for f in synth_lines]
    synth_filenames = monai.data.partition_dataset(
        data=synth_filenames, shuffle=False, num_partitions=world_size, even_divisible=False
    )[local_rank]

    # -------------------------------------------------------------------------
    # Build MONAI transforms
    # -------------------------------------------------------------------------
    transform_list = [
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
    ]

    if enable_resampling:
        transform_list.append(monai.transforms.Spacingd(keys=["image"], pixdim=rs_spacing_tuple, mode=["bilinear"]))

    if enable_padding:
        transform_list.append(
            monai.transforms.SpatialPadd(keys=["image"], spatial_size=target_shape_tuple, mode="constant", value=-1000)
        )

    if enable_center_cropping:
        transform_list.append(monai.transforms.CenterSpatialCropd(keys=["image"], roi_size=target_shape_tuple))

    transform_list.append(
        monai.transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=-1000, b_max=1000, clip=True
        )
    )
    transforms = Compose(transform_list)

    # -------------------------------------------------------------------------
    # Create DataLoaders
    # -------------------------------------------------------------------------
    real_ds = monai.data.Dataset(data=real_filenames, transform=transforms)
    real_loader = monai.data.DataLoader(real_ds, num_workers=6, batch_size=1, shuffle=False)

    synth_ds = monai.data.Dataset(data=synth_filenames, transform=transforms)
    synth_loader = monai.data.DataLoader(synth_ds, num_workers=6, batch_size=1, shuffle=False)

    # -------------------------------------------------------------------------
    # Extract features for Real Dataset
    # -------------------------------------------------------------------------
    real_features_xy, real_features_yz, real_features_zx = [], [], []
    for idx, batch_data in enumerate(real_loader, start=1):
        img = batch_data["image"].to(device)
        fn = img.meta["filename_or_obj"][0]
        logger.info(f"[Rank {local_rank}] Real data {idx}/{len(real_filenames)}: {fn}")

        out_fp = fn.replace(real_dataset_root, output_root_real).replace(".nii.gz", ".pt")
        out_fp = Path(out_fp)
        out_fp.parent.mkdir(parents=True, exist_ok=True)

        if (not ignore_existing) and os.path.isfile(out_fp):
            feats = torch.load(out_fp, weights_only=True)
        else:
            img_t = img.as_tensor()
            logger.info(f"image shape: {tuple(img_t.shape)}")

            feats = get_features_2p5d(
                img_t,
                feature_network,
                center_slices=enable_center_slices,
                center_slices_ratio=center_slices_ratio_final,
                xy_only=False,
            )
            logger.info(f"feats shapes: {feats[0].shape}, {feats[1].shape}, {feats[2].shape}")
            torch.save(feats, out_fp)

        real_features_xy.append(feats[0])
        real_features_yz.append(feats[1])
        real_features_zx.append(feats[2])

    real_features_xy = torch.vstack(real_features_xy)
    real_features_yz = torch.vstack(real_features_yz)
    real_features_zx = torch.vstack(real_features_zx)
    logger.info(
        f"Real feature shapes: {real_features_xy.shape}, " f"{real_features_yz.shape}, {real_features_zx.shape}"
    )

    # -------------------------------------------------------------------------
    # Extract features for Synthetic Dataset
    # -------------------------------------------------------------------------
    synth_features_xy, synth_features_yz, synth_features_zx = [], [], []
    for idx, batch_data in enumerate(synth_loader, start=1):
        img = batch_data["image"].to(device)
        fn = img.meta["filename_or_obj"][0]
        logger.info(f"[Rank {local_rank}] Synth data {idx}/{len(synth_filenames)}: {fn}")

        out_fp = fn.replace(synth_dataset_root, output_root_synth).replace(".nii.gz", ".pt")
        out_fp = Path(out_fp)
        out_fp.parent.mkdir(parents=True, exist_ok=True)

        if (not ignore_existing) and os.path.isfile(out_fp):
            feats = torch.load(out_fp, weights_only=True)
        else:
            img_t = img.as_tensor()
            logger.info(f"image shape: {tuple(img_t.shape)}")

            feats = get_features_2p5d(
                img_t,
                feature_network,
                center_slices=enable_center_slices,
                center_slices_ratio=center_slices_ratio_final,
                xy_only=False,
            )
            logger.info(f"feats shapes: {feats[0].shape}, {feats[1].shape}, {feats[2].shape}")
            torch.save(feats, out_fp)

        synth_features_xy.append(feats[0])
        synth_features_yz.append(feats[1])
        synth_features_zx.append(feats[2])

    synth_features_xy = torch.vstack(synth_features_xy)
    synth_features_yz = torch.vstack(synth_features_yz)
    synth_features_zx = torch.vstack(synth_features_zx)
    logger.info(
        f"Synth feature shapes: {synth_features_xy.shape}, " f"{synth_features_yz.shape}, {synth_features_zx.shape}"
    )

    # -------------------------------------------------------------------------
    # All-reduce / gather features across ranks
    # -------------------------------------------------------------------------
    features = [
        real_features_xy,
        real_features_yz,
        real_features_zx,
        synth_features_xy,
        synth_features_yz,
        synth_features_zx,
    ]

    # 1) Gather local feature sizes across ranks
    local_sizes = []
    for ft_idx in range(len(features)):
        local_size = torch.tensor([features[ft_idx].shape[0]], dtype=torch.int64, device=device)
        local_sizes.append(local_size)

    all_sizes = []
    for ft_idx in range(len(features)):
        rank_sizes = [torch.tensor([0], dtype=torch.int64, device=device) for _ in range(world_size)]
        dist.all_gather(rank_sizes, local_sizes[ft_idx])
        all_sizes.append(rank_sizes)

    # 2) Pad and gather all features
    all_tensors_list = []
    for ft_idx, ft in enumerate(features):
        max_size = max(all_sizes[ft_idx]).item()
        ft_padded = pad_to_max_size(ft, max_size)

        gather_list = [torch.empty_like(ft_padded) for _ in range(world_size)]
        dist.all_gather(gather_list, ft_padded)

        # Trim each gather back to the real size
        for rk in range(world_size):
            gather_list[rk] = gather_list[rk][: all_sizes[ft_idx][rk], :]

        all_tensors_list.append(gather_list)

    # On rank 0, compute FID
    if local_rank == 0:
        real_xy = torch.vstack(all_tensors_list[0])
        real_yz = torch.vstack(all_tensors_list[1])
        real_zx = torch.vstack(all_tensors_list[2])

        synth_xy = torch.vstack(all_tensors_list[3])
        synth_yz = torch.vstack(all_tensors_list[4])
        synth_zx = torch.vstack(all_tensors_list[5])

        logger.info(f"Final Real shapes: {real_xy.shape}, {real_yz.shape}, {real_zx.shape}")
        logger.info(f"Final Synth shapes: {synth_xy.shape}, {synth_yz.shape}, {synth_zx.shape}")

        fid = FIDMetric()
        logger.info(f"Computing FID for: {output_root_real} | {output_root_synth}")
        fid_res_xy = fid(synth_xy, real_xy)
        fid_res_yz = fid(synth_yz, real_yz)
        fid_res_zx = fid(synth_zx, real_zx)

        logger.info(f"FID XY: {fid_res_xy}")
        logger.info(f"FID YZ: {fid_res_yz}")
        logger.info(f"FID ZX: {fid_res_zx}")
        fid_avg = (fid_res_xy + fid_res_yz + fid_res_zx) / 3.0
        logger.info(f"FID Avg: {fid_avg}")

    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
