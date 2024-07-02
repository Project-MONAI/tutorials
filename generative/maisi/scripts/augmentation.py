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

import numpy as np
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from monai.transforms import Rand3DElastic, RandAffine, RandZoom
from monai.utils import ensure_tuple_rep
from .utils import erode_one_img, dilate_one_img

def initialize_tumor_mask(volume: Tensor, tumor_label: Sequence[int]) -> Tensor:
    """
    Initialize tumor mask for tumor augmentation.

    Args:
        volume: input 3D multi-label mask, [H,W,D] torch tensor.
        tumor_label: tumor label in whole_mask, list of int. 

    Return:
        tumor_mask_, initialized tumor mask, [H,W,D] torch tensor.
    """
    tumor_mask_ = torch.zeros_like(volume)
    for idx, label in enumerate(tumor_label):
        tumor_mask_[volume == label] = idx + 1
    tumor_mask_ = tumor_mask_.to(torch.uint8)
    return tumor_mask_
    
def augment_bone_tumor(whole_mask: Tensor, spatial_size: tuple[int, int, int] | int | None = None) -> Tensor:
    """
    Bone tumor augmentation.

    Args:
        whole_mask: input 3D multi-label mask, [1,H,W,D] torch tensor.
        spatial_size: output image spatial size, used in random transform. If not defined, will use (H,W,D). If some components are non-positive values, the transform will use the corresponding components of whole_mask size. For example, spatial_size=(128, 128, -1) will be adapted to (128, 128, 64) if the third spatial dimension size of whole_mask is 64.

    Return:
        augmented mask, with shape of spatial_size and data type as whole_mask.

    Example:

        .. code-block:: python

            # define a multi-label mask
            whole_mask = torch.randint(0, 132, [1,128,128,128])
            augmented_whole_mask = augment_bone_tumor(whole_mask)
    """
    # Initialize binary tumor mask
    volume = whole_mask.squeeze(0) # [M,N,P]
    tumor_label = [128]
    tumor_mask_ = initialize_tumor_mask(volume, tumor_label)

    # Define augmentation transform
    elastic = RandAffine(
        mode="nearest",
        prob=1.0,
        translate_range=(5, 5, 0),
        rotate_range=(0, 0, 0.1),
        scale_range=(0.15, 0.15, 0),
        padding_mode="zeros",
    )

    tumor_size = torch.sum((tumor_mask_ > 0).float())
    ###########################
    # remove pred in pseudo_label in real lesion region
    volume[tumor_mask_ > 0] = 200
    ###########################
    if tumor_size > 0:
        # get organ mask
        organ_mask = (
            torch.logical_and(33 <= volume, volume <= 56).float()
            + torch.logical_and(63 <= volume, volume <= 97).float()
            + (volume == 127).float()
            + (volume == 114).float()
            + tumor_mask_
        )
        organ_mask = (organ_mask > 0).float()
        
        # augment mask
        count = 0        
        while True:
            threshold = 0.8 if count < 40 else 0.75
            tumor_mask = tumor_mask_
            # apply random augmentation
            distorted_mask = elastic(
                (tumor_mask > 0).cuda(), spatial_size=spatial_size
            ).as_tensor()
            # the tumor must be within the organ
            tumor_mask = distorted_mask * organ_mask
            count += 1
            print(torch.sum(tumor_mask), "|", tumor_size * threshold)
            if torch.sum(tumor_mask) >= tumor_size * threshold:
                tumor_mask = dilate_one_img(tumor_mask.squeeze(0), erosion=5, pad_value=1.0)
                tumor_mask = (
                    erode_one_img(tumor_mask, erosion=5, pad_value=1.0).unsqueeze(0).to(torch.uint8)
                )
                break
    else:
        tumor_mask = tumor_mask_

    # update the new tumor mask
    volume[tumor_mask == 1] = tumor_label[0]

    whole_mask = volume.unsqueeze(0)
    return whole_mask


def augmentation_liver_tumor(whole_mask: Tensor, spatial_size: tuple[int, int, int] | int | None = None) -> Tensor:
    """
    Bone liver augmentation.

    Args:
        whole_mask: input 3D multi-label mask, [1,H,W,D] torch tensor.
        spatial_size: output image spatial size, used in random transform. If not defined, will use (H,W,D). If some components are non-positive values, the transform will use the corresponding components of whole_mask size. For example, spatial_size=(128, 128, -1) will be adapted to (128, 128, 64) if the third spatial dimension size of whole_mask is 64.

    Return:
        augmented mask, with shape of spatial_size and data type as whole_mask.

    Example:

        .. code-block:: python

            # define a multi-label mask
            whole_mask = torch.randint(0, 132, [1,128,128,128])
            augmented_whole_mask = augment_liver_tumor(whole_mask)
    """
    # Initialize binary tumor mask
    volume = whole_mask.squeeze(0)
    tumor_label = [1, 26]
    tumor_mask_ = initialize_tumor_mask(volume, tumor_label)

    # Define augmentation transform
    elastic = Rand3DElastic(
        mode="nearest",
        prob=1.0,
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        translate_range=(10, 10, 10),
        rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        scale_range=(0.2, 0.2, 0.2),
        padding_mode="zeros",
    )

    tumor_size = torch.sum(tumor_mask_ == 2)
    ###########################
    # remove pred  organ labels
    volume[volume == 1] = 0
    volume[volume == 26] = 0
    # before move tumor maks, full the original location by organ labels
    volume[tumor_mask_ == 1] = 1
    volume[tumor_mask_ == 2] = 1
    ###########################
    while True:
        tumor_mask = tumor_mask_
        # apply random augmentation
        tumor_mask = elastic(
            (tumor_mask == 2).cuda(), spatial_size=tuple(spatial_size)
        ).as_tensor()
        # get organ mask
        organ_mask = (tumor_mask_ == 1).float() + (tumor_mask_ == 2).float()
        organ_mask = dilate_one_img(organ_mask.squeeze(0), erosion=5, pad_value=1.0)
        organ_mask = erode_one_img(organ_mask, erosion=5, pad_value=1.0).unsqueeze(0)
        
        # the tumor must be within the organ
        tumor_mask = tumor_mask * organ_mask
        print(torch.sum(tumor_mask), "|", tumor_size * 0.80)
        if torch.sum(tumor_mask) >= tumor_size * 0.80:
            tumor_mask = dilate_one_img(tumor_mask.squeeze(0), erosion=5, pad_value=1.0)
            tumor_mask = erode_one_img(tumor_mask, erosion=5, pad_value=1.0).unsqueeze(0)
            break

    volume[tumor_mask == 1] = 26

    whole_mask = volume.unsqueeze(0)
    return whole_mask


def augmentation_lung_tumor(whole_mask: Tensor, spatial_size: tuple[int, int, int] | int | None = None) -> Tensor:
    """
    Lung tumor augmentation.

    Args:
        whole_mask: input 3D multi-label mask, [1,H,W,D] torch tensor.
        spatial_size: output image spatial size, used in random transform. If not defined, will use (H,W,D). If some components are non-positive values, the transform will use the corresponding components of whole_mask size. For example, spatial_size=(128, 128, -1) will be adapted to (128, 128, 64) if the third spatial dimension size of whole_mask is 64.

    Return:
        augmented mask, with shape of spatial_size and data type as whole_mask.

    Example:

        .. code-block:: python

            # define a multi-label mask
            whole_mask = torch.randint(0, 132, [1,128,128,128])
            augmented_whole_mask = augment_lung_tumor(whole_mask)
    """
    # Initialize binary tumor mask
    volume = whole_mask.squeeze(0)
    tumor_label = [23]
    tumor_mask_ = initialize_tumor_mask(volume, tumor_label)

    # Define augmentation transform
    elastic = Rand3DElastic(
        mode="nearest",
        prob=1.0,
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        translate_range=(20, 20, 20),
        rotate_range=(np.pi / 36, np.pi / 36, np.pi),
        scale_range=(0.15, 0.15, 0.15),
        padding_mode="zeros",
    )

    tumor_size = torch.sum(tumor_mask_)
    # before move lung tumor maks, full the original location by lung labels
    new_tumor_mask_ = dilate_one_img(tumor_mask_.squeeze(0), erosion=3, pad_value=1.0)
    new_tumor_mask_ = new_tumor_mask_.unsqueeze(0)
    new_tumor_mask_[tumor_mask_ > 0] = 0
    new_tumor_mask_[volume < 28] = 0
    new_tumor_mask_[volume > 32] = 0
    tmp = volume[(volume * new_tumor_mask_).nonzero(as_tuple=True)].view(-1)

    mode = torch.mode(tmp, 0)[0].item()
    print(mode)
    assert 28 <= mode <= 32
    volume[tumor_mask_.bool()] = mode
    ###########################
    if tumor_size > 0:
        # aug
        while True:
            tumor_mask = tumor_mask_
            # apply random augmentation
            tumor_mask = elastic(
                tumor_mask, spatial_size=tuple(spatial_size)
            ).as_tensor()
            # get lung mask v2 (133 order)
            lung_mask = (
                (volume == 28).float()
                + (volume == 29).float()
                + (volume == 30).float()
                + (volume == 31).float()
                + (volume == 32).float()
            )
            lung_mask = dilate_one_img(lung_mask.squeeze(0), erosion=5, pad_value=1.0)
            lung_mask = erode_one_img(lung_mask, erosion=5, pad_value=1.0).unsqueeze(0)
            
            # the tumor must be within the organ
            tumor_mask = tumor_mask * lung_mask
            print(torch.sum(tumor_mask), "|", tumor_size * 0.85)
            if torch.sum(tumor_mask) >= tumor_size * 0.85:
                tumor_mask = dilate_one_img(tumor_mask.squeeze(0), erosion=5, pad_value=1.0)
                tumor_mask = (
                    erode_one_img(tumor_mask, erosion=5, pad_value=1.0).unsqueeze(0).to(torch.uint8)
                )
                break
    else:
        tumor_mask = tumor_mask_

    volume[tumor_mask == 1] = tumor_label[0]

    whole_mask = volume.unsqueeze(0)
    return whole_mask


def augmentation_pancreas_tumor(whole_mask: Tensor, spatial_size: tuple[int, int, int] | int | None = None) -> Tensor:
    """
    Pancreas tumor augmentation.

    Args:
        whole_mask: input 3D multi-label mask, [1,H,W,D] torch tensor.
        spatial_size: output image spatial size, used in random transform. If not defined, will use (H,W,D). If some components are non-positive values, the transform will use the corresponding components of whole_mask size. For example, spatial_size=(128, 128, -1) will be adapted to (128, 128, 64) if the third spatial dimension size of whole_mask is 64.

    Return:
        augmented mask, with shape of spatial_size and data type as whole_mask.

    Example:

        .. code-block:: python

            # define a multi-label mask
            whole_mask = torch.randint(0, 132, [1,128,128,128])
            augmented_whole_mask = augment_pancreas_tumor(whole_mask)
    """
    # Initialize binary tumor mask
    volume = whole_mask.squeeze(0)
    tumor_label = [4, 24]
    tumor_mask_ = initialize_tumor_mask(volume, tumor_label)

    # Define augmentation transform
    elastic = Rand3DElastic(
        mode="nearest",
        prob=1.0,
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        translate_range=(15, 15, 15),
        rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        scale_range=(0.1, 0.1, 0.1),
        padding_mode="zeros",
    )

    tumor_size = torch.sum(tumor_mask_ == 2)
    ###########################
    # remove pred  organ labels
    volume[volume == 24] = 0
    volume[volume == 4] = 0
    # before move tumor maks, full the original location by organ labels
    volume[tumor_mask_ == 1] = 4
    volume[tumor_mask_ == 2] = 4
    ###########################
    while True:
        tumor_mask = tumor_mask_
        # apply random augmentation
        tumor_mask = elastic(
            (tumor_mask == 2).cuda(), spatial_size=tuple(spatial_size)
        ).as_tensor()
        # get organ mask
        organ_mask = (tumor_mask_ == 1).float() + (tumor_mask_ == 2).float()
        organ_mask = dilate_one_img(organ_mask.squeeze(0), erosion=5, pad_value=1.0)
        organ_mask = erode_one_img(organ_mask, erosion=5, pad_value=1.0).unsqueeze(0)
        
        # the tumor must be within the organ
        tumor_mask = tumor_mask * organ_mask
        print(torch.sum(tumor_mask), "|", tumor_size * 0.80)
        if torch.sum(tumor_mask) >= tumor_size * 0.80:
            tumor_mask = dilate_one_img(tumor_mask.squeeze(0), erosion=5, pad_value=1.0)
            tumor_mask = erode_one_img(tumor_mask, erosion=5, pad_value=1.0).unsqueeze(0)
            break

    volume[tumor_mask == 1] = tumor_label[0]
    
    whole_mask = volume.unsqueeze(0)
    return whole_mask


def augmentation_colon_tumor(whole_mask: Tensor, spatial_size: tuple[int, int, int] | int | None = None) -> Tensor:
    """
    Colon tumor augmentation.

    Args:
        whole_mask: input 3D multi-label mask, [1,H,W,D] torch tensor.
        spatial_size: output image spatial size, used in random transform. If not defined, will use (H,W,D). If some components are non-positive values, the transform will use the corresponding components of whole_mask size. For example, spatial_size=(128, 128, -1) will be adapted to (128, 128, 64) if the third spatial dimension size of whole_mask is 64.

    Return:
        augmented mask, with shape of spatial_size and data type as whole_mask.

    Example:

        .. code-block:: python

            # define a multi-label mask
            whole_mask = torch.randint(0, 132, [1,128,128,128])
            augmented_whole_mask = augment_colon_tumor(whole_mask)
    """
    # Initialize binary tumor mask
    volume = whole_mask.squeeze(0)
    tumor_label = [27]
    tumor_mask_ = initialize_tumor_mask(volume, tumor_label)

    # Define augmentation transform
    elastic = Rand3DElastic(
        mode="nearest",
        prob=1.0,
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        translate_range=(5, 5, 5),
        rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        scale_range=(0.1, 0.1, 0.1),
        padding_mode="zeros",
    )

    tumor_size = torch.sum(tumor_mask_)
    ###########################
    # before move tumor maks, full the original location by organ labels
    volume[tumor_mask_.bool()] = 62
    ###########################
    if tumor_size > 0:
        # get organ mask
        organ_mask = (volume == 62).float()
        organ_mask = dilate_one_img(organ_mask.squeeze(0), erosion=5, pad_value=1.0)
        organ_mask = erode_one_img(organ_mask, erosion=5, pad_value=1.0).unsqueeze(0)
        
        count = 0
        while True:
            threshold = 0.8
            tumor_mask = tumor_mask_
            if count < 20:
                # apply random augmentation
                distorted_mask = elastic(
                    (tumor_mask == 1).cuda(), spatial_size=tuple(spatial_size)
                ).as_tensor()
                tumor_mask = distorted_mask * organ_mask
            elif 20 <= count < 40:
                threshold = 0.75
            else:
                break

            # the tumor must be within the organ
            tumor_mask = tumor_mask * organ_mask
            print(torch.sum(tumor_mask), "|", tumor_size * threshold)
            count += 1
            if torch.sum(tumor_mask) >= tumor_size * threshold:
                tumor_mask = dilate_one_img(tumor_mask.squeeze(0), erosion=5, pad_value=1.0)
                tumor_mask = (
                    erode_one_img(tumor_mask, erosion=5, pad_value=1.0).unsqueeze(0).to(torch.uint8)
                )
                break
    else:
        tumor_mask = tumor_mask_

    volume[tumor_mask == 1] = tumor_label[0]

    whole_mask = volume.unsqueeze(0)
    return whole_mask


def augmentation_body(whole_mask: Tensor) -> Tensor:
    """
    Whole body mask augmentation.

    Args:
        whole_mask: input 3D multi-label mask, [1,H,W,D] torch tensor.

    Return:
        augmented mask, with same shape and data type as whole_mask.

    Example:

        .. code-block:: python

            # define a multi-label mask
            whole_mask = torch.randint(0, 132, [1,128,128,128])
            augmented_whole_mask = augment_body(whole_mask)
    """
    volume = whole_mask.squeeze(0)

    # Define augmentation transform
    zoom = RandZoom(
        min_zoom=0.99,
        max_zoom=1.01,
        mode="nearest",
        align_corners=None,
        prob=1.0,
    )
    # apply random augmentation
    volume = zoom(volume)

    whole_mask = volume.unsqueeze(0)
    return whole_mask


def augmentation(whole_mask: Tensor, spatial_size: tuple[int, int, int] | int | None = None) -> Tensor:
    """
    Tumor or whole body mask augmentation. If tumor exist, augment tumor mask; if not, augment whole body mask

    Args:
        whole_mask: input 3D multi-label mask, [1,H,W,D] torch tensor.
        spatial_size: output image spatial size, used in random transform. If not defined, will use (H,W,D). If some components are non-positive values, the transform will use the corresponding components of whole_mask size. For example, spatial_size=(128, 128, -1) will be adapted to (128, 128, 64) if the third spatial dimension size of whole_mask is 64.

    Return:
        augmented mask, with shape of spatial_size and data type as whole_mask.

    Example:

        .. code-block:: python

            # define a multi-label mask
            whole_mask = torch.randint(0, 132, [1,128,128,128])
            augmented_whole_mask = augmentation(whole_mask)
    """
    label_list = torch.unique(whole_mask)
    label_list = list(label_list.cpu().numpy())

    # Note that we only augment one type of tumor.
    if 128 in label_list:
        print(f"augmenting bone lesion/tumor")
        whole_mask = augmentation_bone_tumor(whole_mask, spatial_size)
    elif 26 in label_list:
        print(f"augmenting liver tumor")
        whole_mask = augmentation_liver_tumor(whole_mask, spatial_size)
    elif 23 in label_list:
        print(f"augmenting lung tumor")
        whole_mask = augmentation_lung_tumor(whole_mask, spatial_size)
    elif 24 in label_list:
        print(f"augmenting pancreas tumor")
        whole_mask = augmentation_pancreas_tumor(whole_mask, spatial_size)
    elif 27 in label_list:
        print(f"augmenting colon tumor")
        whole_mask = augmentation_colon_tumor(whole_mask, spatial_size)
    else:
        print(f"augmenting body")
        whole_mask = augmentation_body(whole_mask)

    return whole_mask
