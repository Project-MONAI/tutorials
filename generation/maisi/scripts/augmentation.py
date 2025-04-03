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
import torch
import torch.nn.functional as F
from monai.transforms import Rand3DElastic, RandAffine, RandZoom
from monai.utils import ensure_tuple_rep


def erode3d(input_tensor, erosion=3):
    # Define the structuring element
    erosion = ensure_tuple_rep(erosion, 3)
    structuring_element = torch.ones(1, 1, erosion[0], erosion[1], erosion[2]).to(input_tensor.device)

    # Pad the input tensor to handle border pixels
    input_padded = F.pad(
        input_tensor.float().unsqueeze(0).unsqueeze(0),
        (erosion[0] // 2, erosion[0] // 2, erosion[1] // 2, erosion[1] // 2, erosion[2] // 2, erosion[2] // 2),
        mode="constant",
        value=1.0,
    )

    # Apply erosion operation
    output = F.conv3d(input_padded, structuring_element, padding=0)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output == torch.sum(structuring_element), 1.0, 0.0)

    return output.squeeze(0).squeeze(0)


def dilate3d(input_tensor, erosion=3):
    # Define the structuring element
    erosion = ensure_tuple_rep(erosion, 3)
    structuring_element = torch.ones(1, 1, erosion[0], erosion[1], erosion[2]).to(input_tensor.device)

    # Pad the input tensor to handle border pixels
    input_padded = F.pad(
        input_tensor.float().unsqueeze(0).unsqueeze(0),
        (erosion[0] // 2, erosion[0] // 2, erosion[1] // 2, erosion[1] // 2, erosion[2] // 2, erosion[2] // 2),
        mode="constant",
        value=1.0,
    )

    # Apply erosion operation
    output = F.conv3d(input_padded, structuring_element, padding=0)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output > 0, 1.0, 0.0)

    return output.squeeze(0).squeeze(0)


def augmentation_tumor_bone(pt_nda, output_size, random_seed=None):
    volume = pt_nda.squeeze(0)
    real_l_volume_ = torch.zeros_like(volume)
    real_l_volume_[volume == 128] = 1
    real_l_volume_ = real_l_volume_.to(torch.uint8)

    elastic = RandAffine(
        mode="nearest",
        prob=1.0,
        translate_range=(5, 5, 0),
        rotate_range=(0, 0, 0.1),
        scale_range=(0.15, 0.15, 0),
        padding_mode="zeros",
    )
    elastic.set_random_state(seed=random_seed)

    tumor_szie = torch.sum((real_l_volume_ > 0).float())
    ###########################
    # remove pred in pseudo_label in real lesion region
    volume[real_l_volume_ > 0] = 200
    ###########################
    if tumor_szie > 0:
        # get organ mask
        organ_mask = (
            torch.logical_and(33 <= volume, volume <= 56).float()
            + torch.logical_and(63 <= volume, volume <= 97).float()
            + (volume == 127).float()
            + (volume == 114).float()
            + real_l_volume_
        )
        organ_mask = (organ_mask > 0).float()
        cnt = 0
        while True:
            threshold = 0.8 if cnt < 40 else 0.75
            real_l_volume = real_l_volume_
            # random distor mask
            distored_mask = elastic((real_l_volume > 0).cuda(), spatial_size=tuple(output_size)).as_tensor()
            real_l_volume = distored_mask * organ_mask
            cnt += 1
            print(torch.sum(real_l_volume), "|", tumor_szie * threshold)
            if torch.sum(real_l_volume) >= tumor_szie * threshold:
                real_l_volume = dilate3d(real_l_volume.squeeze(0), erosion=5)
                real_l_volume = erode3d(real_l_volume, erosion=5).unsqueeze(0).to(torch.uint8)
                break
    else:
        real_l_volume = real_l_volume_

    volume[real_l_volume == 1] = 128

    pt_nda = volume.unsqueeze(0)
    return pt_nda


def augmentation_tumor_liver(pt_nda, output_size, random_seed=None):
    volume = pt_nda.squeeze(0)
    real_l_volume_ = torch.zeros_like(volume)
    real_l_volume_[volume == 1] = 1
    real_l_volume_[volume == 26] = 2
    real_l_volume_ = real_l_volume_.to(torch.uint8)

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
    elastic.set_random_state(seed=random_seed)

    tumor_szie = torch.sum(real_l_volume_ == 2)
    ###########################
    # remove pred  organ labels
    volume[volume == 1] = 0
    volume[volume == 26] = 0
    # before move tumor maks, full the original location by organ labels
    volume[real_l_volume_ == 1] = 1
    volume[real_l_volume_ == 2] = 1
    ###########################
    while True:
        real_l_volume = real_l_volume_
        # random distor mask
        real_l_volume = elastic((real_l_volume == 2).cuda(), spatial_size=tuple(output_size)).as_tensor()
        # get organ mask
        organ_mask = (real_l_volume_ == 1).float() + (real_l_volume_ == 2).float()

        organ_mask = dilate3d(organ_mask.squeeze(0), erosion=5)
        organ_mask = erode3d(organ_mask, erosion=5).unsqueeze(0)
        real_l_volume = real_l_volume * organ_mask
        print(torch.sum(real_l_volume), "|", tumor_szie * 0.80)
        if torch.sum(real_l_volume) >= tumor_szie * 0.80:
            real_l_volume = dilate3d(real_l_volume.squeeze(0), erosion=5)
            real_l_volume = erode3d(real_l_volume, erosion=5).unsqueeze(0)
            break

    volume[real_l_volume == 1] = 26

    pt_nda = volume.unsqueeze(0)
    return pt_nda


def augmentation_tumor_lung(pt_nda, output_size, random_seed=None):
    volume = pt_nda.squeeze(0)
    real_l_volume_ = torch.zeros_like(volume)
    real_l_volume_[volume == 23] = 1
    real_l_volume_ = real_l_volume_.to(torch.uint8)

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
    elastic.set_random_state(seed=random_seed)

    tumor_szie = torch.sum(real_l_volume_)
    # before move lung tumor maks, full the original location by lung labels
    new_real_l_volume_ = dilate3d(real_l_volume_.squeeze(0), erosion=3)
    new_real_l_volume_ = new_real_l_volume_.unsqueeze(0)
    new_real_l_volume_[real_l_volume_ > 0] = 0
    new_real_l_volume_[volume < 28] = 0
    new_real_l_volume_[volume > 32] = 0
    tmp = volume[(volume * new_real_l_volume_).nonzero(as_tuple=True)].view(-1)

    mode = torch.mode(tmp, 0)[0].item()
    print(mode)
    assert 28 <= mode <= 32
    volume[real_l_volume_.bool()] = mode
    ###########################
    if tumor_szie > 0:
        # aug
        while True:
            real_l_volume = real_l_volume_
            # random distor mask
            real_l_volume = elastic(real_l_volume, spatial_size=tuple(output_size)).as_tensor()
            # get lung mask v2 (133 order)
            lung_mask = (
                (volume == 28).float()
                + (volume == 29).float()
                + (volume == 30).float()
                + (volume == 31).float()
                + (volume == 32).float()
            )

            lung_mask = dilate3d(lung_mask.squeeze(0), erosion=5)
            lung_mask = erode3d(lung_mask, erosion=5).unsqueeze(0)
            real_l_volume = real_l_volume * lung_mask
            print(torch.sum(real_l_volume), "|", tumor_szie * 0.85)
            if torch.sum(real_l_volume) >= tumor_szie * 0.85:
                real_l_volume = dilate3d(real_l_volume.squeeze(0), erosion=5)
                real_l_volume = erode3d(real_l_volume, erosion=5).unsqueeze(0).to(torch.uint8)
                break
    else:
        real_l_volume = real_l_volume_

    volume[real_l_volume == 1] = 23

    pt_nda = volume.unsqueeze(0)
    return pt_nda


def augmentation_tumor_pancreas(pt_nda, output_size, random_seed=None):
    volume = pt_nda.squeeze(0)
    real_l_volume_ = torch.zeros_like(volume)
    real_l_volume_[volume == 4] = 1
    real_l_volume_[volume == 24] = 2
    real_l_volume_ = real_l_volume_.to(torch.uint8)

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
    elastic.set_random_state(seed=random_seed)

    tumor_szie = torch.sum(real_l_volume_ == 2)
    ###########################
    # remove pred  organ labels
    volume[volume == 24] = 0
    volume[volume == 4] = 0
    # before move tumor maks, full the original location by organ labels
    volume[real_l_volume_ == 1] = 4
    volume[real_l_volume_ == 2] = 4
    ###########################
    while True:
        real_l_volume = real_l_volume_
        # random distor mask
        real_l_volume = elastic((real_l_volume == 2).cuda(), spatial_size=tuple(output_size)).as_tensor()
        # get organ mask
        organ_mask = (real_l_volume_ == 1).float() + (real_l_volume_ == 2).float()

        organ_mask = dilate3d(organ_mask.squeeze(0), erosion=5)
        organ_mask = erode3d(organ_mask, erosion=5).unsqueeze(0)
        real_l_volume = real_l_volume * organ_mask
        print(torch.sum(real_l_volume), "|", tumor_szie * 0.80)
        if torch.sum(real_l_volume) >= tumor_szie * 0.80:
            real_l_volume = dilate3d(real_l_volume.squeeze(0), erosion=5)
            real_l_volume = erode3d(real_l_volume, erosion=5).unsqueeze(0)
            break

    volume[real_l_volume == 1] = 24

    pt_nda = volume.unsqueeze(0)
    return pt_nda


def augmentation_tumor_colon(pt_nda, output_size, random_seed=None):
    volume = pt_nda.squeeze(0)
    real_l_volume_ = torch.zeros_like(volume)
    real_l_volume_[volume == 27] = 1
    real_l_volume_ = real_l_volume_.to(torch.uint8)

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
    elastic.set_random_state(seed=random_seed)

    tumor_szie = torch.sum(real_l_volume_)
    ###########################
    # before move tumor maks, full the original location by organ labels
    volume[real_l_volume_.bool()] = 62
    ###########################
    if tumor_szie > 0:
        # get organ mask
        organ_mask = (volume == 62).float()
        organ_mask = dilate3d(organ_mask.squeeze(0), erosion=5)
        organ_mask = erode3d(organ_mask, erosion=5).unsqueeze(0)
        #         cnt = 0
        cnt = 0
        while True:
            threshold = 0.8
            real_l_volume = real_l_volume_
            if cnt < 20:
                # random distor mask
                distored_mask = elastic((real_l_volume == 1).cuda(), spatial_size=tuple(output_size)).as_tensor()
                real_l_volume = distored_mask * organ_mask
            elif 20 <= cnt < 40:
                threshold = 0.75
            else:
                break

            real_l_volume = real_l_volume * organ_mask
            print(torch.sum(real_l_volume), "|", tumor_szie * threshold)
            cnt += 1
            if torch.sum(real_l_volume) >= tumor_szie * threshold:
                real_l_volume = dilate3d(real_l_volume.squeeze(0), erosion=5)
                real_l_volume = erode3d(real_l_volume, erosion=5).unsqueeze(0).to(torch.uint8)
                break
    else:
        real_l_volume = real_l_volume_
    #     break
    volume[real_l_volume == 1] = 27

    pt_nda = volume.unsqueeze(0)
    return pt_nda


def augmentation_body(pt_nda, random_seed=None):
    volume = pt_nda.squeeze(0)

    zoom = RandZoom(min_zoom=0.99, max_zoom=1.01, mode="nearest", align_corners=None, prob=1.0)
    zoom.set_random_state(seed=random_seed)

    volume = zoom(volume)

    pt_nda = volume.unsqueeze(0)
    return pt_nda


def augmentation(pt_nda, output_size, random_seed=None):
    label_list = torch.unique(pt_nda)
    label_list = list(label_list.cpu().numpy())

    if 128 in label_list:
        print("augmenting bone lesion/tumor")
        pt_nda = augmentation_tumor_bone(pt_nda, output_size, random_seed)
    elif 26 in label_list:
        print("augmenting liver tumor")
        pt_nda = augmentation_tumor_liver(pt_nda, output_size, random_seed)
    elif 23 in label_list:
        print("augmenting lung tumor")
        pt_nda = augmentation_tumor_lung(pt_nda, output_size, random_seed)
    elif 24 in label_list:
        print("augmenting pancreas tumor")
        pt_nda = augmentation_tumor_pancreas(pt_nda, output_size, random_seed)
    elif 27 in label_list:
        print("augmenting colon tumor")
        pt_nda = augmentation_tumor_colon(pt_nda, output_size, random_seed)
    else:
        print("augmenting body")
        pt_nda = augmentation_body(pt_nda, random_seed)

    return pt_nda
