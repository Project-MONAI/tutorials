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

from typing import Sequence

from monai.apps.generation.maisi.utils.morphological_ops import dilate, erode
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
