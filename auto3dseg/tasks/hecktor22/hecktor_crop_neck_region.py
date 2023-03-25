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

import warnings
import torch
import numpy as np

from monai.data import MetaTensor
from monai.utils.misc import ImageMetaKey
from monai.transforms import CropForegroundd


class HecktorCropNeckRegion(CropForegroundd):
    """
    A simple pre-processing transform to approximately crop the head and neck region based on a PET image.
    This transform relies on several assumptions of patient orientation with a head location on the top,
    and is specific for Hecktor22 dataset, and should not be used for an arbitrary PET image pre-processing.
    """

    def __init__(
        self,
        keys=["image", "image2", "label"],
        source_key="image",
        box_size=[200, 200, 310],
        allow_missing_keys=True,
        **kwargs,
    ) -> None:
        super().__init__(keys=keys, source_key=source_key, allow_missing_keys=allow_missing_keys, **kwargs)
        self.box_size = box_size

    def __call__(self, data):

        d = dict(data)

        im_pet = d["image2"][0]
        box_size = np.array(self.box_size)  # H&N region to crop in mm , defaults to 200x200x310mm
        filename = ""

        if isinstance(im_pet, MetaTensor):
            filename = im_pet.meta[ImageMetaKey.FILENAME_OR_OBJ]
            box_size = (box_size / np.array(im_pet.pixdim)).astype(int)  # compensate for resolution

        box_start, box_end = self.extract_roi(im_pet=im_pet, box_size=box_size)

        use_label = "label" in d and "label" in self.keys and (d["image"].shape[1:] == d["label"].shape[1:])

        if use_label:
            # if label mask is available, let's check if the cropped region includes all foreground
            before_sum = d["label"].sum().item()
            after_sum = (
                (d["label"][0, box_start[0] : box_end[0], box_start[1] : box_end[1], box_start[2] : box_end[2]])
                .sum()
                .item()
            )
            if before_sum != after_sum:
                warnings.warn(
                    "WARNING, H&N crop could be incorrect!!! ",
                    before_sum,
                    after_sum,
                    "image:",
                    d["image"].shape,
                    "pet:",
                    d["image2"].shape,
                    "label:",
                    d["label"].shape,
                    "updated box_size",
                    box_size,
                    "box_start",
                    box_start,
                    "box_end:",
                    box_end,
                    "filename",
                    filename,
                )

        d[self.start_coord_key] = box_start
        d[self.end_coord_key] = box_end

        for key, m in self.key_iterator(d, self.mode):
            if key == "label" and not use_label:
                continue
            d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m)

        return d

    def extract_roi(self, im_pet, box_size):

        crop_len = int(0.75 * im_pet.shape[2])
        im = im_pet[..., crop_len:]

        mask = ((im - im.mean()) / im.std()) > 1
        comp_idx = torch.argwhere(mask)

        center = torch.mean(comp_idx.float(), dim=0).cpu().int().numpy()
        xmin = torch.min(comp_idx, dim=0).values.cpu().int().numpy()
        xmax = torch.max(comp_idx, dim=0).values.cpu().int().numpy()

        xmin[:2] = center[:2] - box_size[:2] // 2
        xmax[:2] = center[:2] + box_size[:2] // 2

        xmax[2] = xmax[2] + crop_len
        xmin[2] = max(0, xmax[2] - box_size[2])

        return xmin.astype(int), xmax.astype(int)
