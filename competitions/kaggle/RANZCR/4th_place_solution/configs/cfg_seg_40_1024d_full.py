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
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    Lambdad,
    NormalizeIntensityd,
    RandAffined,
    RandCoarseDropoutd,
    RandFlipd,
    RandLambdad,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
    SpatialPadd,
)

from default_config import basic_cfg

cfg = basic_cfg

cfg.lr = 0.0006
cfg.backbone = "efficientnet_b8_ap"
cfg.img_size = (896, 896)
cfg.grad_accumulation = 2

cfg.train_aug = Compose(
    [
        Resized(
            keys=("input", "mask"),
            spatial_size=1008,
            size_mode="longest",
            mode="bilinear",
            align_corners=False,
        ),
        SpatialPadd(keys=("input", "mask"), spatial_size=(1008, 1008)),
        RandFlipd(keys=("input", "mask"), prob=0.5, spatial_axis=1),
        RandAffined(
            keys=("input", "mask"),
            prob=0.5,
            rotate_range=np.pi / 14.4,
            translate_range=(63, 63),
            scale_range=(0.1, 0.1),
        ),
        RandSpatialCropd(
            keys=("input", "mask"),
            roi_size=(cfg.img_size[0], cfg.img_size[1]),
            random_size=False,
        ),
        RandScaleIntensityd(keys="input", factors=(-0.2, 0.2), prob=0.7),
        RandShiftIntensityd(keys="input", offsets=(-51, 51), prob=0.7),
        RandLambdad(keys="input", func=lambda x: 255 - x, prob=0.5),
        RandCoarseDropoutd(
            keys=("input", "mask"),
            holes=8,
            spatial_size=(1, 1),
            max_spatial_size=(84, 84),
            prob=0.5,
        ),
        NormalizeIntensityd(keys="input", nonzero=False),
        Lambdad(keys="input", func=lambda x: x.clip(-20, 20)),
    ]
)

cfg.val_aug = Compose(
    [
        Resized(
            keys=("input", "mask"),
            spatial_size=1008,
            size_mode="longest",
            mode="bilinear",
            align_corners=False,
        ),
        SpatialPadd(keys=("input", "mask"), spatial_size=(1008, 1008)),
        CenterSpatialCropd(keys=("input", "mask"), roi_size=(cfg.img_size[0], cfg.img_size[1])),
        NormalizeIntensityd(keys="input", nonzero=False),
        Lambdad(keys="input", func=lambda x: x.clip(-20, 20)),
    ]
)

cfg.test_aug = Compose(
    [
        Resized(
            keys=("input", "mask"),
            spatial_size=1008,
            size_mode="longest",
            mode="bilinear",
            align_corners=False,
        ),
        SpatialPadd(keys=("input", "mask"), spatial_size=(1008, 1008)),
        NormalizeIntensityd(keys="input", nonzero=False),
        Lambdad(keys="input", func=lambda x: x.clip(-20, 20)),
    ]
)
