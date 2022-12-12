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

cfg.lr = 0.001
cfg.backbone = "efficientnet_b7"
cfg.img_size = (1024, 1024)
cfg.grad_accumulation = 3

cfg.thickness = [32, 96]


cfg.train_aug = Compose(
    [
        Resized(
            keys=("input", "mask"),
            spatial_size=1120,
            size_mode="longest",
            mode="bilinear",
            align_corners=False,
        ),
        SpatialPadd(keys=("input", "mask"), spatial_size=(1120, 1120)),
        RandFlipd(keys=("input", "mask"), prob=0.5, spatial_axis=1),
        RandAffined(
            keys=("input", "mask"),
            prob=0.5,
            rotate_range=np.pi / 14.4,
            translate_range=(70, 70),
            scale_range=(0.1, 0.1),
        ),
        RandSpatialCropd(
            keys=("input", "mask"),
            roi_size=(cfg.img_size[0], cfg.img_size[1]),
            random_size=False,
        ),
        RandScaleIntensityd(keys="input", factors=(-0.2, 0.2), prob=0.5),
        RandShiftIntensityd(keys="input", offsets=(-51, 51), prob=0.5),
        RandLambdad(keys="input", func=lambda x: 255 - x, prob=0.5),
        RandCoarseDropoutd(
            keys=("input", "mask"),
            holes=8,
            spatial_size=(1, 1),
            max_spatial_size=(102, 102),
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
            spatial_size=1120,
            size_mode="longest",
            mode="bilinear",
            align_corners=False,
        ),
        SpatialPadd(keys=("input", "mask"), spatial_size=(1120, 1120)),
        CenterSpatialCropd(
            keys=("input", "mask"), roi_size=(cfg.img_size[0], cfg.img_size[1])
        ),
        NormalizeIntensityd(keys="input", nonzero=False),
        Lambdad(keys="input", func=lambda x: x.clip(-20, 20)),
    ]
)

cfg.test_aug = Compose(
    [
        Resized(
            keys=("input", "mask"),
            spatial_size=1120,
            size_mode="longest",
            mode="bilinear",
            align_corners=False,
        ),
        SpatialPadd(keys=("input", "mask"), spatial_size=(1120, 1120)),
        NormalizeIntensityd(keys="input", nonzero=False),
        Lambdad(keys="input", func=lambda x: x.clip(-20, 20)),
    ]
)
