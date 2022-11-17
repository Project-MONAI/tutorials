from default_config import basic_cfg
from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged,
                              RandFlipd, RandRotate90d, Resized)

cfg = basic_cfg

cfg.output_dir = "./output/"
cfg.data_dir = "/raid/surg/image640_blur/"
cfg.backbone = "efficientnet-b4"
cfg.train_df = "cleaned_clf_train_data.csv"
cfg.img_size = (640, 640)
cfg.num_classes = 14
cfg.lr = 0.001
cfg.epochs = 5
cfg.oversample_rate = 4
cfg.clf_threshold = 0.4

cfg.train_aug = Compose(
    [
        LoadImaged(keys="input", image_only=True),
        EnsureChannelFirstd(keys="input"),
        Resized(
            keys="input",
            spatial_size=cfg.img_size,
            mode="bilinear",
            align_corners=False,
        ),
        RandFlipd(keys="input", prob=0.5, spatial_axis=0),
        RandFlipd(keys="input", prob=0.5, spatial_axis=1),
        RandRotate90d(keys="input", prob=0.5),
    ]
)

cfg.val_aug = Compose(
    [
        LoadImaged(keys="input", image_only=True),
        EnsureChannelFirstd(keys="input"),
        Resized(
            keys="input",
            spatial_size=cfg.img_size,
            mode="bilinear",
            align_corners=False,
        ),
    ]
)
