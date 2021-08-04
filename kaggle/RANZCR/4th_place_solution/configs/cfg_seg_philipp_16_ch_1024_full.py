from default_config import basic_cfg

import os

from monai.transforms import (
    Resized,
    SpatialPadd,
    RandFlipd,
    RandAffined,
    RandSpatialCropd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandLambdad,
    RandCoarseDropoutd,
    CenterSpatialCropd,
    CastToTyped,
    Compose,
    EnsureTyped,
    NormalizeIntensityd,
    Lambdad,
)

import numpy as np

cfg = basic_cfg
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.data_dir = '/workspace/data/ranzcr/'
cfg.data_folder = cfg.data_dir + 'train/'
cfg.train_df = '/workspace/data/ranzcr/train_folds.csv'
cfg.output_dir = f"./output/{os.path.basename(__file__).split('.')[0]}"

cfg.lr = 0.001
cfg.epochs = 15
cfg.warmup = 5
cfg.batch_size = 4
#cfg.syncbn = True
cfg.model = 'segment_philipp_v2'
cfg.backbone = 'tf_efficientnet_b7_ns'
cfg.dataset = 'ranzcr_ds_seg_philipp_v5'
cfg.gpu = 0
cfg.num_workers = 8
cfg.tags = 'segment'
cfg.fold = -1
cfg.img_size = (1024,1024)
# cfg.weight_decay = 0.0001
cfg.drop_last=True
cfg.grad_accumulation = 3

cfg.eval_epochs = 1
cfg.train_val = True
cfg.eval_train_epochs = 10

cfg.seg_weight = 50

cfg.dropout = 0

cfg.do_seg = False

cfg.seg_dim = 3

# cfg.clip_grad = 5

cfg.optimizer = "Adam"
cfg.weight_decay = 0

cfg.tubemix = False
cfg.tubemix_proba = 0.75

cfg.tubedrop = False

cfg.reduction = "avg"

cfg.mask_mode = "tube"

cfg.pool = 'max'
#cfg.clip_grad = 5

cfg.train = True

cfg.thickness = [32,96]

# cfg.device = "cuda:0"

cfg.find_unused_parameters = True

cfg.eval_ddp = True

cfg.train_aug = Compose([
    Resized(keys=("input", "mask"), spatial_size=1120, size_mode="longest", mode="bilinear", align_corners=False),
    SpatialPadd(keys=("input", "mask"), spatial_size=(1120, 1120)),
    RandFlipd(keys=("input", "mask"), prob=0.5, spatial_axis=1),
    RandAffined(keys=("input", "mask"), prob=0.5, rotate_range=np.pi/14.4, translate_range=(70, 70), scale_range=(0.1, 0.1), as_tensor_output=False),
    RandSpatialCropd(keys=("input", "mask"), roi_size=(cfg.img_size[0], cfg.img_size[1]), random_size=False),
    RandScaleIntensityd(keys="input", factors=(-0.2,0.2), prob=0.5),
    RandShiftIntensityd(keys="input", offsets=(-51, 51), prob=0.5),
    RandLambdad(keys="input", func=lambda x: 255 - x, prob=0.5),
    RandCoarseDropoutd(keys=("input", "mask"), holes=8, spatial_size=(1, 1), max_spatial_size=(102, 102), prob=0.5),
    CastToTyped(keys="input", dtype=np.float32),
    NormalizeIntensityd(keys="input", nonzero=False),
    Lambdad(keys="input", func=lambda x: x.clip(-20, 20)),
    EnsureTyped(keys=("input", "mask")),
])

cfg.val_aug = Compose([
    Resized(keys=("input", "mask"), spatial_size=1120, size_mode="longest", mode="bilinear", align_corners=False),
    SpatialPadd(keys=("input", "mask"), spatial_size=(1120, 1120)),
    CenterSpatialCropd(keys=("input", "mask"), roi_size=(cfg.img_size[0], cfg.img_size[1])),
    CastToTyped(keys="input", dtype=np.float32),
    NormalizeIntensityd(keys="input", nonzero=False),
    Lambdad(keys="input", func=lambda x: x.clip(-20, 20)),
    EnsureTyped(keys=("input", "mask")),
])
