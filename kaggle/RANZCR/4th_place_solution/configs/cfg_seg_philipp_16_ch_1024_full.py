from default_config import basic_cfg
import albumentations as A
import os
import cv2

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
cfg.normalization = 'image'
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


cfg.train_aug = A.Compose([A.HorizontalFlip(p=0.5),
                           A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.5),
                           A.LongestMaxSize(1120,p=1),
                            A.PadIfNeeded(1120, 1120, border_mode=cv2.BORDER_CONSTANT,p=1),
                            A.RandomCrop(always_apply=False, p=1.0, height=cfg.img_size[0], width=cfg.img_size[1]), 
                            #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
                            A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.5),
                            A.InvertImg(p=0.5),
                           A.Cutout(num_holes=8, max_h_size=102, max_w_size=102, p=0.5),
                          ])

                           #A.RandomResizedCrop(cfg.img_size[0], cfg.img_size[1], scale=(0.85, 1.0)),

cfg.val_aug = A.Compose([#A.PadIfNeeded (min_height=256, min_width=940),
                            A.LongestMaxSize(1120,p=1),
                            A.PadIfNeeded(1120, 1120, border_mode=cv2.BORDER_CONSTANT,p=1),
                            A.CenterCrop(always_apply=False, p=1.0, height=cfg.img_size[0], width=cfg.img_size[1]), 
                         #A.Resize(cfg.img_size[0],cfg.img_size[1])
                         ])
