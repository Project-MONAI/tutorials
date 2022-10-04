import os
from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# data path
cfg.data_dir = "/workspace/data/ranzcr/"
cfg.data_folder = cfg.data_dir + "train/"
cfg.train_df = cfg.data_dir + "train_folds.csv"
cfg.test_df = cfg.data_dir + "sample_submission.csv"
cfg.output_dir = "./output/weights/"

# dataset
cfg.batch_size = 4
cfg.img_size = (896, 896)
cfg.train_aug = None
cfg.val_aug = None

cfg.label_cols = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]
cfg.num_classes = len(cfg.label_cols)

# mask
cfg.thickness = 32
cfg.seg_weight = 50

# model
cfg.backbone = "tf_efficientnet_b8_ap"
cfg.pretrained = True
cfg.pretrained_weights = None
cfg.train = True
cfg.seg_dim = 3
cfg.image_extension = ".jpg"

# training
cfg.fold = -1
cfg.lr = 1e-4
cfg.weight_decay = 0
cfg.epochs = 15
cfg.seed = -1
cfg.calc_loss = True
cfg.train_val = False
cfg.eval_epochs = 1
cfg.eval_train_epochs = 20
cfg.warmup = 5
cfg.compute_auc = True

# ressources
cfg.find_unused_parameters = True
cfg.mixed_precision = True
cfg.grad_accumulation = 1
cfg.gpu = 0
cfg.device = "cuda:%d" % cfg.gpu
cfg.num_workers = 8
cfg.drop_last = True

basic_cfg = cfg
