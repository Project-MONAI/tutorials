from types import SimpleNamespace
from copy import deepcopy

cfg = SimpleNamespace(**{})

#dataset
cfg.dataset = 'rfcx_dataset'
cfg.batch_size = 32
cfg.normalization = 'image'
cfg.img_size = (256,256,1)
cfg.train_aug = None
cfg.val_aug = None
cfg.test_augs = None
cfg.cache_n_img = 0
cfg.cache = False
cfg.train_df2 = None
cfg.label_cols = ['ETT - Abnormal',
                  'ETT - Borderline',
                  'ETT - Normal',
                           'NGT - Abnormal',
                           'NGT - Borderline',
                           'NGT - Incompletely Imaged',
                           'NGT - Normal',
                           'CVC - Abnormal',
                           'CVC - Borderline',
                           'CVC - Normal',
                           'Swan Ganz Catheter Present']

#mask
cfg.thickness = 32
cfg.points_mask = False
cfg.smooth_mask = 0
cfg.rgb = False
cfg.seg_weight = 1
cfg.annotated_only = False
#model
cfg.backbone = 'resnet18'
cfg.pretrained = True
cfg.pretrained_weights = None
cfg.pool = 'mean'
cfg.train = True

cfg.image_extension = ".jpg"

cfg.epoch_weights = None

cfg.calc_loss = True

cfg.bce_seg = "bce"

cfg.alpha = 1
cfg.cls_loss_pos_weight = None
cfg.train_val = True
cfg.eval_epochs = 1
cfg.eval_train_epochs = 1
cfg.drop_path_rate = None
cfg.warmup = 0

cfg.pseudo_df = None

cfg.label_smoothing = 0

cfg.ignore_seg = False

cfg.remove_border = False

#training
cfg.fold = 0
cfg.lr = 1e-4
cfg.schedule = 'cosine'
cfg.weight_decay = 0
cfg.optimizer = 'Adam' # "Adam", "fused_Adam", "SGD", "fused_SGD"
cfg.epochs = 10
cfg.seed = -1
cfg.resume_training = False
cfg.simple_eval = False
cfg.do_test = True
cfg.do_seg = False
cfg.eval_ddp = True
cfg.clip_grad = 0

#ressources
cfg.find_unused_parameters = False
cfg.mixed_precision = True
cfg.grad_accumulation = 1
cfg.syncbn = False
cfg.gpu = 0
cfg.dp = False
cfg.num_workers = 4
cfg.drop_last = True            
#logging,
cfg.neptune_project = None
cfg.tags = None


basic_cfg = cfg











