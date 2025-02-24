from types import SimpleNamespace
from monai import transforms as mt

cfg = SimpleNamespace(**{})

# stages
cfg.train = True
cfg.val = True
cfg.test = True
cfg.train_val = True

# dataset
cfg.batch_size_val = None
cfg.use_custom_batch_sampler = False
cfg.val_df = None
cfg.test_df = None
cfg.val_data_folder = None
cfg.train_aug = None
cfg.val_aug = None
cfg.data_sample = -1

# model

cfg.pretrained = False
cfg.pretrained_weights = None
cfg.pretrained_weights_strict = True
cfg.pop_weights = None
cfg.compile_model = False

# training routine
cfg.fold = 0
cfg.optimizer = "Adam"
cfg.sgd_momentum = 0
cfg.sgd_nesterov = False
cfg.lr = 1e-4
cfg.schedule = "cosine"
cfg.num_cycles = 0.5
cfg.weight_decay = 0
cfg.epochs = 10
cfg.seed = -1
cfg.resume_training = False
cfg.distributed = False
cfg.clip_grad = 0
cfg.save_val_data = True
cfg.gradient_checkpointing = False
cfg.apex_ddp = False
cfg.synchronize_step = True

# eval
cfg.eval_ddp = True
cfg.calc_metric = True
cfg.calc_metric_epochs = 1
cfg.eval_steps = 0
cfg.eval_epochs = 1
cfg.save_pp_csv = True


# ressources
cfg.find_unused_parameters = False
cfg.grad_accumulation = 1
cfg.syncbn = False
cfg.gpu = 0
cfg.dp = False
cfg.num_workers = 8
cfg.drop_last = True
cfg.save_checkpoint = True
cfg.save_only_last_ckpt = False
cfg.save_weights_only = False

# logging,
cfg.neptune_project = None
cfg.neptune_connection_mode = "debug"
cfg.save_first_batch = False
cfg.save_first_batch_preds = False
cfg.clip_mode = "norm"
cfg.data_sample = -1
cfg.track_grad_norm = True
cfg.grad_norm_type = 2.
cfg.track_weight_norm = True
cfg.norm_eps = 1e-4
cfg.disable_tqdm = False




# paths

cfg.data_folder = '/mount/cryo/data/czii-cryo-et-object-identification/train/static/ExperimentRuns/'
cfg.train_df = 'train_folded_v1.csv'


# stages
cfg.test = False
cfg.train = True
cfg.train_val = False

#logging
cfg.neptune_project = None
cfg.neptune_connection_mode = "async"

#model
cfg.model = "mdl_1"
cfg.mixup_p = 1.
cfg.mixup_beta = 1.
cfg.in_channels = 1
cfg.pretrained = False

#data
cfg.dataset = "ds_1"
cfg.classes = ['apo-ferritin','beta-amylase','beta-galactosidase','ribosome','thyroglobulin','virus-like-particle']
cfg.n_classes = len(cfg.classes)

cfg.post_process_pipeline = 'pp_1'
cfg.metric = 'metric_1'



cfg.particle_radi = {'apo-ferritin':60,
        'beta-amylase':65,
        'beta-galactosidase':90,
        'ribosome':150,
        'thyroglobulin':130,
        'virus-like-particle':135
       }

cfg.voxel_spacing = 10.0


# OPTIMIZATION & SCHEDULE

cfg.fold = 0
cfg.epochs = 10

cfg.lr = 1e-3
cfg.optimizer = "Adam"
cfg.weight_decay = 0.
cfg.warmup = 0.
cfg.batch_size = 8
cfg.batch_size_val = 16
cfg.sub_batch_size = 4
cfg.roi_size = [96,96,96]
cfg.train_sub_epochs = 1112
cfg.val_sub_epochs = 1
cfg.mixed_precision = False
cfg.bf16 = True
cfg.force_fp16 = True
cfg.pin_memory = False
cfg.grad_accumulation = 1.
cfg.num_workers = 8






#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = False
cfg.save_val_data = False
cfg.save_checkpoint=True
cfg.save_pp_csv = False



cfg.static_transforms = static_transforms = mt.Compose([mt.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),mt.NormalizeIntensityd(keys="image"),])
cfg.train_aug = mt.Compose([mt.RandSpatialCropSamplesd(keys=["image", "label"],
                                                                        roi_size=cfg.roi_size,
                                                                        num_samples=cfg.sub_batch_size),
                                mt.RandFlipd(
                                    keys=["image", "label"],
                                    prob=0.5,
                                    spatial_axis=0,
                                ),
                                mt.RandFlipd(
                                    keys=["image", "label"],
                                    prob=0.5,
                                    spatial_axis=1,
                                ),
                                mt.RandFlipd(
                                    keys=["image", "label"],
                                    prob=0.5,
                                    spatial_axis=2,
                                ),
                                mt.RandRotate90d(
                                    keys=["image", "label"],
                                    prob=0.75,
                                    max_k=3,
                                    spatial_axes=(0, 1),
                                ),
                                                mt.RandRotated(keys=["image", "label"], prob=0.5,range_x=0.78,range_y=0.,range_z=0., padding_mode='reflection')
                                    
                                            ])

cfg.val_aug = mt.Compose([mt.GridPatchd(keys=["image","label"],patch_size=cfg.roi_size, pad_mode='reflect')])



basic_cfg = cfg
