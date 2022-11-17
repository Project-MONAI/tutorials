from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# dataset
cfg.batch_size = 196
cfg.img_size = (640, 640)
cfg.train_aug = None
cfg.val_aug = None

# training
cfg.fold = -1
cfg.seed = -1
cfg.num_workers = 8
cfg.drop_last = True

# ressources
cfg.gpu = 0
cfg.device = "cuda:%d" % cfg.gpu

basic_cfg = cfg
