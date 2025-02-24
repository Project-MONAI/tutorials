from common_config import basic_cfg
import os
import pandas as pd
import numpy as np

cfg = basic_cfg

# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"/mount/cryo/models/{os.path.basename(__file__).split('.')[0]}"

cfg.backbone = 'resnet34'
cfg.backbone_args = dict(spatial_dims=3,    
                         in_channels=cfg.in_channels,
                         out_channels=cfg.n_classes,
                         backbone=cfg.backbone,
                         pretrained=cfg.pretrained)
cfg.class_weights = np.array([64,64,64,64,64,64,1])
cfg.lvl_weights = np.array([0,0,1,1])
