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


from common_config import basic_cfg
import os
import pandas as pd
import numpy as np
import monai.transforms as mt

cfg = basic_cfg

cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"/mount/cryo/models/{os.path.basename(__file__).split('.')[0]}"

# model
cfg.backbone = "efficientnet-b3"
cfg.backbone_args = dict(
    spatial_dims=3,
    in_channels=cfg.in_channels,
    out_channels=cfg.n_classes,
    backbone=cfg.backbone,
    pretrained=cfg.pretrained,
)
cfg.class_weights = np.array([64, 64, 64, 64, 64, 64, 1])
cfg.lvl_weights = np.array([0, 0, 0, 1])
