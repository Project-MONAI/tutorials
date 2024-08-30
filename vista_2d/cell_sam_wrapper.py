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

import torch
from segment_anything.build_sam import build_sam_vit_b
from torch import nn
from torch.nn import functional as F


class CellSamWrapper(torch.nn.Module):
    def __init__(
        self,
        auto_resize_inputs=True,
        network_resize_roi=[1024, 1024],
        checkpoint="sam_vit_b_01ec64.pth",
        return_features=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        print(
            f"CellSamWrapper auto_resize_inputs {auto_resize_inputs} network_resize_roi {network_resize_roi} checkpoint {checkpoint}"
        )
        self.network_resize_roi = network_resize_roi
        self.auto_resize_inputs = auto_resize_inputs
        self.return_features = return_features

        model = build_sam_vit_b(checkpoint=checkpoint)

        model.prompt_encoder = None
        model.mask_decoder = None

        model.mask_decoder = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
            ),
        )

        self.model = model

    def forward(self, x):
        # print("CellSamWrapper x0", x.shape)
        sh = x.shape[2:]

        if self.auto_resize_inputs:
            x = F.interpolate(x, size=self.network_resize_roi, mode="bilinear")

        # print("CellSamWrapper x1", x.shape)
        x = self.model.image_encoder(x)  # shape: (1, 256, 64, 64)
        # print("CellSamWrapper image_embeddings", x.shape)

        if not self.return_features:
            x = self.model.mask_decoder(x)
            if self.auto_resize_inputs:
                x = F.interpolate(x, size=sh, mode="bilinear")

        # print("CellSamWrapper x final", x.shape)
        return x