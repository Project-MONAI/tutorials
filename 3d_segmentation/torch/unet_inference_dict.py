# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch

from monai.config import print_config
from monai.data import Dataset, DataLoader, create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
)


def main(tempdir):
    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(5):
        im, _ = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
    files = [{"img": img} for img in images]

    # define pre transforms
    pre_transforms = Compose([
        LoadImaged(keys="img"),
        EnsureChannelFirstd(keys="img"),
        Orientationd(keys="img", axcodes="RAS"),
        Resized(keys="img", spatial_size=(96, 96, 96), mode="trilinear", align_corners=True),
        ScaleIntensityd(keys="img"),
    ])
    # define dataset and dataloader
    dataset = Dataset(data=files, transform=pre_transforms)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=4)
    # define post transforms
    post_transforms = Compose([
        Activationsd(keys="pred", sigmoid=True),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=pre_transforms,
            orig_keys="img",  # get the previously applied pre_transforms information on the `img` data field,
                              # then invert `pred` based on this information. we can use same info
                              # for multiple fields, also support different orig_keys for different fields
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                                   # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        AsDiscreted(keys="pred", threshold=0.5),
        SaveImaged(keys="pred", output_dir="./out", output_postfix="seg", resample=False),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    net.load_state_dict(torch.load("best_metric_model_segmentation3d_dict.pth"))

    net.eval()
    with torch.no_grad():
        for d in dataloader:
            images = d["img"].to(device)
            # define sliding window size and batch size for windows inference
            d["pred"] = sliding_window_inference(inputs=images, roi_size=(96, 96, 96), sw_batch_size=4, predictor=net)
            # decollate the batch data into a list of dictionaries, then execute postprocessing transforms
            d = [post_transforms(i) for i in decollate_batch(d)]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
