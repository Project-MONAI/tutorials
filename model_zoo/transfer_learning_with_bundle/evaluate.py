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

import argparse
import logging
import os
import sys
import tempfile
from glob import glob

import monai
import nibabel as nib
import numpy as np
import torch
from monai.data import (
    CacheDataset,
    DataLoader,
    decollate_batch,
    load_decathlon_datalist,
    set_track_meta,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImage,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.visualize import plot_2d_or_3d_image


def main(tempdir, load_pretrained_ckpt=False):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    monai.utils.set_determinism(seed=123)
    torch.backends.cudnn.benchmark = True

    # define path
    data_file_base_dir = "./data/btcv_spleen"
    data_list_file_path = "./data/dataset_0.json"

    if load_pretrained_ckpt:
        save_model = "./models/model_transfer.pt"
    else:
        save_model = "./models/model_from_scratch.pt"

    val_datalist = load_decathlon_datalist(
        data_list_file_path,
        is_segmentation=True,
        data_list_key="validation",
        base_dir=data_file_base_dir,
    )
    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    saver = SaveImage(output_dir="./output", output_ext=".nii.gz", output_postfix="seg")

    # data loader
    val_ds = CacheDataset(
        data=val_datalist,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=2,
    )
    val_loader = DataLoader(val_ds, num_workers=2, batch_size=1, shuffle=False)

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    model.load_state_dict(torch.load(save_model))
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            for val_output in val_outputs:
                saver(val_output)
        # aggregate the final mean dice result
        print("evaluation metric:", dice_metric.aggregate().item())
        # reset the status
        dice_metric.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a train task.")
    parser.add_argument(
        "--load_pretrained_ckpt",
        action="store_true",
        help="whether to load pretrained checkpoint",
    )

    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir, args.load_pretrained_ckpt)
