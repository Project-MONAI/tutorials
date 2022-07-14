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

from monai import config
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AddChannel, AsDiscrete, Compose, SaveImage, ScaleIntensity


def main(tempdir):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(5):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))

    # define transforms for image and segmentation
    imtrans = Compose([ScaleIntensity(), AddChannel()])
    segtrans = Compose([AddChannel()])
    val_ds = ImageDataset(images, segs, transform=imtrans, seg_transform=segtrans, image_only=False)
    # sliding window inference for one image at every iteration
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    saver = SaveImage(output_dir="./output", output_ext=".nii.gz", output_postfix="seg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(torch.load("best_metric_model_segmentation3d_array.pth"))
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
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
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
