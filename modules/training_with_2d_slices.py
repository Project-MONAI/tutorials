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

import logging
import os
import sys
import tempfile
from glob import glob

import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, PatchDataset, create_test_image_3d, list_data_collate
from monai.inferers import SliceInferer
from monai.transforms import (
    AsChannelFirstd,
    Compose,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    Resized,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    SqueezeDimd,
)
from monai.visualize import matshow3d


def main(tempdir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # -----
    # make demo data
    # -----
    # create a temporary directory and 40 random image, mask pairs
    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(40):
        # make the input volumes different spatial shapes for demo purposes
        H, W, D = 30 + i, 40 + i, 50 + i
        im, seg = create_test_image_3d(
            H, W, D, num_seg_classes=1, channel_dim=-1, rad_max=10
        )

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"img{i:d}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:35], segs[:35])]

    # -----
    # volume-level preprocessing
    # -----
    # volume-level transforms for both image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    # 3D dataset with preprocessing transforms
    volume_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=1 to check the volumes because the input volumes have different shapes
    check_loader = DataLoader(volume_ds, batch_size=1, collate_fn=list_data_collate)
    check_data = monai.utils.misc.first(check_loader)
    print("first volume's shape: ", check_data["img"].shape, check_data["seg"].shape)

    # -----
    # volume to patch sampling
    # -----
    # define the sampling transforms, could also be other spatial cropping transforms
    # https://docs.monai.io/en/stable/transforms.html#crop-and-pad-dict
    num_samples = 4
    patch_func = RandCropByPosNegLabeld(
        keys=["img", "seg"],
        label_key="seg",
        spatial_size=[-1, -1, 1],  # dynamic spatial_size for the first two dimensions
        pos=1,
        neg=1,
        num_samples=num_samples,
    )

    # -----
    # patch-level preprocessing
    # -----
    # resize the sampled slices to a consistent size so that we can batch
    # the last spatial dim is always 1, so we squeeze dim.
    patch_transform = Compose(
        [
            SqueezeDimd(keys=["img", "seg"], dim=-1),  # squeeze the last dim
            Resized(keys=["img", "seg"], spatial_size=[48, 48]),
            # ResizeWithPadOrCropd(keys=["img", "seg"], spatial_size=[48, 48], mode="replicate"),
        ]
    )
    patch_ds = PatchDataset(
        volume_ds,
        transform=patch_transform,
        patch_func=patch_func,
        samples_per_image=num_samples,
    )
    train_loader = DataLoader(
        patch_ds,
        batch_size=3,
        shuffle=True,  # this shuffles slices from different volumes
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    check_data = monai.utils.misc.first(train_loader)
    print("first patch's shape: ", check_data["img"].shape, check_data["seg"].shape)

    # -----
    # network defined for 2d inputs
    # -----
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 5e-3)

    # -----
    # training
    # -----
    epoch_loss_values = []
    num_epochs = 5
    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss, step = 0, 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(patch_ds) // train_loader.batch_size
            if step % 25 == 0:
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    print("train completed")

    # -----
    # inference with a SliceInferer
    # -----
    model.eval()
    val_transform = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-3:], segs[-3:])]
    val_ds = monai.data.Dataset(data=val_files, transform=val_transform)
    data_loader = DataLoader(val_ds, pin_memory=torch.cuda.is_available())

    with torch.no_grad():
        for val_data in data_loader:
            val_images = val_data["img"].to(device)
            roi_size = (48, 48)
            sw_batch_size = 3
            slice_inferer = SliceInferer(
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                spatial_dim=2,  # Spatial dim to slice along is defined here
                device=torch.device("cpu"),
                padding_mode="replicate",
            )
            val_output = slice_inferer(val_images, model)
            matshow3d(val_output[0] > 0.5)
            matshow3d(val_images[0])
            plt.show()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
