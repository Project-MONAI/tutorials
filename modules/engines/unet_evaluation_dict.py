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
from ignite.metrics import Accuracy

import monai
from monai.apps import get_logger
from monai.data import create_test_image_3d
from monai.engines import SupervisedEvaluator
from monai.handlers import CheckpointLoader, MeanDice, StatsHandler, from_engine
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    AsDiscreted,
    Compose,
    KeepLargestConnectedComponentd,
    LoadImaged,
    SaveImaged,
    ScaleIntensityd,
    EnsureTyped,
)


def main(tempdir):
    monai.config.print_config()
    # set root log level to INFO and init a evaluation logger, will be used in `StatsHandler`
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    get_logger("eval_log")

    # create a temporary directory and 40 random image, mask pairs
    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(5):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))
        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))
    val_files = [{"image": img, "label": seg} for img, seg in zip(images, segs)]

    # model file path
    model_file = glob("./runs/net_key_metric*")[0]

    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4)

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    val_post_transforms = Compose(
        [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
            SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir="./runs/")
        ]
    )
    val_handlers = [
        # use the logger "eval_log" defined at the beginning of this program
        StatsHandler(name="eval_log", output_transform=lambda x: None),
        CheckpointLoader(load_path=model_file, load_dict={"net": net}),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=from_engine(["pred", "label"]))
        },
        additional_metrics={"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        val_handlers=val_handlers,
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
        amp=True if monai.utils.get_torch_version_tuple() >= (1, 6) else False,
    )
    evaluator.run()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
