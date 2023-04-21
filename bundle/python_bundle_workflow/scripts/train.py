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
from glob import glob

import nibabel as nib
import numpy as np
import torch
from ignite.metrics import Accuracy

from monai.apps import get_logger
from monai.bundle import BundleWorkflow
from monai.config import print_config
from monai.data import create_test_image_3d, CacheDataset, DataLoader
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    MeanDice,
    StatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    AsDiscreted,
    Compose,
    KeepLargestConnectedComponentd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
)
from monai.utils import BundleProperty, BundlePropertyConfig, BundleProperty, set_determinism


class TrainWorkflow(BundleWorkflow):
    """
    Test class simulates the bundle workflow defined by Python script directly.

    """

    def __init__(self, dataset_dir: str = "."):
        super().__init__(workflow="train")
        print_config()
        # set root log level to INFO and init a train logger, will be used in `StatsHandler`
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        get_logger("train_log")

        # create a temporary directory and 40 random image, mask pairs
        print(f"generating synthetic data to {dataset_dir} (this may take a while)")
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(dataset_dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(dataset_dir, f"seg{i:d}.nii.gz"))

        self._props = {}
        self._set_props = {}
        self.dataset_dir = dataset_dir
        # add components to the train properties, these are task specific
        self.properties = dict(self.properties)
        self.properties["network"] = {
            BundleProperty.DESC: "network for the training.",
            BundleProperty.REQUIRED: False,
            BundlePropertyConfig.ID: "network",
        }
        self.properties["loss"] = {
            BundleProperty.DESC: "loss function for the training.",
            BundleProperty.REQUIRED: False,
            BundlePropertyConfig.ID: "loss",
        }
        self.properties["optimizer"] = {
            BundleProperty.DESC: "optimizer for the training.",
            BundleProperty.REQUIRED: False,
            BundlePropertyConfig.ID: "optimizer",
        }

    def initialize(self):
        set_determinism(0)
        self.props = {}

    def run(self):
        self.trainer.run()

    def finalize(self):
        set_determinism(None)

    def _get_property(self, name, property):
        if name in self._props:
            value = self._props[name]
        elif name in self._set_props:
            value = self._set_props[name]
        else:
            if name == "bundle_root":
                value = "."
            elif name == "device":
                value = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif name == "dataset_dir":
                value = "."
            elif name == "network":
                value = UNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=1,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                ).to(self.device)
            elif name == "loss":
                value = DiceLoss(sigmoid=True)
            elif name == "optimizer":
                value = torch.optim.Adam(self.network.parameters(), 1e-3)
            elif name == "trainer":
                value = SupervisedTrainer(
                    device=self.device,
                    max_epochs=self.max_epochs,
                    train_data_loader=DataLoader(self.train_dataset, batch_size=2, shuffle=True, num_workers=4),
                    network=self.network,
                    optimizer=self.optimizer,
                    loss_function=self.loss,
                    inferer=self.train_inferer,
                    postprocessing=self.train_postprocessing,
                    key_train_metric={"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
                    train_handlers=self.train_handlers,
                    # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP training
                    amp=True,
                )
            elif name == "max_epochs":
                value = 5
            elif name == "train_dataset":
                images = sorted(glob(os.path.join(self.dataset_dir, "img*.nii.gz")))
                segs = sorted(glob(os.path.join(self.dataset_dir, "seg*.nii.gz")))
                value = CacheDataset(
                    data=[{"image": img, "label": seg} for img, seg in zip(images[:20], segs[:20])],
                    transform=self.train_preprocessing,
                    cache_rate=0.5,
                )
            elif name == "train_dataset_data":
                value = self.train_dataset.data
            elif name == "train_inferer":
                value = SimpleInferer()
            elif name == "train_handlers":
                value = [
                    ValidationHandler(validator=self.evaluator, interval=self.val_interval, epoch_level=True),
                    # use the logger "train_log" defined at the beginning of this program
                    StatsHandler(
                        name="train_log",
                        tag_name="train_loss",
                        output_transform=from_engine(["loss"], first=True),
                    ),
                ]
            elif name == "train_preprocessing":
                value = Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
                        ScaleIntensityd(keys="image"),
                        RandCropByPosNegLabeld(
                            keys=["image", "label"],
                            label_key="label",
                            spatial_size=[96, 96, 96],
                            pos=1,
                            neg=1,
                            num_samples=4,
                        ),
                        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
                        EnsureTyped(keys=["image", "label"]),
                    ]
                )
            elif name == "train_postprocessing":
                value = Compose(
                    [
                        Activationsd(keys="pred", sigmoid=True),
                        AsDiscreted(keys="pred", threshold=0.5),
                        KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
                    ]
                )
            elif name == "train_key_metric":
                value = ({"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},)
            elif name == "evaluator":
                value = SupervisedEvaluator(
                    device=self.device,
                    val_data_loader=DataLoader(self.val_dataset, batch_size=1, num_workers=4),
                    network=self.network,
                    inferer=self.val_inferer,
                    postprocessing=self.val_postprocessing,
                    key_val_metric=self.val_key_metric,
                    additional_metrics={"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
                    val_handlers=self.val_handlers,
                    amp=True,
                )
            elif name == "val_interval":
                value = 2
            elif name == "val_handlers":
                value = [
                    # use the logger "train_log" defined at the beginning of this program
                    StatsHandler(name="train_log", output_transform=lambda x: None),
                    CheckpointSaver(
                        save_dir=self.bundle_root + "/models/",
                        save_dict={"net": self.network},
                        save_key_metric=True,
                        key_metric_filename="model.pt",
                    ),
                ]
            elif name == "val_dataset":
                images = sorted(glob(os.path.join(self.dataset_dir, "img*.nii.gz")))
                segs = sorted(glob(os.path.join(self.dataset_dir, "seg*.nii.gz")))
                value = CacheDataset(
                    data=[{"image": img, "label": seg} for img, seg in zip(images[-20:], segs[-20:])],
                    transform=self.val_preprocessing,
                    cache_rate=1.0,
                )
            elif name == "val_dataset_data":
                value = self.val_dataset.data
            elif name == "val_inferer":
                value = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5)
            elif name == "val_preprocessing":
                value = Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
                        ScaleIntensityd(keys="image"),
                        EnsureTyped(keys=["image", "label"]),
                    ]
                )
            elif name == "val_postprocessing":
                value = Compose(
                    [
                        EnsureTyped(keys="pred"),
                        Activationsd(keys="pred", sigmoid=True),
                        AsDiscreted(keys="pred", threshold=0.5),
                        KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
                    ]
                )
            elif name == "val_key_metric":
                value = (
                    {
                        "val_mean_dice": MeanDice(
                            include_background=True, output_transform=from_engine(["pred", "label"])
                        )
                    }
                )
            elif property[BundleProperty.REQUIRED]:
                raise ValueError(f"unsupported property '{name}' is required in the bundle properties.")
            self._props[name] = value
        return value

    def _set_property(self, name, property, value):
        self._set_props[name] = value
