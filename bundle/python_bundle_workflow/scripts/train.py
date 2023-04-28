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
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)
from monai.utils import BundleProperty, set_determinism


class TrainWorkflow(BundleWorkflow):
    """
    Test class simulates the bundle training workflow defined by Python script directly.

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

        # define buckets to store the generated properties and set properties
        self._props = {}
        self._set_props = {}
        self.dataset_dir = dataset_dir

        # besides the predefined properties, this bundle workflow can also provide `network`, `loss`, `optimizer`
        self.add_property(name="network", required=True, desc="network for the training.")
        self.add_property(name="loss", required=True, desc="loss function for the training.")
        self.add_property(name="optimizer", required=True, desc="optimizer for the training.")

    def initialize(self):
        set_determinism(0)
        self.props = {}

    def run(self):
        self.trainer.run()

    def finalize(self):
        set_determinism(None)

    def _set_property(self, name, property, value):
        self._set_props[name] = value

    def _get_property(self, name, property):
        """
        Here the customized bundle workflow must implement required properties in:
        https://github.com/Project-MONAI/MONAI/blob/dev/monai/bundle/properties.py.
        If the property is already generated, return from the bucket directly.
        If user explicitly set the property, return it directly.
        Otherwise, generate the expected property as a class private property with prefix "_".

        """
        value = None
        if name in self._props:
            value = self._props[name]
        elif name in self._set_props:
            value = self._set_props[name]
            self._props[name] = value
        else:
            try:
                value = self.__getattribute__(f"_{name}")
            except AttributeError:
                if property[BundleProperty.REQUIRED]:
                    raise ValueError(f"unsupported property '{name}' is required in the bundle properties.")
            self._props[name] = value
        return value

    @property
    def _bundle_root(self):
        return "."

    @property
    def _device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def _dataset_dir(self):
        return "."

    @property
    def _network(self):
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)

    @property
    def _loss(self):
        return DiceLoss(sigmoid=True)

    @property
    def _optimizer(self):
        return torch.optim.Adam(self.network.parameters(), 1e-3)

    @property
    def _trainer(self):
        return SupervisedTrainer(
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
            amp=True,
        )

    @property
    def _max_epochs(self):
        return 5

    @property
    def _train_dataset(self):
        return CacheDataset(data=self.train_dataset_data, transform=self.train_preprocessing, cache_rate=0.5)

    @property
    def _train_dataset_data(self):
        images = sorted(glob(os.path.join(self.dataset_dir, "img*.nii.gz")))
        segs = sorted(glob(os.path.join(self.dataset_dir, "seg*.nii.gz")))
        return [{"image": img, "label": seg} for img, seg in zip(images[:20], segs[:20])]

    @property
    def _train_inferer(self):
        return SimpleInferer()

    @property
    def _train_handlers(self):
        return [
            ValidationHandler(validator=self.evaluator, interval=self.val_interval, epoch_level=True),
            # use the logger "train_log" defined at the beginning of this program
            StatsHandler(
                name="train_log",
                tag_name="train_loss",
                output_transform=from_engine(["loss"], first=True),
            ),
        ]

    @property
    def _train_preprocessing(self):
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
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
            ]
        )

    @property
    def _train_postprocessing(self):
        return Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold=0.5),
                KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
            ]
        )

    @property
    def _train_key_metric(self):
        return ({"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},)

    @property
    def _evaluator(self):
        return SupervisedEvaluator(
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

    @property
    def _val_interval(self):
        return 2

    @property
    def _val_handlers(self):
        return [
            # use the logger "train_log" defined at the beginning of this program
            StatsHandler(name="train_log", output_transform=lambda x: None),
            CheckpointSaver(
                save_dir=self.bundle_root + "/models/",
                save_dict={"net": self.network},
                save_key_metric=True,
                key_metric_filename="model.pt",
            ),
        ]

    @property
    def _val_dataset(self):
        return CacheDataset(data=self.val_dataset_data, transform=self.val_preprocessing, cache_rate=1.0)

    @property
    def _val_dataset_data(self):
        images = sorted(glob(os.path.join(self.dataset_dir, "img*.nii.gz")))
        segs = sorted(glob(os.path.join(self.dataset_dir, "seg*.nii.gz")))
        return [{"image": img, "label": seg} for img, seg in zip(images[-20:], segs[-20:])]

    @property
    def _val_inferer(self):
        return SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5)

    @property
    def _val_preprocessing(self):
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys="image"),
            ]
        )

    @property
    def _val_postprocessing(self):
        return Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold=0.5),
                KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
            ]
        )

    @property
    def _val_key_metric(self):
        return {"val_mean_dice": MeanDice(include_background=True, output_transform=from_engine(["pred", "label"]))}
