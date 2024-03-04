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
from monai.data import create_test_image_3d, Dataset, DataLoader
from monai.engines import SupervisedEvaluator
from monai.handlers import CheckpointLoader, MeanDice, StatsHandler, from_engine
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    SaveImaged,
    ScaleIntensityd,
)
from monai.utils import BundleProperty
from scripts.train import prepare_data


class InferenceWorkflow(BundleWorkflow):
    """
    Test class simulates the bundle workflow defined by Python script directly.

    """

    def __init__(self, dataset_dir: str = "./infer"):
        super().__init__(workflow="inference")
        print_config()
        # set root log level to INFO and init a evaluation logger, will be used in `StatsHandler`
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        get_logger("eval_log")

        # create a temporary directory and 40 random image, mask pairs
        prepare_data(dataset_dir=dataset_dir)

        self._props = {}
        self._set_props = {}
        self.dataset_dir = dataset_dir

    def initialize(self):
        self._props = {}

    def run(self):
        self.evaluator.run()

    def finalize(self):
        pass

    def _set_property(self, name, property, value):
        # stores user-reset initialized objects that should not be re-initialized.
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
        if name in self._set_props:
            value = self._set_props[name]
            self._props[name] = value
        elif name in self._props:
            value = self._props[name]
        else:
            try:
                value = getattr(self, f"get_{name}")()
            except AttributeError:
                if property[BundleProperty.REQUIRED]:
                    raise ValueError(
                        f"unsupported property '{name}' is required in the bundle properties,"
                        f"need to implement a method 'get_{name}' to provide the property."
                    )
            self._props[name] = value
        return value

    def get_bundle_root(self):
        return "."

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_dataset_dir(self):
        return self.dataset_dir

    def get_network_def(self):
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def get_evaluator(self):
        return SupervisedEvaluator(
            device=self.device,
            val_data_loader=DataLoader(self.dataset, batch_size=1, num_workers=4),
            network=self.network_def.to(self.device),
            inferer=self.inferer,
            postprocessing=self.postprocessing,
            key_val_metric=self.key_metric,
            additional_metrics={"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
            val_handlers=self.handlers,
            amp=True,
        )

    def get_handlers(self):
        return [
            # use the logger "eval_log" defined at the beginning of this program
            StatsHandler(name="eval_log", output_transform=lambda x: None),
            CheckpointLoader(load_path=self.bundle_root + "/models/model.pt", load_dict={"net": self.network_def}),
        ]

    def get_dataset(self):
        return Dataset(
            data=self.dataset_data,
            transform=self.preprocessing,
        )

    def get_dataset_data(self):
        images = sorted(glob(os.path.join(self.dataset_dir, "img*.nii.gz")))
        segs = sorted(glob(os.path.join(self.dataset_dir, "seg*.nii.gz")))
        return [{"image": img, "label": seg} for img, seg in zip(images, segs)]

    def get_inferer(self):
        return SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5)

    def get_preprocessing(self):
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys="image"),
            ]
        )

    def get_postprocessing(self):
        return Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold=0.5),
                KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
                SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=self.bundle_root + "/preds/"),
            ]
        )

    def get_key_metric(self):
        return {"val_mean_dice": MeanDice(include_background=True, output_transform=from_engine(["pred", "label"]))}
