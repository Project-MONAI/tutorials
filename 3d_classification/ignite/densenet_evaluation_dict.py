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

import numpy as np
import torch
from ignite.engine import _prepare_batch, create_supervised_evaluator
from ignite.metrics import Accuracy

import monai
from monai.data import DataLoader
from monai.handlers import CheckpointLoader, ClassificationSaver, StatsHandler
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityd


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
    # the path of ixi IXI-T1 dataset
    data_path = os.sep.join([".", "workspace", "data", "medical", "ixi", "IXI-T1"])
    images = [
        "IXI607-Guys-1097-T1.nii.gz",
        "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz",
        "IXI344-Guys-0905-T1.nii.gz",
        "IXI409-Guys-0960-T1.nii.gz",
        "IXI584-Guys-1129-T1.nii.gz",
        "IXI253-HH-1694-T1.nii.gz",
        "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz",
        "IXI585-Guys-1130-T1.nii.gz",
    ]
    images = [os.sep.join([data_path, f]) for f in images]

    # 2 binary labels for gender classification: man and woman
    labels = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    val_files = [{"img": img, "label": label} for img, label in zip(images, labels)]

    # define transforms for image
    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
        ]
    )

    # create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

    def prepare_batch(batch, device=None, non_blocking=False):
        return _prepare_batch((batch["img"], batch["label"]), device, non_blocking)

    metric_name = "Accuracy"
    # add evaluation metric to the evaluator engine
    val_metrics = {metric_name: Accuracy()}
    # Ignite evaluator expects batch=(img, label) and returns output=(y_pred, y) at every iteration,
    # user can add output_transform to return other values
    evaluator = create_supervised_evaluator(net, val_metrics, device, True, prepare_batch=prepare_batch)

    # add stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
    )
    val_stats_handler.attach(evaluator)

    # for the array data format, assume the 3rd item of batch data is the meta_data
    prediction_saver = ClassificationSaver(
        output_dir="tempdir",
        name="evaluator",
        batch_transform=lambda batch: batch["img"].meta,
        output_transform=lambda output: output[0].argmax(1),
    )
    prediction_saver.attach(evaluator)

    # the model was trained by "densenet_training_dict" example
    CheckpointLoader(load_path="./runs_dict/net_checkpoint_20.pt", load_dict={"net": net}).attach(evaluator)

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    state = evaluator.run(val_loader)
    print(state)


if __name__ == "__main__":
    main()
