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
from ignite.engine import (
    Events,
    _prepare_batch,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.handlers import (
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    stopping_fn_from_metric,
)
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
)


def main(tempdir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # create a temporary directory and 40 random image, mask pairs
    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(40):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"img{i:d}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:20], segs[:20])]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-20:], segs[-20:])]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"],
                label_key="seg",
                spatial_size=[96, 96, 96],
                pos=1,
                neg=1,
                num_samples=4,
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )

    # define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(
        check_ds,
        batch_size=2,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["seg"].shape)

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=5,
        num_workers=8,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss = monai.losses.DiceLoss(sigmoid=True)
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr)

    # Ignite trainer expects batch=(img, seg) and returns output=loss at every iteration,
    # user can add output_transform to return other values, like: y_pred, y, etc.
    def prepare_batch(batch, device=None, non_blocking=False):
        return _prepare_batch((batch["img"], batch["seg"]), device, non_blocking)

    trainer = create_supervised_trainer(
        net, opt, loss, device, False, prepare_batch=prepare_batch
    )

    # adding checkpoint handler to save models (network params and optimizer stats) during training
    checkpoint_handler = ModelCheckpoint(
        "./runs_dict/", "net", n_saved=10, require_empty=False
    )
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={"net": net, "opt": opt},
    )

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't set metrics for trainer here, so just print loss, user can also customize print functions
    # and can use output_transform to convert engine.state.output if it's not loss value
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)

    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
    train_tensorboard_stats_handler = TensorBoardStatsHandler(output_transform=lambda x: x)
    train_tensorboard_stats_handler.attach(trainer)

    validation_every_n_iters = 5
    # set parameters for validation
    metric_name = "Mean_Dice"
    # add evaluation metric to the evaluator engine
    val_metrics = {metric_name: MeanDice()}

    post_pred = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    post_label = Compose([EnsureType(), AsDiscrete(threshold_values=True)])

    # Ignite evaluator expects batch=(img, seg) and returns output=(y_pred, y) at every iteration,
    # user can add output_transform to return other values
    evaluator = create_supervised_evaluator(
        net,
        val_metrics,
        device,
        True,
        output_transform=lambda x, y, y_pred: ([post_pred(i) for i in decollate_batch(y_pred)], [post_label(i) for i in decollate_batch(y)]),
        prepare_batch=prepare_batch,
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=validation_every_n_iters))
    def run_validation(engine):
        evaluator.run(val_loader)

    # add early stopping handler to evaluator
    early_stopper = EarlyStopping(
        patience=4, score_function=stopping_fn_from_metric(metric_name), trainer=trainer
    )
    evaluator.add_event_handler(
        event_name=Events.EPOCH_COMPLETED, handler=early_stopper
    )

    # add stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.epoch,
    )  # fetch global epoch number from trainer
    val_stats_handler.attach(evaluator)

    # add handler to record metrics to TensorBoard at every validation epoch
    val_tensorboard_stats_handler = TensorBoardStatsHandler(
        output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.iteration,
    )  # fetch global iteration number from trainer
    val_tensorboard_stats_handler.attach(evaluator)

    # add handler to draw the first image and the corresponding label and model output in the last batch
    # here we draw the 3D output as GIF format along the depth axis, every 2 validation iterations.
    val_tensorboard_image_handler = TensorBoardImageHandler(
        batch_transform=lambda batch: (batch["img"], batch["seg"]),
        output_transform=lambda output: output[0],
        global_iter_transform=lambda x: trainer.state.epoch,
    )
    evaluator.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=2),
        handler=val_tensorboard_image_handler,
    )

    train_epochs = 5
    state = trainer.run(train_loader, train_epochs)
    print(state)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
