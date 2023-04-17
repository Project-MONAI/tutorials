# Copyright 2023 MONAI Consortium
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
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint

import monai
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.handlers import (
    MetricsReloadedBinaryHandler,
    StatsHandler,
    stopping_fn_from_metric,
)
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    RandSpatialCrop,
    Resize,
    ScaleIntensity,
)


def main(tempdir, img_size=96):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Set patch size
    patch_size = (int(img_size / 2.0),) * 3

    # create a temporary directory and 40 random image, mask pairs
    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(40):
        im, seg = create_test_image_3d(img_size, img_size, img_size, num_seg_classes=1)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"lab{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "lab*.nii.gz")))

    # define transforms for image and segmentation
    train_imtrans = Compose(
        [
            ScaleIntensity(),
            EnsureChannelFirst(),
            RandSpatialCrop(patch_size, random_size=False),
        ]
    )
    train_segtrans = Compose([EnsureChannelFirst(), RandSpatialCrop(patch_size, random_size=False)])
    val_imtrans = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize(patch_size)])
    val_segtrans = Compose([EnsureChannelFirst(), Resize(patch_size)])

    # define image dataset, data loader
    check_ds = ImageDataset(images, segs, transform=train_imtrans, seg_transform=train_segtrans)
    check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
    im, seg = monai.utils.misc.first(check_loader)
    print(im.shape, seg.shape)

    # create a training data loader
    train_ds = ImageDataset(images[:20], segs[:20], transform=train_imtrans, seg_transform=train_segtrans)
    train_loader = DataLoader(
        train_ds,
        batch_size=5,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = ImageDataset(images[-20:], segs[-20:], transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=5, num_workers=8, pin_memory=torch.cuda.is_available())

    # Compute UNet levels and strides from image size
    min_size = 4  # minimum size allowed at coarsest resolution level
    num_levels = int(np.maximum(np.ceil(np.log2(np.min(img_size)) - np.log2(min_size)), 1))
    channels = [2 ** (i + 4) for i in range(num_levels)]

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=(2,) * (num_levels - 1),
        num_res_units=2,
    ).to(device)
    loss = monai.losses.DiceLoss(sigmoid=True)
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr)

    # Ignite trainer expects batch=(img, seg) and returns output=loss at every iteration,
    # user can add output_transform to return other values, like: y_pred, y, etc.
    trainer = create_supervised_trainer(net, opt, loss, device, False)

    # adding checkpoint handler to save models (network params and optimizer stats) during training
    checkpoint_handler = ModelCheckpoint("./runs_array/", "net", n_saved=10, require_empty=False)
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={"net": net, "opt": opt},
    )

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't set metrics for trainer here, so just print loss, user can also customize print functions
    # and can use output_transform to convert engine.state.output if it's not a loss value
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)

    # Set parameters for validation
    validation_every_n_epochs = 1
    # Use validation metrics from MetricsReloaded
    metric_name = "Intersection_Over_Union"
    # add evaluation metric to the evaluator engine
    val_metrics = {metric_name: MetricsReloadedBinaryHandler("Intersection Over Union")}

    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    # Ignite evaluator expects batch=(img, seg) and returns output=(y_pred, y) at every iteration,
    # user can add output_transform to return other values
    evaluator = create_supervised_evaluator(
        net,
        val_metrics,
        device,
        True,
        output_transform=lambda x, y, y_pred: (
            [post_pred(i) for i in decollate_batch(y_pred)],
            [post_label(i) for i in decollate_batch(y)],
        ),
    )

    @trainer.on(Events.EPOCH_COMPLETED(every=validation_every_n_epochs))
    def run_validation(engine):
        evaluator.run(val_loader)

    # add early stopping handler to evaluator
    early_stopper = EarlyStopping(patience=4, score_function=stopping_fn_from_metric(metric_name), trainer=trainer)
    evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=early_stopper)

    # add stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.epoch,
    )  # fetch global epoch number from trainer
    val_stats_handler.attach(evaluator)

    train_epochs = 30
    state = trainer.run(train_loader, train_epochs)
    print(state)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
