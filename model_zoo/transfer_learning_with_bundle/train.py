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
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter


def main(tempdir, load_pretrained_ckpt=False):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    monai.utils.set_determinism(seed=123)
    torch.backends.cudnn.benchmark = True

    # define path
    data_file_base_dir = "./data/btcv_spleen"
    data_list_file_path = "./data/dataset_0.json"

    os.makedirs("./models", exist_ok=True)
    if load_pretrained_ckpt:
        # transfer learning
        save_model = "./models/model_transfer.pt"
        max_epochs = 200
    else:
        # train from scratch
        save_model = "./models/model_from_scratch.pt"
        max_epochs = 600

    # create UNet and load pretrained weights from monai model-zoo
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

    if load_pretrained_ckpt:
        print("Load model from monai model-zoo.")
        pretrained_model = monai.bundle.load(
            name="spleen_ct_segmentation", bundle_dir="./", version="0.1.1"
        )
        model.load_state_dict(pretrained_model)

    # define transforms for image and segmentation
    train_transforms = Compose(
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
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )
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

    # data loader
    train_datalist = load_decathlon_datalist(
        data_list_file_path,
        is_segmentation=True,
        data_list_key="training",
        base_dir=data_file_base_dir,
    )

    train_ds = CacheDataset(
        data=train_datalist[: int(0.8 * len(train_datalist))],
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=4,
    )
    train_loader = DataLoader(train_ds, num_workers=2, batch_size=2, shuffle=True)
    val_ds = CacheDataset(
        data=train_datalist[int(0.8 * len(train_datalist)) :],
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=4,
    )
    val_loader = DataLoader(val_ds, num_workers=2, batch_size=1, shuffle=False)

    # DiceLoss and Adam optimizer
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 2e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # start a typical PyTorch training

    val_interval = 5
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(
                device
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["image"].to(device), val_data[
                        "label"
                    ].to(device)
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_images, roi_size, sw_batch_size, model
                    )
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), save_model)
                    print("saved new best metric model to " + save_model)
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(
                    val_outputs, epoch + 1, writer, index=0, tag="output"
                )

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a train task.")
    parser.add_argument(
        "--load_pretrained_ckpt",
        action="store_true",
        help="whether to load pretrained checkpoint from model zoo",
    )

    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir, args.load_pretrained_ckpt)
