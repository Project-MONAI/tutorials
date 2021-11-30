#!/usr/bin/env python

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

import glob
import os
import random
import string
import time

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True

import nvidia_dlprof_pytorch_nvtx
import nvtx

from monai.apps import download_and_extract
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import Range, set_determinism

nvidia_dlprof_pytorch_nvtx.init()

# set directories
random.seed(0)
root_dir = "/tmp/tmp" + ''.join(random.choice(string.ascii_lowercase) for i in range(16))
print(f"root dir is: {root_dir}")

resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
data_root = os.path.join(root_dir, "Task09_Spleen")
if not os.path.exists(data_root):
    download_and_extract(resource, compressed_file, root_dir, md5)

out_dir = "./outputs_base"

train_images = sorted(glob.glob(os.path.join(data_root, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_root, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]

set_determinism(seed=0)

train_transforms = Compose(
    [
        Range("LoadImage")(LoadImaged(keys=["image", "label"])),
        Range()(EnsureChannelFirstd(keys=["image", "label"])),
        Range("Spacing")(
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            )
        ),
        Range()(Orientationd(keys=["image", "label"], axcodes="RAS")),
        Range()(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        ),
        Range()(CropForegroundd(keys=["image", "label"], source_key="image")),
        Range("RandCrop")(
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            )
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ]
)

train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=8
)
train_loader = DataLoader(
    train_ds, num_workers=8, batch_size=4, shuffle=True
)
val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=8
)
val_loader = DataLoader(
    val_ds, num_workers=8, batch_size=1
)

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
max_epochs = 6
learning_rate = 1e-4
val_interval = 2
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(
    to_onehot_y=True, softmax=True
)
optimizer = Adam(model.parameters(), learning_rate)
dice_metric = DiceMetric(
    include_background=True, reduction="mean", get_not_nans=False
)

post_pred = Compose(
    [EnsureType(), AsDiscrete(argmax=True, to_onehot=2)]
)
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
epoch_times = []
total_start = time.time()
writer = SummaryWriter(log_dir=out_dir)

with torch.autograd.profiler.emit_nvtx():
    for epoch in range(max_epochs):
        epoch_start = time.time()
        with nvtx.annotate("epoch", color="red"):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            train_loader_iterator = iter(train_loader)

            for step in range(len(train_loader)):
                step_start = time.time()

                with nvtx.annotate("dataload", color="red"):
                    batch_data = next(train_loader_iterator)
                    inputs, labels = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                    )

                optimizer.zero_grad()

                with nvtx.annotate("forward", color="green"):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                with nvtx.annotate("backward", color="blue"):
                    loss.backward()
                with nvtx.annotate("update", color="yellow"):
                    optimizer.step()

                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                print(
                    f"{step}/{epoch_len}, "
                    f"train_loss: {loss.item():.4f}, "
                    f"step time: {(time.time() - step_start):.4f}"
                )
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loader_iterator = iter(val_loader)
                    for val_step in range(len(val_loader)):

                        with nvtx.annotate("dataload", color="red"):
                            val_data = next(val_loader_iterator)
                            val_inputs, val_labels = (
                                val_data["image"].to(device),
                                val_data["label"].to(device),
                            )

                        roi_size = (160, 160, 160)
                        sw_batch_size = 4

                        with nvtx.annotate("sliding window", color="green"):
                            val_outputs = sliding_window_inference(
                                val_inputs, roi_size, sw_batch_size, model
                            )
                        with nvtx.annotate("decollate batch", color="blue"):
                            val_outputs = [
                                post_pred(i) for i in decollate_batch(val_outputs)
                            ]
                            val_labels = [
                                post_label(i) for i in decollate_batch(val_labels)
                            ]
                        with nvtx.annotate("compute metric", color="yellow"):
                            # compute metric for current iteration
                            dice_metric(y_pred=val_outputs, y=val_labels)

                    metric = dice_metric.aggregate().item()
                    dice_metric.reset()
                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        best_metrics_epochs_and_time[0].append(best_metric)
                        best_metrics_epochs_and_time[1].append(best_metric_epoch)
                        best_metrics_epochs_and_time[2].append(time.time() - total_start)
                        torch.save(
                            model.state_dict(), os.path.join(out_dir, "best_metric_model.pth")
                        )
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} "
                        f"current mean dice: {metric:.4f} "
                        f"best mean dice: {best_metric:.4f} "
                        f" at epoch: {best_metric_epoch}"
                    )
                    writer.add_scalar("val_mean_dice", metric, epoch + 1)
        print(
            f"time consuming of epoch {epoch + 1} is:"
            f" {(time.time() - epoch_start):.4f}"
        )
        epoch_times.append(time.time() - epoch_start)

total_time = time.time() - total_start
print(
    f"train completed, best_metric: {best_metric:.4f}"
    f" at epoch: {best_metric_epoch}"
    f" total time: {total_time:.4f}"
)
