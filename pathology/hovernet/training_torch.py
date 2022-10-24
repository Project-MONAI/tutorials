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

import sys
sys.path.append('/workspace/Code/tutorials/pathology/hovernet/transforms')
sys.path.append('/workspace/Code/tutorials/pathology/hovernet/loss')
from loss import HoVerNetLoss
from transforms import (
    GenerateWatershedMaskd,
    GenerateInstanceBorderd,
    GenerateDistanceMapd,
    GenerateWatershedMarkersd,
    Watershedd,
    GenerateInstanceContour,
    GenerateInstanceCentroid,
    GenerateInstanceType,
    GenerateInstanceCentroid, 
    GenerateInstanceContour, 
    GenerateInstanceType,
)

import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from monai.data import DataLoader, decollate_batch, CacheDataset
from monai.metrics import DiceMetric
from monai.networks.nets import HoVerNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    AsDiscreted,
    Compose,
    ScaleIntensityRanged,
    CastToTyped,
    Lambdad,
    SplitDimd,
    EnsureChannelFirstd,
    ComputeHoVerMapsd,
    RandFlipd,
    RandRotate90d,
    RandGaussianSmoothd,
    FillHoles,
    BoundingRect,
    CenterSpatialCropd,
)

from monai.utils import set_determinism, convert_to_tensor
from monai.utils.enums import HoVerNetBranch
from monai.visualize import plot_2d_or_3d_image
from sklearn.model_selection import StratifiedShuffleSplit



def prepare_datasets(data_dir):
    info = pd.read_csv(os.path.join(data_dir, "patch_info.csv"))
    file_names = np.squeeze(info.to_numpy()).tolist()

    img_sources = [v.split("-")[0] for v in file_names]
    img_sources = np.unique(img_sources)

    cohort_sources = [v.split("_")[0] for v in img_sources]
    _, cohort_sources = np.unique(cohort_sources, return_inverse=True)

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3, random_state=0)

    split_generator = splitter.split(img_sources, cohort_sources)
    for train_indices, valid_test_indices in split_generator:
        train_cohorts = img_sources[train_indices]
        valid_test_sources = img_sources[valid_test_indices]
        valid_test_cohorts = cohort_sources[valid_test_indices]

        _split_generator = splitter.split(valid_test_sources, valid_test_cohorts)
        for valid_indices, test_indices in _split_generator:
            valid_cohorts = valid_test_sources[valid_indices]
            test_cohorts = valid_test_sources[test_indices]
            train_names = [
                file_name for file_name in file_names for source in train_cohorts if source == file_name.split("-")[0]
            ]
            valid_names = [
                file_name for file_name in file_names for source in valid_cohorts if source == file_name.split("-")[0]
            ]
            test_names = [
                file_name for file_name in file_names for source in test_cohorts if source == file_name.split("-")[0]
            ]
            train_names = np.unique(train_names)
            valid_names = np.unique(valid_names)
            test_names = np.unique(test_names)
            print(f"Train: {len(train_names):04d} - Valid: {len(valid_names):04d} - Test: {len(test_names):04d}")
            if np.intersect1d(train_names, valid_names).size != 0:
                raise ValueError("Train and validation cohorts has an overlap.")
            if np.intersect1d(valid_names, test_names).size != 0:
                raise ValueError("Validation and test cohorts has an overlap.")

            train_indices = [file_names.index(v) for v in train_names]
            valid_indices = [file_names.index(v) for v in valid_names]
            test_indices = [file_names.index(v) for v in test_names]

    images = np.load(os.path.join(data_dir, "images.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))

    data = [
        {
            "image": image,
            "image_meta_dict": {"original_channel_dim": -1},
            "label": label,
            "label_meta_dict": {"original_channel_dim": -1},
        }
        for image, label in zip(images, labels)
    ]

    train_data = [data[i] for i in train_indices]
    valid_data = [data[i] for i in valid_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, valid_data, test_data

def post_process(output, device, return_binary=True, return_centroids=False, output_classes=None):
    post_trans_seg = Compose([
        GenerateWatershedMaskd(keys=HoVerNetBranch.NP.value, softmax=True),
        GenerateInstanceBorderd(keys='mask', hover_map_key=HoVerNetBranch.HV, kernel_size=3),
        GenerateDistanceMapd(keys='mask', border_key='border', smooth_fn="gaussian"),
        GenerateWatershedMarkersd(keys='mask', border_key='border', threshold=0.7, radius=2, postprocess_fn=FillHoles()),
        Watershedd(keys='dist', mask_key='mask', markers_key='markers')
    ])
    if HoVerNetBranch.NC.value in output.keys():
        type_pred = Activations(softmax=True)(output[HoVerNetBranch.NC.value])
        type_pred = AsDiscrete(argmax=True)(output[HoVerNetBranch.NC.value])

    pred_inst_dict = post_trans_seg(output)
    pred_inst = pred_inst_dict['dist']

    inst_id_list = np.unique(pred_inst)[1:]  # exclude background

    inst_info_dict = None
    if return_centroids:
        inst_id_list = np.unique(pred_inst)[1:]  # exclude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            inst_bbox = BoundingRect()(inst_map)
            inst_map = inst_map[:, inst_bbox[0][0]: inst_bbox[0][1], inst_bbox[0][2]: inst_bbox[0][3]]
            offset = [inst_bbox[0][2], inst_bbox[0][0]]
            inst_contour = GenerateInstanceContour()(inst_map.squeeze(), offset)
            inst_centroid = GenerateInstanceCentroid()(inst_map, offset)
            if inst_contour is not None:
                inst_info_dict[inst_id] = {  # inst_id should start at 1
                    "bounding_box": inst_bbox,
                    "centroid": inst_centroid,
                    "contour": inst_contour,
                    "type_probability": None,
                    "type": None,
                }

    if output_classes is not None:
        for inst_id in list(inst_info_dict.keys()):
            inst_type, type_prob = GenerateInstanceType()(
                bbox=inst_info_dict[inst_id]["bounding_box"], type_pred=type_pred, seg_pred=pred_inst, instance_id=inst_id)
            inst_info_dict[inst_id]["type"] = inst_type
            inst_info_dict[inst_id]["type_probability"] = type_prob

    pred_inst = convert_to_tensor(pred_inst, device=device)
    if return_binary:
        pred_inst[pred_inst > 0] = 1
    return (pred_inst, inst_info_dict, pred_inst_dict)


def run(data_dir):
    train_transforms = Compose(
        [
            EnsureChannelFirstd(keys=("image", "label"), channel_dim=-1),
            SplitDimd(keys="label", output_postfixes=["inst", "type"]),
            ComputeHoVerMapsd(keys="label_inst"),
            CastToTyped(keys=["image", "label_inst", "label_type", "hover_label_inst"], dtype=torch.float32),
            Lambdad(keys="label", func=lambda x: x[1: 2, ...] > 0),
            AsDiscreted(keys=["label", "label_type"], to_onehot=[2, 7]),
            CenterSpatialCropd(keys=["label", "label_inst", "label_type", "hover_label_inst"], roi_size=(164,164)),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            RandFlipd(keys=["image", "label", "label_inst", "label_type", "hover_label_inst"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label", "label_inst", "label_type", "hover_label_inst"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "label", "label_inst", "label_type", "hover_label_inst"], prob=0.5, max_k=1),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.5,1.15), sigma_y=(0.5,1.15), prob=0.5),
        ]
    )
    val_transforms = Compose(
        [
            EnsureChannelFirstd(keys=("image", "label"), channel_dim=-1),
            SplitDimd(keys="label", output_postfixes=["inst", "type"]),
            ComputeHoVerMapsd(keys="label_inst"),
            CastToTyped(keys=["image", "label_inst", "label_type", "hover_label_inst"], dtype=torch.float32),
            Lambdad(keys="label", func=lambda x: x[1: 2, ...] > 0),
            AsDiscreted(keys=["label", "label_type"], to_onehot=[2, 7]),
            CenterSpatialCropd(keys=["label", "label_inst", "label_type", "hover_label_inst"], roi_size=(164,164)),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    train_data, valid_data, test_data = prepare_datasets(data_dir)
    train_ds = CacheDataset(data=train_data, transform=train_transforms,
                        cache_rate=1.0, num_workers=4)
    valid_ds = CacheDataset(data=valid_data, transform=val_transforms,
                        cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=8, num_workers=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(valid_ds, batch_size=8, num_workers=4, pin_memory=True)


    device = torch.device("cuda:0")
    model = HoVerNet(
        mode="fast",
        in_channels=3,
        out_classes=7,
        act=("relu", {"inplace": True}),
        norm="batch",
        dropout_prob=0.2,
    ).to(device)
    loss_function = HoVerNetLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    max_epochs = 300
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], []]
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()

    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, label, label_type, hover_map = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
                batch_data["label_type"].to(device),
                batch_data["hover_label_inst"].to(device),
            )

            labels = {
                HoVerNetBranch.NP: label,
                HoVerNetBranch.HV: hover_map,
                HoVerNetBranch.NC: label_type,
            }
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_label = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )

                    val_outputs = model(val_inputs)
                    val_outputs = [post_process(i, device=device)[0] for i in decollate_batch(val_outputs)]
                    val_label = [i for i in decollate_batch(val_label)]

                    dice_metric(y_pred=val_outputs, y=val_label)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                dice_metric.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join(writer.log_dir, "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                plot_2d_or_3d_image(val_inputs, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_label, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start
    print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}"
    f"total time: {total_time}")
    writer.close()

if __name__ == "__main__":

    set_determinism(seed=0)
    data_dir = "/workspace/Data/Lizard/Prepared"
    run(data_dir)