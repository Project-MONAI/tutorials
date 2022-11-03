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

import os
import time
import glob
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import scipy.io as sio
from argparse import ArgumentParser
from monai.data import DataLoader, decollate_batch, CacheDataset, GridPatchDataset, PatchIterd, ShuffleBuffer
from monai.metrics import DiceMetric
# from monai.networks.nets import HoVerNet
from monai.transforms import (
    LoadImaged,
    Lambdad,
    Transposed,
    Activations,
    AsDiscrete,
    AsDiscreted,
    Compose,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    CenterSpatialCrop,
    CastToTyped,
    ComputeHoVerMapsd,
    RandFlipd,
    RandRotate90d,
    GaussianSmooth,
    ToMetaTensord,
    RandGaussianSmoothd,
    FillHoles,
    BoundingRect,
    CenterSpatialCropd,
)

from monai.utils import set_determinism, convert_to_tensor
from monai.utils.enums import HoVerNetBranch
from monai.visualize import plot_2d_or_3d_image

from loss import HoVerNetLoss
from net import HoVerNet
from transforms import (
    GenerateWatershedMaskd,
    GenerateInstanceBorderd,
    GenerateDistanceMapd,
    GenerateWatershedMarkersd,
    Watershedd,
    GenerateInstanceContour,
    GenerateInstanceCentroid,
    GenerateInstanceType,
)


def prepare_data(data_dir):
    train_data_dir = os.path.join(data_dir, 'Train')
    test_data_dir = os.path.join(data_dir, 'Test')

    train_images = sorted(
        glob.glob(os.path.join(train_data_dir, "Images/*.png")))
    train_labels = sorted(
        glob.glob(os.path.join(train_data_dir, "Labels/*.mat")))

    labels, inst_maps, type_maps = [], [], []
    for label in train_labels:
        label_data = sio.loadmat(label)
        inst_maps.append(label_data['inst_map'][None].astype(int))
        type_maps.append(label_data['type_map'][None])
        labels.append(np.array(label_data['inst_map'][None] > 0, dtype=int))
        
    data_dicts = [
        {"image": _image, "label": _label, "label_inst": _inst_map, "label_type": _type_map}
        for _image, _label, _inst_map, _type_map in zip(train_images, labels, inst_maps, type_maps)
    ]
    train_data, valid_data = data_dicts[:-7], data_dicts[-7:]

    return train_data, valid_data

def post_process(output, device, return_binary=True, return_centroids=False, output_classes=None):
    post_trans_seg = Compose([
        GenerateWatershedMaskd(keys=HoVerNetBranch.NP.value, softmax=True),
        GenerateInstanceBorderd(keys='mask', hover_map_key=HoVerNetBranch.HV, kernel_size=3),
        GenerateDistanceMapd(keys='mask', border_key='border', smooth_fn=GaussianSmooth()),
        GenerateWatershedMarkersd(keys='mask', border_key='border', threshold=0.7, radius=2, postprocess_fn=FillHoles()),
        Watershedd(keys='dist', mask_key='mask', markers_key='markers')
    ])
    if HoVerNetBranch.NC.value in output.keys():
        type_pred = Activations(softmax=True)(output[HoVerNetBranch.NC.value])
        type_pred = AsDiscrete(argmax=True)(type_pred)

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


def run(data_dir, args):
    train_transforms = Compose(
        [
            LoadImaged(keys="image", image_only=True),
            Transposed(keys="image", indices=[2, 1, 0]), # make channel first
            Lambdad(keys="image", func=lambda x: x[:3, ...]),
            ComputeHoVerMapsd(keys="label_inst"),
            CastToTyped(keys=["image", "label_inst", "label_type", "hover_label_inst"], dtype=torch.float32),
            AsDiscreted(keys=["label", "label_type"], to_onehot=[2, 8]),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            RandCropByPosNegLabeld(
                keys=["image", "label", "label_inst", "label_type", "hover_label_inst"],
                label_key="label",
                spatial_size=[256, 256],
                pos=1,
                neg=0,
                num_samples=4,
            ),
            CenterSpatialCropd(
                keys=["label", "label_inst", "label_type", "hover_label_inst"], 
                roi_size=(164, 164),
            ),
            RandFlipd(keys=["image", "label", "label_inst", "label_type", "hover_label_inst"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label", "label_inst", "label_type", "hover_label_inst"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "label", "label_inst", "label_type", "hover_label_inst"], prob=0.5, max_k=1),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.5,1.15), sigma_y=(0.5,1.15), prob=0.5),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys="image", image_only=True),
            Transposed(keys="image", indices=[2, 1, 0]), # make channel first
            Lambdad(keys="image", func=lambda x: x[:3, ...]),
            ComputeHoVerMapsd(keys="label_inst"),
            CastToTyped(keys=["image", "label_inst", "label_type", "hover_label_inst"], dtype=torch.float32),
            AsDiscreted(keys=["label", "label_type"], to_onehot=[2, 8]),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )
    patch_iter = PatchIterd(
        keys=["image", "label", "label_inst", "label_type", "hover_label_inst"], 
        patch_size=(256, 256), 
        start_pos=(0, 0),
        mode="reflect"
    )
    cropper = Compose([
        ToMetaTensord(keys=["image", "label", "label_inst", "label_type", "hover_label_inst"]),
        CenterSpatialCropd(keys=["label", "label_inst", "label_type", "hover_label_inst"], roi_size=(164,164)),
        ])
    train_data, valid_data = prepare_data(data_dir)

    print("train_files:", len(train_data))
    print("val_files:", len(valid_data))

    train_ds = CacheDataset(data=train_data, transform=train_transforms,
                        cache_rate=1.0, num_workers=4)
    _valid_ds = CacheDataset(data=valid_data, transform=val_transforms,
                        cache_rate=1.0, num_workers=4)
#     train_ds = GridPatchDataset(
#         data=_train_ds,
#         patch_iter=patch_iter,
#         transform=cropper,
#         with_coordinates=False)
    valid_ds = GridPatchDataset(
        data=_valid_ds,
        patch_iter=patch_iter,
        transform=cropper,
        with_coordinates=False)
#     shuffle_ds = ShuffleBuffer(train_ds, buffer_size=30, seed=0)
#     train_loader = DataLoader(shuffle_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(valid_ds, batch_size=args.batch_size*4, num_workers=4, pin_memory=True)

    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)
    model = HoVerNet(
        mode="fast",
        in_channels=3,
        out_classes=8,
        act=("relu", {"inplace": True}),
        norm="batch",
        pretrained=True,
    ).to(device)
    model.freeze_encoder()
    loss_function = HoVerNetLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    if args.amp:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        print("[info] amp enabled")

    max_epochs = args.max_epochs
    val_interval = args.val_freq
    best_metric = -1
    best_metric_epoch = -1
    metric_values = []
    writer = SummaryWriter(comment=f'bs{args.batch_size}_ep{max_epochs}_lr{args.lr}')

    total_start = time.time()
    globel_step = 0
    for epoch in range(max_epochs):
        if epoch > 50:
            model.res_blocks.requires_grad_(True)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25)
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")

        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            globel_step += 1 
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
            if args.amp:
                with autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs.float(), labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            lr_scheduler.step()
            epoch_loss += loss.item()
            print(f"{step}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), globel_step)

        if epoch > 50 and (epoch + 1) % val_interval == 0:
            torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_label = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    with torch.cuda.amp.autocast(enabled=args.amp):
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
                    torch.save(
                        model.state_dict(),
                        os.path.join(writer.log_dir, f"best_metric_model.pth"),
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

            torch.cuda.empty_cache()

        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start
    print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}"
    f"total time: {total_time}")
    writer.flush()
    writer.close()

def main():
    parser = ArgumentParser(description="HoVerNet training torch pipeline")
    parser.add_argument("--ngc", action="store_true", dest="ngc", help="use ngc")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")

    parser.add_argument("--bs", type=int, default=8, dest="batch_size", help="batch size")
    parser.add_argument("--ep", type=int, default=200, dest="max_epochs", help="max epochs")
    parser.add_argument("-f", "--val_freq", type=int, default=10, help="validation frequence")
    parser.add_argument("--lr", type=float, default=1e-4, dest="lr", help="initial learning rate")

    args = parser.parse_args()
    
    set_determinism(seed=0)
    import sys
    if args.ngc:
        data_dir = "/consep"
        sys.path.append('/workspace/pathology/lizard/transforms')
        sys.path.append('/workspace/pathology/lizard/loss')
        sys.path.append('/workspace/pathology/lizard/net')
    else:
        data_dir = "/workspace/Data/Lizard/Prepared"
        sys.path.append('/workspace/Code/tutorials/pathology/hovernet/transforms')
        sys.path.append('/workspace/Code/tutorials/pathology/hovernet/loss')
        sys.path.append('/workspace/Code/tutorials/pathology/hovernet/net')

    run(data_dir, args)

if __name__ == "__main__":
    main()
