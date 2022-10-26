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
import math
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from monai.data import DataLoader, decollate_batch, CacheDataset, partition_dataset
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


def split_dataset(data_dir):
    # using original split in the paper
    info = pd.read_csv(os.path.join(data_dir, "patch_info.csv"))
    file_names = np.squeeze(info.to_numpy()).tolist()
    split_info = pd.read_csv(os.path.join(data_dir, "split_info.csv"))

    indices, splits = [], []
    for i in range(3):
        fold_case = split_info.loc[split_info['Split'] == i+1, 'Filename'].tolist()
        fold_patches = [
            file_name for file_name in file_names for _name in fold_case if _name == file_name.split("-")[0]
        ]
        fold_patches = np.unique(fold_patches)
        print(f"Fold: {i} - {len(fold_patches):04d}")

        fold_indices = [file_names.index(v) for v in fold_patches]
        indices.append(fold_indices)

    for i in range(3):
        _indices = indices.copy()
        splits.append({
            "valid": _indices.pop(i),
            "train": sum(_indices, []),
        })

    return splits

def prepare_data(data_dir, fold, splits):
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
    train_data = [data[i] for i in splits[fold]['train']]
    valid_data = [data[i] for i in splits[fold]['valid']]
    
    return train_data, valid_data

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


def run(data_dir, fold, args):
    print("[info] number of GPUs:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
    else:
        world_size = 1
    print("[info] world_size:", world_size)

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

    splits = split_dataset(data_dir)
    train_data, valid_data = prepare_data(data_dir, fold, splits)

    if torch.cuda.device_count() > 1:
        train_data = partition_dataset(data=train_data[fold], shuffle=True, num_partitions=world_size, even_divisible=True)[
            dist.get_rank()
        ]
        if len(valid_data[fold]) < world_size:
            valid_data[fold] = valid_data[fold] * math.ceil(float(world_size) / float(len(valid_data[fold])))

        valid_data = partition_dataset(data=valid_data[fold], shuffle=False, num_partitions=world_size, even_divisible=False)[
            dist.get_rank()
        ]
    print("train_files:", len(train_data))
    print("val_files:", len(valid_data))

    train_ds = CacheDataset(data=train_data, transform=train_transforms,
                        cache_rate=1.0, num_workers=4)
    valid_ds = CacheDataset(data=valid_data, transform=val_transforms,
                        cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(valid_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    device = torch.device(f"cuda:{dist.get_rank()}") if torch.cuda.device_count() > 1 else torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = HoVerNet(
        mode="fast",
        in_channels=3,
        out_classes=7,
        act=("relu", {"inplace": True}),
        norm="batch",
        dropout_prob=0.2,
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    loss_function = HoVerNetLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)

    if args.amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            print("[info] amp enabled")

    max_epochs = args.max_epochs
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer = SummaryWriter(comment=f'_{fold}')

    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")

        model.train()
        epoch_loss = 0
        loss_torch = torch.zeros(2, dtype=torch.float, device=device)
        step = 0
        for batch_data in train_loader:
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

            epoch_loss += loss.item()
            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0
            epoch_len = len(train_ds) // train_loader.batch_size
            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            lr_scheduler.step()
        
        if torch.cuda.device_count() > 1:
            dist.barrier()
            dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            print(
                f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}"
            )

        if (epoch + 1) % val_interval == 0:
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
                    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                        torch.save(
                            model.state_dict(),
                            os.path.join(writer.log_dir, f"best_metric_model{fold}.pth"),
                        )
                        print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                    writer.add_scalar("val_mean_dice", metric, epoch + 1)
                plot_2d_or_3d_image(val_inputs, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_label, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

                if torch.cuda.device_count() > 1:
                    dist.barrier()
            torch.cuda.empty_cache()

        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start
    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}"
        f"total time: {total_time}")
        writer.flush()
        writer.close()

    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()

def main():
    parser = ArgumentParser(description="HoVerNet training torch pipeline")
    parser.add_argument("--ngc", action="store_true", dest="ngc", help="use ngc")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")

    parser.add_argument("--n", type=int, default=3, dest="n_fold", help="fold of cross validation")
    parser.add_argument("--bs", type=int, default=8, dest="batch_size", help="batch size")
    parser.add_argument("--ep", type=int, default=200, dest="max_epochs", help="max epochs")

    args = parser.parse_args()
    
    set_determinism(seed=0)
    import sys
    if args.ngc:
        data_dir = "/Lizard"
        sys.path.append('/workspace/pathology/lizard/transforms')
        sys.path.append('/workspace/pathology/lizard/loss')
    else:
        data_dir = "/workspace/Data/Lizard/Prepared"
        sys.path.append('/workspace/Code/tutorials/pathology/hovernet/transforms')
        sys.path.append('/workspace/Code/tutorials/pathology/hovernet/loss')

    for i in range(args.n_fold):
        run(data_dir, i, args)

if __name__ == "__main__":
    main()