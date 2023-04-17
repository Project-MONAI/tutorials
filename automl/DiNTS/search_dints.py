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

"""
This example shows how to execute the DiNTS algorithm to search the optimal neural architecture with
distributed training based on PyTorch native `DistributedDataParallel` module.
It can run on several nodes with multiple GPU devices on every node.
This example is a real-world task based on Decathlon challenge Task09: Spleen (CT) segmentation.
Under default settings, each single GPU needs to use ~13GB memory for network training.
Main steps to set up the distributed training:
- Execute `torch.distributed.launch` to create processes on every node for every GPU.
  It receives parameters as below:
  `--nproc_per_node=NUM_GPUS_PER_NODE`
  `--nnodes=NUM_NODES`
  `--node_rank=INDEX_CURRENT_NODE`
  `--master_addr="192.168.1.1"`
  `--master_port=1234`
  For more details, refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py.
- Use `init_process_group` to initialize every process, every GPU runs in a separate process with unique rank.
  Here we use `NVIDIA NCCL` as the backend and must set `init_method="env://"` if use `torch.distributed.launch`.
- Wrap the model with `DistributedDataParallel` after moving to expected device.
- Partition dataset before training, so every rank process will only handle its own data partition.
Note:
    `torch.distributed.launch` will launch `nnodes * nproc_per_node = world_size` processes in total.
    Suggest setting exactly the same software environment for every node, especially `PyTorch`, `nccl`, etc.
    A good practice is to use the same MONAI docker image for all nodes directly.
    Example script to execute this program on every node:
    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
           --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
           --master_addr="192.168.1.1" --master_port=1234
           brats_training_ddp.py -d DIR_OF_TESTDATA
    This example was tested with [Ubuntu 16.04/20.04], [NCCL 2.6.3].
Referring to: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
Some codes are taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py

"""

import argparse
import copy
import json
import logging
import monai
import nibabel as nib
import numpy as np
import os
import pandas as pd
import pathlib
import random
import shutil
import sys
import tempfile
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pdb
from datetime import datetime
from glob import glob
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from monai.apps import download_and_extract
from monai.data import (
    DataLoader,
    ThreadDataLoader,
    decollate_batch,
)
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    apply_transform,
    Randomizable,
    Transform,
    AsDiscrete,
    AsDiscreted,
    AddChannel,
    AddChanneld,
    AsChannelFirstd,
    CastToTyped,
    Compose,
    ConcatItemsd,
    CopyItemsd,
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    KeepLargestConnectedComponent,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityRanged,
    ThresholdIntensityd,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandFlipd,
    RandRotated,
    RandRotate90d,
    RandZoomd,
    Spacingd,
    SpatialPadd,
    SqueezeDimd,
    ToDeviced,
    ToNumpyd,
    ToTensord,
)
from monai.data import Dataset, create_test_image_3d, DistributedSampler, list_data_collate, partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.utils import set_determinism
from scipy import ndimage


def main():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint full path",
    )
    parser.add_argument(
        "--factor_ram_cost",
        default=0.0,
        type=float,
        help="factor to determine RAM cost in the searched architecture",
    )
    parser.add_argument(
        "--fold",
        action="store",
        required=True,
        help="fold index in N-fold cross-validation",
    )
    parser.add_argument(
        "--json",
        action="store",
        required=True,
        help="full path of .json file",
    )
    parser.add_argument(
        "--json_key",
        action="store",
        required=True,
        help="selected key in .json data list",
    )
    parser.add_argument(
        "--local_rank",
        required=int,
        help="local process rank",
    )
    parser.add_argument(
        "--num_folds",
        action="store",
        required=True,
        help="number of folds in cross-validation",
    )
    parser.add_argument(
        "--output_root",
        action="store",
        required=True,
        help="output root",
    )
    parser.add_argument(
        "--root",
        action="store",
        required=True,
        help="data root",
    )
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)

    amp = True
    determ = True
    factor_ram_cost = args.factor_ram_cost
    fold = int(args.fold)
    input_channels = 1
    learning_rate = 0.025
    learning_rate_arch = 0.001
    learning_rate_milestones = np.array([0.4, 0.8])
    num_images_per_batch = 1
    num_epochs = 1430  # around 20k iteration
    num_epochs_per_validation = 100
    num_epochs_warmup = 715
    num_folds = int(args.num_folds)
    num_patches_per_image = 1
    num_sw_batch_size = 6
    output_classes = 3
    overlap_ratio = 0.625
    patch_size = (96, 96, 96)
    patch_size_valid = (96, 96, 96)
    spacing = [1.0, 1.0, 1.0]

    print("factor_ram_cost", factor_ram_cost)

    # deterministic training
    if determ:
        set_determinism(seed=0)

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

    # dist.barrier()
    world_size = dist.get_world_size()

    with open(args.json, "r") as f:
        json_data = json.load(f)

    split = len(json_data[args.json_key]) // num_folds
    list_train = json_data[args.json_key][: (split * fold)] + json_data[args.json_key][(split * (fold + 1)) :]
    list_valid = json_data[args.json_key][(split * fold) : (split * (fold + 1))]

    # training data
    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(args.root, list_train[_i]["image"])
        str_seg = os.path.join(args.root, list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    train_files = files

    random.shuffle(train_files)

    train_files_w = train_files[: len(train_files) // 2]
    train_files_w = partition_dataset(data=train_files_w, shuffle=True, num_partitions=world_size, even_divisible=True)[
        dist.get_rank()
    ]
    print("train_files_w:", len(train_files_w))

    train_files_a = train_files[len(train_files) // 2 :]
    train_files_a = partition_dataset(data=train_files_a, shuffle=True, num_partitions=world_size, even_divisible=True)[
        dist.get_rank()
    ]
    print("train_files_a:", len(train_files_a))

    # validation data
    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(args.root, list_valid[_i]["image"])
        str_seg = os.path.join(args.root, list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    val_files = files
    val_files = partition_dataset(data=val_files, shuffle=False, num_partitions=world_size, even_divisible=False)[
        dist.get_rank()
    ]
    print("val_files:", len(val_files))

    # network architecture
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest"), align_corners=(True, True)),
            CastToTyped(keys=["image"], dtype=(torch.float32)),
            ScaleIntensityRanged(keys=["image"], a_min=-87.0, a_max=199.0, b_min=0.0, b_max=1.0, clip=True),
            CastToTyped(keys=["image", "label"], dtype=(np.float16, np.uint8)),
            CopyItemsd(keys=["label"], times=1, names=["label4crop"]),
            Lambdad(
                keys=["label4crop"],
                func=lambda x: np.concatenate(
                    tuple(
                        [
                            ndimage.binary_dilation((x == _k).astype(x.dtype), iterations=48).astype(float)
                            for _k in range(output_classes)
                        ]
                    ),
                    axis=0,
                ),
                overwrite=True,
            ),
            EnsureTyped(keys=["image", "label"]),
            CastToTyped(keys=["image"], dtype=(torch.float32)),
            SpatialPadd(
                keys=["image", "label", "label4crop"], spatial_size=patch_size, mode=["reflect", "constant", "constant"]
            ),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label4crop",
                num_classes=output_classes,
                ratios=[
                    1,
                ]
                * output_classes,
                spatial_size=patch_size,
                num_samples=num_patches_per_image,
            ),
            Lambdad(keys=["label4crop"], func=lambda x: 0),
            RandRotated(
                keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, mode=["bilinear", "nearest"], prob=0.2
            ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.8,
                max_zoom=1.2,
                mode=["trilinear", "nearest"],
                align_corners=[True, None],
                prob=0.16,
            ),
            RandGaussianSmoothd(
                keys=["image"], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15), prob=0.15
            ),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.uint8)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest"), align_corners=(True, True)),
            CastToTyped(keys=["image"], dtype=(torch.float32)),
            ScaleIntensityRanged(keys=["image"], a_min=-87.0, a_max=199.0, b_min=0.0, b_max=1.0, clip=True),
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
        ]
    )

    train_ds_a = monai.data.CacheDataset(data=train_files_a, transform=train_transforms, cache_rate=1.0, num_workers=8)
    train_ds_w = monai.data.CacheDataset(data=train_files_w, transform=train_transforms, cache_rate=1.0, num_workers=8)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)

    # monai.data.Dataset can be used as alternatives when debugging or RAM space is limited.
    # train_ds_a = monai.data.Dataset(data=train_files_a, transform=train_transforms)
    # train_ds_w = monai.data.Dataset(data=train_files_w, transform=train_transforms)
    # val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)

    train_loader_a = ThreadDataLoader(train_ds_a, num_workers=4, batch_size=num_images_per_batch, shuffle=True)
    train_loader_w = ThreadDataLoader(train_ds_w, num_workers=4, batch_size=num_images_per_batch, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=4, batch_size=1, shuffle=False)

    # DataLoader can be used as alternatives when ThreadDataLoader is less efficient.
    # train_loader_a = DataLoader(train_ds_a, batch_size=num_images_per_batch, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    # train_loader_w = DataLoader(train_ds_w, batch_size=num_images_per_batch, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    # val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    dints_space = monai.networks.nets.TopologySearch(
        channel_mul=0.5,
        num_blocks=12,
        num_depths=4,
        use_downsample=True,
        device=device,
    )

    model = monai.networks.nets.DiNTS(
        dints_space=dints_space,
        in_channels=input_channels,
        num_classes=output_classes,
        use_downsample=True,
    )

    model = model.to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=output_classes)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=output_classes)])

    # loss function
    loss_func = monai.losses.DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        batch=True,
        smooth_nr=0.00001,
        smooth_dr=0.00001,
    )

    # optimizer
    optimizer = torch.optim.SGD(
        model.weight_parameters(), lr=learning_rate * world_size, momentum=0.9, weight_decay=0.00004
    )
    arch_optimizer_a = torch.optim.Adam(
        [dints_space.log_alpha_a], lr=learning_rate_arch * world_size, betas=(0.5, 0.999), weight_decay=0.0
    )
    arch_optimizer_c = torch.optim.Adam(
        [dints_space.log_alpha_c], lr=learning_rate_arch * world_size, betas=(0.5, 0.999), weight_decay=0.0
    )

    print()

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    if args.checkpoint != None and os.path.isfile(args.checkpoint):
        print("[info] fine-tuning pre-trained checkpoint {0:s}".format(args.checkpoint))
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        torch.cuda.empty_cache()
    else:
        print("[info] training from scratch")

    # amp
    if amp:
        from torch.cuda.amp import autocast, GradScaler

        scaler = GradScaler()
        if dist.get_rank() == 0:
            print("[info] amp enabled")

    # start a typical PyTorch training
    val_interval = num_epochs_per_validation
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    idx_iter = 0
    metric_values = list()

    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_root, "Events"))

        with open(os.path.join(args.output_root, "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

    dataloader_a_iterator = iter(train_loader_a)

    start_time = time.time()
    for epoch in range(num_epochs):
        decay = 0.5 ** np.sum(
            [(epoch - num_epochs_warmup) / (num_epochs - num_epochs_warmup) > learning_rate_milestones]
        )
        lr = learning_rate * decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if dist.get_rank() == 0:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{num_epochs}")
            print("learning rate is set to {}".format(lr))

        model.train()
        epoch_loss = 0
        loss_torch = torch.zeros(2, dtype=torch.float, device=device)
        epoch_loss_arch = 0
        loss_torch_arch = torch.zeros(2, dtype=torch.float, device=device)
        step = 0

        for batch_data in train_loader_w:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            if world_size == 1:
                for _ in model.weight_parameters():
                    _.requires_grad = True
            else:
                for _ in model.module.weight_parameters():
                    _.requires_grad = True
            dints_space.log_alpha_a.requires_grad = False
            dints_space.log_alpha_c.requires_grad = False

            optimizer.zero_grad()

            if amp:
                with autocast():
                    outputs = model(inputs)
                    if output_classes == 2:
                        loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                    else:
                        loss = loss_func(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if output_classes == 2:
                    loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                else:
                    loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0
            epoch_len = len(train_loader_w)
            idx_iter += 1

            if dist.get_rank() == 0:
                print("[{0}] ".format(str(datetime.now())[:19]) + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

            if epoch < num_epochs_warmup:
                continue

            try:
                sample_a = next(dataloader_a_iterator)
            except StopIteration:
                dataloader_a_iterator = iter(train_loader_a)
                sample_a = next(dataloader_a_iterator)
            inputs_search, labels_search = sample_a["image"].to(device), sample_a["label"].to(device)
            if world_size == 1:
                for _ in model.weight_parameters():
                    _.requires_grad = False
            else:
                for _ in model.module.weight_parameters():
                    _.requires_grad = False
            dints_space.log_alpha_a.requires_grad = True
            dints_space.log_alpha_c.requires_grad = True

            # linear increase topology and RAM loss
            entropy_alpha_c = torch.tensor(0.0).to(device)
            entropy_alpha_a = torch.tensor(0.0).to(device)
            ram_cost_full = torch.tensor(0.0).to(device)
            ram_cost_usage = torch.tensor(0.0).to(device)
            ram_cost_loss = torch.tensor(0.0).to(device)
            topology_loss = torch.tensor(0.0).to(device)

            probs_a, arch_code_prob_a = dints_space.get_prob_a(child=True)
            entropy_alpha_a = -((probs_a) * torch.log(probs_a + 1e-5)).mean()
            entropy_alpha_c = -(
                F.softmax(dints_space.log_alpha_c, dim=-1) * F.log_softmax(dints_space.log_alpha_c, dim=-1)
            ).mean()
            topology_loss = dints_space.get_topology_entropy(probs_a)

            ram_cost_full = dints_space.get_ram_cost_usage(inputs.shape, full=True)
            ram_cost_usage = dints_space.get_ram_cost_usage(inputs.shape)
            ram_cost_loss = torch.abs(factor_ram_cost - ram_cost_usage / ram_cost_full)

            arch_optimizer_a.zero_grad()
            arch_optimizer_c.zero_grad()

            combination_weights = (epoch - num_epochs_warmup) / (num_epochs - num_epochs_warmup)

            if amp:
                with autocast():
                    outputs_search = model(inputs_search)
                    if output_classes == 2:
                        loss = loss_func(torch.flip(outputs_search, dims=[1]), 1 - labels_search)
                    else:
                        loss = loss_func(outputs_search, labels_search)

                    loss += combination_weights * (
                        (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                    )

                scaler.scale(loss).backward()
                scaler.step(arch_optimizer_a)
                scaler.step(arch_optimizer_c)
                scaler.update()
            else:
                outputs_search = model(inputs_search)
                if output_classes == 2:
                    loss = loss_func(torch.flip(outputs_search, dims=[1]), 1 - labels_search)
                else:
                    loss = loss_func(outputs_search, labels_search)

                loss += 1.0 * (
                    combination_weights * (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                )

                loss.backward()
                arch_optimizer_a.step()
                arch_optimizer_c.step()

            epoch_loss_arch += loss.item()
            loss_torch_arch[0] += loss.item()
            loss_torch_arch[1] += 1.0

            if dist.get_rank() == 0:
                print(
                    "[{0}] ".format(str(datetime.now())[:19])
                    + f"{step}/{epoch_len}, train_loss_arch: {loss.item():.4f}"
                )
                writer.add_scalar("train_loss_arch", loss.item(), epoch_len * epoch + step)

        # synchronizes all processes and reduce results
        dist.barrier()
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)
        loss_torch = loss_torch.tolist()
        loss_torch_arch = loss_torch_arch.tolist()
        if dist.get_rank() == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            print(
                f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
            )

            if epoch >= num_epochs_warmup:
                loss_torch_arch_epoch = loss_torch_arch[0] / loss_torch_arch[1]
                print(
                    f"epoch {epoch + 1} average arch loss: {loss_torch_arch_epoch:.4f}, best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
                )

        if (epoch + 1) % val_interval == 0:
            torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                metric = torch.zeros((output_classes - 1) * 2, dtype=torch.float, device=device)
                metric_sum = 0.0
                metric_count = 0
                metric_mat = []
                val_images = None
                val_labels = None
                val_outputs = None

                _index = 0
                for val_data in val_loader:
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    roi_size = patch_size_valid
                    sw_batch_size = num_sw_batch_size

                    if amp:
                        with torch.cuda.amp.autocast():
                            pred = sliding_window_inference(
                                val_images,
                                roi_size,
                                sw_batch_size,
                                lambda x: model(x),
                                mode="gaussian",
                                overlap=overlap_ratio,
                            )
                    else:
                        pred = sliding_window_inference(
                            val_images,
                            roi_size,
                            sw_batch_size,
                            lambda x: model(x),
                            mode="gaussian",
                            overlap=overlap_ratio,
                        )
                    val_outputs = pred

                    val_outputs = post_pred(val_outputs[0, ...])
                    val_outputs = val_outputs[None, ...]
                    val_labels = post_label(val_labels[0, ...])
                    val_labels = val_labels[None, ...]

                    value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False)

                    print(_index + 1, "/", len(val_loader), value)

                    metric_count += len(value)
                    metric_sum += value.sum().item()
                    metric_vals = value.cpu().numpy()
                    if len(metric_mat) == 0:
                        metric_mat = metric_vals
                    else:
                        metric_mat = np.concatenate((metric_mat, metric_vals), axis=0)

                    for _c in range(output_classes - 1):
                        val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                        val1 = 1.0 - torch.isnan(value[0, 0]).float()
                        metric[2 * _c] += val0 * val1
                        metric[2 * _c + 1] += val1

                    _index += 1

                # synchronizes all processes and reduce results
                dist.barrier()
                dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric = metric.tolist()
                if dist.get_rank() == 0:
                    for _c in range(output_classes - 1):
                        print("evaluation metric - class {0:d}:".format(_c + 1), metric[2 * _c] / metric[2 * _c + 1])
                    avg_metric = 0
                    for _c in range(output_classes - 1):
                        avg_metric += metric[2 * _c] / metric[2 * _c + 1]
                    avg_metric = avg_metric / float(output_classes - 1)
                    print("avg_metric", avg_metric)

                    if avg_metric > best_metric:
                        best_metric = avg_metric
                        best_metric_epoch = epoch + 1
                        best_metric_iterations = idx_iter

                    node_a_d, arch_code_a_d, arch_code_c_d, arch_code_a_max_d = dints_space.decode()
                    torch.save(
                        {
                            "node_a": node_a_d,
                            "arch_code_a": arch_code_a_d,
                            "arch_code_a_max": arch_code_a_max_d,
                            "arch_code_c": arch_code_c_d,
                            "iter_num": idx_iter,
                            "epochs": epoch + 1,
                            "best_dsc": best_metric,
                            "best_path": best_metric_iterations,
                        },
                        os.path.join(args.output_root, "search_code_" + str(idx_iter) + ".pth"),
                    )
                    print("saved new best metric model")

                    dict_file = {}
                    dict_file["best_avg_dice_score"] = float(best_metric)
                    dict_file["best_avg_dice_score_epoch"] = int(best_metric_epoch)
                    dict_file["best_avg_dice_score_iteration"] = int(idx_iter)
                    with open(os.path.join(args.output_root, "progress.yaml"), "w") as out_file:
                        documents = yaml.dump(dict_file, stream=out_file)

                    print(
                        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, avg_metric, best_metric, best_metric_epoch
                        )
                    )

                    current_time = time.time()
                    elapsed_time = (current_time - start_time) / 60.0
                    with open(os.path.join(args.output_root, "accuracy_history.csv"), "a") as f:
                        f.write(
                            "{0:d}\t{1:.5f}\t{2:.5f}\t{3:.5f}\t{4:.1f}\t{5:d}\n".format(
                                epoch + 1, avg_metric, loss_torch_epoch, lr, elapsed_time, idx_iter
                            )
                        )

                dist.barrier()

            torch.cuda.empty_cache()

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    if dist.get_rank() == 0:
        writer.close()

    dist.destroy_process_group()

    return


if __name__ == "__main__":
    main()
