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

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter

from torch.utils.data import DataLoader, DistributedSampler
from utils import collate_fn, reduce_tensor, save_checkpoint, TensorboardLogger

from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
)

from monai.networks.nets import ViTAutoEnc
from monai.data import (
    CacheDataset,
    load_decathlon_datalist,
)

from monai.losses import ContrastiveLoss
from torch.nn import L1Loss


def parse_option():
    parser = argparse.ArgumentParser("ViT Self-Supervised Learning", add_help=False)

    # Set Paths for running SSL training
    parser.add_argument("--data_root", default="/workspace/datasets/tcia/", type=str, help="path to data root")
    parser.add_argument(
        "--json_path",
        default="./datalists/tcia/dataset_split.json",
        type=str,
        help="Json file path for list of data samples",
    )
    parser.add_argument("--logdir_path", default="/to/be/defined", type=str, help="output log directory")
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )

    # DL Training Hyper-parameters
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size for single GPU")
    parser.add_argument("--base_lr", default=5e-4, type=float, help="base learning rate")

    parser.add_argument("--seed", default=19, type=int, help="seed")
    parser.add_argument("--deterministic", help="set seed for deterministic training", action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    json_path = args.json_path
    data_root = args.data_root

    # Define training transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
            RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=2),
            CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                    ),
                    RandCoarseDropoutd(
                        keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                    ),
                ]
            ),
            RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
            # Please note that that if image, image_2 are called via the same transform call because of the determinism
            # they will get augmented the exact same way which is not the required case here, hence two calls are made
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                    ),
                    RandCoarseDropoutd(
                        keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                    ),
                ]
            ),
            RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
        ]
    )

    # Build the data loader
    train_list = load_decathlon_datalist(
        data_list_file_path=json_path, is_segmentation=False, data_list_key="training", base_dir=data_root
    )

    val_list = load_decathlon_datalist(
        data_list_file_path=json_path, is_segmentation=False, data_list_key="validation", base_dir=data_root
    )

    print("Total training data are {} and validation data are {}".format(len(train_list), len(val_list)))

    train_dataset = CacheDataset(data=train_list, transform=train_transforms, cache_rate=1.0, num_workers=4)
    val_dataset = CacheDataset(data=val_list, transform=train_transforms, cache_rate=1.0, num_workers=4)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Data Loaders Built
    # Define the model, losses and optimizer
    model = ViTAutoEnc(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        pos_embed="conv",
        hidden_size=768,
        mlp_dim=3072,
    )

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[int(os.environ["LOCAL_RANK"])], broadcast_buffers=False, find_unused_parameters=True
    )
    model_without_ddp = model.module

    recon_loss = L1Loss()
    contrastive_loss = ContrastiveLoss(temperature=0.05)
    loss_funcs = [recon_loss, contrastive_loss]

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {}".format(n_parameters))

    print("Training Begins ...")

    start_time = time.time()
    val_loss_best = 1e9

    for epoch in range(args.epochs):
        train_loss_avg, train_l1_avg, train_cl_avg = train_one_epoch(
            args=args,
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            loss_functions=loss_funcs,
        )

        val_loss_avg = validate(data_loader=val_loader, model=model, loss_functions=loss_funcs)

        if dist.get_rank() == 0:
            log_writer.update(loss_val_L1=val_loss_avg, head="perf", step=epoch)
            log_writer.update(loss_train_avg=train_loss_avg, head="perf", step=epoch)
            log_writer.update(loss_train_L1=train_l1_avg, head="perf", step=epoch)
            log_writer.update(loss_train_MM=train_cl_avg, head="perf", step=epoch)
            if val_loss_avg <= val_loss_best:
                save_checkpoint(args, epoch, model_without_ddp, 0.0, optimizer, best_model=True)

        if dist.get_rank() == 0 and (epoch == (args.epochs - 1)):
            save_checkpoint(args, epoch, model_without_ddp, 0.0, optimizer, best_model=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))


def train_one_epoch(args, model, data_loader, optimizer, epoch, loss_functions):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_l1_meter = AverageMeter()
    loss_cont_meter = AverageMeter()
    loss_meter = AverageMeter()

    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, batch_data in enumerate(data_loader):
        batch_data = batch_data[0]
        inputs, inputs_2, gt_input = (
            batch_data["image"].cuda(non_blocking=True),
            batch_data["image_2"].cuda(non_blocking=True),
            batch_data["gt_image"].cuda(non_blocking=True),
        )

        optimizer.zero_grad()

        outputs_v1, hidden_v1 = model(inputs)
        outputs_v2, hidden_v2 = model(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

        r_loss = loss_functions[0](outputs_v1, gt_input)
        cl_loss = loss_functions[1](flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        total_loss = r_loss + cl_loss * r_loss

        total_loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        r_loss_t = reduce_tensor(r_loss)
        cl_loss_t = reduce_tensor(cl_loss)
        total_loss_t = reduce_tensor(total_loss)

        loss_l1_meter.update(r_loss_t.item(), inputs.size(0))
        loss_cont_meter.update(cl_loss_t.item(), inputs.size(0))
        loss_meter.update(total_loss_t.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]["lr"]
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        etas = batch_time.avg * (num_steps - idx)
        print(
            f"Train: [{epoch}/{args.epochs}][{idx}/{num_steps}]\t"
            f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
            f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
            f"loss_L1 {loss_l1_meter.val:.4f} ({loss_l1_meter.avg:.4f})\t"
            f"loss_MM {loss_cont_meter.val:.4f} ({loss_cont_meter.avg:.4f})\t"
            f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
            f"mem {memory_used:.0f}MB"
        )

    epoch_time = time.time() - start
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return loss_meter.avg, loss_l1_meter.avg, loss_cont_meter.avg


@torch.no_grad()
def validate(data_loader, model, loss_functions):
    model.eval()
    loss_l1_meter = AverageMeter()

    for idx, batch_data in enumerate(data_loader):
        batch_data = batch_data[0]

        inputs, gt_input = (
            batch_data["image"].cuda(non_blocking=True),
            batch_data["gt_image"].cuda(non_blocking=True),
        )

        outputs, outputs_v2 = model(inputs)
        val_loss = loss_functions[0](outputs, gt_input)
        loss = reduce_tensor(val_loss)
        loss_l1_meter.update(loss.item(), inputs.size(0))

        print(f"Test: [{idx}/{len(data_loader)}]\t" f"Loss_L1 {loss_l1_meter.val:.4f} ({loss_l1_meter.avg:.4f})\t")

    print(f" * Val Loss {loss_l1_meter.avg:.3f}")

    return loss_l1_meter.avg


if __name__ == "__main__":
    args = parse_option()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = args.seed + dist.get_rank()

    if args.deterministic:
        torch.manual_seed(seed)
        np.random.seed(seed)

    cudnn.benchmark = True

    if dist.get_rank() == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
        if not os.path.exists(args.logdir_path):
            os.makedirs(args.logdir_path, exist_ok=True)

    if dist.get_rank() == 0:
        log_writer = TensorboardLogger(log_dir=args.logdir_path)

    main(args)
