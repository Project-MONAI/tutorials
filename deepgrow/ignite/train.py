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

import argparse
import distutils.util
import json
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
from monai.apps.deepgrow.interaction import Interaction
from monai.apps.deepgrow.transforms import (
    AddGuidanceSignald,
    AddInitialSeedPointd,
    AddRandomGuidanced,
    FindAllValidSlicesd,
    FindDiscrepancyRegionsd,
    SpatialCropForegroundd,
)
from monai.data import partition_dataset
from monai.data.dataloader import DataLoader
from monai.data.dataset import PersistentDataset
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.networks.layers import Norm
from monai.networks.nets import BasicUNet, UNet
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    Compose,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    ToNumpyd,
)
from monai.utils import set_determinism


def get_network(network, channels, dimensions):
    if network == "unet":
        if channels == 16:
            features = (16, 32, 64, 128, 256)
        elif channels == 32:
            features = (32, 64, 128, 256, 512)
        else:
            features = (64, 128, 256, 512, 1024)
        logging.info("Using Unet with features: {}".format(features))
        network = UNet(
            spatial_dims=dimensions,
            in_channels=3,
            out_channels=1,
            channels=features,
            strides=[2, 2, 2, 2],
            norm=Norm.BATCH,
        )
    else:
        if channels == 16:
            features = (16, 32, 64, 128, 256, 16)
        elif channels == 32:
            features = (32, 64, 128, 256, 512, 32)
        else:
            features = (64, 128, 256, 512, 1024, 64)
        logging.info("Using BasicUnet with features: {}".format(features))
        network = BasicUNet(spatial_dims=dimensions, in_channels=3, out_channels=1, features=features)
    return network


def get_pre_transforms(roi_size, model_size, dimensions):
    t = [
        LoadImaged(keys=("image", "label")),
        AddChanneld(keys=("image", "label")),
        SpatialCropForegroundd(keys=("image", "label"), source_key="label", spatial_size=roi_size),
        Resized(keys=("image", "label"), spatial_size=model_size, mode=("area", "nearest")),
        NormalizeIntensityd(keys="image", subtrahend=208.0, divisor=388.0),
    ]
    if dimensions == 3:
        t.append(FindAllValidSlicesd(label="label", sids="sids"))
    t.extend(
        [
            AddInitialSeedPointd(label="label", guidance="guidance", sids="sids"),
            AddGuidanceSignald(image="image", guidance="guidance"),
            EnsureTyped(keys=("image", "label")),
        ]
    )
    return Compose(t)


def get_click_transforms():
    return Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            ToNumpyd(keys=("image", "label", "pred")),
            FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy"),
            AddRandomGuidanced(
                guidance="guidance",
                discrepancy="discrepancy",
                probability="probability",
            ),
            AddGuidanceSignald(image="image", guidance="guidance"),
            EnsureTyped(keys=("image", "label")),
        ]
    )


def get_post_transforms():
    return Compose(
        [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ]
    )


def get_loaders(args, pre_transforms, train=True):
    multi_gpu = args.multi_gpu
    local_rank = args.local_rank

    dataset_json = os.path.join(args.input)
    with open(dataset_json) as f:
        datalist = json.load(f)

    total_d = len(datalist)
    datalist = datalist[0 : args.limit] if args.limit else datalist
    total_l = len(datalist)

    if multi_gpu:
        datalist = partition_dataset(
            data=datalist,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
            shuffle=True,
            seed=args.seed,
        )[local_rank]

    if train:
        train_datalist, val_datalist = partition_dataset(
            datalist,
            ratios=[args.split, (1 - args.split)],
            shuffle=True,
            seed=args.seed,
        )

        train_ds = PersistentDataset(train_datalist, pre_transforms, cache_dir=args.cache_dir)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=16)
        logging.info(
            "{}:: Total Records used for Training is: {}/{}/{}".format(local_rank, len(train_ds), total_l, total_d)
        )
    else:
        train_loader = None
        val_datalist = datalist

    val_ds = PersistentDataset(val_datalist, pre_transforms, cache_dir=args.cache_dir)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=8)
    logging.info(
        "{}:: Total Records used for Validation is: {}/{}/{}".format(local_rank, len(val_ds), total_l, total_d)
    )

    return train_loader, val_loader


def create_trainer(args):
    set_determinism(seed=args.seed)

    multi_gpu = args.multi_gpu
    local_rank = args.local_rank
    if multi_gpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda:{}".format(local_rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if args.use_gpu else "cpu")

    pre_transforms = get_pre_transforms(args.roi_size, args.model_size, args.dimensions)
    click_transforms = get_click_transforms()
    post_transform = get_post_transforms()

    train_loader, val_loader = get_loaders(args, pre_transforms)

    # define training components
    network = get_network(args.network, args.channels, args.dimensions).to(device)
    if multi_gpu:
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)

    if args.resume:
        logging.info("{}:: Loading Network...".format(local_rank))
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        network.load_state_dict(torch.load(args.model_filepath, map_location=map_location))

    # define event-handlers for engine
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=args.output, output_transform=lambda x: None),
        CheckpointSaver(
            save_dir=args.output,
            save_dict={"net": network},
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename="model.pt",
        ),
    ]
    val_handlers = val_handlers if local_rank == 0 else None

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        iteration_update=Interaction(
            transforms=click_transforms,
            max_interactions=args.max_val_interactions,
            key_probability="probability",
            train=False,
        ),
        inferer=SimpleInferer(),
        postprocessing=post_transform,
        key_val_metric={
            "val_dice": MeanDice(
                include_background=False,
                output_transform=from_engine(["pred", "label"]),
            )
        },
        val_handlers=val_handlers,
    )

    loss_function = DiceLoss(sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(network.parameters(), args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=args.val_freq, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        TensorBoardStatsHandler(
            log_dir=args.output,
            tag_name="train_loss",
            output_transform=from_engine(["loss"], first=True),
        ),
        CheckpointSaver(
            save_dir=args.output,
            save_dict={"net": network, "opt": optimizer, "lr": lr_scheduler},
            save_interval=args.save_interval * 2,
            save_final=True,
            final_filename="checkpoint.pt",
        ),
    ]
    train_handlers = train_handlers if local_rank == 0 else train_handlers[:2]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        network=network,
        iteration_update=Interaction(
            transforms=click_transforms,
            max_interactions=args.max_train_interactions,
            key_probability="probability",
            train=True,
        ),
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=SimpleInferer(),
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric={
            "train_dice": MeanDice(
                include_background=False,
                output_transform=from_engine(["pred", "label"]),
            )
        },
        train_handlers=train_handlers,
    )
    return trainer


def run(args):
    args.roi_size = json.loads(args.roi_size)
    args.model_size = json.loads(args.model_size)

    if args.local_rank == 0:
        for arg in vars(args):
            logging.info("USING:: {} = {}".format(arg, getattr(args, arg)))
        print("")

    if args.export:
        logging.info("{}:: Loading PT Model from: {}".format(args.local_rank, args.input))
        device = torch.device("cuda" if args.use_gpu else "cpu")
        network = get_network(args.network, args.channels, args.dimensions).to(device)

        map_location = {"cuda:0": "cuda:{}".format(args.local_rank)}
        network.load_state_dict(torch.load(args.input, map_location=map_location))

        logging.info("{}:: Saving TorchScript Model".format(args.local_rank))
        model_ts = torch.jit.script(network)
        torch.jit.save(model_ts, os.path.join(args.output))
        return

    if not os.path.exists(args.output):
        logging.info("output path [{}] does not exist. creating it now.".format(args.output))
        os.makedirs(args.output, exist_ok=True)

    trainer = create_trainer(args)

    start_time = time.time()
    trainer.run()
    end_time = time.time()

    logging.info("Total Training Time {}".format(end_time - start_time))
    if args.local_rank == 0:
        logging.info("{}:: Saving Final PT Model".format(args.local_rank))
        torch.save(trainer.network.state_dict(), os.path.join(args.output, "model-final.pt"))

    if not args.multi_gpu:
        logging.info("{}:: Saving TorchScript Model".format(args.local_rank))
        model_ts = torch.jit.script(trainer.network)
        torch.jit.save(model_ts, os.path.join(args.output, "model-final.ts"))

    if args.multi_gpu:
        dist.destroy_process_group()


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=23)
    parser.add_argument("--dimensions", type=int, default=2)

    parser.add_argument("-n", "--network", default="bunet", choices=["unet", "bunet"])
    parser.add_argument("-c", "--channels", type=int, default=32)
    parser.add_argument(
        "-i",
        "--input",
        default="/workspace/data/deepgrow/2D/MSD_Task09_Spleen/dataset.json",
    )
    parser.add_argument("-o", "--output", default="output")

    parser.add_argument("-g", "--use_gpu", type=strtobool, default="true")
    parser.add_argument("-a", "--amp", type=strtobool, default="false")

    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("-x", "--split", type=float, default=0.9)
    parser.add_argument("-t", "--limit", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("-r", "--resume", type=strtobool, default="false")
    parser.add_argument("-m", "--model_path", default="output/model.pt")
    parser.add_argument("--roi_size", default="[256, 256]")
    parser.add_argument("--model_size", default="[256, 256]")

    parser.add_argument("-f", "--val_freq", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-it", "--max_train_interactions", type=int, default=15)
    parser.add_argument("-iv", "--max_val_interactions", type=int, default=5)

    parser.add_argument("--save_interval", type=int, default=3)
    parser.add_argument("--image_interval", type=int, default=1)
    parser.add_argument("--multi_gpu", type=strtobool, default="false")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--export", type=strtobool, default="false")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    # Single GPU (it will also export)
    # python train.py
    #
    # Multi GPU (run export separate)
    # python -m torch.distributed.launch \
    #   --nproc_per_node=`nvidia-smi -L | wc -l` \
    #   --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1234 \
    #   -m train --multi_gpu true -e 100
    #
    # python train.py --export
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
