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
import logging
import os
import sys
import time
import glob

import torch
import torch.distributed as dist
from monai.apps.deepedit.interaction import Interaction

from monai.apps.deepedit.transforms import (
    AddGuidanceSignalDeepEditd,
    AddRandomGuidanceDeepEditd,
    FindDiscrepancyRegionsDeepEditd,
    NormalizeLabelsInDatasetd,
    FindAllValidSlicesMissingLabelsd,
    AddInitialSeedPointMissingLabelsd,
    SplitPredsLabeld,
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
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet, UNETR
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    Resized,
    ScaleIntensityRanged,
    ToNumpyd,
    ToTensord,
)
from monai.utils import set_determinism


def get_network(network, labels, spatial_size):
    # Network
    if network == "unetr":
        network = UNETR(
            spatial_dims=3,
            in_channels=len(labels) + 1,
            out_channels=len(labels),
            img_size=spatial_size,
            feature_size=64,
            hidden_size=1536,
            mlp_dim=3072,
            num_heads=48,
            pos_embed="conv",
            norm_name="instance",
            res_block=True,
        )
    else:
        network = DynUNet(
            spatial_dims=3,
            in_channels=len(labels) + 1,
            out_channels=len(labels),
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )
    return network


def get_pre_transforms(labels, spatial_size):
    t = [
        LoadImaged(keys=("image", "label"), reader="ITKReader"),
        EnsureChannelFirstd(keys=("image", "label")),
        NormalizeLabelsInDatasetd(keys="label", label_names=labels),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # This transform may not work well for MR images
        ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
        RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
        RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
        RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
        Resized(keys=("image", "label"), spatial_size=spatial_size, mode=("area", "nearest")),
        # Transforms for click simulation
        FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
        AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
        AddGuidanceSignalDeepEditd(keys="image", guidance="guidance"),
        #
        ToTensord(keys=("image", "label")),
    ]

    return Compose(t)


def get_click_transforms():
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        ToNumpyd(keys=("image", "label", "pred")),
        # Transforms for click simulation
        FindDiscrepancyRegionsDeepEditd(keys="label", pred="pred", discrepancy="discrepancy"),
        AddRandomGuidanceDeepEditd(
            keys="NA",
            guidance="guidance",
            discrepancy="discrepancy",
            probability="probability",
        ),
        AddGuidanceSignalDeepEditd(keys="image", guidance="guidance"),
        #
        ToTensord(keys=("image", "label")),
    ]

    return Compose(t)


def get_post_transforms(labels):
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        # This transform is to check dice score per segment/label
        SplitPredsLabeld(keys="pred"),
    ]
    return Compose(t)


def get_loaders(args, pre_transforms):
    multi_gpu = args.multi_gpu
    local_rank = args.local_rank

    all_images = sorted(glob.glob(os.path.join(args.input, "imagesTr", "*.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(args.input, "labelsTr", "*.nii.gz")))
    datalist = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]

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

    train_datalist, val_datalist = partition_dataset(
        datalist,
        ratios=[args.split, (1 - args.split)],
        shuffle=True,
        seed=args.seed,
    )

    train_ds = PersistentDataset(train_datalist, pre_transforms, cache_dir=args.cache_dir)
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=2)
    logging.info("{}:: Total Records used for Training is: {}/{}".format(local_rank, len(train_ds), total_l))

    val_ds = PersistentDataset(val_datalist, pre_transforms, cache_dir=args.cache_dir)
    val_loader = DataLoader(val_ds, num_workers=2)
    logging.info("{}:: Total Records used for Validation is: {}/{}".format(local_rank, len(val_ds), total_l))

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

    pre_transforms = get_pre_transforms(args.labels, args.spatial_size)
    click_transforms = get_click_transforms()
    post_transform = get_post_transforms(args.labels)

    train_loader, val_loader = get_loaders(args, pre_transforms)

    # define training components
    network = get_network(args.network, args.labels, args.spatial_size).to(device)
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
            final_filename="pretrained_deepedit_" + args.network + ".pt",
        ),
    ]
    val_handlers = val_handlers if local_rank == 0 else None

    all_val_metrics = dict()
    all_val_metrics["val_mean_dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False
    )
    for key_label in args.labels:
        if key_label != "background":
            all_val_metrics[key_label + "_dice"] = MeanDice(
                output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
            )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_val,
            transforms=click_transforms,
            click_probability_key="probability",
            train=False,
            label_names=args.labels,
        ),
        inferer=SimpleInferer(),
        postprocessing=post_transform,
        key_val_metric=all_val_metrics,
        val_handlers=val_handlers,
    )

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
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

    all_train_metrics = dict()
    all_train_metrics["train_dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False
    )
    for key_label in args.labels:
        if key_label != "background":
            all_train_metrics[key_label + "_dice"] = MeanDice(
                output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
            )

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_train,
            transforms=click_transforms,
            click_probability_key="probability",
            train=True,
            label_names=args.labels,
        ),
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=SimpleInferer(),
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=all_train_metrics,
        train_handlers=train_handlers,
    )
    return trainer


def run(args):
    if args.local_rank == 0:
        for arg in vars(args):
            logging.info("USING:: {} = {}".format(arg, getattr(args, arg)))
        print("")

    if args.export:
        logging.info("{}:: Loading PT Model from: {}".format(args.local_rank, args.input))
        device = torch.device("cuda" if args.use_gpu else "cpu")
        network = get_network(args.network, args.labels, args.spatial_size).to(device)

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
        torch.save(
            trainer.network.state_dict(), os.path.join(args.output, "pretrained_deepedit_" + args.network + "-final.pt")
        )

    if not args.multi_gpu:
        logging.info("{}:: Saving TorchScript Model".format(args.local_rank))
        model_ts = torch.jit.script(trainer.network)
        torch.jit.save(model_ts, os.path.join(args.output, "pretrained_deepedit_" + args.network + "-final.ts"))

    if args.multi_gpu:
        dist.destroy_process_group()


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=36)

    parser.add_argument("-n", "--network", default="dynunet", choices=["dynunet", "unetr"])
    parser.add_argument(
        "-i",
        "--input",
        default="/Datasets/MSD_datasets/Task09_Spleen",
    )
    parser.add_argument("-o", "--output", default="output")

    parser.add_argument("-g", "--use_gpu", type=strtobool, default="true")
    parser.add_argument("-a", "--amp", type=strtobool, default="false")

    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-x", "--split", type=float, default=0.9)
    parser.add_argument("-t", "--limit", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("-r", "--resume", type=strtobool, default="false")

    parser.add_argument("-f", "--val_freq", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-it", "--max_train_interactions", type=int, default=15)
    parser.add_argument("-iv", "--max_val_interactions", type=int, default=5)

    parser.add_argument("-dpt", "--deepgrow_probability_train", type=float, default=0.4)
    parser.add_argument("-dpv", "--deepgrow_probability_val", type=float, default=1.0)

    parser.add_argument("--save_interval", type=int, default=3)
    parser.add_argument("--image_interval", type=int, default=1)
    parser.add_argument("--multi_gpu", type=strtobool, default="false")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--export", type=strtobool, default="false")

    args = parser.parse_args()
    args.spatial_size = [128, 128, 128]
    # For single label using one of the Medical Segmentation Decathlon
    args.labels = {"spleen": 1, "background": 0}

    # # For multiple label using the BTCV dataset (https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
    # # For this, remember to update accordingly the function 'get_loaders' in lines 151-152
    # args.labels = {
    #               "spleen": 1,
    #               "right kidney": 2,
    #               "left kidney": 3,
    #               "gallbladder": 4,
    #               "esophagus": 5,
    #               "liver": 6,
    #               "stomach": 7,
    #               "aorta": 8,
    #               "background": 0,
    #             }

    # Restoring previous model if resume flag is True
    args.model_filepath = args.output + "/net_key_metric=0.8566.pt"
    run(args)


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
