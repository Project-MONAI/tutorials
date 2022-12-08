import os
import glob
import time
import logging
import torch
import numpy as np
import torch.distributed as dist
from argparse import ArgumentParser
from monai.data import DataLoader, partition_dataset, CacheDataset
from monai.networks.nets import HoVerNet
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.transforms import (
    LoadImaged,
    TorchVisiond,
    Lambdad,
    Activationsd,
    OneOf,
    MedianSmoothd,
    AsDiscreted,
    Compose,
    CastToTyped,
    ComputeHoVerMapsd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandFlipd,
    RandAffined,
    RandGaussianSmoothd,
    CenterSpatialCropd,
)
from monai.handlers import (
    MeanDice,
    CheckpointSaver,
    LrScheduleHandler,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.utils import set_determinism
from monai.utils.enums import HoVerNetBranch
from monai.apps.pathology.handlers.utils import from_engine_hovernet
from monai.apps.pathology.engines.utils import PrepareBatchHoVerNet
from monai.apps.pathology.losses import HoVerNetLoss
from skimage import measure


def create_log_dir(cfg):
    timestamp = time.strftime("%y%m%d-%H%M")
    run_folder_name = (
        f"{timestamp}_hovernet_bs{cfg['batch_size']}_ep{cfg['n_epochs']}_lr{cfg['lr']}_seed{cfg['seed']}_stage{cfg['stage']}"
    )
    log_dir = os.path.join(cfg["logdir"], run_folder_name)
    print(f"Logs and model are saved at '{log_dir}'.")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def prepare_data(data_dir, phase):
    # prepare datalist
    images = sorted(
        glob.glob(os.path.join(data_dir, f"{phase}/*image.npy")))
    inst_maps = sorted(
        glob.glob(os.path.join(data_dir, f"{phase}/*inst_map.npy")))
    type_maps = sorted(
        glob.glob(os.path.join(data_dir, f"{phase}/*type_map.npy")))

    data_dicts = [
        {"image": _image, "label_inst": _inst_map, "label_type": _type_map}
        for _image, _inst_map, _type_map in zip(images, inst_maps, type_maps)
    ]

    return data_dicts


def get_loaders(cfg, train_transforms, val_transforms):
    multi_gpu = True if torch.cuda.device_count() > 1 else False

    train_data = prepare_data(cfg["root"], "Train")
    valid_data = prepare_data(cfg["root"], "Test")
    if multi_gpu:
        train_data = partition_dataset(
            data=train_data,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
            shuffle=True,
            seed=cfg["seed"],
        )[dist.get_rank()]
        valid_data = partition_dataset(
            data=valid_data,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
            shuffle=False,
            seed=cfg["seed"],
        )[dist.get_rank()]

    print("train_files:", len(train_data))
    print("val_files:", len(valid_data))

    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0, num_workers=4)
    valid_ds = CacheDataset(data=valid_data, transform=val_transforms, cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        valid_ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def create_model(cfg, device):
    # Each user is responsible for checking the content of models/datasets and the applicable licenses and
    # determining if suitable for the intended use.
    # The license for the below pre-trained model is different than MONAI license.
    # Please check the source where these weights are obtained from:
    # https://github.com/vqdang/hover_net#data-format
    pretrained_model = "https://drive.google.com/u/1/uc?id=1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5&export=download"
    if cfg["stage"] == 0:
        model = HoVerNet(
            mode=cfg["mode"],
            in_channels=3,
            out_classes=cfg["out_classes"],
            act=("relu", {"inplace": True}),
            norm="batch",
            pretrained_url=pretrained_model,
            freeze_encoder=True,
        ).to(device)
        print(f'stage{cfg["stage"]} start!')
    else:
        model = HoVerNet(
            mode=cfg["mode"],
            in_channels=3,
            out_classes=cfg["out_classes"],
            act=("relu", {"inplace": True}),
            norm="batch",
            pretrained_url=None,
            freeze_encoder=False,
        ).to(device)
        model.load_state_dict(torch.load(cfg["ckpt_path"])['net'])
        print(f'stage{cfg["stage"]}, success load weight!')

    return model


def run(log_dir, cfg):
    set_determinism(seed=cfg["seed"])

    if cfg["mode"].lower() == "original":
        cfg["patch_size"] = [270, 270]
        cfg["out_size"] = [80, 80]
    elif cfg["mode"].lower() == "fast":
        cfg["patch_size"] = [256, 256]
        cfg["out_size"] = [164, 164]

    multi_gpu = True if torch.cuda.device_count() > 1 else False
    if multi_gpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda:{}".format(dist.get_rank()))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if cfg["use_gpu"] else "cpu")

    # --------------------------------------------------------------------------
    # Data Loading and Preprocessing
    # --------------------------------------------------------------------------
    # __________________________________________________________________________
    # __________________________________________________________________________
    # Build MONAI preprocessing
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label_inst", "label_type"], image_only=True),
            Lambdad(keys="label_inst", func=lambda x: measure.label(x)),
            RandAffined(
                keys=["image", "label_inst", "label_type"],
                prob=1.0,
                rotate_range=((np.pi), 0),
                scale_range=((0.2), (0.2)),
                shear_range=((0.05), (0.05)),
                translate_range=((6), (6)),
                padding_mode="zeros",
                mode=("nearest"),
            ),
            CenterSpatialCropd(
                keys="image",
                roi_size=cfg["patch_size"],
            ),
            RandFlipd(keys=["image", "label_inst", "label_type"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label_inst", "label_type"], prob=0.5, spatial_axis=1),
            OneOf(transforms=[
                RandGaussianSmoothd(keys=["image"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
                MedianSmoothd(keys=["image"], radius=1),
                RandGaussianNoised(keys=["image"], prob=1.0, std=0.05)
            ]),
            CastToTyped(keys="image", dtype=np.uint8),
            TorchVisiond(
                keys=["image"],
                name="ColorJitter",
                brightness=(229 / 255.0, 281 / 255.0),
                contrast=(0.95, 1.10),
                saturation=(0.8, 1.2),
                hue=(-0.04, 0.04)
            ),
            AsDiscreted(keys=["label_type"], to_onehot=[5]),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            CastToTyped(keys="label_inst", dtype=torch.int),
            ComputeHoVerMapsd(keys="label_inst"),
            Lambdad(keys="label_inst", func=lambda x: x > 0, overwrite="label"),
            CenterSpatialCropd(
                keys=["label", "hover_label_inst", "label_inst", "label_type"],
                roi_size=cfg["out_size"],
            ),
            AsDiscreted(keys=["label"], to_onehot=2),
            CastToTyped(keys=["image", "label_inst", "label_type"], dtype=torch.float32),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label_inst", "label_type"], image_only=True),
            Lambdad(keys="label_inst", func=lambda x: measure.label(x)),
            CastToTyped(keys=["image", "label_inst"], dtype=torch.int),
            CenterSpatialCropd(
                keys="image",
                roi_size=cfg["patch_size"],
            ),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            ComputeHoVerMapsd(keys="label_inst"),
            Lambdad(keys="label_inst", func=lambda x: x > 0, overwrite="label"),
            CenterSpatialCropd(
                keys=["label", "hover_label_inst", "label_inst", "label_type"],
                roi_size=cfg["out_size"],
            ),
            CastToTyped(keys=["image", "label_inst", "label_type"], dtype=torch.float32),
        ]
    )

    # __________________________________________________________________________
    # Create MONAI DataLoaders
    train_loader, val_loader = get_loaders(cfg, train_transforms, val_transforms)

    # --------------------------------------------------------------------------
    # Create Model, Loss, Optimizer, lr_scheduler
    # --------------------------------------------------------------------------
    # __________________________________________________________________________
    # initialize model
    model = create_model(cfg, device)
    if multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.get_rank()], output_device=dist.get_rank()
        )
    loss_function = HoVerNetLoss(lambda_hv_mse=1.0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25)
    post_process_np = Compose([
        Activationsd(keys=HoVerNetBranch.NP.value, softmax=True),
        AsDiscreted(keys=HoVerNetBranch.NP.value, argmax=True),
    ])
    post_process = Lambdad(keys="pred", func=post_process_np)

    # --------------------------------------------
    # Ignite Trainer/Evaluator
    # --------------------------------------------
    # Evaluator
    val_handlers = [
        CheckpointSaver(
            save_dir=log_dir,
            save_dict={"net": model},
            save_key_metric=True,
        ),
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: None),
    ]
    if multi_gpu:
        val_handlers = val_handlers if dist.get_rank() == 0 else None
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        prepare_batch=PrepareBatchHoVerNet(extra_keys=['label_type', 'hover_label_inst']),
        network=model,
        postprocessing=post_process,
        key_val_metric={"val_dice": MeanDice(include_background=False, output_transform=from_engine_hovernet(keys=["pred", "label"], nested_key=HoVerNetBranch.NP.value))},
        val_handlers=val_handlers,
        amp=cfg["amp"],
    )

    # Trainer
    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=cfg["val_freq"], epoch_level=True),
        CheckpointSaver(
            save_dir=log_dir,
            save_dict={"net": model, "opt": optimizer},
            save_interval=cfg["save_interval"],
            epoch_level=True,
        ),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        TensorBoardStatsHandler(
            log_dir=log_dir, tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
    ]
    if multi_gpu:
        train_handlers = train_handlers if dist.get_rank() == 0 else train_handlers[:2]
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=cfg["n_epochs"],
        train_data_loader=train_loader,
        prepare_batch=PrepareBatchHoVerNet(extra_keys=['label_type', 'hover_label_inst']),
        network=model,
        optimizer=optimizer,
        loss_function=loss_function,
        postprocessing=post_process,
        key_train_metric={"train_dice": MeanDice(include_background=False, output_transform=from_engine_hovernet(keys=["pred", "label"], nested_key=HoVerNetBranch.NP.value))},
        train_handlers=train_handlers,
        amp=cfg["amp"],
    )
    trainer.run()

    if multi_gpu:
        dist.destroy_process_group()


def main():
    parser = ArgumentParser(description="Tumor detection on whole slide pathology images.")
    parser.add_argument(
        "--root",
        type=str,
        default="/workspace/Data/CoNSeP/Prepared",
        help="root data dir",
    )
    parser.add_argument("--logdir", type=str, default="./logs/", dest="logdir", help="log directory")
    parser.add_argument("-s", "--seed", type=int, default=24)

    parser.add_argument("--bs", type=int, default=16, dest="batch_size", help="batch size")
    parser.add_argument("--ep", type=int, default=3, dest="n_epochs", help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, dest="lr", help="initial learning rate")
    parser.add_argument("--step", type=int, default=25, dest="step_size", help="period of learning rate decay")
    parser.add_argument("-f", "--val_freq", type=int, default=1, help="validation frequence")
    parser.add_argument("--stage", type=int, default=0, dest="stage", help="training stage")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")
    parser.add_argument("--classes", type=int, default=5, dest="out_classes", help="output classes")
    parser.add_argument("--mode", type=str, default="original", help="choose either `original` or `fast`")

    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--cpu", type=int, default=8, dest="num_workers", help="number of workers")
    parser.add_argument("--no-gpu", action="store_false", dest="use_gpu", help="deactivate use of gpu")
    parser.add_argument("--ckpt", type=str, dest="ckpt_path", help="checkpoint path")

    args = parser.parse_args()
    cfg = vars(args)
    print(cfg)

    logging.basicConfig(level=logging.INFO)
    log_dir = create_log_dir(cfg)
    run(log_dir, cfg)


if __name__ == "__main__":
    main()
