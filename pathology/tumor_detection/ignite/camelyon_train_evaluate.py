import logging
import os
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from ignite.metrics import Accuracy
from torch.optim import SGD, lr_scheduler

import monai
from monai.data import DataLoader, PatchWSIDataset, CSVDataset
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.networks.nets import TorchVisionFCModel
from monai.optimizers import Novograd
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CastToTyped,
    Compose,
    GridSplitd,
    Lambdad,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    ScaleIntensityRanged,
    ToNumpyd,
    TorchVisiond,
    ToTensord,
)
from monai.utils import first, set_determinism

torch.backends.cudnn.enabled = True
set_determinism(seed=0, additional_settings=None)


def create_log_dir(cfg):
    timestamp = time.strftime("%y%m%d-%H%M%S")
    run_folder_name = (
        f"{timestamp}_resnet18_ps{cfg['patch_size']}_bs{cfg['batch_size']}_ep{cfg['n_epochs']}_lr{cfg['lr']}"
    )
    log_dir = os.path.join(cfg["logdir"], run_folder_name)
    print(f"Logs and model are saved at '{log_dir}'.")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def set_device(cfg):
    # Define the device, GPU or CPU
    gpus = [int(n.strip()) for n in cfg["gpu"].split(",")]
    gpus = set(gpus) & set(range(16))  # limit to 16-gpu machines
    if gpus and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(n) for n in gpus])
        device = torch.device("cuda")
        print(f'CUDA is being used with GPU Id(s): {os.environ["CUDA_VISIBLE_DEVICES"]}')
    else:
        device = torch.device("cpu")
        print("CPU only!")
    return device


def train(cfg):
    log_dir = create_log_dir(cfg)
    device = set_device(cfg)
    # --------------------------------------------------------------------------
    # Data Loading and Preprocessing
    # --------------------------------------------------------------------------
    # __________________________________________________________________________
    # Build MONAI preprocessing
    train_preprocess = Compose(
        [
            Lambdad(keys="label", func=lambda x: x.reshape((1, cfg["grid_shape"], cfg["grid_shape"]))),
            GridSplitd(
                keys=("image", "label"),
                grid=(cfg["grid_shape"], cfg["grid_shape"]),
                size={"image": cfg["patch_size"], "label": 1},
            ),
            ToTensord(keys=("image")),
            TorchVisiond(
                keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
            ),
            ToNumpyd(keys="image"),
            RandFlipd(keys="image", prob=0.5),
            RandRotate90d(keys="image", prob=0.5, max_k=3, spatial_axes=(-2, -1)),
            CastToTyped(keys="image", dtype=np.float32),
            RandZoomd(keys="image", prob=0.5, min_zoom=0.9, max_zoom=1.1),
            ScaleIntensityRanged(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            ToTensord(keys=("image", "label")),
        ]
    )
    valid_preprocess = Compose(
        [
            Lambdad(keys="label", func=lambda x: x.reshape((1, cfg["grid_shape"], cfg["grid_shape"]))),
            GridSplitd(
                keys=("image", "label"),
                grid=(cfg["grid_shape"], cfg["grid_shape"]),
                size={"image": cfg["patch_size"], "label": 1},
            ),
            CastToTyped(keys="image", dtype=np.float32),
            ScaleIntensityRanged(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            ToTensord(keys=("image", "label")),
        ]
    )
    # __________________________________________________________________________
    # Create MONAI dataset
    train_data_list = CSVDataset(
        cfg["train_file"],
        col_groups={"image": 0, "location": [2, 1], "label": [3, 6, 9, 4, 7, 10, 5, 8, 11]},
        kwargs_read_csv={"header": None},
        transform=Lambdad("image", lambda x: os.path.join(cfg["root"], "training/images", x + ".tif")),
    )
    train_dataset = PatchWSIDataset(
        data=train_data_list,
        patch_size=cfg["region_size"],
        patch_level=0,
        transform=train_preprocess,
        reader="openslide" if cfg["use_openslide"] else "cuCIM",
    )

    valid_data_list = CSVDataset(
        cfg["valid_file"],
        col_groups={"image": 0, "location": [2, 1], "label": [3, 6, 9, 4, 7, 10, 5, 8, 11]},
        kwargs_read_csv={"header": None},
        transform=Lambdad("image", lambda x: os.path.join(cfg["root"], "training/images", x + ".tif")),
    )
    valid_dataset = PatchWSIDataset(
        data=valid_data_list,
        patch_size=cfg["region_size"],
        patch_level=0,
        transform=valid_preprocess,
        reader="openslide" if cfg["use_openslide"] else "cuCIM",
    )

    # __________________________________________________________________________
    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset, num_workers=cfg["num_workers"], batch_size=cfg["batch_size"], pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, num_workers=cfg["num_workers"], batch_size=cfg["batch_size"], pin_memory=True
    )

    # Check first sample
    first_sample = first(train_dataloader)
    if first_sample is None:
        raise ValueError("First sample is None!")
    print("image: ")
    print("    shape", first_sample["image"].shape)
    print("    type: ", type(first_sample["image"]))
    print("    dtype: ", first_sample["image"].dtype)
    print("labels: ")
    print("    shape", first_sample["label"].shape)
    print("    type: ", type(first_sample["label"]))
    print("    dtype: ", first_sample["label"].dtype)
    print(f"batch size: {cfg['batch_size']}")
    print(f"train number of batches: {len(train_dataloader)}")
    print(f"valid number of batches: {len(valid_dataloader)}")

    # --------------------------------------------------------------------------
    # Deep Learning Classification Model
    # --------------------------------------------------------------------------
    # __________________________________________________________________________
    # initialize model
    model = TorchVisionFCModel("resnet18", num_classes=1, use_conv=True, pretrained=cfg["pretrain"])
    model = model.to(device)

    # loss function
    loss_func = torch.nn.BCEWithLogitsLoss()
    loss_func = loss_func.to(device)

    # optimizer
    if cfg["novograd"]:
        optimizer = Novograd(model.parameters(), cfg["lr"])
    else:
        optimizer = SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"])

    # --------------------------------------------
    # Ignite Trainer/Evaluator
    # --------------------------------------------
    # Evaluator
    val_handlers = [
        CheckpointSaver(save_dir=log_dir, save_dict={"net": model}, save_key_metric=True),
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: None),
    ]
    val_postprocessing = Compose([Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold=0.5)])
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=valid_dataloader,
        network=model,
        postprocessing=val_postprocessing,
        key_val_metric={"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        val_handlers=val_handlers,
        amp=cfg["amp"],
    )

    # Trainer
    train_handlers = [
        LrScheduleHandler(lr_scheduler=scheduler, print_lr=True),
        CheckpointSaver(
            save_dir=cfg["logdir"], save_dict={"net": model, "opt": optimizer}, save_interval=1, epoch_level=True
        ),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        TensorBoardStatsHandler(
            log_dir=cfg["logdir"], tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
    ]
    train_postprocessing = Compose([Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold=0.5)])

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=cfg["n_epochs"],
        train_data_loader=train_dataloader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_func,
        postprocessing=train_postprocessing,
        key_train_metric={"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        train_handlers=train_handlers,
        amp=cfg["amp"],
    )
    trainer.run()


def main():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(description="Tumor detection on whole slide pathology images.")
    parser.add_argument(
        "--root",
        type=str,
        default="/workspace/data/medical/pathology",
        help="path to image folder containing training/validation",
    )
    parser.add_argument("--train-file", type=str, default="training.csv", help="path to training data file")
    parser.add_argument("--valid-file", type=str, default="validation.csv", help="path to training data file")
    parser.add_argument("--logdir", type=str, default="./logs/", dest="logdir", help="log directory")

    parser.add_argument("--rs", type=int, default=256 * 3, dest="region_size", help="region size")
    parser.add_argument("--gs", type=int, default=3, dest="grid_shape", help="image grid shape e.g 3 means 3x3")
    parser.add_argument("--ps", type=int, default=224, dest="patch_size", help="patch size")
    parser.add_argument("--bs", type=int, default=64, dest="batch_size", help="batch size")
    parser.add_argument("--ep", type=int, default=10, dest="n_epochs", help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, dest="lr", help="initial learning rate")

    parser.add_argument("--openslide", action="store_true", dest="use_openslide", help="use OpenSlide")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")
    parser.add_argument("--no-novograd", action="store_false", dest="novograd", help="deactivate novograd optimizer")
    parser.add_argument("--no-pretrain", action="store_false", dest="pretrain", help="deactivate Imagenet weights")

    parser.add_argument("--cpu", type=int, default=8, dest="num_workers", help="number of workers")
    parser.add_argument("--gpu", type=str, default="0", dest="gpu", help="which gpu to use")

    args = parser.parse_args()
    config_dict = vars(args)
    print(config_dict)
    train(config_dict)


if __name__ == "__main__":
    main()
