import logging
import os
import time
from argparse import ArgumentParser

import monai
import numpy as np
import torch
from monai.apps.pathology.data import PatchWSIDataset
from monai.data import DataLoader, load_decathlon_datalist
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.handlers.nvtx_handlers import RangeHandler
from monai.networks.nets import TorchVisionFCModel
from monai.optimizers import Novograd
from monai.transforms import (
    ActivationsD,
    AsDiscreteD,
    CastToTypeD,
    Compose,
    RandFlipD,
    RandRotate90D,
    RandZoomD,
    ScaleIntensityRangeD,
    ToNumpyD,
    TorchVisionD,
    ToTensorD,
)
from monai.transforms.nvtx import RandRangeD, RangeD, RangePopD, RangePushD
from monai.utils import first, set_determinism
from torch.optim import SGD, lr_scheduler

from ignite.metrics import Accuracy

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
        print(f'CUDA is being used with GPU ID(s): {os.environ["CUDA_VISIBLE_DEVICES"]}')
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
            RangePushD("Preprocessing"),
            RangeD(ToTensorD(keys="image"), "ToTensorD_1"),
            RangeD(
                TorchVisionD(
                    keys="image",
                    name="ColorJitter",
                    brightness=64.0 / 255.0,
                    contrast=0.75,
                    saturation=0.25,
                    hue=0.04,
                ),
                "ColorJitter",
            ),
            RangeD(ToNumpyD(keys="image")),
            RandRangeD(RandFlipD(keys="image", prob=0.5)),
            RandRangeD(RandRotate90D(keys="image", prob=0.5)),
            RangeD(CastToTypeD(keys="image", dtype=np.float32)),
            RandRangeD(RandZoomD(keys="image", prob=0.5, min_zoom=0.9, max_zoom=1.1)),
            RangeD(ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0)),
            RangeD(ToTensorD(keys=("image", "label")), "ToTensorD_2"),
            RangePopD(),
        ]
    )
    valid_preprocess = Compose(
        [
            CastToTypeD(keys="image", dtype=np.float32),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            ToTensorD(keys=("image", "label")),
        ]
    )
    # __________________________________________________________________________
    # Create MONAI dataset
    train_json_info_list = load_decathlon_datalist(
        data_list_file_path=cfg["dataset_json"],
        data_list_key="training",
        base_dir=cfg["data_root"],
    )
    valid_json_info_list = load_decathlon_datalist(
        data_list_file_path=cfg["dataset_json"],
        data_list_key="validation",
        base_dir=cfg["data_root"],
    )

    train_dataset = PatchWSIDataset(
        train_json_info_list,
        cfg["region_size"],
        cfg["grid_shape"],
        cfg["patch_size"],
        train_preprocess,
        image_reader_name="openslide" if cfg["use_openslide"] else "cuCIM",
    )
    valid_dataset = PatchWSIDataset(
        valid_json_info_list,
        cfg["region_size"],
        cfg["grid_shape"],
        cfg["patch_size"],
        valid_preprocess,
        image_reader_name="openslide" if cfg["use_openslide"] else "cuCIM",
    )

    # __________________________________________________________________________
    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=cfg["num_workers"],
        batch_size=cfg["batch_size"],
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        num_workers=cfg["num_workers"],
        batch_size=cfg["batch_size"],
        pin_memory=True,
    )

    # __________________________________________________________________________
    # Get sample batch and some info
    first_sample = first(train_dataloader)
    if first_sample is None:
        raise ValueError("Fist sample is None!")

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
    model = TorchVisionFCModel("resnet18", n_classes=1, use_conv=True, pretrained=cfg["pretrain"])
    model = model.to(device)

    # loss function
    loss_func = torch.nn.BCEWithLogitsLoss()
    loss_func = loss_func.to(device)

    # optimizer
    if cfg["novograd"]:
        optimizer = Novograd(model.parameters(), cfg["lr"])
    else:
        optimizer = SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

    # AMP scaler
    if cfg["amp"]:
        cfg["amp"] = True if monai.utils.get_torch_version_tuple() >= (1, 6) else False
    else:
        cfg["amp"] = False

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
    val_postprocessing = Compose(
        [
            ActivationsD(keys="pred", sigmoid=True),
            AsDiscreteD(keys="pred", threshold_values=True),
        ]
    )
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
        RangeHandler("Iteration"),
        RangeHandler("Batch"),
        LrScheduleHandler(lr_scheduler=scheduler, print_lr=True),
        CheckpointSaver(
            save_dir=cfg["logdir"],
            save_dict={"net": model, "opt": optimizer},
            save_interval=1,
            epoch_level=True,
        ),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        TensorBoardStatsHandler(
            log_dir=cfg["logdir"],
            tag_name="train_loss",
            output_transform=from_engine(["loss"], first=True),
        ),
    ]
    train_postprocessing = Compose(
        [
            RangePushD("Postprocessing"),
            RangeD(ActivationsD(keys="pred", sigmoid=True)),
            RangeD(AsDiscreteD(keys="pred", threshold_values=True)),
            RangePopD(),
        ]
    )

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
        "--dataset",
        type=str,
        default="../dataset_0.json",
        dest="dataset_json",
        help="path to dataset json file",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/workspace/data/medical/pathology/",
        dest="data_root",
        help="path to root folder of images containing training folder",
    )
    parser.add_argument("--logdir", type=str, default="./logs/", dest="logdir", help="log directory")

    parser.add_argument("--rs", type=int, default=256 * 3, dest="region_size", help="region size")
    parser.add_argument("--gs", type=int, default=3, dest="grid_shape", help="image grid shape (3x3)")
    parser.add_argument("--ps", type=int, default=224, dest="patch_size", help="patch size")
    parser.add_argument("--bs", type=int, default=64, dest="batch_size", help="batch size")
    parser.add_argument("--ep", type=int, default=10, dest="n_epochs", help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, dest="lr", help="initial learning rate")

    parser.add_argument("--openslide", action="store_true", dest="use_openslide", help="use OpenSlide")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")
    parser.add_argument(
        "--no-novograd",
        action="store_false",
        dest="novograd",
        help="deactivate novograd optimizer",
    )
    parser.add_argument(
        "--no-pretrain",
        action="store_false",
        dest="pretrain",
        help="deactivate Imagenet weights",
    )

    parser.add_argument("--cpu", type=int, default=0, dest="num_workers", help="number of workers")
    parser.add_argument("--gpu", type=str, default="0", dest="gpu", help="which gpu to use")

    args = parser.parse_args()
    config_dict = vars(args)
    print(config_dict)
    train(config_dict)


if __name__ == "__main__":
    main()
