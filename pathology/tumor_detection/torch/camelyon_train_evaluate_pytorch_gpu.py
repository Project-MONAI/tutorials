import json
import logging
import os
import time
from argparse import ArgumentParser
from shutil import copyfile

import monai
import numpy as np
from monai.data import CSVDataset, DataLoader, PatchWSIDataset
from monai.networks.nets import TorchVisionFCModel
from monai.optimizers import Novograd
from monai.transforms import (
    Activations,
    AsDiscrete,
    CastToType,
    CastToTyped,
    Compose,
    CuCIM,
    GridSplitd,
    Lambdad,
    RandCuCIM,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    ScaleIntensityRanged,
    ToCupy,
    ToNumpyd,
    TorchVisiond,
    ToTensor,
    ToTensord,
)
from monai.utils import first, set_determinism

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter


def create_log_dir(cfg):
    timestamp = time.strftime("%y%m%d-%H%M%S")
    run_folder_name = (
        f"run_{cfg['backend']}_bs{cfg['batch_size']}_cpu{cfg['num_workers']}_gpu{cfg['gpu']}"
        f"_pin{int(cfg['pin'])}_amp{int(cfg['amp'])}_{timestamp}"
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


def training(
    summary,
    model,
    loss_fn,
    optimizer,
    scaler,
    amp,
    dataloader,
    pre_process,
    post_process,
    device,
    writer: SummaryWriter,
    print_step,
):
    summary["epoch"] += 1

    model.train()

    n_steps = len(dataloader)
    iter_data = iter(dataloader)

    for step in range(n_steps):
        summary["step"] += 1

        batch = next(iter_data)
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        if pre_process is not None:
            x = pre_process(x)

        with autocast(enabled=amp):
            output = model(x)
            loss = loss_fn(output, y)

        optimizer.zero_grad()

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pred = post_process(output)

        acc_data = (pred == y).float().mean().item()
        loss_data = loss.item()

        writer.add_scalar("train/loss", loss_data, summary["step"])
        writer.add_scalar("train/accuracy", acc_data, summary["step"])

        if step % print_step == 0:
            logging.info(
                f"[Training] Epoch: {summary['epoch']}/{summary['n_epochs']}, "
                f"Step: {step + 1}/{n_steps} -- "
                f"train_loss: {loss_data:.5f}, train_acc: {acc_data:.3f}"
            )

    return summary


def validation(model, loss_fn, amp, dataloader, pre_process, post_process, device, print_step):
    model.eval()

    n_steps = len(dataloader)
    iter_data = iter(dataloader)
    total_acc = 0
    total_loss = 0
    total_n_samples = 0
    for step in range(n_steps):
        batch = next(iter_data)
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        if pre_process is not None:
            x = pre_process(x)

        with autocast(enabled=amp):
            output = model(x)
            loss = loss_fn(output, y)

        pred = post_process(output)

        acc_data = (pred == y).float().sum().item()
        loss_data = loss.item()
        n_samples = y.shape[0]

        total_acc += acc_data
        total_loss += loss_data * n_samples
        total_n_samples += n_samples

        if step % print_step == 0:
            logging.info(
                f"[Validation] "
                f"Step : {step + 1}/{n_steps} -- "
                f"valid_loss : {loss_data:.3f}, valid_acc : {acc_data / n_samples:.2f}"
            )
    return (total_loss / total_n_samples, total_acc / total_n_samples)


def main(cfg):
    # -------------------------------------------------------------------------
    # Configs
    # -------------------------------------------------------------------------
    # Create log/model dir
    log_dir = create_log_dir(cfg)

    # Set the logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)2s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_name = os.path.join(log_dir, "logs.txt")
    logger = logging.getLogger()
    fh = logging.FileHandler(log_name)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Set TensorBoard summary writer
    writer = SummaryWriter(log_dir)

    # Save configs
    logging.info(json.dumps(cfg))
    with open(os.path.join(log_dir, "config.json"), "w") as fp:
        json.dump(cfg, fp, indent=4)

    # Set device cuda/cpu
    device = set_device(cfg)

    # Set cudnn benchmark/deterministic
    if cfg["benchmark"]:
        torch.backends.cudnn.benchmark = True
    else:
        set_determinism(seed=0)
    # -------------------------------------------------------------------------
    # Transforms and Datasets
    # -------------------------------------------------------------------------
    # Pre-processing
    preprocess_cpu_train = None
    preprocess_gpu_train = None
    preprocess_cpu_valid = None
    preprocess_gpu_valid = None
    if cfg["backend"] == "cucim":
        preprocess_cpu_train = Compose(
            [
                Lambdad(keys="label", func=lambda x: x.reshape((1, cfg["grid_shape"], cfg["grid_shape"]))),
                GridSplitd(
                    keys=("image", "label"),
                    grid=(cfg["grid_shape"], cfg["grid_shape"]),
                    size={"image": cfg["patch_size"], "label": 1},
                ),
                ToTensord(keys="label"),
            ]
        )
        preprocess_gpu_train = Compose(
            [
                ToCupy(),
                RandCuCIM(
                    name="rand_color_jitter",
                    prob=1.0,
                    brightness=64.0 / 255.0,
                    contrast=0.75,
                    saturation=0.25,
                    hue=0.04,
                ),
                RandCuCIM(name="rand_image_flip", prob=cfg["prob"], spatial_axis=-1),
                RandCuCIM(name="rand_image_rotate_90", prob=cfg["prob"], max_k=3, spatial_axis=(-2, -1)),
                CastToType(dtype=np.float32),
                RandCuCIM(name="rand_zoom", prob=cfg["prob"], min_zoom=0.9, max_zoom=1.1),
                CuCIM(name="scale_intensity_range", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
                ToTensor(device=device),
            ]
        )
        preprocess_cpu_valid = Compose(
            [
                Lambdad(keys="label", func=lambda x: x.reshape((1, cfg["grid_shape"], cfg["grid_shape"]))),
                GridSplitd(
                    keys=("image", "label"),
                    grid=(cfg["grid_shape"], cfg["grid_shape"]),
                    size={"image": cfg["patch_size"], "label": 1},
                ),
                ToTensord(keys="label"),
            ]
        )
        preprocess_gpu_valid = Compose(
            [
                ToCupy(dtype=np.float32),
                CuCIM(name="scale_intensity_range", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
                ToTensor(device=device),
            ]
        )
    elif cfg["backend"] == "numpy":
        preprocess_cpu_train = Compose(
            [
                Lambdad(keys="label", func=lambda x: x.reshape((1, cfg["grid_shape"], cfg["grid_shape"]))),
                GridSplitd(
                    keys=("image", "label"),
                    grid=(cfg["grid_shape"], cfg["grid_shape"]),
                    size={"image": cfg["patch_size"], "label": 1},
                ),
                ToTensord(keys=("image", "label")),
                TorchVisiond(
                    keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
                ),
                ToNumpyd(keys="image"),
                RandFlipd(keys="image", prob=cfg["prob"], spatial_axis=-1),
                RandRotate90d(keys="image", prob=cfg["prob"]),
                CastToTyped(keys="image", dtype=np.float32),
                RandZoomd(keys="image", prob=cfg["prob"], min_zoom=0.9, max_zoom=1.1),
                ScaleIntensityRanged(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
                ToTensord(keys="image"),
            ]
        )
        preprocess_cpu_valid = Compose(
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
    else:
        raise ValueError(f"Backend should be either numpy or cucim! ['{cfg['backend']}' is provided.]")

    # Post-processing
    postprocess = Compose(
        [
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
        ]
    )

    # Create train dataset and dataloader
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
        transform=preprocess_cpu_train,
        reader="openslide" if cfg["use_openslide"] else "cuCIM",
    )
    train_dataloader = DataLoader(
        train_dataset, num_workers=cfg["num_workers"], batch_size=cfg["batch_size"], pin_memory=cfg["pin"]
    )

    # Create validation dataset and dataloader
    if not cfg["no_validate"]:
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
            transform=preprocess_cpu_valid,
            reader="openslide" if cfg["use_openslide"] else "cuCIM",
        )
        valid_dataloader = DataLoader(
            valid_dataset, num_workers=cfg["num_workers"], batch_size=cfg["batch_size"], pin_memory=cfg["pin"]
        )

    # Get sample batch and some info
    first_sample = first(train_dataloader)
    if first_sample is None:
        raise ValueError("First sample is None!")
    for d in ["image", "label"]:
        logging.info(
            f"[{d}] \n"
            f"  {d} shape: {first_sample[d].shape}\n"
            f"  {d} type:  {type(first_sample[d])}\n"
            f"  {d} dtype: {first_sample[d].dtype}"
        )
    logging.info(f"Batch size: {cfg['batch_size']}")
    logging.info(f"[Training] number of batches: {len(train_dataloader)}")
    if not cfg["no_validate"]:
        logging.info(f"[Validation] number of batches: {len(valid_dataloader)}")
    # -------------------------------------------------------------------------
    # Deep Learning Model and Configurations
    # -------------------------------------------------------------------------
    # Initialize model
    model = TorchVisionFCModel("resnet18", num_classes=1, use_conv=True, pretrained=cfg["pretrain"])
    model = model.to(device)

    # Loss function
    loss_func = torch.nn.BCEWithLogitsLoss()
    loss_func = loss_func.to(device)

    # Optimizer
    if cfg["novograd"] is True:
        optimizer = Novograd(model.parameters(), lr=cfg["lr"])
    else:
        optimizer = SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

    # AMP scaler
    if cfg["amp"] is True:
        scaler = GradScaler()
    else:
        scaler = None

    # Learning rate scheduler
    if cfg["cos"] is True:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"])
    else:
        scheduler = None

    # -------------------------------------------------------------------------
    # Training/Evaluating
    # -------------------------------------------------------------------------
    train_counter = {"n_epochs": cfg["n_epochs"], "epoch": 0, "step": 0}

    total_valid_time, total_train_time = 0.0, 0.0
    t_start = time.perf_counter()
    if cfg["no_validate"]:
        metric_summary = {}
    else:
        metric_summary = {"loss": np.Inf, "accuracy": 0, "best_epoch": 1}
    # Training/Validation Loop
    for _ in range(cfg["n_epochs"]):
        t_epoch = time.perf_counter()
        logging.info(f"[Training] learning rate: {optimizer.param_groups[0]['lr']}")

        # Training
        train_counter = training(
            train_counter,
            model,
            loss_func,
            optimizer,
            scaler,
            cfg["amp"],
            train_dataloader,
            preprocess_gpu_train,
            postprocess,
            device,
            writer,
            cfg["print_step"],
        )
        if scheduler is not None:
            scheduler.step()
        if not cfg["no_save"]:
            torch.save(model.state_dict(), os.path.join(log_dir, f"model_epoch_{train_counter['epoch']}.pt"))
        t_train = time.perf_counter()
        train_time = t_train - t_epoch
        total_train_time += train_time

        # Validation
        if cfg["no_validate"]:
            logging.info(f"[Epoch: {train_counter['epoch']}/{cfg['n_epochs']}] Train time: {train_time:.1f}s")
        else:
            valid_loss, valid_acc = validation(
                model,
                loss_func,
                cfg["amp"],
                valid_dataloader,
                preprocess_gpu_valid,
                postprocess,
                device,
                cfg["print_step"],
            )
            t_valid = time.perf_counter()
            valid_time = t_valid - t_train
            total_valid_time += valid_time
            if valid_loss < metric_summary["loss"]:
                metric_summary["loss"] = min(valid_loss, metric_summary["loss"])
                metric_summary["accuracy"] = max(valid_acc, metric_summary["accuracy"])
                metric_summary["best_epoch"] = train_counter["epoch"]
            writer.add_scalar("valid/loss", valid_loss, train_counter["epoch"])
            writer.add_scalar("valid/accuracy", valid_acc, train_counter["epoch"])

            logging.info(
                f"[Epoch: {train_counter['epoch']}/{cfg['n_epochs']}] loss: {valid_loss:.3f}, accuracy: {valid_acc:.3f}, "
                f"time: {t_valid - t_epoch:.1f}s (train: {train_time:.1f}s, valid: {valid_time:.1f}s)"
            )
        writer.flush()
    t_end = time.perf_counter()

    # Save final metrics
    metric_summary["train_time_per_epoch"] = total_train_time / cfg["n_epochs"]
    metric_summary["total_time"] = t_end - t_start
    writer.add_hparams(hparam_dict=cfg, metric_dict=metric_summary, run_name=log_dir)
    writer.close()
    logging.info(f"Metric Summary: {metric_summary}")

    # Save the best and final model
    if not cfg["no_validate"] and not cfg["no_save"]:
        copyfile(
            os.path.join(log_dir, f"model_epoch_{metric_summary['best_epoch']}.pt"),
            os.path.join(log_dir, "model_best.pt"),
        )
        copyfile(
            os.path.join(log_dir, f"model_epoch_{cfg['n_epochs']}.pt"),
            os.path.join(log_dir, "model_final.pt"),
        )

    # Final prints
    logging.info(
        f"[Completed] {train_counter['epoch']} epochs -- time: {t_end - t_start:.1f}s "
        f"(training: {total_train_time:.1f}s, validation: {total_valid_time:.1f}s)",
    )
    logging.info(f"Logs and model was saved at: {log_dir}")


def parse_arguments():
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
    parser.add_argument("--ep", type=int, default=4, dest="n_epochs", help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, dest="lr", help="initial learning rate")
    parser.add_argument("--pr", type=float, default=10, dest="print_step", help="print info each number of steps")
    parser.add_argument("--prob", type=float, default=0.5, help="probability for random transforms")

    parser.add_argument("--openslide", action="store_true", dest="use_openslide", help="use OpenSlide")
    parser.add_argument("--pin", action="store_true", help="pin memory for dataloader")
    parser.add_argument("--amp", action="store_true", help="activate amp")
    parser.add_argument("--novograd", action="store_true", help="activate novograd optimizer")
    parser.add_argument("--cos", action="store_true", help="activate cosine annealing")
    parser.add_argument("--pretrain", action="store_true", help="activate Imagenet weights")
    parser.add_argument("--benchmark", action="store_true", help="activate Imagenet weights")

    parser.add_argument("--no-save", action="store_true", help="save model at each epoch")
    parser.add_argument("--no-validate", action="store_true", help="use optimized parameters")
    parser.add_argument("--baseline", action="store_true", help="use baseline parameters")
    parser.add_argument("--optimized", action="store_true", help="use optimized parameters")
    parser.add_argument("-b", "--backend", type=str, dest="backend", help="backend for transforms")

    parser.add_argument("--cpu", type=int, default=8, dest="num_workers", help="number of workers")
    parser.add_argument("--gpu", type=str, default="0", dest="gpu", help="which gpu to use")

    args = parser.parse_args()
    config_dict = vars(args)

    if config_dict["optimized"] and config_dict["baseline"]:
        raise ValueError("Either --optimized or --baseline should be set!")
    if config_dict["optimized"] is True:
        config_dict["benchmark"] = True
        config_dict["novograd"] = True
        config_dict["pretrain"] = True
        config_dict["cos"] = True
        config_dict["pin"] = True
        config_dict["amp"] = True
        if config_dict["backend"] is None:
            config_dict["backend"] = "cucim"

    if config_dict["baseline"] is True:
        config_dict["benchmark"] = False
        config_dict["novograd"] = False
        config_dict["pretrain"] = True
        config_dict["cos"] = False
        config_dict["pin"] = False
        config_dict["amp"] = False

    if config_dict["backend"] is None:
        config_dict["backend"] = "numpy"
    config_dict["backend"] = config_dict["backend"].lower()

    return config_dict


if __name__ == "__main__":
    cfg = parse_arguments()
    main(cfg)
