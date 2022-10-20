import sys
sys.path.append('/workspace/Code/tutorials/pathology/hovernet/loss')
sys.path.append('/workspace/Code/tutorials/pathology/hovernet/transforms')
from functools import partial
import logging
import os
import time
from argparse import ArgumentParser
import torch
import numpy as np
import pandas as pd
from monai.data import DataLoader, Dataset
from monai.networks.nets import HoVerNet
from monai.engines import SupervisedEvaluator, SupervisedTrainer, PrepareBatchExtraInput
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
    CenterSpatialCropd,
    FillHoles,
    BoundingRect,
    ThresholdIntensity,
    RandFlipd,
    RandRotate90d,
    RandGaussianSmoothd,
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
from monai.utils import set_determinism, ensure_tuple, convert_to_tensor
from monai.utils.enums import HoVerNetBranch
from sklearn.model_selection import StratifiedShuffleSplit

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
)


set_determinism(seed=0)

def create_log_dir(cfg):
    timestamp = time.strftime("%y%m%d-%H%M%S")
    run_folder_name = (
        f"{timestamp}_hovernet_bs{cfg['batch_size']}_ep{cfg['n_epochs']}_lr{cfg['lr']}"
    )
    log_dir = os.path.join(cfg["logdir"], run_folder_name)
    print(f"Logs and model are saved at '{log_dir}'.")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def set_device(cfg):
    return torch.device(cfg['gpu'] if torch.cuda.is_available() else "cpu")

def prepare_datasets(data_dir):
    info = pd.read_csv(os.path.join(data_dir, "patch_info.csv"))
    file_names = np.squeeze(info.to_numpy()).tolist()

    img_sources = [v.split("-")[0] for v in file_names]
    img_sources = np.unique(img_sources)

    cohort_sources = [v.split("_")[0] for v in img_sources]
    _, cohort_sources = np.unique(cohort_sources, return_inverse=True)

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=0)

    split_generator = splitter.split(img_sources, cohort_sources)
    for train_indices, valid_indices in split_generator:
        train_cohorts = img_sources[train_indices]
        valid_cohorts = img_sources[valid_indices]
        if np.intersect1d(train_cohorts, valid_cohorts).size != 0:
            raise ValueError("Train and validation cohorts has an overlap.")
        train_names = [
            file_name for file_name in file_names for source in train_cohorts if source == file_name.split("-")[0]
        ]
        valid_names = [
            file_name for file_name in file_names for source in valid_cohorts if source == file_name.split("-")[0]
        ]
        train_names = np.unique(train_names)
        valid_names = np.unique(valid_names)
        print(f"Train: {len(train_names):04d} - Valid: {len(valid_names):04d}")
        if np.intersect1d(train_names, valid_names).size != 0:
            raise ValueError("Train and validation cohorts has an overlap.")

        train_indices = [file_names.index(v) for v in train_names]
        valid_indices = [file_names.index(v) for v in valid_names]

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

    train_data = [data[i] for i in train_indices]
    valid_data = [data[i] for i in valid_indices]

    return train_data, valid_data

def _from_engine(keys):
    keys = ensure_tuple(keys)

    def _wrapper(data):
        ret = [[i[k][HoVerNetBranch.NP.value] for i in data] for k in keys]
        return tuple(ret) if len(ret) > 1 else ret[0]

    return _wrapper

def post_process(output, return_binary=True, return_centroids=False, output_classes=None):
    pred = output["pred"]
    device = pred[HoVerNetBranch.NP.value].device
    if HoVerNetBranch.NC.value in pred.keys():
        type_pred = Activations(softmax=True)(pred[HoVerNetBranch.NC.value])
        type_pred = AsDiscrete(argmax=True)(pred[HoVerNetBranch.NC.value])

    post_trans_seg = Compose([
        GenerateWatershedMaskd(keys=HoVerNetBranch.NP.value, softmax=True),
        GenerateInstanceBorderd(keys='mask', hover_map_key=HoVerNetBranch.HV, kernel_size=3),
        GenerateDistanceMapd(keys='mask', border_key='border', smooth_fn="gaussian"),
        GenerateWatershedMarkersd(keys='mask', border_key='border', threshold=0.6, radius=2, postprocess_fn=FillHoles()),
        Watershedd(keys='dist', mask_key='mask', markers_key='markers')
    ])
    pred_inst_dict = post_trans_seg(pred)
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
        pred_inst = ThresholdIntensity(threshold=0)(pred_inst)
    output["pred"][HoVerNetBranch.NP.value] = pred_inst
    output["pred"]["inst_info_dict"] = inst_info_dict
    output["pred"]["pred_inst_dict"] = pred_inst_dict
    return output


class PrepareBatchExtraInput_v2():
    def __init__(self, extra_keys) -> None:
        self.prepare_batch = PrepareBatchExtraInput(extra_keys)

    def __call__(self, batchdata, device, non_blocking, **kwargs):
        image, label, extra_label, _ = self.prepare_batch(batchdata, device, non_blocking, **kwargs)
        all_label = {
            HoVerNetBranch.NP: label,
            HoVerNetBranch.NC: extra_label[0],
            HoVerNetBranch.HV: extra_label[1],
        }

        return image, all_label


def run(cfg):
    log_dir = create_log_dir(cfg)
    device = set_device(cfg)
    # --------------------------------------------------------------------------
    # Data Loading and Preprocessing
    # --------------------------------------------------------------------------
    # __________________________________________________________________________
    # __________________________________________________________________________
    # Build MONAI preprocessing
    train_data, valid_data = prepare_datasets(cfg["root"])
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
    # __________________________________________________________________________
    # Create MONAI dataset
    train_ds = Dataset(data=train_data, transform=train_transforms)
    valid_ds = Dataset(data=valid_data, transform=val_transforms)
    # __________________________________________________________________________
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())

    # --------------------------------------------------------------------------
    # Create Model, Loss, Optimizer, lr_scheduler
    # --------------------------------------------------------------------------
    # __________________________________________________________________________
    # initialize model
    model = HoVerNet(
        mode="fast",
        in_channels=3,
        out_classes=7,
        act=("relu", {"inplace": True}),
        norm="batch",
        dropout_prob=0.2,
    ).to(device)
    loss_function = HoVerNetLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step_size"])

    # --------------------------------------------
    # Ignite Trainer/Evaluator
    # --------------------------------------------
    # Evaluator
    val_handlers = [
        CheckpointSaver(save_dir=log_dir, save_dict={"net": model}, save_key_metric=True),
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: None),
    ]
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        prepare_batch=PrepareBatchExtraInput_v2(extra_keys=['label_type', 'hover_label_inst']),
        network=model,
        postprocessing=partial(post_process, return_binary=True, return_centroids=False, output_classes=None),
        key_val_metric={"val_dice": MeanDice(include_background=False, output_transform=_from_engine(keys=["pred", "label"]))},
        val_handlers=val_handlers,
        amp=cfg["amp"],
    )

    # Trainer
    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        CheckpointSaver(
            save_dir=cfg["logdir"], save_dict={"net": model, "opt": optimizer}, save_interval=1, epoch_level=True
        ),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        TensorBoardStatsHandler(
            log_dir=cfg["logdir"], tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
    ]
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=cfg["n_epochs"],
        train_data_loader=train_loader,
        prepare_batch=PrepareBatchExtraInput_v2(extra_keys=['label_type', 'hover_label_inst']),
        network=model,
        optimizer=optimizer,
        loss_function=loss_function,
        postprocessing=partial(post_process, return_binary=True, return_centroids=False, output_classes=None),
        key_train_metric={"train_dice": MeanDice(include_background=False, output_transform=_from_engine(keys=["pred", "label"]))},
        train_handlers=train_handlers,
        amp=cfg["amp"],
    )
    trainer.run()


def parse_arguments():
    parser = ArgumentParser(description="Tumor detection on whole slide pathology images.")
    parser.add_argument(
        "--root",
        type=str,
        default="/workspace/Data/Lizard/Prepared",
        help="root data dir",
    )
    parser.add_argument("--logdir", type=str, default="./logs/", dest="logdir", help="log directory")

    parser.add_argument("--bs", type=int, default=15, dest="batch_size", help="batch size")
    parser.add_argument("--ep", type=int, default=300, dest="n_epochs", help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, dest="lr", help="initial learning rate")
    parser.add_argument("--step", type=int, default=25, dest="step_size", help="period of learning rate decay")

    parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")

    parser.add_argument("--cpu", type=int, default=8, dest="num_workers", help="number of workers")
    parser.add_argument("--gpu", type=str, default="cuda:1", dest="gpu", help="which gpu to use")

    args = parser.parse_args()
    config_dict = vars(args)
    print(config_dict)
    return config_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = parse_arguments()
    run(cfg)
