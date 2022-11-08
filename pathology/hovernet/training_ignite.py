import sys

from functools import partial
import logging
import os
import glob
import time
import torch
import numpy as np
import scipy.io as sio
import torch.distributed as dist
from argparse import ArgumentParser
from monai.data import DataLoader, decollate_batch, CacheDataset, GridPatchDataset, PatchIterd, ShuffleBuffer, partition_dataset
from monai.networks.nets import HoVerNet
from monai.engines import SupervisedEvaluator, SupervisedTrainer, PrepareBatchExtraInput
from monai.transforms import (
    LoadImaged,
    Transposed,
    Activations,
    AsDiscrete,
    AsDiscreted,
    Compose,
    ScaleIntensityRanged,
    CastToTyped,
    Lambdad,
    ToMetaTensord,
    ComputeHoVerMapsd,
    CenterSpatialCropd,
    FillHoles,
    BoundingRect,
    GaussianSmooth,
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
    TensorBoardImageHandler,
    ValidationHandler,
    from_engine,
)
from monai.utils import set_determinism, ensure_tuple, convert_to_tensor
from monai.utils.enums import HoVerNetBranch

from loss import HoVerNetLoss
from net import HoVerNet
from transforms import (
    GenerateWatershedMaskd,
    GenerateInstanceBorderd,
    GenerateDistanceMapd,
    GenerateWatershedMarkersd,
    Watershedd,
    GenerateInstanceContour,
    GenerateInstanceCentroid,
    GenerateInstanceType,
    ComputeHoVerMapsd,
)


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

def prepare_data(data_dir):
    train_data_dir = os.path.join(data_dir, 'Train')

    train_images = sorted(
        glob.glob(os.path.join(train_data_dir, "Images/*.png")))
    train_labels = sorted(
        glob.glob(os.path.join(train_data_dir, "Labels/*.mat")))

    labels, inst_maps, type_maps = [], [], []
    for label in train_labels:
        label_data = sio.loadmat(label)
        inst_maps.append(label_data['inst_map'][None].astype(int))
        type_maps.append(label_data['type_map'][None])
        labels.append(np.array(label_data['inst_map'][None] > 0, dtype=int))
        
    data_dicts = [
        {"image": _image, "label": _label, "label_inst": _inst_map, "label_type": _type_map}
        for _image, _label, _inst_map, _type_map in zip(train_images, labels, inst_maps, type_maps)
    ]
    train_data, valid_data = data_dicts[:5], data_dicts[-5:]

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
        type_pred = AsDiscrete(argmax=True)(type_pred)

    post_trans_seg = Compose([
        GenerateWatershedMaskd(keys=HoVerNetBranch.NP.value, softmax=True),
        GenerateInstanceBorderd(keys='mask', hover_map_key=HoVerNetBranch.HV, kernel_size=3),
        GenerateDistanceMapd(keys='mask', border_key='border', smooth_fn=GaussianSmooth()),
        GenerateWatershedMarkersd(keys='mask', border_key='border', threshold=0.7, radius=2, postprocess_fn=FillHoles()),
        Watershedd(keys='dist', mask_key='mask', markers_key='markers')
    ])
    pred_inst_dict = post_trans_seg(pred)
    pred_inst = pred_inst_dict['dist']

    inst_id_list = np.unique(pred_inst)[1:]  # exclude background

    inst_info_dict = None
    if return_centroids:
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            inst_bbox = BoundingRect()(inst_map)
            inst_map = inst_map[:, inst_bbox[0][0]: inst_bbox[0][1], inst_bbox[0][2]: inst_bbox[0][3]]
            offset = [inst_bbox[0][2], inst_bbox[0][0]]
            inst_contour = GenerateInstanceContour()(inst_map, offset)
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


def get_loaders(cfg, train_transforms, val_transforms):
    multi_gpu = True if torch.cuda.device_count() > 1 else False

    train_data, valid_data = prepare_data(cfg["root"])
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

    patch_iter = PatchIterd(
        keys=["image", "label", "label_inst", "label_type", "hover_label_inst"], 
        patch_size=(256, 256), 
        start_pos=(0, 0),
        mode="reflect"
    )
    cropper = Compose([
        ToMetaTensord(keys=["image", "label", "label_inst", "label_type", "hover_label_inst"]),
        CenterSpatialCropd(keys=["label", "label_inst", "label_type", "hover_label_inst"], roi_size=(164,164)),
        ])

    _train_ds = CacheDataset(data=train_data, transform=train_transforms,
                        cache_rate=1.0, num_workers=4)
    _valid_ds = CacheDataset(data=valid_data, transform=val_transforms,
                        cache_rate=1.0, num_workers=4)
    train_ds = GridPatchDataset(
        data=_train_ds,
        patch_iter=patch_iter,
        transform=cropper,
        with_coordinates=False)
    valid_ds = GridPatchDataset(
        data=_valid_ds,
        patch_iter=patch_iter,
        transform=cropper,
        with_coordinates=False)
    shuffle_ds = ShuffleBuffer(train_ds, buffer_size=30, seed=cfg.seed)
    train_loader = DataLoader(shuffle_ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())

    return train_loader, val_loader

def run(cfg):
    log_dir = create_log_dir(cfg)
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
            LoadImaged(keys="image", image_only=True),
            Transposed(keys="image", indices=[2, 1, 0]), # make channel first
            Lambdad(keys="image", func=lambda x: x[:3, ...]),
            ComputeHoVerMapsd(keys="label_inst"),
            CastToTyped(keys=["image", "label_inst", "label_type", "hover_label_inst"], dtype=torch.float32),
            AsDiscreted(keys=["label", "label_type"], to_onehot=[2, 8]),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            RandFlipd(keys=["image", "label", "label_inst", "label_type", "hover_label_inst"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label", "label_inst", "label_type", "hover_label_inst"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "label", "label_inst", "label_type", "hover_label_inst"], prob=0.5, max_k=1),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.5,1.15), sigma_y=(0.5,1.15), prob=0.5),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys="image", image_only=True),
            Transposed(keys="image", indices=[2, 1, 0]), # make channel first
            Lambdad(keys="image", func=lambda x: x[:3, ...]),
            ComputeHoVerMapsd(keys="label_inst"),
            CastToTyped(keys=["image", "label_inst", "label_type", "hover_label_inst"], dtype=torch.float32),
            AsDiscreted(keys=["label", "label_type"], to_onehot=[2, 8]),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
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
    model = HoVerNet(
        mode="fast",
        in_channels=3,
        out_classes=8,
        act=("relu", {"inplace": True}),
        norm="batch",
        pretrained=True,
    ).to(device)
    if cfg["resume_checkpoint"]:
        model.load_state_dict(torch.load(cfg["checkpoint_dir"]))
        print('resume training from a given checkpoint...')
    else:
        model.freeze_encoder()
    if multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.get_rank()], output_device=dist.get_rank()
        )
    loss_function = HoVerNetLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step_size"])

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

    if multi_gpu:
        dist.destroy_process_group()


def main():
    parser = ArgumentParser(description="Tumor detection on whole slide pathology images.")
    parser.add_argument(
        "--root",
        type=str,
        default="/consep",
        help="root data dir",
    )
    parser.add_argument("--logdir", type=str, default="./logs/", dest="logdir", help="log directory")
    parser.add_argument("-s", "--seed", type=int, default=23)

    parser.add_argument("--bs", type=int, default=8, dest="batch_size", help="batch size")
    parser.add_argument("--ep", type=int, default=100, dest="n_epochs", help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, dest="lr", help="initial learning rate")
    parser.add_argument("--step", type=int, default=25, dest="step_size", help="period of learning rate decay")
    parser.add_argument("-f", "--val_freq", type=int, default=1, help="validation frequence")
    parser.add_argument(
        "--resume_checkpoint",
        default=False,
        type=bool,
        help="if True, training statrts from a model checkpoint"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        help="model checkpoint path to resume training from"
    )

    parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--cpu", type=int, default=8, dest="num_workers", help="number of workers")
    parser.add_argument("--use_gpu", type=bool, default=True, dest="use_gpu", help="whether to use gpu")
    parser.add_argument("--ngc", action="store_true", dest="ngc", help="use ngc")

    args = parser.parse_args()
    cfg = vars(args)
    print(cfg)
    set_determinism(seed=cfg["seed"])

    if cfg["ngc"]:
        sys.path.append('/workspace/pathology/lizard/transforms')
        sys.path.append('/workspace/pathology/lizard/loss')
        sys.path.append('/workspace/pathology/lizard/net')
    else:
        sys.path.append('/workspace/Code/tutorials/pathology/hovernet/transforms')
        sys.path.append('/workspace/Code/tutorials/pathology/hovernet/loss')
        sys.path.append('/workspace/Code/tutorials/pathology/hovernet/net')


    logging.basicConfig(level=logging.INFO)
    run(cfg)

    # export CUDA_VISIBLE_DIVICE=0; python training_ignite.py --root /Lizard
if __name__ == "__main__":
    main()