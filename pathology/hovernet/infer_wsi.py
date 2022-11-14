import sys

from functools import partial
import logging
import os
import time
from argparse import ArgumentParser
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from monai.data import DataLoader, MaskedPatchWSIDataset
from monai.networks.nets import HoVerNet
from monai.engines import SupervisedEvaluator
from monai.apps.pathology.transforms import (
    GenerateWatershedMaskd,
    GenerateInstanceBorderd,
    GenerateDistanceMapd,
    GenerateWatershedMarkersd,
    Watershedd,
    GenerateInstanceContour,
    GenerateInstanceCentroid,
    GenerateInstanceType,
)
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
    GaussianSmooth,
)
from monai.handlers import (
    MeanDice,
    StatsHandler,
    TensorBoardStatsHandler,
    from_engine,
)
from monai.utils import convert_to_tensor, first, HoVerNetBranch


def create_output_dir(cfg):
    timestamp = time.strftime("%y%m%d-%H%M%S")
    run_folder_name = f"{timestamp}_inference_hovernet_ps{cfg['patch_size']}"
    log_dir = os.path.join(cfg["logdir"], run_folder_name)
    print(f"Logs and outputs are saved at '{log_dir}'.")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def post_process(output, return_binary=True, return_centroids=False, output_classes=None):
    pred = output["pred"]
    device = pred[HoVerNetBranch.NP.value].device
    if HoVerNetBranch.NC.value in pred.keys():
        type_pred = Activations(softmax=True)(pred[HoVerNetBranch.NC.value])
        type_pred = AsDiscrete(argmax=True)(type_pred)

    post_trans_seg = Compose(
        [
            GenerateWatershedMaskd(keys=HoVerNetBranch.NP.value, softmax=True),
            GenerateInstanceBorderd(keys="mask", hover_map_key=HoVerNetBranch.HV, kernel_size=3),
            GenerateDistanceMapd(keys="mask", border_key="border", smooth_fn=GaussianSmooth()),
            GenerateWatershedMarkersd(
                keys="mask", border_key="border", threshold=0.7, radius=2, postprocess_fn=FillHoles()
            ),
            Watershedd(keys="dist", mask_key="mask", markers_key="markers"),
        ]
    )
    pred_inst_dict = post_trans_seg(pred)
    pred_inst = pred_inst_dict["dist"]

    inst_id_list = np.unique(pred_inst)[1:]  # exclude background

    inst_info_dict = None
    if return_centroids:
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            inst_bbox = BoundingRect()(inst_map)
            inst_map = inst_map[:, inst_bbox[0][0] : inst_bbox[0][1], inst_bbox[0][2] : inst_bbox[0][3]]
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
                bbox=inst_info_dict[inst_id]["bounding_box"],
                type_pred=type_pred,
                seg_pred=pred_inst,
                instance_id=inst_id,
            )
            inst_info_dict[inst_id]["type"] = inst_type
            inst_info_dict[inst_id]["type_probability"] = type_prob

    pred_inst = convert_to_tensor(pred_inst, device=device)
    if return_binary:
        pred_inst[pred_inst > 0] = 1
    output["pred"][HoVerNetBranch.NP.value] = pred_inst
    output["pred"]["inst_info_dict"] = inst_info_dict
    output["pred"]["pred_inst_dict"] = pred_inst_dict
    return output


def run(cfg):
    # --------------------------------------------------------------------------
    # Set Directory and Device
    # --------------------------------------------------------------------------
    log_dir = create_output_dir(cfg)
    multi_gpu = True if cfg["use_gpu"] and torch.cuda.device_count() > 1 else False
    if multi_gpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda:{}".format(dist.get_rank()))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if cfg["use_gpu"] else "cpu")
    # --------------------------------------------------------------------------
    # Data Loading and Preprocessing
    # --------------------------------------------------------------------------
    # Preprocessing transforms
    pre_transforms = Compose(
        [
            CastToTyped(keys=["image"], dtype=torch.float32),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )
    # List of whole slide images
    data_list = [
        {"image": "TCGA-A1-A0SP-01Z-00-DX1.20D689C6-EFA5-4694-BE76-24475A89ACC0.svs"},
        {"image": "TCGA-A2-A0D0-01Z-00-DX1.4FF6B8E5-703B-400F-920A-104F56E0F874.svs"},
    ]
    # Dataset of patches
    dataset = MaskedPatchWSIDataset(
        data_list,
        patch_size=cfg["patch_size"],
        patch_level=0,
        mask_level=3,
        transform=pre_transforms,
        reader=cfg["reader"],
    )
    # Dataloader
    data_loader = DataLoader(dataset, num_workers=cfg["num_workers"], batch_size=cfg["batch_size"], pin_memory=True)
    # --------------------------------------------------------------------------
    # Run some sanity checks
    # --------------------------------------------------------------------------
    #  Check first sample
    first_sample = first(data_loader)
    if first_sample is None:
        raise ValueError("First sample is None!")
    print("image: ")
    print("    shape", first_sample["image"].shape)
    print("    type: ", type(first_sample["image"]))
    print("    dtype: ", first_sample["image"].dtype)
    print(f"batch size: {cfg['batch_size']}")
    print(f"number of batches: {len(data_loader)}")

    # --------------------------------------------------------------------------
    # Model and Handlers
    # --------------------------------------------------------------------------
    # Create model and load weights
    model = HoVerNet(
        mode="fast", in_channels=3, out_classes=7, act=("relu", {"inplace": True}), norm="batch", dropout_prob=0.2
    ).to(device)
    model.load_state_dict(torch.load(cfg["ckpt"]))
    model.eval()
    # Handlers
    inference_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: None),
    ]
    if multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.get_rank()], output_device=dist.get_rank()
        )
        inference_handlers = inference_handlers if dist.get_rank() == 0 else None
    # --------------------------------------------
    # Inference
    # --------------------------------------------
    inference = SupervisedEvaluator(
        device=device,
        val_data_loader=data_loader,
        network=model,
        postprocessing=partial(post_process, return_binary=True, return_centroids=False, output_classes=None),
        val_handlers=inference_handlers,
        amp=cfg["amp"],
    )
    inference.run()

    if multi_gpu:
        dist.destroy_process_group()


def main():
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser(description="Tumor detection on whole slide pathology images.")
    parser.add_argument("--root", type=str, default="./", help="root WSI dir")
    parser.add_argument("--logdir", type=str, default="./logs/", dest="logdir", help="log directory")
    parser.add_argument("--ckpt", type=str, default="./", dest="ckpt", help="Path to the pytorch checkpoint")
    parser.add_argument("--ps", type=int, default=256, dest="patch_size", help="patch size")
    parser.add_argument("--bs", type=int, default=8, dest="batch_size", help="batch size")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--cpu", type=int, default=1, dest="num_workers", help="number of workers")
    parser.add_argument("--use_gpu", type=bool, default=False, help="whether to use gpu")
    parser.add_argument("--reader", type=str, default="OpenSlide", help="WSI reader backend")
    args = parser.parse_args()

    config_dict = vars(args)
    print(config_dict)
    run(config_dict)


if __name__ == "__main__":
    main()
