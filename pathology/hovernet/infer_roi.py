import sys

from functools import partial
from glob import glob
import logging
import os
import time
from argparse import ArgumentParser
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from monai.data import DataLoader, Dataset
from monai.networks.nets import HoVerNet
from monai.engines import SupervisedEvaluator
from monai.inferers import SlidingWindowInferer
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
    EnsureChannelFirstd,
    CastToTyped,
    LoadImaged,
    FillHoles,
    BoundingRect,
    ThresholdIntensity,
    GaussianSmooth,
    ScaleIntensityRanged,
)
from monai.handlers import (
    MeanDice,
    StatsHandler,
    TensorBoardStatsHandler,
    from_engine,
)
from monai.utils import convert_to_tensor, first, HoVerNetBranch


def create_output_dir(cfg):
    if cfg["mode"].lower() == "original":
        cfg["patch_size"] = 270
        cfg["out_size"] = 80
    elif cfg["mode"].lower() == "fast":
        cfg["patch_size"] = 256
        cfg["out_size"] = 164

    timestamp = time.strftime("%y%m%d-%H%M%S")
    run_folder_name = f"{timestamp}_inference_hovernet_ps{cfg['patch_size']}"
    log_dir = os.path.join(cfg["logdir"], run_folder_name)
    print(f"Logs and outputs are saved at '{log_dir}'.")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def post_process(output, return_binary=True, return_centroids=False, output_classes=None):
    device = output[HoVerNetBranch.NP.value].device
    if HoVerNetBranch.NC.value in output.keys():
        type_pred = Activations(softmax=True)(output[HoVerNetBranch.NC.value])
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
    pred_inst_dict = post_trans_seg(output)
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
    output[HoVerNetBranch.NP.value] = pred_inst
    output["inst_info_dict"] = inst_info_dict
    output["pred_inst_dict"] = pred_inst_dict
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
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            CastToTyped(keys=["image"], dtype=torch.float32),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )
    # List of whole slide images
    data_list = [{"image": image} for image in glob(os.path.join(cfg["root"], "*.png"))]

    dataset = Dataset(data_list, transform=pre_transforms)

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
        mode="original",
        in_channels=3,
        out_classes=5,
        act=("relu", {"inplace": True}),
        norm="batch",
    ).to(device)
    model.load_state_dict(torch.load(cfg["ckpt"], map_location=device))
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
    inference = SlidingWindowInferer(
        roi_size=cfg["patch_size"],
        sw_batch_size=8,
        overlap=1.0 - float(cfg["out_size"]) / float(cfg["patch_size"]),
        # overlap=0,
        padding_mode="constant",
        cval=0,
        sw_device=device,
        device=device,
        progress=True,
        extra_input_padding=(cfg["patch_size"] - cfg["out_size"],) * 4,
        pad_output=True,
    )

    for data in data_loader:
        image = data["image"]
        output = inference(image, model)
        result = post_process(output)

    if multi_gpu:
        dist.destroy_process_group()


def main():
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser(description="Tumor detection on whole slide pathology images.")
    parser.add_argument(
        "--root",
        type=str,
        default="/Users/bhashemian/workspace/project-monai/tutorials/pathology/hovernet/CoNSeP/Test/Images",
        help="image root dir",
    )
    parser.add_argument("--logdir", type=str, default="./logs/", dest="logdir", help="log directory")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/Users/bhashemian/workspace/project-monai/tutorials/pathology/hovernet/model_CoNSeP_new.pth",
        dest="ckpt",
        help="Path to the pytorch checkpoint",
    )

    parser.add_argument("--mode", type=str, default="original", help="HoVerNet mode (original/fast)")
    parser.add_argument("--bs", type=int, default=1, dest="batch_size", help="batch size")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--cpu", type=int, default=0, dest="num_workers", help="number of workers")
    parser.add_argument("--use_gpu", type=bool, default=False, help="whether to use gpu")
    parser.add_argument("--reader", type=str, default="OpenSlide", help="WSI reader backend")
    args = parser.parse_args()

    config_dict = vars(args)
    print(config_dict)
    run(config_dict)


if __name__ == "__main__":
    main()
