import logging
import os
import time
from argparse import ArgumentParser
from glob import glob

import torch
import torch.distributed as dist

from monai.apps.pathology.inferers import SlidingWindowHoVerNetInferer
from monai.apps.pathology.transforms import (
    GenerateDistanceMapd,
    GenerateInstanceBorderd,
    GenerateWatershedMarkersd,
    GenerateWatershedMaskd,
    HoVerNetNuclearTypePostProcessingd,
    Watershedd,
)
from monai.data import DataLoader, Dataset, PILReader, partition_dataset
from monai.engines import SupervisedEvaluator
from monai.networks.nets import HoVerNet
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    FillHoles,
    FromMetaTensord,
    GaussianSmooth,
    LoadImaged,
    PromoteChildItemsd,
    SaveImaged,
    ScaleIntensityRanged,
)
from monai.utils import HoVerNetBranch, first


def create_output_dir(cfg):
    timestamp = time.strftime("%y%m%d-%H%M%S")
    run_folder_name = f"{timestamp}_inference_hovernet_ps{cfg['patch_size']}"
    output_dir = os.path.join(cfg["output"], run_folder_name)
    print(f"Outputs are saved at '{output_dir}'.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def run(cfg):
    # --------------------------------------------------------------------------
    # Set Directory, Device,
    # --------------------------------------------------------------------------
    output_dir = create_output_dir(cfg)
    multi_gpu = cfg["use_gpu"] if torch.cuda.device_count() > 1 else False
    if multi_gpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda:{}".format(dist.get_rank()))
        torch.cuda.set_device(device)
        if dist.get_rank() == 0:
            print(f"Running multi-gpu with {dist.get_world_size()} GPUs")
    else:
        device = torch.device("cuda" if cfg["use_gpu"] and torch.cuda.is_available() else "cpu")
    # --------------------------------------------------------------------------
    # Transforms
    # --------------------------------------------------------------------------
    # Preprocessing transforms
    pre_transforms = Compose(
        [
            LoadImaged(keys=["image"], reader=PILReader, converter=lambda x: x.convert("RGB")),
            EnsureChannelFirstd(keys=["image"]),
            CastToTyped(keys=["image"], dtype=torch.float32),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )
    # Postprocessing transforms
    post_transforms = Compose(
        [
            PromoteChildItemsd(
                keys="pred",
                child_keys=[HoVerNetBranch.NC.value, HoVerNetBranch.NP.value, HoVerNetBranch.HV.value],
                delete_keys=True,
            ),
            GenerateWatershedMaskd(keys=HoVerNetBranch.NP.value, softmax=True),
            GenerateInstanceBorderd(keys="mask", hover_map_key=HoVerNetBranch.HV.value, kernel_size=3),
            GenerateDistanceMapd(keys="mask", border_key="border", smooth_fn=GaussianSmooth()),
            GenerateWatershedMarkersd(
                keys="mask",
                border_key="border",
                threshold=0.7,
                radius=2,
                postprocess_fn=FillHoles(),
            ),
            Watershedd(keys="dist", mask_key="mask", markers_key="markers"),
            HoVerNetNuclearTypePostProcessingd(
                type_pred_key=HoVerNetBranch.NC.value, instance_pred_key="dist", output_classes=5
            ),
            FromMetaTensord(keys=["image", "pred_binary"]),
            SaveImaged(
                keys="pred_binary",
                meta_keys="image_meta_dict",
                output_ext="png",
                output_dir=output_dir,
                output_postfix="pred",
                output_dtype="uint8",
                separate_folder=False,
                scale=255,
            ),
        ]
    )
    # --------------------------------------------------------------------------
    # Data and Data Loading
    # --------------------------------------------------------------------------
    # List of whole slide images
    data_list = [{"image": image} for image in glob(os.path.join(cfg["root"], "*.png"))]

    if multi_gpu:
        data = partition_dataset(data=data_list, num_partitions=dist.get_world_size())[dist.get_rank()]
    else:
        data = data_list

    # Dataset
    dataset = Dataset(data, transform=pre_transforms)

    # Dataloader
    data_loader = DataLoader(dataset, num_workers=cfg["ncpu"], batch_size=cfg["batch_size"], pin_memory=True)

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
    # Model
    # --------------------------------------------------------------------------
    # Create model and load weights
    model = HoVerNet(
        mode=cfg["mode"],
        in_channels=3,
        out_classes=5,
        act=("relu", {"inplace": True}),
        norm="batch",
    ).to(device)
    model.load_state_dict(torch.load(cfg["ckpt"], map_location=device))
    model.eval()
    if multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.get_rank()], output_device=dist.get_rank()
        )

    # --------------------------------------------
    # Inference
    # --------------------------------------------
    # Inference engine
    sliding_inferer = SlidingWindowHoVerNetInferer(
        roi_size=cfg["patch_size"],
        sw_batch_size=cfg["sw_batch_size"],
        overlap=1.0 - float(cfg["out_size"]) / float(cfg["patch_size"]),
        padding_mode="constant",
        cval=0,
        sw_device=device,
        device=device,
        progress=True,
        extra_input_padding=((cfg["patch_size"] - cfg["out_size"]) // 2,) * 4,
    )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=data_loader,
        network=model,
        postprocessing=post_transforms,
        inferer=sliding_inferer,
        amp=cfg["use_amp"],
    )
    evaluator.run()

    if multi_gpu:
        dist.destroy_process_group()


def main():
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser(description="Tumor detection on whole slide pathology images.")
    parser.add_argument("--root", type=str, default="./CoNSeP/Test/Images", help="image root dir")
    parser.add_argument("--output", type=str, default="./logs/", dest="output", help="log directory")
    parser.add_argument("--ckpt", type=str, default="./model_CoNSeP_new.pth", help="Path to the pytorch checkpoint")
    parser.add_argument("--mode", type=str, default="original", help="HoVerNet mode (original/fast)")
    parser.add_argument("--bs", type=int, default=1, dest="batch_size", help="batch size")
    parser.add_argument("--swbs", type=int, default=8, dest="sw_batch_size", help="sliding window batch size")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp", help="deactivate use of amp")
    parser.add_argument("--no-gpu", action="store_false", dest="use_gpu", help="deactivate use of gpu")
    parser.add_argument("--ncpu", type=int, default=0, help="number of CPU workers")
    args = parser.parse_args()

    config_dict = vars(args)
    if config_dict["mode"].lower() == "original":
        config_dict["patch_size"] = 270
        config_dict["out_size"] = 80
    elif config_dict["mode"].lower() == "fast":
        config_dict["patch_size"] = 256
        config_dict["out_size"] = 164
    else:
        raise ValueError("`--mode` should be either `original` or `fast`.")

    print(config_dict)
    run(config_dict)


if __name__ == "__main__":
    main()
