import logging
import os
import time
from argparse import ArgumentParser
from glob import glob

import torch
import torch.distributed as dist
from imageio import imsave

from monai.apps.pathology.inferers import SlidingWindowHoVerNetInferer
from monai.apps.pathology.transforms import (
    GenerateDistanceMapd, GenerateInstanceBorderd, GenerateWatershedMarkersd,
    GenerateWatershedMaskd, HoVerNetNuclearTypePostProcessingd, Watershedd)
from monai.data import DataLoader, Dataset, PILReader
from monai.engines import SupervisedEvaluator
from monai.networks.nets import HoVerNet
from monai.transforms import (Activationsd, AsDiscreted, CastToTyped,
                              CenterSpatialCropd, Compose, Decollated,
                              EnsureChannelFirstd, FillHoles, GaussianSmooth,
                              LoadImaged, ScaleIntensityRanged)
from monai.transforms.utils import apply_transform
from monai.utils import HoVerNetBranch, first


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


def run(cfg):
    # --------------------------------------------------------------------------
    # Set Directory and Device
    # --------------------------------------------------------------------------
    output_dir = create_output_dir(cfg)
    multi_gpu = True if cfg["use_gpu"] and torch.cuda.device_count() > 1 else False
    if multi_gpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda:{}".format(dist.get_rank()))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if cfg["use_gpu"] else "cpu")
    # --------------------------------------------------------------------------
    # Transforms
    # --------------------------------------------------------------------------
    # Preprocessing transforms
    pre_transforms = Compose(
        [
            LoadImaged(
                keys=["image"], reader=PILReader, converter=lambda x: x.convert("RGB")
            ),
            EnsureChannelFirstd(keys=["image"]),
            # CenterSpatialCropd(keys=["image"], roi_size=(80, 80)),  # for testing only
            CastToTyped(keys=["image"], dtype=torch.float32),
            ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
            ),
        ]
    )
    # Postprocessing transforms
    post_transforms = Compose(
        [
            Decollated(
                keys=[
                    HoVerNetBranch.NC.value,
                    HoVerNetBranch.NP.value,
                    HoVerNetBranch.HV.value,
                ]
            ),
            Activationsd(keys=HoVerNetBranch.NC.value, softmax=True),
            AsDiscreted(keys=HoVerNetBranch.NC.value, argmax=True),
            GenerateWatershedMaskd(keys=HoVerNetBranch.NP.value, softmax=True),
            GenerateInstanceBorderd(
                keys="mask", hover_map_key=HoVerNetBranch.HV.value, kernel_size=3
            ),
            GenerateDistanceMapd(
                keys="mask", border_key="border", smooth_fn=GaussianSmooth()
            ),
            GenerateWatershedMarkersd(
                keys="mask",
                border_key="border",
                threshold=0.7,
                radius=2,
                postprocess_fn=FillHoles(),
            ),
            Watershedd(keys="dist", mask_key="mask", markers_key="markers"),
            HoVerNetNuclearTypePostProcessingd(
                type_pred_key=HoVerNetBranch.NC.value, instance_pred_key="dist"
            ),
        ]
    )
    # --------------------------------------------------------------------------
    # Data and Data Loading
    # --------------------------------------------------------------------------
    # List of whole slide images
    data_list = [
        {"image": image, "filename": image}
        for image in glob(os.path.join(cfg["root"], "*.png"))
    ]

    # Dataset
    dataset = Dataset(data_list, transform=pre_transforms)

    # Dataloader
    data_loader = DataLoader(
        dataset,
        num_workers=cfg["num_workers"],
        batch_size=cfg["batch_size"],
        pin_memory=True,
    )

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
        mode="original",
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
    sliding_inferer = SlidingWindowHoVerNetInferer(
        roi_size=cfg["patch_size"],
        sw_batch_size=8,
        overlap=1.0 - float(cfg["out_size"]) / float(cfg["patch_size"]),
        # overlap=0,
        padding_mode="constant",
        cval=0,
        sw_device=device,
        device=device,
        progress=True,
        extra_input_padding=((cfg["patch_size"] - cfg["out_size"]) // 2,) * 4,
    )

    for i, data in enumerate(data_loader):
        print(">>>>> ", data["filename"])
        image = data["image"]
        output = sliding_inferer(image, model)
        results = apply_transform(post_transforms, output)
        for i, res in enumerate(results):
            filename = os.path.join(
                output_dir,
                os.path.basename(data["filename"][i]).replace(".png", "_pred.png"),
            )
            print(f"Saving {filename}...")
            imsave(filename, res["pred_binary"].permute(1, 2, 0).numpy())

    # evaluator = SupervisedEvaluator(
    #     device=device,
    #     val_data_loader=data_loader,
    #     network=model,
    #     postprocessing=post_transforms,
    #     inferer=sliding_inferer,
    #     amp=cfg["amp"],
    # )
    # evaluator.run()

    if multi_gpu:
        dist.destroy_process_group()


def main():
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser(
        description="Tumor detection on whole slide pathology images."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/Users/bhashemian/workspace/project-monai/tutorials/pathology/hovernet/CoNSeP/Test/Images",
        help="image root dir",
    )
    parser.add_argument(
        "--logdir", type=str, default="./logs/", dest="logdir", help="log directory"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/Users/bhashemian/workspace/project-monai/tutorials/pathology/hovernet/model_CoNSeP_new.pth",
        dest="ckpt",
        help="Path to the pytorch checkpoint",
    )

    parser.add_argument(
        "--mode", type=str, default="original", help="HoVerNet mode (original/fast)"
    )
    parser.add_argument(
        "--bs", type=int, default=1, dest="batch_size", help="batch size"
    )
    parser.add_argument(
        "--no-amp", action="store_false", dest="amp", help="deactivate amp"
    )
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument(
        "--cpu", type=int, default=0, dest="num_workers", help="number of workers"
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=False, help="whether to use gpu"
    )
    parser.add_argument(
        "--reader", type=str, default="OpenSlide", help="WSI reader backend"
    )
    args = parser.parse_args()

    config_dict = vars(args)
    print(config_dict)
    run(config_dict)


if __name__ == "__main__":
    main()
