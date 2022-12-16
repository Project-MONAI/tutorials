import os
import glob
import logging
import torch
from argparse import ArgumentParser
from monai.data import DataLoader, CacheDataset
from monai.networks.nets import HoVerNet
from monai.engines import SupervisedEvaluator
from monai.transforms import (
    LoadImaged,
    Lambdad,
    Activationsd,
    Compose,
    CastToTyped,
    ComputeHoVerMapsd,
    ScaleIntensityRanged,
    CenterSpatialCropd,
)
from monai.handlers import (
    MeanDice,
    StatsHandler,
    CheckpointLoader,
)
from monai.utils.enums import HoVerNetBranch
from monai.apps.pathology.handlers.utils import from_engine_hovernet
from monai.apps.pathology.engines.utils import PrepareBatchHoVerNet
from skimage import measure


def prepare_data(data_dir, phase):
    """prepare data list"""

    data_dir = os.path.join(data_dir, phase)
    images = sorted(glob.glob(os.path.join(data_dir, "*image.npy")))
    inst_maps = sorted(glob.glob(os.path.join(data_dir, "*inst_map.npy")))
    type_maps = sorted(glob.glob(os.path.join(data_dir, "*type_map.npy")))

    data_list = [
        {"image": _image, "label_inst": _inst_map, "label_type": _type_map}
        for _image, _inst_map, _type_map in zip(images, inst_maps, type_maps)
    ]
    return data_list


def run(cfg):
    if cfg["mode"].lower() == "original":
        cfg["patch_size"] = [270, 270]
        cfg["out_size"] = [80, 80]
    elif cfg["mode"].lower() == "fast":
        cfg["patch_size"] = [256, 256]
        cfg["out_size"] = [164, 164]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Create MONAI DataLoaders
    valid_data = prepare_data(cfg["root"], "Test")
    valid_ds = CacheDataset(data=valid_data, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(
        valid_ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available()
    )

    # initialize model
    model = HoVerNet(
        mode=cfg["mode"],
        in_channels=3,
        out_classes=cfg["out_classes"],
        act=("relu", {"inplace": True}),
        norm="batch",
        pretrained_url=None,
        freeze_encoder=False,
    ).to(device)

    post_process_np = Compose(
        [
            Activationsd(keys=HoVerNetBranch.NP.value, softmax=True),
            Lambdad(keys=HoVerNetBranch.NP.value, func=lambda x: x[1:2, ...] > 0.5),
        ]
    )
    post_process = Lambdad(keys="pred", func=post_process_np)

    # Evaluator
    val_handlers = [
        CheckpointLoader(load_path=cfg["ckpt"], load_dict={"net": model}),
        StatsHandler(output_transform=lambda x: None),
    ]
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        prepare_batch=PrepareBatchHoVerNet(extra_keys=["label_type", "hover_label_inst"]),
        network=model,
        postprocessing=post_process,
        key_val_metric={
            "val_dice": MeanDice(
                include_background=False,
                output_transform=from_engine_hovernet(keys=["pred", "label"], nested_key=HoVerNetBranch.NP.value),
            )
        },
        val_handlers=val_handlers,
        amp=cfg["amp"],
    )

    state = evaluator.run()
    print(state)


def main():
    parser = ArgumentParser(description="Tumor detection on whole slide pathology images.")
    parser.add_argument(
        "--root",
        type=str,
        default="/workspace/Data/Pathology/CoNSeP/Prepared",
        help="root data dir",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./logs/model.pt",
        help="Path to the pytorch checkpoint",
    )
    parser.add_argument("--bs", type=int, default=16, dest="batch_size", help="batch size")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")
    parser.add_argument("--classes", type=int, default=5, dest="out_classes", help="output classes")
    parser.add_argument("--mode", type=str, default="fast", help="choose either `original` or `fast`")

    parser.add_argument("--cpu", type=int, default=8, dest="num_workers", help="number of workers")
    parser.add_argument("--use_gpu", type=bool, default=True, dest="use_gpu", help="whether to use gpu")

    args = parser.parse_args()
    cfg = vars(args)
    print(cfg)

    logging.basicConfig(level=logging.INFO)
    run(cfg)


if __name__ == "__main__":
    main()
