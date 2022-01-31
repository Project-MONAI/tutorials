import argparse
import json
import logging
import sys

import monai
import numpy as np
from monai.apps.detection.networks.nets.detection.anchor_utils import AnchorGenerator
from monai.apps.detection.networks.nets.detection.retinanet import (
    retinanet_resnet50_fpn,
)
from monai.config import print_config
from monai.data import DataLoader, Dataset, box_utils, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.metrics.detection.coco_evaluator import COCOMetric
from monai.metrics.detection.matching import matching_batch
from monai.transforms import (
    BoxClipToImaged,
    BoxConvertToStandardd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    # Orientationd,
)

import torch


def box_iou_np(bbox1: np.array, bbox2: np.array, **kwargs):
    return (
        box_utils.box_iou(torch.FloatTensor(bbox1), torch.FloatTensor(bbox2), **kwargs)
        .cpu().detach().numpy()
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Testing")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train.json",
        help="config json file that stores hyper-parameters",
    )
    args = parser.parse_args()

    monai.config.print_config()

    envDict = json.load(open(args.environment_file, "r"))
    configDict = json.load(open(args.config_file, "r"))

    for k, v in envDict.items():
        setattr(args, k, v)
    for k, v in configDict.items():
        setattr(args, k, v)

    # 1. define transform
    gt_box_mode = "xxyyzz"
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image", "box", "label"]),
            NormalizeIntensityd(keys=["image"]),
            # Orientationd(keys=["image"], axcodes="RAS"),
            BoxConvertToStandardd(box_keys=["box"], box_mode=gt_box_mode),
            BoxClipToImaged(box_keys=["box"], image_key="image", remove_empty=True),
            EnsureTyped(keys=["image", "box", "label"]),
        ]
    )

    # create a validation data loader
    val_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key="validation",
        base_dir=args.data_base_dir,
    )
    val_ds = Dataset(
        data=val_data,
        transform=val_transforms,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
    )

    # 3. build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(args.returned_layers) == len(args.base_anchor_size) - 1
    anchor_sizes = tuple(
        (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
        for x in args.base_anchor_size
    )
    print(anchor_sizes)
    aspect_ratios = [args.base_aspect_ratios] * len(anchor_sizes)
    anchor_generator = AnchorGenerator(args.spatial_dims, anchor_sizes, aspect_ratios)
    model = retinanet_resnet50_fpn(
        spatial_dims=args.spatial_dims,
        pretrained=False,
        progress=True,
        num_classes=args.num_classes,
        n_input_channels=args.n_input_channels,
        pretrained_backbone=False,
        trainable_backbone_layers=None,
        anchor_generator=anchor_generator,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
    ).to(device)
    model.load_state_dict(torch.load(envDict["model_path"]))

    # 4. apply trained model
    results_dict = {"validation": []}
    val_outputs_all = []
    val_targets_all = []
    model.eval()

    coco_metric = COCOMetric(classes=["nodule"], iou_list=[0.1, 0.3], max_detection=[7])
    with torch.no_grad():
        for val_data in val_loader:
            val_img_filenames = [
                val_data_i["image_meta_dict"]["filename_or_obj"]
                for val_data_i in val_data
            ]
            val_inputs = [val_data_i["image"].to(device) for val_data_i in val_data]
            val_outputs = model(val_inputs)

            val_outputs_all += [
                {
                    "labels": val_data_i["labels"].cpu().detach().numpy(),
                    "boxes": val_data_i["boxes"].cpu().detach().numpy(),
                    "scores": val_data_i["scores"].cpu().detach().numpy(),
                }
                for val_data_i in val_outputs
            ]
            val_targets_all += [
                {
                    "labels": val_data_i["label"].cpu().detach().numpy(),
                    "boxes": val_data_i["box"].cpu().detach().numpy(),
                    "ignore": np.zeros(val_data_i["box"].shape[0]),
                }
                for val_data_i in val_data
            ]

            for val_img_filename, result in zip(val_img_filenames, val_outputs):
                result.update({"image": val_img_filename})
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.detach().cpu().tolist()
                results_dict["validation"].append(result)

    with open(args.result_list_file_path, "w") as outfile:
        json.dump(results_dict, outfile, indent=4)

    results_metric = matching_batch(
        iou_fn=box_iou_np,
        iou_thresholds=coco_metric.iou_thresholds,
        pred_boxes=[val_data_i["boxes"] for val_data_i in val_outputs_all],
        pred_classes=[val_data_i["labels"] for val_data_i in val_outputs_all],
        pred_scores=[val_data_i["scores"] for val_data_i in val_outputs_all],
        gt_boxes=[val_data_i["boxes"] for val_data_i in val_targets_all],
        gt_classes=[val_data_i["labels"] for val_data_i in val_targets_all],
        gt_ignore=[val_data_i["ignore"] for val_data_i in val_targets_all],
    )

    final_metric = coco_metric(results_metric)

    print(final_metric)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
