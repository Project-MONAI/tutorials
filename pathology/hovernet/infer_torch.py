# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
)

import os
import glob
import torch
import numpy as np
from argparse import ArgumentParser
from monai.data import DataLoader, decollate_batch, Dataset, CacheDataset
from monai.metrics import DiceMetric
from monai.metrics.confusion_matrix import ConfusionMatrixMetric
from monai.losses import DiceLoss
# from monai.networks.nets import HoVerNet
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
    RandFlipd,
    RandRotate90d,
    RandGaussianSmoothd,
    GaussianSmooth,
    FillHoles,
    BoundingRect,
    CenterSpatialCropd,
    SaveImage,
)
# from monai.apps.pathology.transforms.post import (
#     GenerateInstanceContour,
#     GenerateInstanceCentroid,
#     GenerateInstanceType
# )

from monai.utils import set_determinism, convert_to_tensor
from monai.utils.enums import HoVerNetBranch
from monai.visualize import plot_2d_or_3d_image
from skimage import measure


def prepare_data(data_dir, phase):
    data_dir = os.path.join(data_dir, phase)

    files = sorted(
        glob.glob(os.path.join(data_dir, "*/*.npy")))

    images, labels, inst_maps, type_maps = [], [], [], []
    for file in files:
        data = np.load(file)
        images.append(data[..., :3].transpose(2, 0, 1))
        inst_maps.append(measure.label(data[..., 3][None]).astype(int))
        type_maps.append(data[..., 4][None])
        labels.append(np.array(data[..., 3][None] > 0, dtype=int))

    data_dicts = [
        {"image": _image, "label": _label, "label_inst": _inst_map, "label_type": _type_map}
        for _image, _label, _inst_map, _type_map in zip(images, labels, inst_maps, type_maps)
    ]

    return data_dicts


def _dice_info(true, pred, label):
    true = np.array(true == label, np.int32)
    pred = np.array(pred == label, np.int32)
    inter = (pred * true).sum()
    total = (pred + true).sum()
    return inter, total


def post_process_WS(output, device, return_binary=True, return_centroids=False, output_classes=None):
    post_trans_seg = Compose([
        GenerateWatershedMaskd(keys=HoVerNetBranch.NP.value, softmax=True),
        GenerateInstanceBorderd(keys='mask', hover_map_key=HoVerNetBranch.HV, kernel_size=21),
        GenerateDistanceMapd(keys='mask', border_key='border', smooth_fn=GaussianSmooth()),
        GenerateWatershedMarkersd(keys='mask', border_key='border', threshold=0.99, radius=3, postprocess_fn=FillHoles(connectivity=2)),
        Watershedd(keys='dist', mask_key='mask', markers_key='markers')
    ])
    if HoVerNetBranch.NC.value in output.keys():
        type_pred = Activations(softmax=True)(output[HoVerNetBranch.NC.value])
        type_pred = AsDiscrete(argmax=True)(output[HoVerNetBranch.NC.value])

    pred_inst_dict = post_trans_seg(output)
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
            try:
                inst_contour = GenerateInstanceContour()(inst_map, offset)
            except:
                inst_contour = GenerateInstanceContour()(FillHoles(connectivity=2)(inst_map), offset)
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
    pred_type_map = torch.zeros_like(pred_inst)
    for key, value in inst_info_dict.items():
        pred_type_map[pred_inst == key] = value['type']
    pred_type_map = AsDiscrete(to_onehot=5)(pred_type_map)

    if return_binary:
        pred_inst[pred_inst > 0] = 1
    return (pred_inst, pred_type_map, inst_info_dict, pred_inst_dict)


def main(data_dir, args):

    test_transforms = Compose(
        [
            CastToTyped(keys=["image", "label_inst", "label_type"], dtype=torch.float32),
            AsDiscreted(keys="label_type", to_onehot=5),
            CenterSpatialCropd(
                keys="image",
                roi_size=(270, 270),
            ),
            CenterSpatialCropd(
                keys=["label", "label_inst", "label_type"],
                roi_size=(80, 80),
            ),
            CastToTyped(keys="label_inst", dtype=torch.int),
            ComputeHoVerMapsd(keys="label_inst"),
            CastToTyped(keys=["image", "label_inst", "label_type"], dtype=torch.float32),
        ]
    )

    post_process = Compose([Activations(softmax=True)])

    valid_data = prepare_data(data_dir, "valid")
    test_ds = Dataset(data=valid_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    matrix_metric = ConfusionMatrixMetric(include_background=False, metric_name="f1 score")
    device = torch.device("cuda:0")
    model = HoVerNet(
        mode="original",
        in_channels=3,
        out_classes=args.out_classes,
        act=("relu", {"inplace": True}),
        norm="batch",
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt_path))

    model.eval()
    over_inter = 0
    over_total = 0
    with torch.no_grad():
        for test_data in test_loader:
            test_inputs, test_label, test_label_type, test_hover_map = (
                    test_data["image"].to(device),
                    test_data["label"].to(device),
                    test_data["label_type"].to(device),
                    test_data["hover_label_inst"].to(device),
                )

            test_outputs = model(test_inputs)

            test_outputs_seg, test_outputs_type = [], []
            for i in decollate_batch(test_outputs):
                out = post_process_WS(i, device=device, return_binary=True, return_centroids=True, output_classes=args.out_classes)
                test_outputs_seg.append(out[0])
                test_outputs_type.append(out[1])
            test_outputs = [post_process(i[HoVerNetBranch.NP.value])[1:2, ...] > 0.5 for i in decollate_batch(test_outputs)]

            test_label = [i for i in decollate_batch(test_label)]
            test_label_type = [i for i in decollate_batch(test_label_type)]
            for i, out in enumerate(test_outputs):
                inter, total = _dice_info(test_label[i].detach().cpu(), out.detach().cpu(), 1)
                over_inter += inter
                over_total += total

            # compute metric for current iteration
            dice_metric(y_pred=test_outputs, y=test_label)
            matrix_metric(y_pred=test_outputs_type, y=test_label_type)

        metric = dice_metric.aggregate().item()
        f1 = matrix_metric.aggregate()[0].item()
        dice_np = 2 * over_inter / (over_total + 1.0e-8)
        # aggregate the final mean dice result
        print("evaluation metric:", metric, f1, dice_np)
        # reset the status
        dice_metric.reset()
        matrix_metric.reset()


# -
if __name__ == "__main__":
    parser = ArgumentParser(description="HoVerNet inference torch pipeline")
    parser.add_argument("--ngc", action="store_true", dest="ngc", help="use ngc")
    parser.add_argument("--bs", type=int, default=16, dest="batch_size", help="batch size")
    parser.add_argument("--ckpt", type=str, dest="ckpt_path", help="checkpoint path")
    parser.add_argument("--classes", type=int, default=5, dest="out_classes", help="output classes")


    args = parser.parse_args()

    set_determinism(seed=0)
    import sys
    if args.ngc:
        data_dir = "/consep/Prepared/consep"
        sys.path.append('/workspace/pathology/lizard/transforms')
        sys.path.append('/workspace/pathology/lizard/net')
    else:
        data_dir = "/workspace/Data/Lizard/Prepared"
        sys.path.append('/workspace/Code/tutorials/pathology/hovernet/transforms')
        sys.path.append('/workspace/Code/tutorials/pathology/hovernet/net')

    main(data_dir, args)
