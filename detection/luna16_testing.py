# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import sys
import time
from copy import deepcopy

import numpy as np
import torch
from generate_transforms import generate_detection_inference_transform

import monai
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.transforms.dictionary import ClipBoxToImaged
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.networks.nets import resnet
from monai.transforms import Compose, DeleteItemsd, Invertd, ScaleIntensityRanged


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
        default="./config/config_test.json",
        help="config json file that stores hyper-parameters",
    )
    args = parser.parse_args()

    amp = True

    monai.config.print_config()

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    patch_size = args.val_patch_size

    # 1. define transform
    intensity_transform = ScaleIntensityRanged(
        keys=["image"],
        a_min=-1024,
        a_max=300.0,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    inference_transforms, post_transforms = generate_detection_inference_transform(
        "image",
        "pred_box",
        "pred_label",
        "pred_score",
        args.gt_box_mode,
        intensity_transform,
        affine_lps_to_ras=True,
        amp=amp,
    )

    # 2. create a inference data loader
    inference_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key="validation",
        base_dir=args.data_base_dir,
    )
    inference_ds = Dataset(
        data=inference_data,
        transform=inference_transforms,
    )
    inference_loader = DataLoader(
        inference_ds,
        batch_size=1,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
    )

    # 3. build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) build anchor generator
    # returned_layers: when target boxes are small, set it smaller
    # base_anchor_shapes: anchor shape for the most high-resolution output,
    #   when target boxes are small, set it smaller
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(args.returned_layers) + 1)],
        base_anchor_shapes=args.base_anchor_shapes,
    )

    # 2) build network
    net = torch.jit.load(env_dict["model_path"]).to(device)
    print(f"Load model from {env_dict['model_path']}")


    # 3) build detector
    detector = RetinaNetDetector(
        network=net, anchor_generator=anchor_generator, debug=False
    )

    # set inference components
    detector.set_box_selector_parameters(
        score_thresh=args.score_thresh,
        topk_candidates_per_level=1000,
        nms_thresh=args.nms_thresh,
        detections_per_img=100,
    )
    detector.set_sliding_window_inferer(
        roi_size=patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="gaussian",
        device="cpu",
    )

    # 4. apply trained model
    results_dict = {"validation": []}
    detector.eval()

    with torch.no_grad():
        start_time = time.time()
        for inference_data in inference_loader:
            inference_img_filenames = [
                inference_data_i["image_meta_dict"]["filename_or_obj"]
                for inference_data_i in inference_data
            ]
            print(inference_img_filenames)
            use_inferer = not all(
                [
                    inference_data_i["image"][0, ...].numel() < np.prod(patch_size)
                    for inference_data_i in inference_data
                ]
            )
            inference_inputs = [inference_data_i["image"].to(device) for inference_data_i in inference_data]

            if amp:
                with torch.cuda.amp.autocast():
                    inference_outputs = detector(inference_inputs, use_inferer=use_inferer)
            else:
                inference_outputs = detector(inference_inputs, use_inferer=use_inferer)
            del inference_inputs

            # update inference_data for post transform
            for i in range(len(inference_outputs)):
                inference_data_i, inference_pred_i = inference_data[i], inference_outputs[i]
                inference_data_i["pred_box"] = inference_pred_i[detector.target_box_key].to(
                    torch.float32
                )
                inference_data_i["pred_label"] = inference_pred_i[detector.target_label_key]
                inference_data_i["pred_score"] = inference_pred_i[detector.pred_score_key].to(
                    torch.float32
                )
                inference_data[i] = post_transforms(inference_data_i)

            for inference_img_filename, inference_pred_i in zip(inference_img_filenames, inference_data):
                result = {
                    "label": inference_pred_i["pred_label"].cpu().detach().numpy().tolist(),
                    "box": inference_pred_i["pred_box"].cpu().detach().numpy().tolist(),
                    "score": inference_pred_i["pred_score"].cpu().detach().numpy().tolist(),
                }
                result.update({"image": inference_img_filename})
                results_dict["validation"].append(result)

    end_time = time.time()
    print("Testing time: ", end_time - start_time)

    with open(args.result_list_file_path, "w") as outfile:
        json.dump(results_dict, outfile, indent=4)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
