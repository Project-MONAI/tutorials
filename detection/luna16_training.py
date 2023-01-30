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
import gc
import json
import logging
import sys
import time

import cv2
import numpy as np
import torch
from generate_transforms import (
    generate_detection_train_transform,
    generate_detection_val_transform,
)
from torch.utils.tensorboard import SummaryWriter
from visualize_image import visualize_one_xy_slice_in_3d_image
from warmup_scheduler import GradualWarmupScheduler

import monai
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import DataLoader, Dataset, box_utils, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.networks.nets import resnet
from monai.transforms import ScaleIntensityRanged
from monai.utils import set_determinism


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
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

    set_determinism(seed=0)

    amp = True
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    monai.config.print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    # 1. define transform
    intensity_transform = ScaleIntensityRanged(
        keys=["image"],
        a_min=-1024,
        a_max=300.0,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    train_transforms = generate_detection_train_transform(
        "image",
        "box",
        "label",
        args.gt_box_mode,
        intensity_transform,
        args.patch_size,
        args.batch_size,
        affine_lps_to_ras=True,
        amp=amp,
    )

    val_transforms = generate_detection_val_transform(
        "image",
        "box",
        "label",
        args.gt_box_mode,
        intensity_transform,
        affine_lps_to_ras=True,
        amp=amp,
    )

    # 2. prepare training data
    # create a training data loader
    train_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key="training",
        base_dir=args.data_base_dir,
    )
    train_ds = Dataset(
        data=train_data[: int(0.95 * len(train_data))],
        transform=train_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=7,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True,
    )

    # create a validation data loader
    val_ds = Dataset(
        data=train_data[int(0.95 * len(train_data)) :],
        transform=val_transforms,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True,
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
    conv1_t_size = [max(7, 2 * s + 1) for s in args.conv1_t_stride]
    backbone = resnet.ResNet(
        block=resnet.ResNetBottleneck,
        layers=[3, 4, 6, 3],
        block_inplanes=resnet.get_inplanes(),
        n_input_channels=args.n_input_channels,
        conv1_t_stride=args.conv1_t_stride,
        conv1_t_size=conv1_t_size,
    )
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=args.spatial_dims,
        pretrained_backbone=False,
        trainable_backbone_layers=None,
        returned_layers=args.returned_layers,
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    size_divisible = [s * 2 * 2 ** max(args.returned_layers) for s in feature_extractor.body.conv1.stride]
    net = torch.jit.script(
        RetinaNet(
            spatial_dims=args.spatial_dims,
            num_classes=len(args.fg_labels),
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
        )
    )

    # 3) build detector
    detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=False).to(device)

    # set training components
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(
        batch_size_per_image=64,
        positive_fraction=args.balanced_sampler_pos_fraction,
        pool_size=20,
        min_neg=16,
    )
    detector.set_target_keys(box_key="box", label_key="label")

    # set validation components
    detector.set_box_selector_parameters(
        score_thresh=args.score_thresh,
        topk_candidates_per_level=1000,
        nms_thresh=args.nms_thresh,
        detections_per_img=100,
    )
    detector.set_sliding_window_inferer(
        roi_size=args.val_patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device="cpu",
    )

    # 4. Initialize training
    # initlize optimizer
    optimizer = torch.optim.SGD(
        detector.network.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True,
    )
    after_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=after_scheduler)
    scaler = torch.cuda.amp.GradScaler() if amp else None
    optimizer.zero_grad()
    optimizer.step()

    # initialize tensorboard writer
    tensorboard_writer = SummaryWriter(args.tfevent_path)

    # 5. train
    val_interval = 5  # do validation every val_interval epochs
    coco_metric = COCOMetric(classes=["nodule"], iou_list=[0.1], max_detection=[100])
    best_val_epoch_metric = 0.0
    best_val_epoch = -1  # the epoch that gives best validation metrics

    max_epochs = 300
    epoch_len = len(train_ds) // train_loader.batch_size
    w_cls = config_dict.get("w_cls", 1.0)  # weight between classification loss and box regression loss, default 1.0
    for epoch in range(max_epochs):
        # ------------- Training -------------
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        detector.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_box_reg_loss = 0
        step = 0
        start_time = time.time()
        scheduler_warmup.step()
        # Training
        for batch_data in train_loader:
            step += 1
            inputs = [
                batch_data_ii["image"].to(device) for batch_data_i in batch_data for batch_data_ii in batch_data_i
            ]
            targets = [
                dict(
                    label=batch_data_ii["label"].to(device),
                    box=batch_data_ii["box"].to(device),
                )
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]

            for param in detector.network.parameters():
                param.grad = None

            if amp and (scaler is not None):
                with torch.cuda.amp.autocast():
                    outputs = detector(inputs, targets)
                    loss = w_cls * outputs[detector.cls_key] + outputs[detector.box_reg_key]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = detector(inputs, targets)
                loss = w_cls * outputs[detector.cls_key] + outputs[detector.box_reg_key]
                loss.backward()
                optimizer.step()

            # save to tensorboard
            epoch_loss += loss.detach().item()
            epoch_cls_loss += outputs[detector.cls_key].detach().item()
            epoch_box_reg_loss += outputs[detector.box_reg_key].detach().item()
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            tensorboard_writer.add_scalar("train_loss", loss.detach().item(), epoch_len * epoch + step)

        end_time = time.time()
        print(f"Training time: {end_time-start_time}s")
        del inputs, batch_data
        torch.cuda.empty_cache()
        gc.collect()

        # save to tensorboard
        epoch_loss /= step
        epoch_cls_loss /= step
        epoch_box_reg_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        tensorboard_writer.add_scalar("avg_train_loss", epoch_loss, epoch + 1)
        tensorboard_writer.add_scalar("avg_train_cls_loss", epoch_cls_loss, epoch + 1)
        tensorboard_writer.add_scalar("avg_train_box_reg_loss", epoch_box_reg_loss, epoch + 1)
        tensorboard_writer.add_scalar("train_lr", optimizer.param_groups[0]["lr"], epoch + 1)

        # save last trained model
        torch.jit.save(detector.network, env_dict["model_path"][:-3] + "_last.pt")
        print("saved last model")

        # ------------- Validation for model selection -------------
        if (epoch + 1) % val_interval == 0:
            detector.eval()
            val_outputs_all = []
            val_targets_all = []
            start_time = time.time()
            with torch.no_grad():
                for val_data in val_loader:
                    # if all val_data_i["image"] smaller than args.val_patch_size, no need to use inferer
                    # otherwise, need inferer to handle large input images.
                    use_inferer = not all(
                        [val_data_i["image"][0, ...].numel() < np.prod(args.val_patch_size) for val_data_i in val_data]
                    )
                    val_inputs = [val_data_i.pop("image").to(device) for val_data_i in val_data]

                    if amp:
                        with torch.cuda.amp.autocast():
                            val_outputs = detector(val_inputs, use_inferer=use_inferer)
                    else:
                        val_outputs = detector(val_inputs, use_inferer=use_inferer)

                    # save outputs for evaluation
                    val_outputs_all += val_outputs
                    val_targets_all += val_data

            end_time = time.time()
            print(f"Validation time: {end_time-start_time}s")

            # visualize an inference image and boxes to tensorboard
            draw_img = visualize_one_xy_slice_in_3d_image(
                gt_boxes=val_data[0]["box"].cpu().detach().numpy(),
                image=val_inputs[0][0, ...].cpu().detach().numpy(),
                pred_boxes=val_outputs[0][detector.target_box_key].cpu().detach().numpy(),
            )
            tensorboard_writer.add_image("val_img_xy", draw_img.transpose([2, 1, 0]), epoch + 1)

            # compute metrics
            del val_inputs
            torch.cuda.empty_cache()
            results_metric = matching_batch(
                iou_fn=box_utils.box_iou,
                iou_thresholds=coco_metric.iou_thresholds,
                pred_boxes=[
                    val_data_i[detector.target_box_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                pred_classes=[
                    val_data_i[detector.target_label_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                pred_scores=[
                    val_data_i[detector.pred_score_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                gt_boxes=[val_data_i[detector.target_box_key].cpu().detach().numpy() for val_data_i in val_targets_all],
                gt_classes=[
                    val_data_i[detector.target_label_key].cpu().detach().numpy() for val_data_i in val_targets_all
                ],
            )
            val_epoch_metric_dict = coco_metric(results_metric)[0]
            print(val_epoch_metric_dict)

            # write to tensorboard event
            for k in val_epoch_metric_dict.keys():
                tensorboard_writer.add_scalar("val_" + k, val_epoch_metric_dict[k], epoch + 1)
            val_epoch_metric = val_epoch_metric_dict.values()
            val_epoch_metric = sum(val_epoch_metric) / len(val_epoch_metric)
            tensorboard_writer.add_scalar("val_metric", val_epoch_metric, epoch + 1)

            # save best trained model
            if val_epoch_metric > best_val_epoch_metric:
                best_val_epoch_metric = val_epoch_metric
                best_val_epoch = epoch + 1
                torch.jit.save(detector.network, env_dict["model_path"])
                print("saved new best metric model")
            print(
                "current epoch: {} current metric: {:.4f} "
                "best metric: {:.4f} at epoch {}".format(
                    epoch + 1, val_epoch_metric, best_val_epoch_metric, best_val_epoch
                )
            )

    print(f"train completed, best_metric: {best_val_epoch_metric:.4f} " f"at epoch: {best_val_epoch}")
    tensorboard_writer.close()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
