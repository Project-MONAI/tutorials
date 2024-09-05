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

import json
import os

import cv2
import fastremap
import numpy as np
import PIL
import tifffile
import torch
import torch.nn.functional as F
from cellpose.dynamics import compute_masks, masks_to_flows
from cellpose.metrics import _intersection_over_union, _true_positive
from monai.apps import get_logger
from monai.data import MetaTensor
from monai.transforms import MapTransform
from monai.utils import ImageMetaKey, convert_to_dst_type

logger = get_logger("VistaCell")


class LoadTiffd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            filename = d[key]

            extension = os.path.splitext(filename)[1][1:]
            image_size = None

            if extension in ["tif", "tiff"]:
                img_array = tifffile.imread(filename)  # use tifffile for tif images
                image_size = img_array.shape
                if len(img_array.shape) == 3 and img_array.shape[-1] <= 3:
                    img_array = np.transpose(img_array, (2, 0, 1))  # channels first without transpose
            else:
                img_array = np.array(PIL.Image.open(filename))  # PIL for all other images (png, jpeg)
                image_size = img_array.shape
                if len(img_array.shape) == 3:
                    img_array = np.transpose(img_array, (2, 0, 1))  # channels first

            if len(img_array.shape) not in [2, 3]:
                raise ValueError(
                    "Unsupported image dimensions, filename " + str(filename) + " shape " + str(img_array.shape)
                )

            if len(img_array.shape) == 2:
                img_array = img_array[np.newaxis]  # add channels_first if no channel

            if key == "label":
                if img_array.shape[0] > 1:
                    print(
                        f"Strange case, label with several channels {filename} shape {img_array.shape}, keeping only first"
                    )
                    img_array = img_array[[0]]

            elif key == "image":
                if img_array.shape[0] == 1:
                    img_array = np.repeat(img_array, repeats=3, axis=0)  # if grayscale, repeat as 3 channels
                elif img_array.shape[0] == 2:
                    print(
                        f"Strange case, image with 2 channels {filename} shape {img_array.shape}, appending first channel to make 3"
                    )
                    img_array = np.stack(
                        (img_array[0], img_array[1], img_array[0]), axis=0
                    )  # this should not happen, we got 2 channel input image
                elif img_array.shape[0] > 3:
                    print(f"Strange case, image with >3 channels,  {filename} shape {img_array.shape}, keeping first 3")
                    img_array = img_array[:3]

            meta_data = {
                ImageMetaKey.FILENAME_OR_OBJ: filename,
                ImageMetaKey.SPATIAL_SHAPE: image_size,
            }
            d[key] = MetaTensor.ensure_torch_and_prune_meta(img_array, meta_data)

        return d


class LabelsToFlows(MapTransform):
    # This transform is useful in cell segmentation tasks where flow-based representations
    # help in identifying cell boundaries and directions. The flow representation are used as
    # input to models that are designed to work with such data, potentially improving
    # segmentation accuracy.
    # based on dynamics labels_to_flows()
    # created a 3 channel output (foreground, flowx, flowy) and saves under flow (new) key

    def __init__(self, flow_key, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flow_key = flow_key

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key].int().numpy()

            label = fastremap.renumber(label, in_place=True)[0]
            veci = masks_to_flows(label[0], device=None)

            flows = np.concatenate((label > 0.5, veci), axis=0).astype(np.float32)
            flows = convert_to_dst_type(flows, d[key], dtype=torch.float, device=d[key].device)[0]
            d[self.flow_key] = flows
            # meta_data = {ImageMetaKey.FILENAME_OR_OBJ : filename}
            # d[key] = MetaTensor.ensure_torch_and_prune_meta(img_array, meta_data)
        return d


class LogitsToLabels:
    def __call__(self, logits, filename=None):
        device = logits.device
        logits = logits.float().cpu().numpy()
        dP = logits[1:]  # vectors
        cellprob = logits[0]  # foreground prob (logit)

        try:
            pred_mask, p = compute_masks(
                dP,
                cellprob,
                niter=200,
                cellprob_threshold=0.4,
                flow_threshold=0.4,
                interp=True,
                device=device,
            )
        except RuntimeError as e:
            logger.warning(f"compute_masks failed on GPU retrying on CPU {logits.shape} file {filename} {e}")
            pred_mask, p = compute_masks(
                dP,
                cellprob,
                niter=200,
                cellprob_threshold=0.4,
                flow_threshold=0.4,
                interp=True,
                device=None,
            )

        return pred_mask, p


# Loss (adopted from Cellpose)
class CellLoss:
    def __call__(self, y_pred, y):
        loss = 0.5 * F.mse_loss(y_pred[:, 1:], 5 * y[:, 1:]) + F.binary_cross_entropy_with_logits(
            y_pred[:, [0]], y[:, [0]]
        )
        return loss


# Accuracy (adopted from Cellpose)
class CellAcc:
    def __call__(self, mask_pred, mask_true):
        if isinstance(mask_true, torch.Tensor):
            mask_true = mask_true.cpu().numpy()

        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.cpu().numpy()

        # print("CellAcc mask_true", mask_true.shape, 'max', np.max(mask_true), ",
        #       "'mask_pred', mask_pred.shape,  'max', np.max(mask_pred) )

        iou = _intersection_over_union(mask_true, mask_pred)[1:, 1:]
        tp = _true_positive(iou, th=0.5)

        fp = np.max(mask_pred) - tp
        fn = np.max(mask_true) - tp
        ap = tp / (tp + fp + fn)

        # print("CellAcc ap", ap, 'tp', tp, 'fp', fp,  'fn', fn)
        return ap
