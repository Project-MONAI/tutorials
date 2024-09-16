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

import torch
import numpy as np
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByPosNegLabeld,
    RandZoomd,
    RandFlipd,
    RandRotate90d,
    MapTransform,
)
from monai.transforms.utility.dictionary import ApplyTransformToPointsd
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    ConvertBoxModed,
    StandardizeEmptyBoxd,
)
from monai.config import KeysCollection
from monai.utils.type_conversion import convert_data_type
from monai.data.box_utils import clip_boxes_to_image
from monai.apps.detection.transforms.box_ops import convert_box_to_mask


def generate_detection_train_transform(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    patch_size,
    batch_size,
    point_key="points",
    affine_lps_to_ras=False,
    amp=True,
):
    """
    Generate training transform for detection.

    Args:
        image_key: the key to represent images in the input json files
        box_key: the key to represent boxes in the input json files
        label_key: the key to represent box labels in the input json files
        point_key: the key to represent points to save the box coordinates
        gt_box_mode: ground truth box mode in the input json files
        intensity_transform: transform to scale image intensities,
            usually ScaleIntensityRanged for CT images, and NormalizeIntensityd for MR images.
        patch_size: cropped patch size for training
        batch_size: number of cropped patches from each image
        affine_lps_to_ras: Usually False.
            Set True only when the original images were read by itkreader with affine_lps_to_ras=True
        amp: whether to use half precision

    Return:
        training transform for detection
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    train_transforms = Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            EnsureTyped(keys=[image_key], dtype=torch.float16),
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            ConvertBoxToPointsd(keys=[box_key]),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            # generate box mask based on the input boxes which used for cropping
            GenerateExtendedBoxMask(
                keys=box_key,
                image_key=image_key,
                spatial_size=patch_size,
                whole_box=True,
            ),
            RandCropByPosNegLabeld(
                keys=[image_key],
                label_key="mask_image",
                spatial_size=patch_size,
                num_samples=batch_size,
                pos=1,
                neg=1,
            ),
            RandZoomd(
                keys=[image_key],
                prob=0.2,
                min_zoom=0.7,
                max_zoom=1.4,
                padding_mode="constant",
                keep_size=True,
            ),
            RandFlipd(
                keys=[image_key],
                prob=0.5,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=[image_key],
                prob=0.5,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=[image_key],
                prob=0.5,
                spatial_axis=2,
            ),
            RandRotate90d(
                keys=[image_key],
                prob=0.75,
                max_k=3,
                spatial_axes=(0, 1),
            ),
            # apply the same affine matrix which already applied on the images to the points
            ApplyTransformToPointsd(keys=[point_key], refer_keys=image_key, affine_lps_to_ras=affine_lps_to_ras),
            # convert points back to boxes
            ConvertPointsToBoxesd(keys=[point_key]),
            ClipBoxToImaged(
                box_keys=box_key,
                label_keys=[label_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            BoxToMaskd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                box_ref_image_keys=image_key,
                min_fg_label=0,
                ellipse_mask=True,
            ),
            RandRotated(
                keys=[image_key, "box_mask"],
                mode=["nearest", "nearest"],
                prob=0.2,
                range_x=np.pi / 6,
                range_y=np.pi / 6,
                range_z=np.pi / 6,
                keep_size=True,
                padding_mode="zeros",
            ),
            MaskToBoxd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
            ),
            DeleteItemsd(keys=["box_mask"]),
            RandGaussianNoised(keys=[image_key], prob=0.1, mean=0, std=0.1),
            RandGaussianSmoothd(
                keys=[image_key],
                prob=0.1,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=[image_key], prob=0.15, factors=0.25),
            RandShiftIntensityd(keys=[image_key], prob=0.15, offsets=0.1),
            RandAdjustContrastd(keys=[image_key], prob=0.3, gamma=(0.7, 1.5)),
            EnsureTyped(keys=[image_key], dtype=compute_dtype),
            EnsureTyped(keys=[label_key], dtype=torch.long),
        ]
    )
    return train_transforms


def generate_detection_val_transform(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    affine_lps_to_ras=False,
    amp=True,
):
    """
    Generate validation transform for detection.

    Args:
        image_key: the key to represent images in the input json files
        box_key: the key to represent boxes in the input json files
        label_key: the key to represent box labels in the input json files
        gt_box_mode: ground truth box mode in the input json files
        intensity_transform: transform to scale image intensities,
            usually ScaleIntensityRanged for CT images, and NormalizeIntensityd for MR images.
        affine_lps_to_ras: Usually False.
            Set True only when the original images were read by itkreader with affine_lps_to_ras=True
        amp: whether to use half precision

    Return:
        validation transform for detection
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    val_transforms = Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
            EnsureTyped(keys=label_key, dtype=torch.long),
        ]
    )
    return val_transforms


def generate_detection_inference_transform(
    image_key,
    pred_box_key,
    pred_label_key,
    pred_score_key,
    gt_box_mode,
    intensity_transform,
    affine_lps_to_ras=False,
    amp=True,
):
    """
    Generate validation transform for detection.

    Args:
        image_key: the key to represent images in the input json files
        pred_box_key: the key to represent predicted boxes
        pred_label_key: the key to represent predicted box labels
        pred_score_key: the key to represent predicted classification scores
        gt_box_mode: ground truth box mode in the input json files
        intensity_transform: transform to scale image intensities,
            usually ScaleIntensityRanged for CT images, and NormalizeIntensityd for MR images.
        affine_lps_to_ras: Usually False.
            Set True only when the original images were read by itkreader with affine_lps_to_ras=True
        amp: whether to use half precision

    Return:
        validation transform for detection
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    test_transforms = Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key], dtype=torch.float32),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            EnsureTyped(keys=[image_key], dtype=compute_dtype),
        ]
    )
    post_transforms = Compose(
        [
            ClipBoxToImaged(
                box_keys=[pred_box_key],
                label_keys=[pred_label_key, pred_score_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            AffineBoxToWorldCoordinated(
                box_keys=[pred_box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            ConvertBoxModed(box_keys=[pred_box_key], src_mode="xyzxyz", dst_mode=gt_box_mode),
            DeleteItemsd(keys=[image_key]),
        ]
    )
    return test_transforms, post_transforms


class GenerateExtendedBoxMask(MapTransform):
    """
    Generate box mask based on the input boxes.
    """

    def __init__(
        self,
        keys: KeysCollection,
        image_key: str,
        spatial_size: tuple[int, int, int],
        whole_box: bool,
        mask_image_key: str = "mask_image",
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            image_key: key for the image data in the dictionary.
            spatial_size: size of the spatial dimensions of the mask.
            whole_box: whether to use the whole box for generating the mask.
            mask_image_key: key to store the generated box mask.
        """
        super().__init__(keys)
        self.image_key = image_key
        self.spatial_size = spatial_size
        self.whole_box = whole_box
        self.mask_image_key = mask_image_key

    def generate_fg_center_boxes_np(self, boxes, image_size, whole_box=True):
        # We don't require crop center to be within the boxes.
        # As along as the cropped patch contains a box, it is considered as a foreground patch.
        # Positions within extended_boxes are crop centers for foreground patches
        spatial_dims = len(image_size)
        boxes_np, *_ = convert_data_type(boxes, np.ndarray)

        extended_boxes = np.zeros_like(boxes_np, dtype=int)
        boxes_start = np.ceil(boxes_np[:, :spatial_dims]).astype(int)
        boxes_stop = np.floor(boxes_np[:, spatial_dims:]).astype(int)
        for axis in range(spatial_dims):
            if not whole_box:
                extended_boxes[:, axis] = boxes_start[:, axis] - self.spatial_size[axis] // 2 + 1
                extended_boxes[:, axis + spatial_dims] = boxes_stop[:, axis] + self.spatial_size[axis] // 2 - 1
            else:
                # extended box start
                extended_boxes[:, axis] = boxes_stop[:, axis] - self.spatial_size[axis] // 2 - 1
                extended_boxes[:, axis] = np.minimum(extended_boxes[:, axis], boxes_start[:, axis])
                # extended box stop
                extended_boxes[:, axis + spatial_dims] = extended_boxes[:, axis] + self.spatial_size[axis] // 2
                extended_boxes[:, axis + spatial_dims] = np.maximum(
                    extended_boxes[:, axis + spatial_dims], boxes_stop[:, axis]
                )
        extended_boxes, _ = clip_boxes_to_image(extended_boxes, image_size, remove_empty=True)  # type: ignore
        return extended_boxes

    def generate_mask_img(self, boxes, image_size, whole_box=True):
        extended_boxes_np = self.generate_fg_center_boxes_np(boxes, image_size, whole_box)
        mask_img = convert_box_to_mask(
            extended_boxes_np, np.ones(extended_boxes_np.shape[0]), image_size, bg_label=0, ellipse_mask=False
        )
        mask_img = np.amax(mask_img, axis=0, keepdims=True)[0:1, ...]
        return mask_img

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[self.image_key]
            boxes = d[key]
            data[self.mask_image_key] = self.generate_mask_img(boxes, image.shape[1:], whole_box=self.whole_box)
        return data
