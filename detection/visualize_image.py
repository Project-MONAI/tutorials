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

import cv2
import numpy as np


def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def visualize_one_xy_slice_in_3d_image(gt_boxes, image, pred_boxes, gt_box_index=0):
    """
    Prepare a 2D xy-plane image slice from a 3D image for visualization.
    It draws the (gt_box_index)-th GT box and predicted boxes on the same slice.
    The GT box will be green rect overlayed on the image.
    The predicted boxes will be red boxes overlayed on the image.

    Args:
        gt_boxes: numpy sized (M, 6)
        image: image numpy array, sized (H, W, D)
        pred_boxes: numpy array sized (N, 6)
    """
    draw_box = gt_boxes[gt_box_index, :]
    draw_box_center = [
        round((draw_box[axis] + draw_box[axis + 3] - 1) / 2.0) for axis in range(3)
    ]
    draw_box = np.round(draw_box).astype(int).tolist()
    draw_box_z = draw_box_center[2]  # the z-slice we will visualize

    # draw image
    draw_img = normalize_image_to_uint8(image[:, :, draw_box_z])
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)

    # draw GT box, notice that cv2 uses Cartesian indexing instead of Matrix indexing.
    # so the xy position needs to be transposed.
    cv2.rectangle(
        draw_img,
        pt1=(draw_box[1], draw_box[0]),
        pt2=(draw_box[4], draw_box[3]),
        color=(0, 255, 0),  # green for GT
        thickness=1,
    )
    # draw predicted boxes
    for bbox in pred_boxes:
        bbox = np.round(bbox).astype(int).tolist()
        if bbox[5] < draw_box[2] or bbox[2] > draw_box[5]:
            continue
        cv2.rectangle(
            draw_img,
            pt1=(bbox[1], bbox[0]),
            pt2=(bbox[4], bbox[3]),
            color=(255, 0, 0),  # red for predicted box
            thickness=1,
        )
    return draw_img
