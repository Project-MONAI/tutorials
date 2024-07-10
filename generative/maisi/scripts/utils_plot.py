# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import AsDiscrete


def find_label_center_loc(x):
    """
    Find the center location of non-zero elements in a binary mask.

    Args:
        x (torch.Tensor): Binary mask tensor. Expected shape: [H, W, D] or [C, H, W, D].

    Returns:
        list: Center locations for each dimension. Each element is either
              the middle index of non-zero locations or None if no non-zero elements exist.
    """
    label_loc = torch.where(x != 0)
    center_loc = []
    for loc in label_loc:
        unique_loc = torch.unique(loc)
        if len(unique_loc) == 0:
            center_loc.append(None)
        else:
            center_loc.append(unique_loc[len(unique_loc) // 2])

    return center_loc


def normalize_label_to_uint8(colorize, label, n_label):
    """
    Normalize and colorize a label tensor to a uint8 image.

    Args:
        colorize (torch.Tensor): Weight tensor for colorization. Expected shape: [3, n_label, 1, 1].
        label (torch.Tensor): Input label tensor. Expected shape: [1, H, W].
        n_label (int): Number of unique labels.

    Returns:
        numpy.ndarray: Normalized and colorized image as uint8 numpy array. Shape: [H, W, 3].
    """
    with torch.no_grad():
        post_label = AsDiscrete(to_onehot=n_label)
        label = post_label(label).permute(1, 0, 2, 3)
        label = F.conv2d(label, weight=colorize)
        label = torch.clip(label, 0, 1).squeeze().permute(1, 2, 0).cpu().numpy()

    draw_img = (label * 255).astype(np.uint8)

    return draw_img


def visualize_one_slice_in_3d(image, axis: int = 2, center=None, mask_bool=True, n_label=105, colorize=None):
    """
    Extract and visualize a 2D slice from a 3D image or label tensor.

    Args:
        image (torch.Tensor): Input 3D image or label tensor. Expected shape: [1, H, W, D].
        axis (int, optional): Axis along which to extract the slice (0, 1, or 2). Defaults to 2.
        center (int, optional): Index of the slice to extract. If None, the middle slice is used.
        mask_bool (bool, optional): If True, treat the input as a label mask and normalize it. Defaults to True.
        n_label (int, optional): Number of labels in the mask. Used only if mask_bool is True. Defaults to 105.
        colorize (torch.Tensor, optional): Colorization weights for label normalization.
                                           Expected shape: [3, n_label, 1, 1] if provided.

    Returns:
        numpy.ndarray: 2D slice of the input. If mask_bool is True, returns a normalized uint8 array
                       with shape [3, H, W]. Otherwise, returns a float32 array with shape [3, H, W].

    Raises:
        ValueError: If the specified axis is not 0, 1, or 2.
    """
    # draw image
    if center is None:
        center = image.shape[2:][axis] // 2
    if axis == 0:
        draw_img = image[..., center, :, :]
    elif axis == 1:
        draw_img = image[..., :, center, :]
    elif axis == 2:
        draw_img = image[..., :, :, center]
    else:
        raise ValueError("axis should be in [0,1,2]")
    if mask_bool:
        draw_img = normalize_label_to_uint8(colorize, draw_img, n_label)
    else:
        draw_img = draw_img.squeeze().cpu().numpy().astype(np.float32)
        draw_img = np.stack((draw_img,) * 3, axis=-1)
    return draw_img


def show_image(image, title="mask"):
    """
    Plot and display an input image.

    Args:
        image (numpy.ndarray): Image to be displayed. Expected shape: [H, W] for grayscale or [H, W, 3] for RGB.
        title (str, optional): Title for the plot. Defaults to "mask".
    """
    plt.figure("check", (24, 12))
    plt.subplot(1, 2, 1)
    plt.title(title)
    plt.imshow(image)
    plt.show()


def to_shape(a, shape):
    """
    Pad an image to a desired shape.

    This function pads a 3D numpy array (image) with zeros to reach the specified shape.
    The padding is added equally on both sides of each dimension, with any odd padding
    added to the end.

    Args:
        a (numpy.ndarray): Input 3D array to be padded. Expected shape: [X, Y, Z].
        shape (tuple): Desired output shape as (x_, y_, z_).

    Returns:
        numpy.ndarray: Padded array with the desired shape [x_, y_, z_].

    Note:
        If the input shape is larger than the desired shape in any dimension,
        no padding is removed; the original size is maintained for that dimension.
        Padding is done using numpy's pad function with 'constant' mode (zero-padding).
    """
    x_, y_, z_ = shape
    x, y, z = a.shape
    x_pad = x_ - x
    y_pad = y_ - y
    z_pad = z_ - z
    return np.pad(
        a,
        (
            (x_pad // 2, x_pad // 2 + x_pad % 2),
            (y_pad // 2, y_pad // 2 + y_pad % 2),
            (z_pad // 2, z_pad // 2 + z_pad % 2),
        ),
        mode="constant",
    )


def get_xyz_plot(image, center_loc_axis, mask_bool=True, n_label=105, colorize=None, target_class_index=0):
    """
    Generate a concatenated XYZ plot of 2D slices from a 3D image.

    This function creates visualizations of three orthogonal slices (XY, XZ, YZ) from a 3D image
    and concatenates them into a single 2D image.

    Args:
        image (torch.Tensor): Input 3D image tensor. Expected shape: [1, H, W, D].
        center_loc_axis (list): List of three integers specifying the center locations for each axis.
        mask_bool (bool, optional): Whether to apply masking. Defaults to True.
        n_label (int, optional): Number of labels for visualization. Defaults to 105.
        colorize (torch.Tensor, optional): Colorization weights. Expected shape: [3, n_label, 1, 1] if provided.
        target_class_index (int, optional): Index of the target class. Defaults to 0.

    Returns:
        numpy.ndarray: Concatenated 2D image of the three orthogonal slices. Shape: [max(H,W,D), 3*max(H,W,D), 3].

    Note:
        The output image is padded to ensure all slices have the same dimensions.
    """
    target_shape = list(image.shape[1:])  # [1,H,W,D]
    img_list = []

    for axis in range(3):
        center = center_loc_axis[axis]

        img = visualize_one_slice_in_3d(
            torch.flip(image.unsqueeze(0), [-3, -2, -1]),
            axis,
            center=center,
            mask_bool=mask_bool,
            n_label=n_label,
            colorize=colorize,
        )
        img = img.transpose([2, 1, 0])

        img = to_shape(img, (3, max(target_shape), max(target_shape)))
        img_list.append(img)
        img = np.concatenate(img_list, axis=2).transpose([1, 2, 0])
    return img
