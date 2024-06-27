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

from typing import Sequence

import torch
import torch.nn.functional as F
import copy
import numpy as np
import skimage
from scipy import stats

from monai.utils import (
    TransformBackends,
    convert_data_type,
    convert_to_dst_type,
    get_equivalent_dtype,
    ensure_tuple_rep,
)
from monai.config import DtypeLike, NdarrayOrTensor
from monai.bundle import ConfigParser


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def get_index_arr(img):
    return np.moveaxis(
        np.moveaxis(
            np.stack(np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]))), 0, 3
        ),
        0,
        1,
    )


def supress_non_largest_components(img, target_label, default_val=0):
    """As a last step, supress all non largest components"""
    index_arr = get_index_arr(img)
    img_mod = copy.deepcopy(img)
    new_background = np.zeros(img.shape, dtype=np.bool_)
    for label in target_label:
        label_cc = skimage.measure.label(img == label, connectivity=3)
        uv, uc = np.unique(label_cc, return_counts=True)
        dominant_vals = uv[np.argsort(uc)[::-1][:2]]
        if len(dominant_vals) >= 2:  # Case: no predictions
            new_background = np.logical_or(
                new_background,
                np.logical_not(np.logical_or(label_cc == dominant_vals[0], label_cc == dominant_vals[1])),
            )

    for voxel in index_arr[new_background]:
        img_mod[tuple(voxel)] = default_val
    diff = np.sum((img - img_mod) > 0)

    return img_mod, diff


def erode3d(input_tensor, erosion=3, value=0.0):
    # Define the structuring element
    erosion = ensure_tuple_rep(erosion, 3)
    structuring_element = torch.ones(1, 1, erosion[0], erosion[1], erosion[2]).to(input_tensor.device)

    # Pad the input tensor to handle border pixels
    input_padded = F.pad(
        input_tensor.float().unsqueeze(0).unsqueeze(0),
        (erosion[0] // 2, erosion[0] // 2, erosion[1] // 2, erosion[1] // 2, erosion[2] // 2, erosion[2] // 2),
        mode="constant",
        value=value,
    )

    # Apply erosion operation
    output = F.conv3d(input_padded, structuring_element, padding=0)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output == torch.sum(structuring_element), 1.0, 0.0)

    return output.squeeze(0).squeeze(0)


def dilate3d(input_tensor, erosion=3, value=0.0):
    # Define the structuring element
    erosion = ensure_tuple_rep(erosion, 3)
    structuring_element = torch.ones(1, 1, erosion[0], erosion[1], erosion[2]).to(input_tensor.device)

    # Pad the input tensor to handle border pixels
    input_padded = F.pad(
        input_tensor.float().unsqueeze(0).unsqueeze(0),
        (erosion[0] // 2, erosion[0] // 2, erosion[1] // 2, erosion[1] // 2, erosion[2] // 2, erosion[2] // 2),
        mode="constant",
        value=value,
    )

    # Apply erosion operation
    output = F.conv3d(input_padded, structuring_element, padding=0)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output > 0, 1.0, 0.0)

    return output.squeeze(0).squeeze(0)


def organ_fill_by_closing(data, target_label, device):
    mask = (data == target_label).astype(np.uint8)
    mask = dilate3d(torch.from_numpy(mask).to(device), erosion=3, value=0.0)
    mask = erode3d(mask, erosion=3, value=0.0)
    mask = dilate3d(mask, erosion=3, value=0.0)
    mask = erode3d(mask, erosion=3, value=0.0).cpu().numpy()
    return mask.astype(np.bool_)


def organ_fill_by_removed_mask(data, target_label, remove_mask, device):
    mask = (data == target_label).astype(np.uint8)
    mask = dilate3d(torch.from_numpy(mask).to(device), erosion=3, value=0.0)
    mask = dilate3d(mask, erosion=3, value=0.0)
    roi_oragn_mask = dilate3d(mask, erosion=3, value=0.0).cpu().numpy()
    return (roi_oragn_mask * remove_mask).astype(np.bool_)


def get_body_region_index_from_mask(input_mask):
    region_indices = {}
    # head and neck
    region_indices["region_0"] = [22, 120]
    # thorax
    region_indices["region_1"] = [28, 29, 30, 31, 32]
    # abdomen
    region_indices["region_2"] = [1, 2, 3, 4, 5, 14]
    # pelvis and lower
    region_indices["region_3"] = [93, 94]

    nda = input_mask.cpu().numpy().squeeze()
    unique_elements = np.lib.arraysetops.unique(nda)
    unique_elements = list(unique_elements)
    print(f"nda: {nda.shape} {unique_elements}.")
    overlap_array = np.zeros(len(region_indices), dtype=np.uint8)
    for _j in range(len(region_indices)):
        overlap = any(element in region_indices[f"region_{_j}"] for element in unique_elements)
        overlap_array[_j] = np.uint8(overlap)
    overlap_array_indices = np.nonzero(overlap_array)[0]
    top_region_index = np.eye(len(region_indices), dtype=np.uint8)[np.amin(overlap_array_indices), ...]
    top_region_index = list(top_region_index)
    top_region_index = [int(_k) for _k in top_region_index]
    bottom_region_index = np.eye(len(region_indices), dtype=np.uint8)[np.amax(overlap_array_indices), ...]
    bottom_region_index = list(bottom_region_index)
    bottom_region_index = [int(_k) for _k in bottom_region_index]
    print(f"{top_region_index} {bottom_region_index}")
    return top_region_index, bottom_region_index


def general_mask_generation_post_process(volume_t, target_tumor_label=None, device="cuda:0"):
    # assume volume_t is np array with shape (H,W,D)
    hepatic_vessel = volume_t == 25
    airway = volume_t == 132

    # ------------ refine body mask pred
    body_region_mask = erode3d(torch.from_numpy((volume_t > 0)).to(device), erosion=3, value=0.0).cpu().numpy()
    body_region_mask, _ = supress_non_largest_components(body_region_mask, [1])
    body_region_mask = (
        dilate3d(torch.from_numpy(body_region_mask).to(device), erosion=3, value=0.0).cpu().numpy().astype(np.uint8)
    )
    volume_t = volume_t * body_region_mask

    # ------------ refine tumor pred
    tumor_organ_dict = {23: 28, 24: 4, 26: 1, 27: 62, 128: 200}
    for t in [23, 24, 26, 27, 128]:
        if t != target_tumor_label:
            volume_t[volume_t == t] = tumor_organ_dict[t]
        else:
            volume_t[organ_fill_by_closing(volume_t, target_label=t, device=device)] = t
            volume_t[organ_fill_by_closing(volume_t, target_label=t, device=device)] = t
    # we only keep the largest connected componet for tumors except hepatic tumor and bone lesion
    if target_tumor_label != 26 and target_tumor_label != 128:
        volume_t, _ = supress_non_largest_components(volume_t, [target_tumor_label], default_val=200)
    target_tumor = volume_t == target_tumor_label

    # ------------ remove undesired organ pred
    # general post-process non-largest components suppression
    # process 4 ROI organs + spleen + 2 kidney + 5 lung lobes + duodenum + inferior vena cava
    oran_list = [1, 4, 10, 12, 3, 28, 29, 30, 31, 32, 5, 14, 13, 6, 7, 8, 9, 10]
    if target_tumor_label != 128:
        oran_list += list(range(33, 60))  # + list(range(63,87))
    data, _ = supress_non_largest_components(volume_t, oran_list, default_val=200)  # 200 is body region
    organ_remove_mask = (volume_t - data).astype(np.bool_)
    # process intestinal system (stomach 12, duodenum 13, small bowel 19, colon 62)
    intestinal_mask_ = (
        (data == 12).astype(np.uint8)
        + (data == 13).astype(np.uint8)
        + (data == 19).astype(np.uint8)
        + (data == 62).astype(np.uint8)
    )
    intestinal_mask, _ = supress_non_largest_components(intestinal_mask_, [1], default_val=0)
    # process small bowel 19
    small_bowel_remove_mask = (data == 19).astype(np.uint8) - (data == 19).astype(np.uint8) * intestinal_mask
    # process colon 62
    colon_remove_mask = (data == 62).astype(np.uint8) - (data == 62).astype(np.uint8) * intestinal_mask
    intestinal_remove_mask = (small_bowel_remove_mask + colon_remove_mask).astype(np.bool_)
    data[intestinal_remove_mask] = 200

    # ------------ full correponding organ in removed regions
    for organ_label in oran_list:
        data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label

    if target_tumor_label == 23 and np.sum(target_tumor) > 0:
        # speical process for cases with lung tumor
        dia_lung_tumor_mask = dilate3d(torch.from_numpy((data == 23)).to(device), erosion=3, value=0.0).cpu().numpy()
        tmp = (
            (data * (dia_lung_tumor_mask.astype(np.uint8) - (data == 23).astype(np.uint8))).astype(np.float32).flatten()
        )
        tmp[tmp == 0] = float("nan")
        mode = int(stats.mode(tmp.flatten(), nan_policy="omit")[0])
        if mode in [28, 29, 30, 31, 32]:
            dia_lung_tumor_mask = (
                dilate3d(torch.from_numpy(dia_lung_tumor_mask).to(device), erosion=3, value=0.0).cpu().numpy()
            )
            lung_remove_mask = dia_lung_tumor_mask.astype(np.uint8) - (data == 23).astype(np.uint8).astype(np.uint8)
            data[
                organ_fill_by_removed_mask(data, target_label=mode, remove_mask=lung_remove_mask, device=device)
            ] = mode
        dia_lung_tumor_mask = (
            dilate3d(torch.from_numpy(dia_lung_tumor_mask).to(device), erosion=3, value=0.0).cpu().numpy()
        )
        data[
            organ_fill_by_removed_mask(
                data, target_label=23, remove_mask=dia_lung_tumor_mask * organ_remove_mask, device=device
            )
        ] = 23
        for organ_label in [28, 29, 30, 31, 32]:
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label

    if target_tumor_label == 26 and np.sum(target_tumor) > 0:
        # speical process for cases with hepatic tumor
        # process liver 1
        data[organ_fill_by_removed_mask(data, target_label=1, remove_mask=intestinal_remove_mask, device=device)] = 1
        data[organ_fill_by_removed_mask(data, target_label=1, remove_mask=intestinal_remove_mask, device=device)] = 1
        # process spleen 2
        data[organ_fill_by_removed_mask(data, target_label=3, remove_mask=organ_remove_mask, device=device)] = 3
        data[organ_fill_by_removed_mask(data, target_label=3, remove_mask=organ_remove_mask, device=device)] = 3
        dia_tumor_mask = (
            dilate3d(torch.from_numpy((data == target_tumor_label)).to(device), erosion=3, value=0.0).cpu().numpy()
        )
        dia_tumor_mask = dilate3d(torch.from_numpy(dia_tumor_mask).to(device), erosion=3, value=0.0).cpu().numpy()
        data[
            organ_fill_by_removed_mask(
                data, target_label=target_tumor_label, remove_mask=dia_tumor_mask * organ_remove_mask, device=device
            )
        ] = target_tumor_label
        # refine hepatic tumor
        hepatic_tumor_vessel_liver_mask_ = (
            (data == 26).astype(np.uint8) + (data == 25).astype(np.uint8) + (data == 1).astype(np.uint8)
        )
        hepatic_tumor_vessel_liver_mask_ = (hepatic_tumor_vessel_liver_mask_ > 1).astype(np.uint8)
        hepatic_tumor_vessel_liver_mask, _ = supress_non_largest_components(
            hepatic_tumor_vessel_liver_mask_, [1], default_val=0
        )
        removed_region = (hepatic_tumor_vessel_liver_mask_ - hepatic_tumor_vessel_liver_mask).astype(np.bool_)
        data[removed_region] = 200
        target_tumor = (target_tumor * hepatic_tumor_vessel_liver_mask).astype(np.bool_)
        # refine liver
        data[organ_fill_by_closing(data, target_label=1, device=device)] = 1
        data[organ_fill_by_closing(data, target_label=1, device=device)] = 1
        data[organ_fill_by_closing(data, target_label=1, device=device)] = 1

    if target_tumor_label == 27 and np.sum(target_tumor) > 0:
        # speical process for cases with colon tumor
        dia_tumor_mask = (
            dilate3d(torch.from_numpy((data == target_tumor_label)).to(device), erosion=3, value=0.0).cpu().numpy()
        )
        dia_tumor_mask = dilate3d(torch.from_numpy(dia_tumor_mask).to(device), erosion=3, value=0.0).cpu().numpy()
        data[
            organ_fill_by_removed_mask(
                data, target_label=target_tumor_label, remove_mask=dia_tumor_mask * organ_remove_mask, device=device
            )
        ] = target_tumor_label

    if target_tumor_label == 129 and np.sum(target_tumor) > 0:
        # speical process for cases with kidney tumor
        for organ_label in [5, 14]:
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label
            data[organ_fill_by_closing(data, target_label=organ_label, device=device)] = organ_label
    # TODO: current model does not support hepatic vessel by size control.
    # we treat it as liver for better visiaulization
    print(
        "Current model does not support hepatic vessel by size control, "
        "so we treat generated hepatic vessel as part of liver for better visiaulization."
    )
    data[hepatic_vessel] = 1
    data[airway] = 132
    if target_tumor_label is not None:
        data[target_tumor] = target_tumor_label

    return data


class MapLabelValue:
    """
    Utility to map label values to another set of values.
    For example, map [3, 2, 1] to [0, 1, 2], [1, 2, 3] -> [0.5, 1.5, 2.5], ["label3", "label2", "label1"] -> [0, 1, 2],
    [3.5, 2.5, 1.5] -> ["label0", "label1", "label2"], etc.
    The label data must be numpy array or array-like data and the output data will be numpy array.

    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, orig_labels: Sequence, target_labels: Sequence, dtype: DtypeLike = np.float32) -> None:
        """
        Args:
            orig_labels: original labels that map to others.
            target_labels: expected label values, 1: 1 map to the `orig_labels`.
            dtype: convert the output data to dtype, default to float32.
                if dtype is from PyTorch, the transform will use the pytorch backend, else with numpy backend.

        """
        if len(orig_labels) != len(target_labels):
            raise ValueError("orig_labels and target_labels must have the same length.")

        self.orig_labels = orig_labels
        self.target_labels = target_labels
        self.pair = tuple((o, t) for o, t in zip(self.orig_labels, self.target_labels) if o != t)
        type_dtype = type(dtype)
        if getattr(type_dtype, "__module__", "") == "torch":
            self.use_numpy = False
            self.dtype = get_equivalent_dtype(dtype, data_type=torch.Tensor)
        else:
            self.use_numpy = True
            self.dtype = get_equivalent_dtype(dtype, data_type=np.ndarray)

    def __call__(self, img: NdarrayOrTensor):
        if self.use_numpy:
            img_np, *_ = convert_data_type(img, np.ndarray)
            _out_shape = img_np.shape
            img_flat = img_np.flatten()
            try:
                out_flat = img_flat.astype(self.dtype)
            except ValueError:
                # can't copy unchanged labels as the expected dtype is not supported, must map all the label values
                out_flat = np.zeros(shape=img_flat.shape, dtype=self.dtype)
            for o, t in self.pair:
                out_flat[img_flat == o] = t
            out_t = out_flat.reshape(_out_shape)
        else:
            img_t, *_ = convert_data_type(img, torch.Tensor)
            out_t = img_t.detach().clone().to(self.dtype)  # type: ignore
            for o, t in self.pair:
                out_t[img_t == o] = t
        out, *_ = convert_to_dst_type(src=out_t, dst=img, dtype=self.dtype)
        return out


def load_autoencoder_ckpt(load_autoencoder_path):
    checkpoint_autoencoder = torch.load(load_autoencoder_path)
    new_state_dict = {}
    for k, v in checkpoint_autoencoder.items():
        if "decoder" in k and "conv" in k:
            new_key = (
                k.replace("conv.weight", "conv.conv.weight")
                if "conv.weight" in k
                else k.replace("conv.bias", "conv.conv.bias")
            )
            new_state_dict[new_key] = v
        elif "encoder" in k and "conv" in k:
            new_key = (
                k.replace("conv.weight", "conv.conv.weight")
                if "conv.weight" in k
                else k.replace("conv.bias", "conv.conv.bias")
            )
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    checkpoint_autoencoder = new_state_dict
    return checkpoint_autoencoder


def binarize_labels(x, bits=8):
    """
    x: the input tensor with shape (B, 1, H, W, D)
    bits: the num of channel to represent the data.
    """
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().squeeze(1).permute(0, 4, 1, 2, 3)
