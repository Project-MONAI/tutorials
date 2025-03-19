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
from typing import Sequence

from monai.apps.utils import extractall
from monai.utils import ensure_tuple_rep


def convert_body_region(body_region: str | Sequence[str]) -> Sequence[int]:
    """
    Convert body region string to body region index.
    Args:
        body_region: list of input body region string. If single str, will be converted to list of str.
    Return:
        body_region_indices, list of input body region index.
    """
    if type(body_region) is str:
        body_region = [body_region]

    # body region mapping for maisi
    region_mapping_maisi = {
        "head": 0,
        "chest": 1,
        "thorax": 1,
        "chest/thorax": 1,
        "abdomen": 2,
        "pelvis": 3,
        "lower": 3,
        "pelvis/lower": 3,
    }

    # perform mapping
    body_region_indices = []
    for region in body_region:
        normalized_region = region.lower()  # norm str to lower case
        if normalized_region not in region_mapping_maisi:
            raise ValueError(f"Invalid region: {normalized_region}")
        body_region_indices.append(region_mapping_maisi[normalized_region])

    return body_region_indices


def find_masks(
    body_region: str | Sequence[str],
    anatomy_list: int | Sequence[int],
    spacing: Sequence[float] | float = 1.0,
    output_size: Sequence[int] = [512, 512, 512],
    check_spacing_and_output_size: bool = False,
    database_filepath: str = "./configs/database.json",
    mask_foldername: str = "./datasets/masks/",
):
    """
    Find candidate masks that fullfills all the requirements.
    They shoud contain all the body region in `body_region`, all the anatomies in `anatomy_list`.
    If there is no tumor specified in `anatomy_list`, we also expect the candidate masks to be tumor free.
    If check_spacing_and_output_size is True, the candidate masks need to have the expected `spacing` and `output_size`.
    Args:
        body_region: list of input body region string. If single str, will be converted to list of str.
            The found candidate mask will include these body regions.
        anatomy_list: list of input anatomy. The found candidate mask will include these anatomies.
        spacing: list of three floats, voxel spacing. If providing a single number, will use it for all the three dimensions.
        output_size: list of three int, expected candidate mask spatial size.
        check_spacing_and_output_size: whether we expect candidate mask to have spatial size of `output_size` and voxel size of `spacing`.
        database_filepath: path for the json file that stores the information of all the candidate masks.
        mask_foldername: directory that saves all the candidate masks.
    Return:
        candidate_masks, list of dict, each dict contains information of one candidate mask that fullfills all the requirements.
    """
    # check and preprocess input
    body_region = convert_body_region(body_region)

    if isinstance(anatomy_list, int):
        anatomy_list = [anatomy_list]

    spacing = ensure_tuple_rep(spacing, 3)

    if not os.path.exists(mask_foldername):
        zip_file_path = mask_foldername + ".zip"

        if not os.path.isfile(zip_file_path):
            raise ValueError(f"Please download {zip_file_path} following the instruction in ./datasets/README.md.")

        print(f"Extracting {zip_file_path} to {os.path.dirname(zip_file_path)}")
        extractall(filepath=zip_file_path, output_dir=os.path.dirname(zip_file_path), file_type="zip")
        print(f"Unzipped {zip_file_path} to {mask_foldername}.")

    if not os.path.isfile(database_filepath):
        raise ValueError(f"Please download {database_filepath} following the instruction in ./datasets/README.md.")
    with open(database_filepath, "r") as f:
        db = json.load(f)

    # select candidate_masks
    candidate_masks = []
    for _item in db:
        if not set(anatomy_list).issubset(_item["label_list"]):
            continue

        # whether to keep this mask, default to be True.
        keep_mask = True

        # extract region indice (top_index and bottom_index) for candidate mask
        include_body_region = "top_region_index" in _item.keys()
        if include_body_region:
            top_index = [index for index, element in enumerate(_item["top_region_index"]) if element != 0]
            top_index = top_index[0]
            bottom_index = [index for index, element in enumerate(_item["bottom_region_index"]) if element != 0]
            bottom_index = bottom_index[0]

            # if candiate mask does not contain all the body_region, skip it
            for _idx in body_region:
                if _idx > bottom_index or _idx < top_index:
                    keep_mask = False

        for tumor_label in [23, 24, 26, 27, 128]:
            # we skip those mask with tumors if users do not provide tumor label in anatomy_list
            if tumor_label not in anatomy_list and tumor_label in _item["label_list"]:
                keep_mask = False

        if check_spacing_and_output_size:
            # if the output_size and spacing are different with user's input, skip it
            for axis in range(3):
                if _item["dim"][axis] != output_size[axis] or _item["spacing"][axis] != spacing[axis]:
                    keep_mask = False

        if keep_mask:
            # if decide to keep this mask, we pack the information of this mask and add to final output.
            candidate = {
                "pseudo_label": os.path.join(mask_foldername, _item["pseudo_label_filename"]),
                "spacing": _item["spacing"],
                "dim": _item["dim"],
            }
            if include_body_region:
                candidate["top_region_index"] = _item["top_region_index"]
                candidate["bottom_region_index"] = _item["bottom_region_index"]

            # Conditionally add the label to the candidate dictionary
            if "label_filename" in _item:
                candidate["label"] = os.path.join(mask_foldername, _item["label_filename"])

            candidate_masks.append(candidate)

    if len(candidate_masks) == 0 and not check_spacing_and_output_size:
        raise ValueError("Cannot find body region with given anatomy list.")

    return candidate_masks
