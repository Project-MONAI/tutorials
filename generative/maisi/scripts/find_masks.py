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
import zipfile


def convert_body_region(
    body_region: list[int],
):
    body_region_indices = []

    for _k in range(len(body_region)):
        region = body_region[_k].lower()

        idx = None
        if "head" in region:
            idx = 0
        elif "chest" in region or "thorax" in region or "chest/thorax" in region:
            idx = 1
        elif "abdomen" in region:
            idx = 2
        elif "pelvis" in region or "lower" in region or "pelvis/lower" in region:
            idx = 3
        else:
            raise ValueError("Input region information is incorrect.")

        body_region_indices.append(idx)

    return body_region_indices


def find_masks(
    body_region: str | list[str],
    anatomy_list: int | list[int],
    spacing: list[float],
    output_size: list[int],
    check_spacing_and_output_size: bool = False,
    database_filepath: str = "./database.json",
    mask_foldername: str = "./masks",
):
    if type(body_region) is str:
        body_region = [body_region]

    body_region = convert_body_region(body_region)

    if type(anatomy_list) == int:
        anatomy_list = [anatomy_list]

    if not os.path.isfile(database_filepath):
        raise ValueError(f"Please download {database_filepath}.")

    if not os.path.exists(mask_foldername):
        zip_file_path = mask_foldername + ".zip"

        if not os.path.isfile(zip_file_path):
            raise ValueError(f"Please downloaded {zip_file_path}.")

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            print(mask_foldername)
            zip_ref.extractall(path="./datasets")
        print(f"Unzipped {zip_file_path} to {mask_foldername}.")

    with open(database_filepath, "r") as f:
        db = json.load(f)

    candidates = []
    for _i in range(len(db)):
        _item = db[_i]
        if not set(anatomy_list).issubset(_item["label_list"]):
            continue

        top_index = [index for index, element in enumerate(_item["top_region_index"]) if element != 0]
        top_index = top_index[0]
        bottom_index = [index for index, element in enumerate(_item["bottom_region_index"]) if element != 0]
        bottom_index = bottom_index[0]

        flag = False
        for _idx in body_region:
            if _idx > bottom_index or _idx < top_index:
                flag = True

        # check if candiate mask contains tumors
        for tumor_label in [23, 24, 26, 27, 128]:
            # we skip those mask with tumors if users do not provide tumor label in anatomy_list
            if tumor_label not in anatomy_list and tumor_label in _item["label_list"]:
                flag = True

        if check_spacing_and_output_size:
            # check if the output_size and spacing are same as user's input
            for axis in range(3):
                if _item["dim"][axis] != output_size[axis] or _item["spacing"][axis] != spacing[axis]:
                    flag = True

        if flag == True:
            continue

        candidate = {}
        if "label_filename" in _item:
            candidate["label"] = os.path.join(mask_foldername, _item["label_filename"])
        candidate["pseudo_label"] = os.path.join(mask_foldername, _item["pseudo_label_filename"])
        candidate["spacing"] = _item["spacing"]
        candidate["dim"] = _item["dim"]
        candidate["top_region_index"] = _item["top_region_index"]
        candidate["bottom_region_index"] = _item["bottom_region_index"]

        candidates.append(candidate)

    if len(candidates) == 0 and not check_spacing_and_output_size:
        raise ValueError("Cannot find body region with given organ list.")

    return candidates
