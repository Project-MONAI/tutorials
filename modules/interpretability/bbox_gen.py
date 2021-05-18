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
"""
Script used to generate lesion/non-lesion patches from MSD lung.
(requires numpy/torch/skimage.measure/monai)

The original dataset could be downloaded via the MONAI API:

    import monai
    monai.apps.DecathlonDataset(root_dir="./", task="Task06_Lung", section="training", download=True)

- After running these command, the dataset will be downloaded and unzipped at `root_dir`.
- Lesion bounding boxes are generated from the connected components in the segmentation masks.
- Non-lesion patches are generated from randomly from the volume.

"""

import glob
import os
import sys

import numpy as np
import skimage.measure as measure
from monai.data import write_nifti
from monai.transforms import (
    AddChanneld,
    BoundingRect,
    LoadImaged,
    RandWeightedCropd,
    Resized,
    SpatialCropd,
)
from monai.utils import set_determinism

# optionally give folder
folder = sys.argv[1] if len(sys.argv) > 1 else "."
# create output folder
os.makedirs(folder + "/patch", exist_ok=True)
set_determinism(0)

image_names = sorted(glob.glob(folder + "/Task06_Lung/imagesTr/*"))
label_names = sorted(glob.glob(folder + "/Task06_Lung/labelsTr/*"))
if len(image_names) * len(label_names) == 0:
    raise AssertionError("no images and/or labels found")

data_names = [{"label": ll, "image": ii} for ll, ii in zip(label_names, image_names)]

patch_size = (72, 72, 48)

for name in data_names:
    print(f"---on {name['label']}---")
    name_id = os.path.basename(name["label"])
    keys = ("image", "label")
    data = LoadImaged(keys)(name)
    labels, n_comp = measure.label(data["label"] == 1, connectivity=3, return_num=True)
    print("total components", n_comp)
    for i in range(n_comp + 1):
        if i == 0:
            continue  # skipping background
        b_label = labels == i
        bb = BoundingRect()(b_label[None])
        area = (bb[0, 1] - bb[0, 0]) * (bb[0, 3] - bb[0, 2]) * (bb[0, 5] - bb[0, 4])
        if area <= 500:
            continue
        print(bb, area)
        s = [bb[0, 0] - 16, bb[0, 2] - 16, bb[0, 4] - 16]
        e = [bb[0, 1] + 16, bb[0, 3] + 16, bb[0, 5] + 16]

        # generate lesion patches based on the bounding boxes
        data_out = AddChanneld(keys)(data)
        data_out = SpatialCropd(keys, roi_start=s, roi_end=e)(data_out)
        resize = Resized(keys, patch_size, mode=("trilinear", "nearest"))
        data_out = resize(data_out)

        patch_out = (
            f"{folder}/patch/lesion_{s[0]}_{s[1]}_{s[2]}_{e[0]}_{e[1]}_{e[2]}_{name_id}"
        )
        label_out = (
            f"{folder}/patch/labels_{s[0]}_{s[1]}_{s[2]}_{e[0]}_{e[1]}_{e[2]}_{name_id}"
        )
        write_nifti(data_out["image"][0], file_name=patch_out)
        write_nifti(data_out["label"][0], file_name=label_out)

        # generate random negative samples
        rand_data_out = AddChanneld(keys)(data)
        rand_data_out["inv_label"] = (
            rand_data_out["label"] == 0
        )  # non-lesion sampling map
        rand_crop = RandWeightedCropd(keys, "inv_label", patch_size, num_samples=3)
        rand_data_out = rand_crop(rand_data_out)
        for idx, d in enumerate(rand_data_out):
            if np.sum(d["label"]) > 0:
                continue
            write_nifti(d["image"][0], file_name=f"{folder}/patch/norm_{idx}_{name_id}")
