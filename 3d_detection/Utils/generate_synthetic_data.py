#!/usr/bin/env python

# Copyright 2021 - 2022 MONAI Consortium
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
import monai
import nibabel as nib
import numpy as np
import os

from skimage.measure import label, regionprops


def generate_data(prefix, num_vols):

    output_dict = []

    for _i in range(num_vols):
        _, nda_labels = monai.data.synthetic.create_test_image_3d(
            height = 128,
            width = 128,
            depth = 128,
            num_objs = 6,
            rad_max = 18,
            rad_min = 6,
            noise_max = 0.0,
            num_seg_classes = 1,
            channel_dim = None,
            random_state = None,
        )
        nda_labels = (nda_labels > 0).astype(np.uint8)
        nda_labels = label(nda_labels)
        print(np.mean(nda_labels), np.unique(nda_labels))

        data_point = {}
        data_point["image"] = prefix + "_" + str(_i) + ".nii.gz"
        data_point["label"] = []
        data_point["box"] = []

        props = regionprops(nda_labels)
        for _j in range(len(props)):
            bbox = props[_j].bbox

            data_point["label"].extend([0])
            data_point["box"].append(
                [
                    bbox[0],
                    bbox[3],
                    bbox[1],
                    bbox[4],
                    bbox[2],
                    bbox[5],
                ]
            )

        output_dict.append(data_point)

        img = nib.Nifti1Image((nda_labels > 0).astype(np.uint8), np.eye(4))
        nib.save(img, prefix + "_" + str(_i) + ".nii.gz")

    return output_dict

def main():
    os.system("clear")
    monai.utils.set_determinism(seed=0, additional_settings=None)

    json_data = {}
    for key in ["training", "validation"]:
        json_data[key] = generate_data(
            prefix=key,
            num_vols=5,
        )

    with open("dataset_synthetic.json", "w") as outfile:
        json.dump(json_data, outfile, sort_keys=True, indent=4, ensure_ascii=False)

    return

if __name__ == "__main__":
    main()
