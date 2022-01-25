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

import argparse
import csv
import json
import nibabel as nib
import numpy as np
import os


def save_obj(vertices, faces, filename):

    with open(filename, "w") as f:

        for v in vertices:
            f.write("v {} {} {}\n".format(*np.array(v)))

        for t in faces:
            f.write("f {} {} {} {}\n".format(*(np.array(t) + 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        action="store",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_dataset_json",
        action="store",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lpi_to_ras",
        action="store_true",
        default=False,
        help="Convert from LPI to RAS",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.input_dataset_json)) as f:
        input_dataset = json.load(f)

    for key in input_dataset.keys():
        section = input_dataset[key]

        for _k in range(len(section)):
            image_name = os.path.join(args.input_dir, section[_k]["image"])
            img = nib.load(image_name)
            affine = img.affine
            affine = np.array(affine)

            if args.lpi_to_ras:
                # LPI to RAS
                new_affine = np.diag(np.array([-1, -1, 1, 1]))@affine
            else:
                new_affine = np.diag(np.array([1, 1, 1, 1]))

            box_data = section[_k]["box"]
            box_filename = section[_k]["image"]
            box_filename = box_filename.replace(".gz", "").replace(".nii", "")

            vertices = []
            faces = []
            _i = 0
            for vec in box_data:
                xmin = vec[0]
                xmax = vec[1]
                ymin = vec[2]
                ymax = vec[3]
                zmin = vec[4]
                zmax = vec[5]

                out = new_affine@np.transpose(np.array([xmin, ymin, zmin, 1]))
                xmin = out[0]
                ymin = out[1]
                zmin = out[2]

                out = new_affine@np.transpose(np.array([xmax, ymax, zmax, 1]))
                xmax = out[0]
                ymax = out[1]
                zmax = out[2]

                vertices += [
                    (xmax, ymax, zmin),
                    (xmax, ymin, zmin),
                    (xmin, ymin, zmin),
                    (xmin, ymax, zmin),
                    (xmax, ymax, zmax),
                    (xmax, ymin, zmax),
                    (xmin, ymin, zmax),
                    (xmin, ymax, zmax),
                ]

                faces += [
                    (0 + 8 * _i, 1 + 8 * _i, 2 + 8 * _i, 3 + 8 * _i),
                    (4 + 8 * _i, 7 + 8 * _i, 6 + 8 * _i, 5 + 8 * _i),
                    (0 + 8 * _i, 4 + 8 * _i, 5 + 8 * _i, 1 + 8 * _i),
                    (1 + 8 * _i, 5 + 8 * _i, 6 + 8 * _i, 2 + 8 * _i),
                    (2 + 8 * _i, 6 + 8 * _i, 7 + 8 * _i, 3 + 8 * _i),
                    (4 + 8 * _i, 0 + 8 * _i, 3 + 8 * _i, 7 + 8 * _i),
                ]

                _i += 1

            save_obj(vertices, faces, os.path.join(args.output_dir, box_filename + ".obj"))

    return


if __name__ == "__main__":
    main()
