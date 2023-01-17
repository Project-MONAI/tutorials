#!/usr/bin/env python

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

import argparse
import csv
import json
import monai
import nibabel as nib
import numpy as np
import os

from monai.data.box_utils import convert_box_mode


def save_obj(vertices, faces, filename):

    with open(filename, "w") as f:

        for v in vertices:
            f.write("v {} {} {}\n".format(*np.array(v)))

        for t in faces:
            f.write("f {} {} {} {}\n".format(*(np.array(t) + 1)))

    return


def main():
    parser = argparse.ArgumentParser(
        description="Save .obj files of boxes for visualization using 3D Slicer.",
    )
    parser.add_argument(
        "--image_coordinate",
        action="store_true",
        help="if box coordinates in image coordinate",
    )
    parser.add_argument(
        "--image_data_root",
        type=str,
        default="",
        help="image data root",
    )
    parser.add_argument(
        "--input_box_mode",
        action="store",
        type=str,
        required=True,
        help="input box coordinate mode",
    ),
    parser.add_argument(
        "--input_dataset_json",
        action="store",
        type=str,
        required=True,
        help="the dataset .json with box information",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        required=True,
        help="output directory",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.input_dataset_json)) as f:
        input_dataset = json.load(f)

    if args.image_coordinate:
        image_loader = monai.transforms.LoadImage(reader=None, image_only=False)

    for key in input_dataset.keys():
        section = input_dataset[key]

        for _k in range(len(section)):
            box_data = section[_k]["box"]
            box_filename = section[_k]["image"]
            box_filename = box_filename.split(os.sep)[-1]
            print("-- {0:d}th case name:".format(_k + 1), box_filename)

            if args.image_coordinate:
                image_name = os.path.join(args.image_data_root, section[_k]["image"])
                image_data = image_loader(image_name)
                affine = image_data[1]["original_affine"]

                # convert to RAS coordinate system (required by 3D Slicer)
                for _i in range(3):
                    if affine[_i, _i] < 0:
                        affine[_i, _i] *= -1.0
                        affine[_i, 3] *= -1.0

            vertices = []
            faces = []
            _i = 0
            for _vec in box_data:
                vec = convert_box_mode(
                    np.expand_dims(np.array(_vec), axis=0),
                    src_mode=args.input_box_mode,
                    dst_mode="xyzxyz",
                )
                vec = vec.squeeze()
                xmin, ymin, zmin = vec[0], vec[1], vec[2]
                xmax, ymax, zmax = vec[3], vec[4], vec[5]

                if args.image_coordinate:
                    _out = affine @ np.transpose(np.array([xmin, ymin, zmin, 1]))
                    xmin, ymin, zmin = _out[0], _out[1], _out[2]

                    _out = affine @ np.transpose(np.array([xmax, ymax, zmax, 1]))
                    xmax, ymax, zmax = _out[0], _out[1], _out[2]

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

            save_obj(
                vertices, faces, os.path.join(args.output_dir, box_filename + ".obj")
            )

    return


if __name__ == "__main__":
    main()
