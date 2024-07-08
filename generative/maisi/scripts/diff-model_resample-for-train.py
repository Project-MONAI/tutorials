#!/usr/bin/env python

import json
import monai
import multiprocessing
import nibabel as nib
import numpy as np
import os
import torch

from monai.transforms import Compose
from pathlib import Path


def create_transforms(new_dim):
    new_transforms = Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.EnsureTyped(
                keys=["image", "label"], dtype=[torch.float32, torch.short]
            ),
            monai.transforms.Resized(
                keys=["image", "label"],
                spatial_size=new_dim,
                mode=["trilinear", "nearest"],
            ),
        ]
    )
    return new_transforms


def round_number(number):
    new_number = max(round(float(number) / 128.0), 1.0) * 128.0
    new_number = int(new_number)
    return new_number


dataroot = "/mnt/drive2/data"
list_filepath = "/localhome/local-dongy/projects/monai/generative/utils/lists/filenames_nii_common.txt"
output_dir = "/mnt/drive2/data_128"
pl_root = "/mnt/drive2/V2_pseudo_12Feb2024"


transforms = Compose(
    [
        monai.transforms.LoadImaged(keys="image"),
        monai.transforms.EnsureChannelFirstd(keys="image"),
        monai.transforms.Orientationd(keys="image", axcodes="RAS"),
    ]
)


def process_string(data):
    print(f"{data}")

    filepath, index = data

    out_filepath_base = os.path.join(
        output_dir, filepath.replace(".gz", "").replace(".nii", "")
    )
    if os.path.isfile(out_filepath_base + "_image.nii.gz") and os.path.isfile(
        out_filepath_base + "_label.nii.gz"
    ):
        return

    test_data = {"image": os.path.join(dataroot, filepath)}

    if True:
        data = transforms(test_data)
        nda = data["image"]
    else:
        try:
            data = transforms(test_data)
            nda = data["image"]
        except:
            print(test_data)
            return

    dim = nda.meta["dim"]
    dim = dim[1:4]
    dim = [int(dim[_i]) for _i in range(3)]

    spacing = nda.meta["pixdim"]
    spacing = spacing[1:4]
    spacing = [float(spacing[_i]) for _i in range(3)]

    print(dim, spacing)

    new_dim = [round_number(dim[_i]) for _i in range(3)]
    new_dim = tuple(new_dim)
    new_transforms = create_transforms(new_dim)

    new_test_data = {
        "image": os.path.join(dataroot, filepath),
        "label": os.path.join(pl_root, filepath),
    }

    new_data = new_transforms(new_test_data)
    nda_image = new_data["image"]
    nda_label = new_data["label"]

    affine = nda_image.meta["affine"]
    affine = affine.numpy()

    print("new", nda_image.size(), nda_label.size(), affine)

    nda_image = nda_image.numpy().squeeze().astype(np.int16)
    out_filename = out_filepath_base + f"_image.nii.gz"
    out_path = Path(out_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(nda_image, affine=affine)
    nib.save(out_img, out_filename)
    print(f"out_filename: {out_filename}")

    nda_label = nda_label.numpy().squeeze().astype(np.uint8)
    out_filename = out_filepath_base + f"_label.nii.gz"
    out_path = Path(out_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(nda_label, affine=affine)
    nib.save(out_img, out_filename)
    print(f"out_filename: {out_filename}")

    return f"Finished {data}"


if __name__ == "__main__":
    with open(list_filepath, "r") as file:
        filepaths = file.readlines()
    filepaths = [_item.strip() for _item in filepaths]

    filepaths_with_indices = [(filepath, i) for i, filepath in enumerate(filepaths)]

    print(f"multiprocessing.cpu_count(): {multiprocessing.cpu_count()}")
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # pool = multiprocessing.Pool(processes=1)
    results = pool.map(process_string, filepaths_with_indices)

    pool.close()
    pool.join()
