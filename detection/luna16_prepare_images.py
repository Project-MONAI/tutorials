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
import json
import logging
import sys

import monai
import torch
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    SaveImaged,
    Spacingd,
)


def main():
    parser = argparse.ArgumentParser(description="LUNA16 Detection Image Resampling")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment_luna16_prepare.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_luna16_16g.json",
        help="config json file that stores hyper-parameters",
    )
    args = parser.parse_args()

    monai.config.print_config()

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    # 1. define transform
    # resample images to args.spacing defined in args.config_file.
    process_transforms = Compose(
        [
            LoadImaged(
                keys=["image"],
                image_only=False,
                meta_key_postfix="meta_dict",
                reader="itkreader",
                affine_lps_to_ras=True,
            ),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float16),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=args.spacing, padding_mode="border"),
        ]
    )
    # saved images to Nifti
    post_transforms = Compose(
        [
            SaveImaged(
                keys="image",
                meta_keys="image_meta_dict",
                output_dir=args.data_base_dir,
                output_postfix="",
                resample=False,
            ),
        ]
    )

    # 2. prepare data
    for data_list_key in ["training", "validation"]:
        # create a data loader
        process_data = load_decathlon_datalist(
            args.data_list_file_path,
            is_segmentation=True,
            data_list_key=data_list_key,
            base_dir=args.orig_data_base_dir,
        )
        process_ds = Dataset(
            data=process_data,
            transform=process_transforms,
        )
        process_loader = DataLoader(
            process_ds,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            collate_fn=no_collation,
        )

        print("-" * 10)
        for batch_data in process_loader:
            for batch_data_i in batch_data:
                batch_data_i = post_transforms(batch_data_i)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
