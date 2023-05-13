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

import torch
from monai.config import print_config
from monai.utils import set_determinism
from utils import prepare_brats2d_dataloader


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    args = parser.parse_args()

    # Step 0: configuration
    rank = 0
    world_size = 1
    device = 0

    torch.cuda.set_device(device)
    print(f"Using {device}")

    print_config()

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    # Step 1: set data loader
    size_divisible = 2 ** (len(args.autoencoder_def["num_channels"]) - 1)
    train_loader, val_loader = prepare_brats2d_dataloader(
        args,
        args.autoencoder_train["batch_size"],
        args.autoencoder_train["patch_size"],
        sample_axis=args.sample_axis,
        randcrop=True,
        rank=rank,
        world_size=world_size,
        cache=0.0,
        download=True,
        size_divisible=size_divisible,
    )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
