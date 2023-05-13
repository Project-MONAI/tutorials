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

from monai.config import print_config
from monai.utils import set_determinism
from monai.apps import DecathlonDataset


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    args = parser.parse_args()

    # Step 0: configuration
    print_config()

    env_dict = json.load(open(args.environment_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    # Step 1: set data loader
    train_ds = DecathlonDataset(
        root_dir=args.data_base_dir,
        task="Task01_BrainTumour",
        section="training",  # validation
        cache_rate=0.0,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=8,
        download=True,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
    )
    val_ds = DecathlonDataset(
        root_dir=args.data_base_dir,
        task="Task01_BrainTumour",
        section="validation",  # validation
        cache_rate=0.0,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=8,
        download=True,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
    )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
