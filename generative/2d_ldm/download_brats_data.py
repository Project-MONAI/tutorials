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
import os
import shutil
import sys
import tempfile

from monai.apps import download_and_extract
from monai.config import print_config


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

    # Step 1: set data loader
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory

    msd_task = "Task01_BrainTumour"
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/" + msd_task + ".tar"
    compressed_file = os.path.join(root_dir, msd_task + ".tar")

    os.makedirs(args.data_base_dir, exist_ok=True)
    download_and_extract(resource, compressed_file, args.data_base_dir)

    if directory is None:
        shutil.rmtree(root_dir)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
