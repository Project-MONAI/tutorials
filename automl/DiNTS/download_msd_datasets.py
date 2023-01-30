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
import os

from monai.apps import download_and_extract


def main():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument(
        "--msd_task",
        action="store",
        default="Task07_Pancreas",
        help="msd task",
    )
    parser.add_argument(
        "--root",
        action="store",
        default="./data_msd",
        help="data root",
    )
    args = parser.parse_args()

    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/" + args.msd_task + ".tar"
    compressed_file = os.path.join(args.root, args.msd_task + ".tar")
    if not os.path.exists(args.root):
        download_and_extract(resource, compressed_file, args.root)


if __name__ == "__main__":
    main()
