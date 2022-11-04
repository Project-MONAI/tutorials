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
""" A demo of overwriting the configuration content using the command line input """

from monai.bundle import ConfigParser


def run(config_file, **kwargs):
    parser = ConfigParser()
    parser.read_config(config_file)

    for k, v in kwargs.items():
        parser[k] = v
    print(parser)
    print(f"new value set: {k}: {parser[k]}")


if __name__ == "__main__":
    """
    Assuming the config_file is at '../configs/net_inferer.yaml
    this function adds support of overwriting the config elements:

        python train_demo.py run --config_file=../configs/net_inferer.yaml --network#in_channels=1

    """
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
