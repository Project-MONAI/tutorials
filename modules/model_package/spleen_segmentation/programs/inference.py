
# Copyright 2020 - 2021 MONAI Consortium
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

import torch
from monai.data import decollate_batch
from monai.inferers import Inferer
from monai.transforms import Transform
from monai.utils.enums import CommonKeys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='file path of config file that defines network', required=True)
    parser.add_argument('--meta', '-e', type=str, help='file path of the meta data')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file
    with open(args.config, "r") as f:
        cofnig_dict = json.load(f)
    # load meta data
    with open(args.meta, "r") as f:
        meta_dict = json.load(f)

    net: torch.nn.Module = None
    dataloader: torch.utils.data.DataLoader = None
    inferer: Inferer = None
    post_transforms: Transform = None
    # TODO: parse inference config file and construct instances
    # config_parser = ConfigParser(config_dict, meta_dict)
    # net = config_parser.get_component("model").to(device)
    # dataloader = config_parser.get_component("dataloader")
    # inferer = config_parser.get_component("inferer")
    # post_transforms = config_parser.get_component("post_transforms")

    net.eval()
    with torch.no_grad():
        for d in dataloader:
            images = d[CommonKeys.IMAGE].to(device)
            # define sliding window size and batch size for windows inference
            d[CommonKeys.PRED] = inferer(inputs=images, predictor=net)
            # decollate the batch data into a list of dictionaries, then execute postprocessing transforms
            d = [post_transforms(i) for i in decollate_batch(d)]


if __name__ == '__main__':
    main()
