# Copyright 2022 MONAI Consortium
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

from torch.utils.model_zoo import load_url as load_state_dict_from_url

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_url",
    type=str,
    default="https://download.pytorch.org/models/resnet18-f37072fd.pth",
)
args = parser.parse_args()

# will download
model = load_state_dict_from_url(args.model_url)
