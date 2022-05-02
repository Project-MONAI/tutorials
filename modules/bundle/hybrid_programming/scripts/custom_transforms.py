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

from monai.config import KeysCollection
from monai.transforms import EnsureTyped


class PrintEnsureTyped(EnsureTyped):
    """
    Extend the `EnsureTyped` transform to print the image shape.

     Args:
        keys: keys of the corresponding items to be transformed.

    """

    def __init__(self, keys: KeysCollection, data_type: str = "tensor") -> None:
        super().__init__(keys, data_type=data_type)

    def __call__(self, data):
        d = dict(super().__call__(data=data))
        for key in self.key_iterator(d):
            print(f"data shape of {key}: {d[key].shape}")
        return d
