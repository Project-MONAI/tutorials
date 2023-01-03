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

import os
from glob import glob

import numpy as np
from monai.data import create_test_image_2d
from PIL import Image


class ExampleImageGenerator():
    def __init__(self, num_image=40, image_size=(128, 128)):
        self.num_image = num_image
        self.image_size = image_size

    def generate(self, tempdir):
        for i in range(self.num_image):
            im, seg = create_test_image_2d(
                self.image_size[0], self.image_size[1], num_seg_classes=1,random_state=np.random.RandomState(42)
            )
            Image.fromarray((im * 255).astype("uint8")).save(os.path.join(tempdir, f"img{i:d}.png"))
            Image.fromarray((seg * 255).astype("uint8")).save(os.path.join(tempdir, f"seg{i:d}.png"))

        images = sorted(glob(os.path.join(tempdir, "img*.png")))
        segs = sorted(glob(os.path.join(tempdir, "seg*.png")))
        return (images, seg)
