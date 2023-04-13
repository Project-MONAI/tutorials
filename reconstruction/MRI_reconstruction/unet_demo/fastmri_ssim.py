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

from skimage.metrics import structural_similarity as compare_ssim
from numpy import ndarray


# monai.losses.ssim_loss can be used as a metric
# but in order to match numbers with the fastMRI leaderboard,
# we use scikit-image ssim metric
def skimage_ssim(gt: ndarray, rec: ndarray) -> float:
    """
    Compute SSIM quality assessment metric
    inputs:
        gt: ground truth with the last two dims being spatial and the number of slices
        rec: reconstruction with the same dimensions as gt
    outputs:
        skimage SSIM score between gt and rec
    """
    # assumes 3D inputs
    return compare_ssim(gt.transpose(1, 2, 0), rec.transpose(1, 2, 0), channel_axis=2, data_range=gt.max())
