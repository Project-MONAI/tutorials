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


import glob
import os
import shutil

import monai
import torch
from monai.apps import download_and_extract
from monai.data.torchscript_utils import save_net_with_metadata
from monai.networks.nets import SegResNet
from monai.networks.utils import convert_to_trt


def prepare_test_datalist(root_dir):
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar"

    compressed_file = os.path.join(root_dir, "Task03_Liver.tar")
    data_root = os.path.join(root_dir, "Task03_Liver")
    if not os.path.exists(data_root):
        download_and_extract(resource, compressed_file, root_dir)

    nii_dir = os.path.join(data_root, "imagesTs_nii")
    if not os.path.exists(nii_dir):
        os.makedirs(nii_dir, exist_ok=True)
        train_gz_files = sorted(glob.glob(os.path.join(data_root, "imagesTs", "*.nii.gz")))
        for file in train_gz_files:
            new_file = file.replace(".nii.gz", ".nii")
            if not os.path.exists(new_file):
                os.system(f"gzip -dc {file} > {new_file}")
            shutil.copy(new_file, nii_dir)
    else:
        print(f"Test data already exists at {nii_dir}")

    files = sorted(glob.glob(os.path.join(nii_dir, "*.nii")))
    return files


def prepare_model_weights(root_dir, bundle_name="spleen_ct_segmentation"):
    bundle_path = os.path.join(root_dir, bundle_name)
    weights_path = os.path.join(root_dir, "model.pt")
    if not os.path.exists(weights_path):
        monai.bundle.download(name=bundle_name, bundle_dir=root_dir)

        weights_original_path = os.path.join(bundle_path, "models", "model.pt")
        shutil.copy(weights_original_path, weights_path)
    else:
        print(f"Weights already exists at {weights_path}")

    return weights_path


def prepare_tensorrt_model(root_dir, weights_path, trt_model_name="model_trt.ts"):
    trt_path = os.path.join(root_dir, trt_model_name)
    if not os.path.exists(trt_path):
        model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=105,
            init_filters=32,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2,
        )
        weights = torch.load(weights_path)
        model.load_state_dict(weights)
        torchscript_model = convert_to_trt(
            model=model,
            precision="fp32",
            input_shape=[1, 1, 96, 96, 96],
            dynamic_batchsize=[1, 1, 1],
            use_trace=True,
            verify=False,
        )

        save_net_with_metadata(torchscript_model, trt_model_name.split(".")[0])
    else:
        print(f"TensorRT model already exists at {trt_path}")

    return os.path.join(root_dir, trt_model_name)
