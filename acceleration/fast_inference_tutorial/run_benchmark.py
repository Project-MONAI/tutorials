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
import gc
import os
from timeit import default_timer as timer

import pandas as pd
import torch
import torch_tensorrt
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (Activationsd, AsDiscreted, Compose,
                              EnsureChannelFirstd, EnsureTyped, Invertd,
                              LoadImaged, NormalizeIntensityd, Orientationd,
                              ScaleIntensityd, Spacingd)

from utils import (prepare_model_weights, prepare_tensorrt_model,
                   prepare_test_datalist)


def get_transforms(device, gpu_loading_flag=False, gpu_transforms_flag=False):
    preprocess_transforms = [
        LoadImaged(keys="image", reader="NibabelReader", to_gpu=gpu_loading_flag),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True),
        ScaleIntensityd(
            keys=["image"],
            minv=-1.0,
            maxv=1.0,
        ),
    ]

    if gpu_transforms_flag and not gpu_loading_flag:
        preprocess_transforms.insert(1, EnsureTyped(keys="image", device=device, track_meta=True))
    infer_transforms = Compose(preprocess_transforms)

    return infer_transforms


def get_post_transforms(infer_transforms):
    post_transforms = Compose(
        [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            Invertd(
                keys="pred",
                transform=infer_transforms,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
        ]
    )
    return post_transforms


def get_model(device, weights_path, trt_model_path, trt_flag=False):
    if not trt_flag:
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
        model.to(device)
        model.eval()
    else:
        model = torch.jit.load(trt_model_path)
    return model


def run_inference(data_list, infer_transforms, model, device, benchmark_type):
    total_time_dict = {}
    roi_size = (96, 96, 96)
    sw_batch_size = 4

    for idx, sample in enumerate(data_list):
        start = timer()
        data = infer_transforms({"image": sample})

        with torch.no_grad():
            input_image = (
                data["image"].unsqueeze(0).to(device)
                if benchmark_type in ["trt", "original"]
                else data["image"].unsqueeze(0)
            )

            output_image = sliding_window_inference(input_image, roi_size, sw_batch_size, model)
            output_image = output_image.cpu()

            end = timer()

        del data
        del input_image
        del output_image
        torch.cuda.empty_cache()
        gc.collect()

        sample_name = sample.split("/")[-1]
        if idx > 0:
            total_time_dict[sample_name] = end - start
            print(f"Time taken for {sample_name}: {end - start} seconds")
    return total_time_dict


def main():
    parser = argparse.ArgumentParser(description="Run inference benchmark.")
    parser.add_argument("--benchmark_type", type=str, default="original", help="Type of benchmark to run")
    args = parser.parse_args()

    ### Prepare the environment
    root_dir = "."
    torch.backends.cudnn.benchmark = True
    torch_tensorrt.runtime.set_multi_device_safe_mode(True)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    train_files = prepare_test_datalist(root_dir)
    # since the dataset is too large, the smallest 31 files are used for warm up (1 file) and benchmarking (30 files)
    train_files = sorted(train_files, key=lambda x: os.path.getsize(x), reverse=False)[:31]
    weights_path = prepare_model_weights(root_dir=root_dir, bundle_name="wholeBody_ct_segmentation")
    trt_model_name = "model_trt.ts"
    trt_model_path = prepare_tensorrt_model(root_dir, weights_path, trt_model_name)

    gpu_transforms_flag = "gpu_transforms" in args.benchmark_type
    gpu_loading_flag = "gds" in args.benchmark_type
    trt_flag = "trt" in args.benchmark_type
    # Get components
    infer_transforms = get_transforms(device, gpu_loading_flag, gpu_transforms_flag)
    model = get_model(device, weights_path, trt_model_path, trt_flag)
    # Run inference
    total_time_dict = run_inference(train_files, infer_transforms, model, device, args.benchmark_type)
    # Save the results
    df = pd.DataFrame(list(total_time_dict.items()), columns=["file_name", "time"])
    df.to_csv(os.path.join(root_dir, f"time_{args.benchmark_type}.csv"), index=False)


if __name__ == "__main__":
    main()
