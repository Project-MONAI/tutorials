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
import sys

import torch
import torch.profiler

import monai
from benchmark_utils import (
    get_bundle_network,
    get_bundle_input_shape,
    get_bundle_trt_model,
    inference_random_python_timer,
    inference_random_torch_timer,
    get_current_device,
    get_bundle_evaluator,
    get_trt_evaluator,
    inference_bundle_torch_timer,
)


if __name__ == "__main__":
    support_precsion = ["fp32", "fp16"]
    current_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="spleen_ct_segmentation", help="Bundle name in MONAI model zoo.")
    parser.add_argument("-v", "--version", default=None, help="Version of the corresponding bundle.")
    parser.add_argument(
        "-p",
        "--precision",
        default="fp32",
        help="Precision of converting TensorRT models. Can be chosen from [fp32, fp16].",
    )
    parser.add_argument(
        "-m", "--model", default=False, action="store_true", help="Only benchmark the model if input this parameter."
    )
    parser.add_argument("-b", "--batchsize", default=4, type=int, help="The static input batchsize for the bundle.")
    parser.add_argument(
        "-c", "--convert", default=False, action="store_true", help="Convert the model without benchmark."
    )
    parser.add_argument(
        "-t",
        "--timer",
        default="torch_timer",
        help="How to time the inference. Can be chosen from [timer, torch_timer], default to torch_timer",
    )
    args = parser.parse_args()

    bundle_name = args.name
    bundle_version = args.version
    convert_precision = args.precision
    only_benchmark_model = args.model
    timer_type = args.timer
    only_convert = args.convert
    batch_size = args.batchsize
    device = get_current_device()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"Cudnn tf32 : {torch.backends.cudnn.allow_tf32}, CUDA tf32 : {torch.backends.cuda.matmul.allow_tf32}")

    print(f"================Benchmarking on {bundle_name}.================")

    if not convert_precision in support_precsion:
        raise ValueError(f"Not supported precision {convert_precision}.")

    convert_precision = torch.float32 if convert_precision == "fp32" else torch.half
    # TODO add constrain for torch-tensorrt version
    bundle_path = os.path.join(current_path, bundle_name)
    if not os.path.exists(bundle_path):
        monai.bundle.download(name=bundle_name, version=bundle_version, bundle_dir=current_path)

    sys.path.append(bundle_path)
    model = get_bundle_network(bundle_path)
    model.to(device)
    model_input_shape = get_bundle_input_shape(bundle_path)

    trt_model = get_bundle_trt_model(bundle_path, model, model_input_shape, convert_precision, batch_size)
    sys.path.remove(bundle_path)

    if only_convert:
        sys.exit()
    # inference_random(trt_model, spatial_shape)

    print(f"Input spatial shape : {model_input_shape}.")
    if only_benchmark_model:
        if timer_type == "torch_timer":
            torch_inference_time = inference_random_torch_timer(bundle_path, model, model_input_shape, torch.float32)
            trt_inference_time = inference_random_torch_timer(
                bundle_path, trt_model, model_input_shape, convert_precision
            )
        else:
            torch_inference_time = inference_random_python_timer(bundle_path, model, model_input_shape, torch.float32)
            trt_inference_time = inference_random_python_timer(
                bundle_path, trt_model, model_input_shape, convert_precision
            )
    else:
        if timer_type == "torch_timer":
            bundle_evaluator = get_bundle_evaluator(bundle_path)
            bundle_evaluator.amp = False
            trt_evalator = get_trt_evaluator(bundle_path, convert_precision)
            trt_evalator.amp = False
            torch_inference_time = inference_bundle_torch_timer(bundle_path, bundle_evaluator, torch.float32, "torch")
            trt_inference_time = inference_bundle_torch_timer(bundle_path, trt_evalator, convert_precision, "trt")
        else:
            pass
    print(
        (
            f"Torch : {torch_inference_time}.\n"
            f"Trt : {trt_inference_time}.\n"
            f"Acc ratio : {torch_inference_time/(trt_inference_time + 1e-12):.2f}"
        )
    )
    print(f"====================End {bundle_name}.========================")
