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


import copy
import json
import logging
import os
import pathlib
from tempfile import NamedTemporaryFile

import cupy
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from cucim.skimage.measure import label
from torch.utils.dlpack import from_dlpack, to_dlpack

from monai.transforms import (
    AddChannel,
    AsDiscrete,
    Compose,
    CropForeground,
    EnsureType,
    KeepLargestConnectedComponent,
    LabelToContour,
    Resize,
    ScaleIntensityRange,
    ToCupy,
    ToTensor,
)

logger = logging.getLogger(__name__)


class TritonPythonModel:
    """
    Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """
        `initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        """

        # load model configuration
        self.model_config = json.loads(args["model_config"])
        input0_config = pb_utils.get_input_config_by_name(self.model_config, "INPUT0")
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "MASK")

        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(self.model_config, "CONTOUR")

        # Convert Triton types to numpy types
        self.input0_dtype = pb_utils.triton_string_to_numpy(input0_config["data_type"])
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config["data_type"])

        # create inferer engine and load PyTorch model
        self.inference_device_kind = args.get("model_instance_kind", None)
        # print(f"Inference device: {self.inference_device_kind}")

        if self.inference_device_kind is None:
            self.inference_device_kind = "CPU"
        elif self.inference_device_kind == "GPU":
            self.inference_device_id = args.get("model_instance_device_id", "0")
            logger.info(f"Inference device id: {self.inference_device_id}")

        infer_transforms = []
        if self.inference_device_kind == "GPU":
            # print("*********************I'm here********************8")
            infer_transforms.append(EnsureType(device=torch.device(f"cuda:{self.inference_device_id}")))
        else:
            infer_transforms.append(EnsureType())
        infer_transforms.append(AddChannel())
        infer_transforms.append(ScaleIntensityRange(a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True))
        infer_transforms.append(CropForeground())
        infer_transforms.append(Resize(spatial_size=(224, 224, 224)))
        self.pre_transforms = Compose(infer_transforms)
        self.post_transforms = Compose([])

    def GetLargestConnectedComponent(self, x):
        x_cupy = ToCupy()(x)
        x_cupy_dtype = x_cupy.dtype
        x_label = label(x_cupy)
        vals, counts = cupy.unique(x_label[cupy.nonzero(x_label)], return_counts=True)
        comp = x_label == vals[cupy.ndarray.argmax(counts)]
        out = comp.astype(x_cupy_dtype)
        out = ToTensor(device=f"cuda:{self.inference_device_id}")(out)
        return out

    def execute(self, requests):
        """
        `execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        """
        responses = []

        for request in requests:

            # get the input by name (as configured in config.pbtxt)
            input_triton_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_torch_tensor = from_dlpack(input_triton_tensor.to_dlpack())
            logger.info(f"the shape of the input tensor is: {input_torch_tensor.shape}")
            transform_output = self.pre_transforms(input_torch_tensor[0])
            logger.info(f"the shape of the transformed tensor is: {transform_output.shape}")
            transform_output_batched = transform_output.unsqueeze(0)
            logger.info(f"the shape of the unsqueezed transformed tensor is: {transform_output_batched.shape}")
            # if(transform_output_batched.is_cuda):
            # print("the transformed pytorch tensor is on GPU")

            # print(transform_output.shape)
            transform_tensor = pb_utils.Tensor.from_dlpack("INPUT__0", to_dlpack(transform_output_batched))
            # if(transform_tensor.is_cpu()):
            #     print("the transformed triton tensor is on CPU")
            inference_request = pb_utils.InferenceRequest(
                model_name="segmentation_3d", requested_output_names=["OUTPUT__0"], inputs=[transform_tensor]
            )

            infer_response = inference_request.exec()
            output1 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT__0")
            output_tensor = from_dlpack(output1.to_dlpack())
            if output1.is_cpu():
                # print("the output tensor is on CPU")
                if self.inference_device_kind == "GPU":
                    output_tensor.to(f"cuda:{self.inference_device_id}")
                    logger.info("run post process on GPU")
            else:
                # print("the output tensor is on GPU")
                if self.inference_device_kind == "CPU":
                    output_tensor = output_tensor.to("cpu")
                    # if(output_tensor.is_cuda == False):
                    #     print("the output tensor is now moved to CPU")

            argmax = AsDiscrete(argmax=True)(output_tensor[0])
            if self.inference_device_kind == "GPU":
                # print("I'm here to do GetLargest...")
                largest = self.GetLargestConnectedComponent(argmax)
            else:
                largest = KeepLargestConnectedComponent(applied_labels=1)(argmax)
            contour = LabelToContour()(largest)
            out_tensor_0 = pb_utils.Tensor.from_dlpack("MASK", to_dlpack(largest.unsqueeze(0)))
            out_tensor_1 = pb_utils.Tensor.from_dlpack("CONTOUR", to_dlpack(contour.unsqueeze(0)))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """
        `finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass
