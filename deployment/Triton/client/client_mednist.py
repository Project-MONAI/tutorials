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


# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import argparse
import numpy as np
import os
import sys
import time
from uuid import uuid4
import glob

from monai.apps.utils import download_and_extract
from monai.transforms.utils import convert_to_numpy

MEDNIST_CLASSES = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]



model_name = "mednist_class"
gdrive_path = "https://drive.google.com/uc?id=1HQk4i4vXKUX_aAYR4wcZQKd-qk5Lcm_W"
mednist_filename = "MedNIST_demo.tar.gz"
md5_check = "3f24a5833bb0455a7815c4e0ecc8a810"


def open_jpeg_files(input_path):
    return sorted(glob.glob(os.path.join(input_path, "*.jpeg")))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Triton CLI for MedNist classification inference from JPEG data')
    parser.add_argument(
        'input',
        type=str,
        help="Path to JPEG file or directory containing JPEG files to send for MedNist classification"
    )
    args = parser.parse_args()

    jpeg_files = []
    extract_dir = "./client/test_data/MedNist"
    tar_save_path = os.path.join(extract_dir, mednist_filename)
    if os.path.isdir(args.input):  # check for directory existence
        # Grab files from Google Drive and place in directory
        download_and_extract(gdrive_path, tar_save_path, output_dir=extract_dir, hash_val=md5_check, hash_type="md5")
        jpeg_files = open_jpeg_files(args.input)

    elif os.path.isfile(args.input):
        jpeg_files = [args.input]

    if not jpeg_files:
        print("No valid inputs provided")
        sys.exit(1)

    with httpclient.InferenceServerClient("localhost:7555") as client:
        image_bytes = b''
        for jpeg_file in jpeg_files:
            with open(jpeg_file, 'rb') as f:
                image_bytes = f.read()

            input0_data = np.array([[image_bytes]], dtype=np.bytes_)
            
            inputs = [
                httpclient.InferInput("INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)),
            ]

            inputs[0].set_data_from_numpy(input0_data)

            outputs = [
                httpclient.InferRequestedOutput("OUTPUT0"),
            ]

            inference_start_time = time.time() * 1000
            response = client.infer(model_name,
                                    inputs,
                                    request_id=str(uuid4().hex),
                                    outputs=outputs,)
            inference_time = time.time() * 1000 - inference_start_time

            result = response.get_response()
            print("Classification result for `{}`: {}.  (Inference time: {:6.0f} ms)".format(
                jpeg_file,
                response.as_numpy("OUTPUT0").astype(str)[0],
                inference_time,
            ))
