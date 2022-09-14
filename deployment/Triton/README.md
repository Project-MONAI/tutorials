<!--
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


# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
-->

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Deploying MONAI Code via Triton Python Backend
================================================

Simple demo to introduce a standard way for model developers to incorporate Python based projects in a standard way.
In addition, this code will demonstrate how Users/Developers can easily deploy MONAI inference code for field testing.
Finally, the code will demonstrate a method to do low latency classification and validation inference with Triton.

The steps below describe how to set-up a model repository, pull the Triton container, launch the Triton inference server, and then send inference requests to the running server.

This demo and description borrows heavily from the [Triton Python Backend](https://github.com/triton-inference-server/python_backend) repo. The demo assumes you have at least one GPU

Get The Demo Source Code
-------------------------------
Pull down the demo repository and start with the [Quick Start] (#quick-start) guide.

```
$ git clone https://github.com/Project-MONAI/tutorials.git
```
# Python Backend

The Triton backend for Python. The goal of Python backend is to let you serve
models written in Python by Triton Inference Server without having to write
any C++ code. We will use this to demonstrate implementing MONAI code inside Triton.

## User Documentation

* [Quick Start](#quick-start)
* [Examples](#examples)
* [Usage](#usage)
* [Model Config File](#model-config-file)
* [Error Hanldling](#error-handling)


## Quick Start

1. Build Triton Container Image and Copy Model repository files using shell script

```
$ ./triton_build.sh
```
2. Run Triton Container Image in Background Terminal using provided shell script
The supplied script will start the demo container with Triton and expose the three ports to localhost needed for the application to send inference requests.
```
$ ./run_triton_local.sh
```
3. Install environment for client
The client environment should have Python 3 installed and should have the necessary packages installed.
```
$ python3 -m pip install -r requirements.txt
```
4. Other dependent libraries for the Python Triton client are available as a Python packages
```
$ pip install nvidia-pyindex
$ pip install tritonclient[all]
```
5. Run the client program
The [client](./client/client_mednist.py) program will take an optional file input and perform classification on body parts using the MedNIST data set.  A small subset of the database is included.
```
$ mkdir -p client/test_data/MedNist
$ python -u client/client_mednist.py client/test_data/MedNist
```
Alternatively, the user can just run the shell script provided the previous steps 1 -4 in the [Quick Start](#quick-start) were followed.
```
$ ./mednist_client_run.sh
```
The expected result is variety of classification results for body images and local inference times.
```

## Examples:
The example demonstrates running a Triton Python Backend on a single image classification problem.
1. First, a Dockerfile and build script is used to build a container to Run the Triton Service and copy the model specific files in the container.
```Dockerfile:
# use desired Triton container as base image for our app
FROM nvcr.io/nvidia/tritonserver:21.04-py3

# create model directory in container
RUN mkdir -p /models/monai_covid/1

# install project-specific dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

# copy contents of model project into model repo in container image
COPY models/monai_covid/config.pbtxt /models/monai_covid
COPY models/monai_covid/1/model.py /models/monai_covid/1


ENTRYPOINT [ "tritonserver", "--model-repository=/models"]
```
Note: The Triton service expects a certain directory structure discussed in [Model Config File](#model-config-file) to load the model definitions.

2. Next, the container with the Triton Service runs as a service (in background or separate terminal for demo).  In this example, the ports used by the Triton Service are set to `8000` for client communications.
```Dockerfile:
demo_app_image_name="monai_triton:demo"
docker run --shm-size=128G --rm -p 127.0.0.1:8000:8000 -p 127.0.0.1:8001:8001 -p 127.0.0.1:8090:8002 ${demo_app_image_name}
```
3. See [Model Config File](#model-config-file) to see the expected file structure for Triton.
- Modify the models/monai_prostrate/1/model.py file to satisfy any model configuration requirements while keeping the required components in the model definition. See the * [Usage](#usage) section for background.
- In the models/monai_prostrate/1/config.pbtxt file configure the number of GPUs and which ones are used.
e.g. Using two available GPUs and two parallel versions of the model per GPU
```
instance_group [
  {
    kind: KIND_GPU
    count: 2
    gpus: [ 0, 1 ]
  }
```
e.g. Using three of four available GPUs and four parallel versions of the model per GPU
```
instance_group [
  {
    kind: KIND_GPU
    count: 4
    gpus: [ 0, 1, 3 ]
  }
```
Also, other configurations like dynamic batching and corresponding sizes can be configured. See the [Triton Service Documentation model configurations](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) documentation for more information.

- Finally, be sure to include Tensors or Torchscript definition *.ts files in the directory structure. In this example, a COVID19 classificatiion model based in PyTorch is used.
```
covid19_model.ts
```
The Dockerfile will copy the model definition structure into the Triton container Service.  When the container is run, the python backend implementation will pull the covid19_model.ts file from a Google Drive for the demo.  So the container should be rebuilt after any modifications to the GPU configuration or model configurations for the example.

4. A Python [client](./client/client.py) program configures the model and makes an http request to Triton as a Service. Note: Triton supports other interfaces like gRPC.
The client reads an input image converted from Nifti to a byte array for classification.

- In this example, a model trained to detect COVID-19 is provided an image with COVID or without.
```python:
filename = 'client/test_data/volume-covid19-A-0000.nii.gz'
```
- The client calls the Triton Service using the external port configured previously.
```python:
with httpclient.InferenceServerClient("localhost:8000") as client:
```
- The Triton inference response is returned :
```python:
response = client.infer(model_name,
    inputs,
    request_id=str(uuid4().hex),
    outputs=outputs,)

result = response.get_response()
```
-------
## MedNIST Classification Example

- Added to this demo as alternate demo using the MedNIST dataset in a classification example.
- To run the MedNIST example use the same steps as shown in the [Quick Start](#quick-start) with the following changes at step 5.
5. Run the client program (for the MedNIST example)
The [client](./client/client_mednist.py) program will take an optional file input and perform classification on body parts using the MedNIST data set.  A small subset of the database is included.
```
$ mkdir -p client/test_data/MedNist
$ python -u client/client_mednist.py client/test_data/MedNist
```
Alternatively, the user can just run the shell script provided the previous steps 1 -4 in the [Quick Start](#quick-start) were followed.
```
$ ./mednist_client_run.sh
```
The expected result is variety of classification results for body images and local inference times.

## Notes about the `requirements.txt` file and installed CUDA Drivers
- The requirements.txt file is used to place requirements into the Triton Server Container, but also for the client environment.
- Take care with the version of PyTorch (torch) used based on the specific GPU and installed driver versions. The --extra-index-url flag may need to be modified to correspond with the CUDA version installed on the local GPU.
- Determine your driver and CUDA version with the following command:
```
nvidia-smi
```
- Then choose the appropriate library to load for PyTorch by adding the helper flag in the `requirements.txt` file.
```
--extra-index-url https://download.pytorch.org/whl/cu116
```
- Note: in the above example the cu116 instructs to install the latest torch version that supports CUDA 11.6
-------
## Usage
[See Triton Inference Server/python_backend documentation](https://github.com/triton-inference-server/python_backend#usage)
## Model Config
[See Triton Inference Server/python_backend documentation](https://github.com/triton-inference-server/python_backend#model-config-file)
## Error Handling
[See Triton Inference Server/python_backend documentation](https://github.com/triton-inference-server/python_backend#error-handling)
