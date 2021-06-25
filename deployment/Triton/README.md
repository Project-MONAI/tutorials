<!--
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
$ ./triton_run_local.sh
```
3. Install environment for client
The client environment should have Python 3 installed and should have the necessary packages installed. 
```
$ python3 -m pip install -r requirements.txt
```
4. Other dependent libraries for the Python Triton client are available as a Python package than can installed using pip
```
$ pip install nvidia-pyindex
$ pip install tritonclient[all]
```
5. Run the client program
The [client](./client/client.py) program will take an optional file input and perform  classification to determine whether the study shows COVID-19 or not COVID-19.  See the [NVIDIA COVID-19 Classification ](https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_covid19_3d_ct_classification) example in NGC for more background. 
```
$ python -u client/client.py [filename/directory]
```
and the program returns
``` Default input for the client is client/test_data/prostate_24.nii.gz
$ Classification result: ['Prostate segmented']
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
COPY models/monai_covid/1/covid19_model.ts /models/monai_covid/1

ENTRYPOINT [ "tritonserver", "--model-repository=/models"]
#ENTRYPOINT [ "tritonserver", "--model-repository=/models", "--log-verbose=1"]
```
Note: The Triton service expects a certain directory structure discussed in [Model Config File](#model-config-file) to load the model definitions.

2. Next, the container with the Triton Service is run as a service (in background or separate terminal for demo).  In this example, the ports used by the Triton Service are set to `8090` for client communications.
```Dockerfile:
demo_app_image_name="monai_triton:demo"
docker run --shm-size=128G --rm -p 127.0.0.1:8090:8000 -p 127.0.0.1:8091:8001 -p 127.0.0.1:8092:8002 ${demo_app_image_name}
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
 
- Finally, be sure to include you Tensors or Torchscript definition *.ts files in the directory structure. In this example,a prostate segmentation model based in PyTorch is used. 
```
prostate_model.ts
```
The Dockerfile will copy the model definition structure into the  Triton container Service.  So the container should be rebuilt after any modifications to the GPU or model configurations for the example.

4. A Python [client](./client/client.py) program configures the model and makes an http request to Triton as a Service. Note: Triton supports other interfaces like gRPC.
The client reads an input image converted from Nifti to a byte array for classification. 

- In this example, a model trained to detect COVID-19 is provide an image with COVID or without.    
```python:
filename = 'client/test_data/volume-covid19-A-0000.nii.gz'
```
- The client calls the Triton Service using the external port configured previously.
```python:
with httpclient.InferenceServerClient("localhost:8090") as client:
```
- The Triton inference request is returned in the response 
```python:
response = client.infer(model_name,
    inputs,
    request_id=str(uuid4().hex),
    outputs=outputs,)

result = response.get_response()
```
-------
## Usage

In order to use the Python backend, you need to create a Python file that
has a structure with the three components descibed below.

Every Python backend can implement three main functions:

### `initialize`

`initialize` is called once the model is being loaded. Implementing
`initialize` is optional. `initialize` allows you to do any necessary
initializations before execution. In the `initialize` function, you are given
an `args` variable. `args` is a Python dictionary. Both keys and
values for this Python dictionary are strings. You can find the available
keys in the `args` dictionary along with their description in the table
below:

| key                      | description                                      |
| ------------------------ | ------------------------------------------------ |
| model_config             | A JSON string containing the model configuration |
| model_instance_kind      | A string containing model instance kind          |
| model_instance_device_id | A string containing model instance device ID     |
| model_repository         | Model repository path                            |
| model_version            | Model version                                    |
| model_name               | Model name                                       |

### `execute`

`execute` function is called whenever an inference request is made. Every Python
model must implement `execute` function. In the `execute` function you are given
a list of `InferenceRequest` objects. In this function, your `execute` function
must return a list of `InferenceResponse` objects that has the same length as
`requests`.

In case one of the inputs has an error, you can use the `TritonError` object
to set the error message for that specific request. Below is an example of
setting errors for an `InferenceResponse` object:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    ...

    def execute(self, requests):
        responses = []

        for request in requests:
            if an_error_occurred:
              # If there is an error, the output_tensors are ignored
              responses.append(pb_utils.InferenceResponse(
                output_tensors=[], error=pb_utils.TritonError("An Error Occurred")))

        return responses
```

### `finalize`

Implementing `finalize` is optional. This function allows you to do any clean
ups necessary before the model is unloaded from Triton server.

You can look at the [add_sub example](examples/add_sub.py) which contains
a complete example of implementing all these functions for a Python model
that adds and subtracts the inputs given to it. After implementing all the
necessary functions, you should save this file as `model.py`.


## Model Config File

Every Python Triton model must provide a `config.pbtxt` file describing
the model configuration. In order to use this backend you must set the `backend`
field of your model `config.pbtxt` file to `python`. You shouldn't set
`platform` field of the configuration.

Your models directory should look like below:
```
models
└── add_sub
    ├── 1
    │   └── model.py
    └── config.pbtxt
```

## Error Handling

If there is an error that affects the `initialize`, `execute`, or `finalize`
function of the Python model you can use `TritonInferenceException`.
Example below shows how you can do error handling in `finalize`:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    ...

    def finalize(self):
      if error_during_finalize:
        raise pb_utils.TritonModelException("An error occurred during finalize.")
```


