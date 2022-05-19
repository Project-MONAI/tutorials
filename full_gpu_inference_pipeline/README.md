# Deploy MONAI pipeline by Triton and run the full pipeline on GPU step by step
- [Deploy MONAI pipeline by Triton and run the full pipeline on GPU step by step](#deploy-monai-pipeline-by-triton-and-run-the-full-pipeline-on-gpu-step-by-step)
  * [overview](#overview)
  * [Prepare the model repository](#prepare-the-model-repository)
    + [Prepare the model repository file directories](#prepare-the-model-repository-file-directories)
  * [Environment Setup](#environment-setup)
    + [Setup Triton environment](#setup-triton-environment)
    + [Setup python execution environment](#setup-python-execution-environment)
  * [Run Triton server](#run-triton-server)
  * [Run Triton client](#run-triton-client)
  * [Benchmark](#benchmark)
    + [Understanding the benchmark output](#understanding-the-benchmark-output)
    + [HTTP vs. gRPC vs. shared memory](#http-vs-grpc-vs-shared-memory)
    + [Pre/Post-processing on GPU vs. CPU](#pre-post-processing-on-gpu-vs-cpu)

## overview

This example is to implement a 3D medical imaging AI inference pipeline using the model and transforms of MONAI, and deploy the pipeline using Triton. the goal of it is to test the influence brought by different features of MONAI and Triton to medical imaging AI inference performance.

In this repository, I will try following features:
- [Python backend BLS](https://github.com/triton-inference-server/python_backend#business-logic-scripting) (Triton), which allows you to execute inference requests on other models being served by Triton as a part of executing your Python model.
- Transforms on GPU(MONAI), by using which, you can compose GPU accelerated pre/post processing chains.

Before starting, I highly recommand you to read the the following two links to get familiar with the basic features of Triton python backend and MONAI:
- https://github.com/triton-inference-server/python_backend
- https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md

## Prepare the model repository
The full pipeline is as below:

<img src="https://github.com/Project-MONAI/tutorials/raw/main/full_gpu_inference_pipeline/pics/Picture3.png">

### Prepare the model repository file directories
The Triton model repository of the experiment can be fast set up by: 
```bash
git clone https://github.com/Project-MONAI/tutorials.git
cd full_gpu_inference_pipeline
bash download_model_repo.sh
```
The model repository is in folder triton_models. The file structure of the model repository should be:
```
triton_models/
├── segmentation_3d
│   ├── 1
│   │   └── model.pt
│   └── config.pbtxt
└── spleen_seg
├── 1
│   └── model.py
└── config.pbtxt
```

## Environment Setup
### Setup Triton environment
Triton environment can be quickly setup by running a Triton docker container:
```bash
docker run --gpus=1 -it --name='triton_monai' --ipc=host -p18100:8000 -p18101:8001 -p18102:8002 --shm-size=1g -v /yourfolderpath:/triton_monai nvcr.io/nvidia/tritonserver:21.12-py3
```
Please note that when starting the docker container, --ipc=host should be set, so that shared memory can be used to do the data transmission between server and client. Also you should allocate a relatively large shared memory using --shm-size option, because starting from 21.04 release, Python backend uses shared memory to connect user's code to Triton.
### Setup python execution environment
Since we will use MONAI transforms in Triton python backend, we should set up the python execution environment in Triton container by following the instructions in [Triton python backend repository](https://github.com/triton-inference-server/python_backend#using-custom-python-execution-environments). For the installation steps of MONAI, you can refer to [monai install](https://docs.monai.io/en/latest/installation.html "monai install"). Below are the steps used to setup the proper environments for this experiment:

Install the software packages below:
- conda
- cmake
- rapidjson and libarchive ([instructions](https://github.com/triton-inference-server/python_backend#building-from-source "instructions") for installing these packages in Ubuntu or Debian are included in Building from Source Section)
- conda-pack

Create and activate a conda environment.
```bash
conda create -n monai python=3.8
conda activate monai
```
Since Triton 21.12 NGC docker image is used, in which python version is 3.8, we can create a conda env of python3.8 for convenience. You can also specify other python versions. If the python version you use is not equal to that of triton container's, please make sure you go through these extra [steps](https://github.com/triton-inference-server/python_backend#1-building-custom-python-backend-stub "steps"). 
Before installing the packages in your conda environment, make sure that you have exported PYTHONNOUSERSITE environment variable:
```bash
export PYTHONNOUSERSITE=True
```
If this variable is not exported and similar packages are installed outside your conda environment, your tar file may not contain all the dependencies required for an isolated Python environment.
Install Pytorch with CUDA 11.3 support. 
```bash
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Install MONAI and the recommended dependencies.
```bash
BUILD_MONAI=1 pip install --no-build-isolation git+https://github.com/Project-MONAI/MONAI#egg=monai
```
Then we can verify the installation of MONAI and all its dependencies:
```bash
python -c 'import monai; monai.config.print_config()'
```
You'll see the output below, which lists the versions of MONAI and relevant dependencies.

```bash
MONAI version: 0.8.0+65.g4bd13fe
Numpy version: 1.21.4
Pytorch version: 1.10.1+cu113
MONAI flags: HAS_EXT = True, USE_COMPILED = False
MONAI rev id: 4bd13fefbafbd0076063201f0982a2af8b56ff09
MONAI __file__: /usr/local/lib/python3.8/dist-packages/monai/__init__.py
Optional dependencies:
Pytorch Ignite version: NOT INSTALLED or UNKNOWN VERSION.
Nibabel version: 3.2.1
scikit-image version: 0.19.1
Pillow version: 9.0.0
Tensorboard version: NOT INSTALLED or UNKNOWN VERSION.
gdown version: NOT INSTALLED or UNKNOWN VERSION.
TorchVision version: NOT INSTALLED or UNKNOWN VERSION.
tqdm version: NOT INSTALLED or UNKNOWN VERSION.
lmdb version: NOT INSTALLED or UNKNOWN VERSION.
psutil version: NOT INSTALLED or UNKNOWN VERSION.
pandas version: NOT INSTALLED or UNKNOWN VERSION.
einops version: NOT INSTALLED or UNKNOWN VERSION.
transformers version: NOT INSTALLED or UNKNOWN VERSION.
mlflow version: NOT INSTALLED or UNKNOWN VERSION.

For details about installing the optional dependencies, please visit:
    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies
```
Install the dependencies of MONAI:
```bash
pip install nibabel scikit-image pillow tensorboard gdown ignite torchvision itk tqdm lmdb psutil cucim  pandas einops transformers mlflow matplotlib tensorboardX tifffile cupy
```
Next, we should package the conda environment by using `conda-pack` command, which will produce a package of monai.tar.gz. This file contains all the environments needed by the python backend model and is portable. Then put the created monai.tar.gz under the spleen_seg folder, and the config.pbtxt should be set as:
```bash
parameters: {
key: "EXECUTION_ENV_PATH",
value: {string_value: "$$TRITON_MODEL_DIRECTORY/monai.tar.gz"}
}
```
Also, please note that in the config.pbtxt, the parameter `FORCE_CPU_ONLY_INPUT_TENSORS` is set to `no`, so that Triton will not move input tensors to CPU for the Python model. Instead, Triton will provide the input tensors to the Python model in either CPU or GPU memory, depending on how those tensors were last used.
And now the file structure of the model repository should be:
```
triton_models/
├── segmentation_3d
│   ├── 1
│   │   └── model.pt
│   └── config.pbtxt
└── spleen_seg
├── 1
│   └── model.py
├── config.pbtxt
└── monai.tar.gz
```
## Run Triton server
Then you can start the triton server by the command:
```bash
tritonserver --model-repository=/ROOT_PATH_OF_YOUR_MODEL_REPOSITORY
```
## Run Triton Client
We assume that the server and client are both on the same machine. Open a new bash terminal and run the commands below to setup the client environment.
```bash
nvidia-docker run -it --ipc=host --shm-size=1g --name=triton_client --net=host nvcr.io/nvidia/tritonserver:21.12-py3-sdk
pip install monai
pip install nibabel
pip install jupyter
```
Then you can run the jupyter nootbook in the client folder of this example.
Please note that when starting the docker container, --ipc=host should be set, so that we can use shared memory to do the data transmission between server and client.

## Benchmark
The benchmark was run on RTX 8000 and tested by using perf_analyzer.
```bash
perf_analyzer -m spleen_seg -u localhost:18100 --input-data zero --shape "INPUT0":512,512,114 --shared-memory system
```

### Understanding the benchmark output
- HTTP: `send/recv` indicates the time on the client spent sending the request and receiving the response. `response wait` indicates time waiting for the response from the server.
- GRPC: `(un)marshal request/response` indicates the time spent marshalling the request data into the GRPC protobuf and unmarshalling the response data from the GRPC protobuf. `response wait` indicates time writing the GRPC request to the network, waiting for the response, and reading the GRPC response from the network.
- compute_input : The count and cumulative duration to prepare input tensor data as required by the model framework / backend. For example, this duration should include the time to copy input tensor data to the GPU.
- compute_infer : The count and cumulative duration to execute the model.
- compute_output : The count and cumulative duration to extract output tensor data produced by the model framework / backend. For example, this duration should include the time to copy output tensor data from the GPU.

### HTTP vs. gRPC vs. shared memory
Since 3D medical images are generally big, the overhead brought by protocols cannot be ignored. For most common cases of medical image AI, the clients are on the same machine as the server, so shared memory is an appliable way to reduce the send/receive overhead. In this experiment, perf_analyzer is used to compare different ways of communicating between client and server.
Note that all the processes (pre/post and AI inference) are on GPU.
From the result, we can come to a conclusion that using shared memory will greatly reduce the latency when data transfer is huge.

![](https://github.com/Project-MONAI/tutorials/raw/main/full_gpu_inference_pipeline/pics/Picture2.png)

### Pre/Post-processing on GPU vs. CPU 
After doing pre and post-processing on GPU, we can get a 12x speedup for the full pipeline.

![](https://github.com/Project-MONAI/tutorials/raw/main/full_gpu_inference_pipeline/pics/Picture1.png)
