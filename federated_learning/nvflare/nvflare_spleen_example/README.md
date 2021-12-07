# Federated learning with NVIDIA FLARE

## Brief Introduction

This repository contains an end-to-end Federated training example based on MONAI trainers and [NVIDIA FLARE](https://github.com/nvidia/nvflare). Please also download the `hello-monai` folder in [NVFlare/examples](https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-monai), and copy it into this directory.



This example requires Python 3.8.10

Inside this directory:
- All Jupiter notebooks are used to build an FL experiment step-by-step.
- hello-monai (needs to be downloaded) is a folder containing all required config files for the experiment (in `config/`) and the customized trainer (in `custom`) and its components.

## Installation

The following installation steps refer to the official installation guide of NVIDIA FLARE, [please check that guide for more details](https://nvidia.github.io/NVFlare/installation.html)

### Virtual Environment

It is recommended to create a virtual engironment via `venv` to install all packages:

```
python3 -m venv nvflare-env
source nvflare-env/bin/activate
```
### Libraries

Please run:
```
pip install -U -r requirements.txt
```

### Prepare Startup Kit

NVIDIA FLARE provides the Open Provision API to build the startup kit flexibly, the corresponding guide is in [here](https://nvidia.github.io/NVFlare/user_guide/provisioning_tool.html).

In this example, we simply use the `poc` command to create one startup kit, this way is also used in [an official example of NVIDIA FLARE](https://nvidia.github.io/NVFlare/examples/hello_cross_val.html?highlight=poc).

Please run:
```
poc -n 2
```
and type `y`, then a working folder named `poc` will be created (the related readme file is in `poc/Readme.rst`), the folder works for one server, two clients and one admin.

## Build Experiment

The following step-by-step process will be shown in Jupyter Notebooks, please run:

`jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --no-browser --NotebookApp.token=MONAIFLExample`

and enter the following link:

`http://localhost:8888/?token=MONAIFLExample`

Then run `1-Server.ipynb`. You should follow the steps in the notebook, which will guide you through the process of building an FL experiment based on 2 clients and 1 server.
