# Federated learning with NVIDIA FLARE

## Brief Introduction

This repository contains an end-to-end Federated training example based on MONAI trainers and [NVIDIA FLARE](https://github.com/nvidia/nvflare). Please also download the `hello-monai` folder in [NVFlare/examples](https://github.com/NVIDIA/NVFlare/tree/dev/examples/hello-monai), and copy it into this directory.



This example requires Python 3.8. It may work with Python 3.7 but currently is not compatible with Python 3.9 and above.

Inside this directory:
- All Jupiter notebooks are used to build an FL experiment step-by-step.
- hello-monai (needs to be downloaded) is a folder containing all required config files for the experiment (in `app/config`) and the customized trainer (in `app/custom`) and its components.

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

### Prepare POC

In this example, we use the `poc` command to create a proof of concept folder, please check the [official document](https://nvflare.readthedocs.io/en/main/quickstart.html#setting-up-the-application-environment-in-poc-mode) for more details.

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
