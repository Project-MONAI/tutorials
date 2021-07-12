# Federated Learning with MONAI using NVFLare

## Brief Introduction

This repository contains an end-to-end Federated training example based on MONAI trainers and [NVFlare](https://pypi.org/project/nvflare/).

Inside this folder:
- All Jupiter notebooks are used to build an FL experiment step-by-step.
- `demo_figs` is a folder containing all referred example figures in the notebooks.
- `spleen_example` is the experiment config folder. Some of the experiment related hyperparameters are set in `spleen_example/config/config_train.json`. You
may need to modify `multi_gpu` and some other parameters. Please check the docstrings in `spleen_example/custom/train_configer.py` for more details.
- `build_docker_provision.sh` is the script to build the docker image and do provision.
- `docker_files` is a folder containing all files to build the docker image.
- `expr_files` is a folder containing all files to be used for the experiment.

Inside `expr_files`:

`project.yml` is the project yml file to describe the FL project, it defines project name, participants, server name and othe settings. You can keep the default settings, but may need to change the `cn` name to the server name as:
```
server:
  cn: <your server name >
```
`authz_config.json` is the authorization configuration json file, it defines groups, roles and rights for all users, organizations and sites. If you modified `project.yml`, please change this file to keep the consistency.
`prerpare_expr_files.sh` is the script to do provision and (optional) download the Decathlon Spleen dataset for this experiment.


## Provision Package Preparation

We need to build a docker image for the experiment. It will be based on MONAI's latest docker image in Docker Hub as well as `nvflare` library in PyPI.

Please ensure that you have installed Docker (https://docs.docker.com/engine/install/).

Please run `bash build_docker_provision.sh`.


## Build Experiment

Please ensure that you have installed JupyterLab (https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). The following is an example to install JupyterLab in Python Virtual Environment:
```
python3 -m venv venv/fl_startup
source venv/fl_startup/bin/activate
pip install --upgrade pip
pip install wheel
pip install jupyterlab
```

Please run the following command:

`jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --no-browser --NotebookApp.token=MONAIFLExample`

The link is: `http://localhost:8888/?token=MONAIFLExample`

Then run `1-Startup.ipynb`. You should follow the steps in the notebook, which will guide you through the process of building an FL experiment based on 2 clients and 1 server.
