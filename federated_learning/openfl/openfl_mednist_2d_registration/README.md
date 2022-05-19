# Federated learning with [OpenFL](https://github.com/intel/openfl)

## Introduction

This repository contains an end-to-end Federated training example based on [MONAI](https://github.com/Project-MONAI/MONAI) 2d mednist registration [tutorial](../../../2d_registration/registration_mednist.ipynb). In this federated learning experiment we will run two collabolators with separated samples from MedNIST dataset.


This example requires Python 3.6 - 3.8

This folder contains:
* `director` folder for running director server.
* `envoy` folder for running collaborators.
* `workspace` folder with jupyter-notebook for running an FL experiment.
* `requirements.txt` with dependencies.

## Installation
First of all, clone the repository and go to [openfl_mednist_2d_registration](./) directory:

```sh
git clone https://github.com/Project-MONAI/tutorials.git
cd tutorials/federated_learning/openfl/openfl_mednist_2d_registration/
```
Then follow the steps to set up federation and run your experiment.
### Virtual Environment

We will use several `Python3.8` virtual environments  with `openfl`, `monai` and other packages.

## Set up your federation
You can set up your federation in one of the following modes: with or without TLS connections. For simplicity, here are the commands for running a federation without TLS and locally, so the connection bettween envoys is over the local network). Information on how to establish a secure TLS connection in federation can be found in the related [section](https://openfl.readthedocs.io/en/latest/running_the_federation.html#optional-step-create-pki-certificates-using-step-ca) of the documentation.


### 1\. Start director:
- Create director virtual environment  and activate it:
```sh
python3.8 -m venv director_env
source director_env/bin/activate
```

- Install required packages:
```sh
pip install -r requirements.txt
```

- Run director:
```sh
cd director
fx director start --disable-tls --director-config-path director_config.yaml
```
After that, the director will be started on `localhost:50051`, this behavior is specified in [director_config.yaml](./director/director_config.yaml)

### 2\. Start first envoy:
- Open a new terminal.

- Before starting the first envoy, we should create a copy of the [envoy](./envoy) folder. Just copy this directory to another location.

```sh
cp -r envoy envoy_two
```
This one will be needed to start the second envoy.

- Create envoy virtual environment and activate it:
```sh
python3.8 -m venv envoy_one_env
source envoy_one_env/bin/activate
```

- Install required packages:
```sh
pip install -r requirements.txt
cd envoy
pip install -r sd_requirements.txt
```


- Run the next command from [envoy](./envoy) directory to start envoy:
```sh
fx envoy start --shard-name env_one --disable-tls --envoy-config-path envoy_config_one.yaml --director-host localhost --director-port 50051
```

The envoy `env_one` will be connected to the director via given host and port. Also, you can choose the GPU to be used during the experiment, you can specify it in [envoy_config_one.yaml](./envoy/envoy_config_one.yaml). By default we will use `CUDA:0` on the `env_one`.

### 3\. Start second envoy:

- Open a new terminal.

- Create envoy virtual environment and activate it:
```sh
python3.8 -m venv envoy_two_env
source envoy_two_env/bin/activate
```

- Install required packages:
```sh
pip install -r requirements.txt
cd envoy_two
pip install -r sd_requirements.txt
```

- Run the next command from `envoy_two` directory to start envoy:
```sh
fx envoy start --shard-name env_two --disable-tls --envoy-config-path envoy_config_two.yaml --director-host localhost --director-port 50051
```

The envoy `env_two` will be connected to the director via given host and port. Also, you can choose the GPU to be used during the experiment, you can specify it in [envoy_config_two.yaml](./envoy/envoy_config_two.yaml). By default we will use `CUDA:1` on the `env_one`.

---
**_NOTE:_**  If your want to run this example in distributed mode, you have to change some variables:

- On director node:

    - Change `listen_host` director variable to [FQDN](https://en.wikipedia.org/wiki/Fully_qualified_domain_name) name of machine (see [director_config.yaml](./director/director_config.yaml) file).
    - Optional: change `listen_port` director variable to any free port (see [director_config.yaml](./director/director_config.yaml) file).

- On envoy nodes:

    - Change `--director-host` variable in the command for envoy running on the current director FQDN
    - Change the `--director-port` variable in the command for envoy running on the current director port
---


## Running  FL 2d mednist registration experiment
Now we are ready to start federated experiment
1. Open a new terminal.

2. Create new virtual environment and activate it:
```sh
python3.8 -m venv workspace_env
source workspace_env/bin/activate
```

3. Install required packages:
```sh
pip install -r requirements.txt
```

4. Run `Monai_MedNIST.ipynb` jupyter notebook:
```sh
cd workspace
jupyter notebook Monai_MedNIST.ipynb
```
