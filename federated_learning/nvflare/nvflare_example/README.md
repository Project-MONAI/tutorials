# Federated Learning with MONAI using NVFlare (without docker)
The purpose of this tutorial is to show how to run NVFlare with MONAI on a local machine to simulate a FL setting (server and client communicate over localhost).
It is based on the [tutorial]() showing how to run FL with MONAI and NVFlare which using a docker container for the server and each client.

## Installation
(If needed) install pip and virtualenv (on macOS and Linux):
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```
set up virtual environment
```
source ./set_env.sh
```
install required packages
```
pip install --upgrade pip
pip install -r ${projectpath}/requirements.txt
```

## Download the data
Download the Spleen segmentation task dataset from http://medicaldecathlon.com and place under `./data/Task09_Spleen` following the prepared folder structure.

## Start server and clients
To start the server and clients, run the following script.
```
export n_clients=2
${projectpath}/fl_workspace/start_fl.sh ${n_clients}
```
*Note:* We have prepared startup kits for up to 8 clients. If you need to modify the number of clients, please follow instructions here and modify the provided project.yml file. Then re-run the provisioning tool: `provision -p project.yml -a authz_config.json` (see [here](https://docs.nvidia.com/clara/clara-train-sdk/federated-learning/fl_provisioning_tool.html) for details).
Optionally, modify the `export CUDA_VISIBLE_DEVICES` command in `start_fl.sh` to set which GPU a client should run on.

## Start admin tool
In new terminal, start environment again
```
source ./set_env.sh
```
Then, start admin tool
```
${projectpath}/fl_workspace/admin/startup/fl_admin.sh
```
*Note:* user name: admin@nvidia.com

Then, use admin to control the FL process:
```
> check_status server
> check_status client
> set_run_number 1 
```
Upload and deploy the training configurations. The files implementing the configuration parser and MONAI-based trainer are under `fl_workspace/admin/transfer`.
```
> upload_folder spleen_example
> deploy spleen_example server
> deploy spleen_example client
```
Inside the server/client terminal, deploy the training configurations that specify the data json for each client
```
${projectpath}/fl_workspace/deploy_train_configs.sh ${n_clients}
```
Next, you can start the FL server in the admin terminal and begin training:
```
> start server
> start client
```
(Optional) monitor the training progress
```
> check_status server
> check_status client
```
(Optional) shutdown FL system:
```
> shutdown client
admin@nvidia.com
> shutdown server
```
(Optional) clean up previous runs
```
cd fl_workspace
./clean_up.sh
```

## Visualize the training progress
To visualize the training progress, run tensorboard in the server/client terminal:
```
tensorboard --logdir="./"
```
and point your browser to `http://localhost:6006/#scalars`. You should see the performance of the global model to be the same at the beginning of each round, as the clients in this example all share the same validation set.
![Validation curve for two clients](tensorboard.png)

For more details visit the [NVFlare documentation](https://pypi.org/project/nvflare).
For more examples using NVFlare, see [here](https://github.com/NVIDIA/clara-train-examples/tree/master/PyTorch/NoteBooks).
