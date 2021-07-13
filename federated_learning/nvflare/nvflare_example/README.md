# Federated Learning with MONAI using NVFlare (without docker)
The purpose of this tutorial is to show how to run [NVFlare](https://pypi.org/project/nvflare) with MONAI on a local machine to simulate a FL setting (server and client communicate over localhost).
It is based on the [tutorial](../nvflare_example_docker) showing how to run FL with MONAI and NVFlare which using a docker container for the server and each client.

## Installation
(If needed) install pip and virtualenv (on macOS and Linux):
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```
(If needed) make shell scripts executable using
```
chmod +x ./*.sh
chmod +x ./*/*.sh
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
Download the Spleen segmentation task dataset from http://medicaldecathlon.com. 
```
./data/download_dataset.sh
```

## Start server and clients
To start the server and clients, run the following script.
```
export n_clients=2
${projectpath}/start_fl.sh ${n_clients}
```
*Note:* We have prepared startup kits for up to 8 clients. If you need to modify the number of clients, please follow instructions here and modify the provided project.yml file. Then re-run the provisioning tool: `provision -p project.yml -a authz_config.json` (see [here](https://docs.nvidia.com/clara/clara-train-sdk/federated-learning/fl_provisioning_tool.html) for details).
Optionally, modify the `export CUDA_VISIBLE_DEVICES` command in `start_fl.sh` to set which GPU a client should run on.

## Start admin client
In new terminal, start environment again
```
source ./set_env.sh
```
Then, start admin client
```
${projectpath}/admin/startup/fl_admin.sh
```
*Note:* user name: admin@nvidia.com

Use the admin client to control the FL process:
```
> check_status server
> check_status client
> set_run_number 1 
```
*Note:* For more details about the admin client and its commands, see [here](https://docs.nvidia.com/clara/clara-train-sdk/federated-learning/fl_admin_commands.html).

Next, upload and deploy the training configurations. The files implementing the configuration parser and MONAI-based trainer are under `fl_workspace/admin/transfer`.
```
> upload_folder spleen_example
> deploy spleen_example server
> deploy spleen_example client
```
Inside the server/client terminal, deploy the training configurations that specify the data json for each client
```
${projectpath}/deploy_train_configs.sh ${n_clients}
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
admin@nvidia.com
```
(Optional) clean up previous runs
```
cd fl_workspace
./clean_up.sh ${n_clients}
```

## Automate running FL
The following commands automate the above described steps. Executing the `run_fl.sh` script will automatically start the server and clients, upload the configuration folders and deploy them with the client-specific data list. It will also set the minimum number of clients needed for each global model update depending on the given argument.
```
export n_clients=2
${projectpath}/run_fl.sh ${n_clients}
```
*Note:* This script will automatically shutdown the server and client in case of an error or misconfiguration. You can check if a nvflare process is still running before starting a new experiment via `ps -as | grep nvflare`. It is best to not keep old processes running while trying to start a new experiment.

## Visualize the training progress
To visualize the training progress, run tensorboard in the server/client terminal:
```
tensorboard --logdir="./"
```
and point your browser to `http://localhost:6006/#scalars`. You should see the performance of the global model to be the same at the beginning of each round, as the clients in this example all share the same validation set.
![Validation curve for two clients](tensorboard.png)

## Further reading
For more details visit the [NVFlare documentation](https://pypi.org/project/nvflare).
For more examples using NVFlare, see [here](https://github.com/NVIDIA/clara-train-examples/tree/master/PyTorch/NoteBooks/FL).
