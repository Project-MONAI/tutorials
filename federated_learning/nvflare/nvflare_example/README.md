# Federated Learning with MONAI using NVFlare (without docker)
The purpose of this tutorial is to show how to run [NVFlare](https://pypi.org/project/nvflare) with MONAI on a local machine to simulate a FL setting (server and client communicate over localhost).
It is based on the [tutorial](../nvflare_example_docker) showing how to run FL with MONAI and NVFlare which using a docker container for the server and each client.

## Environment setup
(If needed) install pip and virtualenv (on macOS and Linux):
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```
(If needed) make all shell scripts executable using
```
find . -name ".sh" -exec chmod +x {} \;
```
initialize virtual environment and set the current folder as projectpath
```
source ./virtualenv/set_env.sh
export projectpath="."
```
install required packages
```
pip install --upgrade pip
pip install -r ${projectpath}/virtualenv/requirements.txt
```

## FL workspace prepareation for NVFlare
NVFlare has a "provision" mechanisum to automatically generate the fl workspace, see [here](https://docs.nvidia.com/clara/clara-train-sdk/federated-learning/fl_provisioning_tool.html) for details.

In this example for convenience, we included a pregenerated workspace supporting up to 8 clients. 
```
unzip ${projectpath}/fl_workspace_pregenerated.zip
```
If needed (changing the number of max clients, client names, etc.), please follow the instructions from above link. We included the sample project.yml and authz_config.json files used for generating the 8-client workspace under `${projectpath}/fl_utils/workspace_gen`. After modification, the provisioning tool can be run as: `provision -p project.yml -a authz_config.json` 

## Example task - spleen segmentation with MONAI
In this example, we used spleen segmentation task with a MONAI-based client trainer under `${projectpath}/spleen_example`
### Download the data
Download the Spleen segmentation task dataset from http://medicaldecathlon.com. 
```
${projectpath}/spleen_example/data/download_dataset.sh
```
This will creat a `${projectpath}/data` folder containing the dataset and pre-assigned 8-client datalists.
### Copy the MONAI-based trainer files to workspace.
```
cp -r ${projectpath}/spleen_example/ ${projectpath}/fl_workspace/admin/transfer/
```
## Running the federated learning
Two steps for running the federated learning using NVFlare+MONAI: 
1. start the server, clients, and admin under NVFlare workspace
2. start the actual training process with MONAI implementation
### Start server and clients
To start the server and clients, run the following script (example with 2 clients).
```
export n_clients=2
${projectpath}/fl_utils/fl_run/start_fl.sh ${n_clients}
```
For more clients/GPUs, modify the `n_clients` and `export CUDA_VISIBLE_DEVICES` command in `start_fl.sh` to set which GPU a client should run on. Note that multiple clients can run on a single GPU as long as the memory is sufficient.

### Start admin client
In new terminal, start environment again
```
source ./virtualenv/set_env.sh
```
Then, start admin client
```
${projectpath}/fl_workspace/admin/startup/fl_admin.sh
```
*Note:* user name: admin@nvidia.com

Use the admin client to control the FL process:
```
> check_status server
> check_status client
> set_run_number 1 
```
*Note:* For more details about the admin client and its commands, see [here](https://docs.nvidia.com/clara/clara-train-sdk/federated-learning/fl_admin_commands.html).

### Start actual fl with spleen_example
Upload and deploy the training configurations. 
Then in admin, 
```
> upload_folder spleen_example
> deploy spleen_example server
> deploy spleen_example client
```
Inside the server/client terminal, deploy the training configurations that specify the data json for each client
```
${projectpath}/fl_utils/fl_run/deploy_train_configs.sh ${n_clients}
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
${projectpath}/fl_utils/fl_run/clean_up.sh ${n_clients}
```

## Automate running FL
Alternatively, the following commands automate the above described steps. It makes use of NVFlare's AdminAPI. The script will automatically start the server and clients, upload the configuration folders and deploy them with the client-specific data list. It will also set the minimum number of clients needed for each global model update depending on the given argument.
```
export n_clients=2
${projectpath}/fl_utils/fl_run_auto/run_fl.sh ${n_clients}
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
