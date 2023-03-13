# Training Auto3dSeg in AzureML

It is possible to train Auto3dSeg using your AzureML resources. This page documents the minimal set up required to utilise AzureML hardware to run your Auto3dSeg training tasks.

## AzureML Setup

Before being able to train in the cloud you will need to have your AzureML resources set up properly. If you have not already completed this, please follow [this tutorial](https://hi-ml.readthedocs.io/en/latest/azure_setup.html) from Microsoft's Medical Imaging research team.

### GPU requirements

The GPU requirements needed to successfully run Auto3dSeg depend on the size of your data. For running the data used by the [Hello World Example](https://github.com/Project-MONAI/tutorials/blob/main/auto3dseg/notebooks/auto3dseg_hello_world.ipynb) then a single GPU with 8GB RAM is sufficient. The GPU nodes you have available in AzureML will depend upon your region and Azure subscription. For larger datasets it is highly recommended to use nodes with multiple GPUs and more RAM (>= 16GB).

## Configuration

### Pre-requisites

As well as the requirements outlined in the [tutorials README](https://github.com/Project-MONAI/tutorials), you will need to install [hi-ml-azure](https://pypi.org/project/hi-ml-azure/) in your environment by running the following command:

- `pip install "hi-ml-azure>=0.2.19"`

### Configure environment - `environment(-azure).yaml`

- To be discussed with Dong et. al.

### Configure AzureML connection - `config.json`

To connect Auto3dSeg to your AzureML workspace you will need to download the configuration file associated with that workspace by following the instructions [here](https://hi-ml.readthedocs.io/en/latest/azure_setup.html#accessing-the-workspace). Place this file in your new directory:

```shell
.
└── your_new_dir/
    └── config.json
```

### Configure data - `datalist.json`

The same as when running Auto3dSeg locally, you will need to define a `datalist.json` file for your data and place it in your directory. Check out the [tasks folder](https://github.com/Project-MONAI/tutorials/tree/main/auto3dseg/tasks) for example datalists. Make sure that all paths are relative to the head of your Azure storage blob that contains your dataset.

Your directory should now look like so:

```shell
.
└── your_new_dir/
    ├── config.json
    └── datalist.json
```

### Configure Auto3dSeg - `task.yaml`

All arguments passed to the Auto3dSeg autorunner and the AzureML submission will need to be defined in a configuration file. Create a new `task.yaml` file from this template:

```yaml
name: <your_task_name>
task: segmentation

modality: <MRI or CT>
datalist: <your_new_datalist>.json
dataroot: /path/to/local/data # Only necessary if you also want to run locally

multigpu: <True/False> # set to true if you want to use multiple GPUs for training

azureml_config:
  compute_cluster_name: "<name of your compute cluster>"
  input_datasets:
  - "<name of your AzureML data asset>" # currently only 1 input dataset is supported
  default_datastore: "<name of your AzureML datastore>"
```

Your directory should now look like this:

```dir
.
└── your_new_dir/
    ├── config.json
    ├── datalist.json
    └── task.yaml
```

## Running

Once the above configuration is completed, to run the Auto3dSeg AutoRunner in AzureML simply use the following command (in `your_new_dir/`):

```python
python -m monai.apps.auto3dseg AutoRunner run --input='./task.yaml` --azureml
```

Make sure to follow any instructions in the terminal regarding authenticating on AzureML. After the job is successfully uploaded, you should see the following output:

```shell
URL to job: https://ml.azure.com/runs/<run_id>?wsid=/subscriptions/<subscription_id>/resourcegroups/<resource_group>/workspaces/<workspace>&tid=<tenant_id>
```

Following this link will take you to your AzureML job where the AutoRunner is running. Below the job name you will see the following tabs, the most important of which are explained below.

![azureml_job_tabs](../figures/azureml_job_tabs.png)

### Outputs

### Metrics

### Hardware Monitoring

## Additional Configuration Options

all other configuration options + some examples?
