
# Get started to MONAI bundle

A MONAI bundle usually includes the stored weights of a model, TorchScript model, JSON files which include configs and metadata about the model, information for constructing training, inference, and post-processing transform sequences, plain-text description, legal information, and other data the model creator wishes to include.

For more information about MONAI bundle, please read the description: https://docs.monai.io/en/latest/bundle_intro.html.

This is a step-by-step tutorial to help get started to develop a bundle package, which contains a config file to construct the training pipeline and also has a `metadata.json` file to define the metadata information.

Mainly contains the below sections:
- Define a training config with `JSON` or `YAML` format.
- Execute training based on bundle scripts and configs.
- Execute other scripts for bundle functionalities.
- Hybrid programming with config and python code.

You can find the usage examples of MONAI bundle key features and syntax in this tutorial, like:
- Instantiate a python object from a dictionary config with `_target_` indicating class or function name or module path.
- Execute python expression from a string config with the `$` syntax.
- Refer to other python object with the `@` syntax.
- Macro text replacement with the `%` syntax to simplify the config content.
- Leverage the `_disabled_` syntax to tune or debug different components.
- Override config content at runtime.
- Hybrid programming with config and python code.

## Download dataset

Downloads and extracts the dataset for this example.
The dataset comes from http://medicaldecathlon.com/.
Here specify a directory with the `MONAI_DATA_DIRECTORY` environment variable to save downloaded dataset and outputs, if no environment, save to the temorary directory.

```python
import os
import tempfile
from monai.apps import download_and_extract

resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
data_dir = os.path.join(root_dir, "Task09_Spleen")
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir, md5)
```

## Define train config - Set imports and input / output environments

Now let's start to define the config file for a regular training task. MONAI bundle support both `JSON` and `YAML` format, here we use `JSON` as the example. the [whole config for training](spleen_segmentation/configs/train.json) is available and can be a reference.

According to the predefined syntax of MONAI bundle, `$` indicates an expression to evaluate and `@` refers to another object in the config content. For more details about the syntax in bundle config, please check: https://docs.monai.io/en/latest/config_syntax.html.

Please note that a MONAI bundle doesn't require any hard-coded logic in the config, so users can define the config content in any structure.

For the first step, import `os` and `glob` to use in the `python expressions` (start with `$`), then define input / output environments and enable `cudnn.benchmark` for better performance.

The `dataset_dir` in the config is the directory of downloaded dataset. Please check the `root_dir` and update this accordingly when you are writing your config.

Note that the `imports` are only used to execute the `python expressions`, and already imported `monai`, `numpy`, `np`, `torch` internally as these are mininum dependencies of MONAI.

```json
{
    "imports": [
        "$import glob",
        "$import os",
        "$import ignite"
    ],
    "device": "$torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')",
    "ckpt_path": "/workspace/data/models/model.pt",
    "dataset_dir": "/workspace/data/Task09_Spleen",
    "images": "$list(sorted(glob.glob(@dataset_dir + '/imagesTr/*.nii.gz')))",
    "labels": "$list(sorted(glob.glob(@dataset_dir + '/labelsTr/*.nii.gz')))"
}
```

## Define train config - Define network, optimizer, loss function

Define `UNet` of MONAI as the training network, and use the `Adam` optimizer of PyTorch, `DiceCELoss` of MONAI.

An instantiable config component uses `_target_` keyword to define the class / function name or module path, other keys are args for the component.

Note that for all the MONAI classes and functions, we can use its name in `_target_` directly, for any other packages, please provide the `full module path` in `_target_`.

```json
"network_def": {
    "_target_": "UNet",
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 2,
    "channels": [16, 32, 64, 128, 256],
    "strides": [2, 2, 2, 2],
    "num_res_units": 2,
    "norm": "batch"
}
```

Move the network to the expected device which was defined earlier by `"device": "$torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"`.

```json
"network": "$@network_def.to(@device)"
```

Define optimizer and loss function, for MONAI classes, we can use the class name directly, other classes should provide the module path (like `Adam`).

```json
"loss": {
    "_target_": "DiceCELoss",
    "to_onehot_y": true,
    "softmax": true,
    "squared_pred": true,
    "batch": true
},
"optimizer": {
    "_target_": "torch.optim.Adam",
    "params": "$@network.parameters()",
    "lr": 1e-4
}
```

## Define train config - Define data loading and preprocessing logic

Define `transforms` and `dataset`, `dataloader` to generate training data for network.

To make the config stucture clear, here we split the `train` and `validate` related components into 2 sections:
```json
"train": {...},
"validate": {...}
```
The composed transforms are for preprocessing.

```json
"train": {
    "preprocessing": {
        "_target_": "Compose",
        "transforms": [
            {
                "_target_": "LoadImaged",
                "keys": ["image", "label"]
            },
            {
                "_target_": "EnsureChannelFirstd",
                "keys": ["image", "label"]
            },
            {
                "_target_": "Orientationd",
                "keys": ["image", "label"],
                "axcodes": "RAS"
            },
            {
                "_target_": "Spacingd",
                "keys": ["image", "label"],
                "pixdim": [1.5, 1.5, 2.0],
                "mode": ["bilinear", "nearest"]
            },
            {
                "_target_": "ScaleIntensityRanged",
                "keys": "image",
                "a_min": -57,
                "a_max": 164,
                "b_min": 0,
                "b_max": 1,
                "clip": true
            },
            {
                "_target_": "RandCropByPosNegLabeld",
                "keys": ["image", "label"],
                "label_key": "label",
                "spatial_size": [96, 96, 96],
                "pos": 1,
                "neg": 1,
                "num_samples": 4,
                "image_key": "image",
                "image_threshold": 0
            },
            {
                "_target_": "EnsureTyped",
                "keys": ["image", "label"]
            }
        ]
    }
}
```

The train and validation image file names are organized into a list of dictionaries.

Here we use `dataset` instance as 1 argument of `dataloader` by the `@` syntax, and please note that `"#"` in the reference id are interpreted as special characters to go one level further into the nested config structures. For example: `"dataset": "@train#dataset"`.

```json
"dataset": {
    "_target_": "CacheDataset",
    "data": "$[{'image': i, 'label': l} for i, l in zip(@images[:-9], @labels[:-9])]",
    "transform": "@train#preprocessing",
    "cache_rate": 1.0,
    "num_workers": 4
},
"dataloader": {
    "_target_": "DataLoader",
    "dataset": "@train#dataset",
    "batch_size": 2,
    "shuffle": false,
    "num_workers": 4
}
```

## Define train config - Define inference method, post-processing and event-handlers

Here we use `SimpleInferer` to execute `forward()` computation for the network and add post-processing methods like `activation`, `argmax`, `one-hot`, etc. And logging into stdout and TensorBoard based on event handlers.

```json
"inferer": {
    "_target_": "SimpleInferer"
},
"postprocessing": {
    "_target_": "Compose",
    "transforms": [
        {
            "_target_": "Activationsd",
            "keys": "pred",
            "softmax": true
        },
        {
            "_target_": "AsDiscreted",
            "keys": ["pred", "label"],
            "argmax": [true, false],
            "to_onehot": 2
        }
    ]
},
"handlers": [
    {
        "_target_": "StatsHandler",
        "tag_name": "train_loss",
        "output_transform": "$monai.handlers.from_engine(['loss'], first=True)"
    },
    {
        "_target_": "TensorBoardStatsHandler",
        "log_dir": "eval",
        "tag_name": "train_loss",
        "output_transform": "$monai.handlers.from_engine(['loss'], first=True)"
    }
]
```

## Define train config - Define Accuracy metric for training data to avoid over-fitting

Here we define the `Accuracy` metric to compute on training data to help check whether the converge is expected and avoid over-fitting. Note that it's not validation step during the training.

```json
"key_metric": {
    "train_accuracy": {
        "_target_": "ignite.metrics.Accuracy",
        "output_transform": "$monai.handlers.from_engine(['pred', 'label'])"
    }
}
```

## Define train config - Define the trainer

Here we use MONAI engine `SupervisedTrainer` to execute a regular training.

If users have customized logic, then can put the logic in the `iteration_update` arg or implement their own `trainer` in python code and set `_target_` to the class directly.

```json
"trainer": {
    "_target_": "SupervisedTrainer",
    "max_epochs": 100,
    "device": "@device",
    "train_data_loader": "@train#dataloader",
    "network": "@network",
    "loss_function": "@loss",
    "optimizer": "@optimizer",
    "inferer": "@train#inferer",
    "postprocessing": "@train#postprocessing",
    "key_train_metric": "@train#key_metric",
    "train_handlers": "@train#handlers",
    "amp": true
}
```

## Define train config - Define the validation section

Usually we need to execute validation for every N epochs during training to verify the model and save the best model.

Here we don't define the `validate` section step by step as it's similar to the `train` section, please refer to the full training config of the spleen bundle example.

Just show an example of `macro text replacement` to simplify the config content and avoid duplicated text. Please note that it's just token text replacement of the config content, not refer to the instantiated python objects.

```json
"validate": {
    "preprocessing": {
        "_target_": "Compose",
        "transforms": [
            "%train#preprocessing#transforms#0",
            "%train#preprocessing#transforms#1",
            "%train#preprocessing#transforms#2",
            "%train#preprocessing#transforms#3",
            "%train#preprocessing#transforms#4",
            "%train#preprocessing#transforms#6"
        ]
    }
}
```

## Define metadata information

We can define a `metadata` file in the bundle, which contains the metadata information relating to the model, including what the shape and format of inputs and outputs are, what the meaning of the outputs are, what type of model is present, and other information. The structure is a dictionary containing a defined set of keys with additional user-specified keys.

A typical [metadata example](spleen_segmentation/configs/metadata.json) is available.

## Execute training with bundle script - `run`

There are several predefined scripts in MONAI bundle module, here we leverage the `run` script and specify the ID of trainer in the config.

Just define the entry point expressions in the config to execute in order, and specify the `runner_id` in CLI script.

```json
"training": [
    "$monai.utils.set_determinism(seed=123)",
    "$setattr(torch.backends.cudnn, 'benchmark', True)",
    "$@train#trainer.run()"
]
```

```shell
python -m monai.bundle run training --config_file configs/train.json
```

## Execute training with bundle script - Override config at runtime

To override some config items at runtime, users can specify the target `id` and `value` at command line, or override the `id` with some content in another config file. Here we set the device to `cuda:1` at runtime.

Please note that "#" and "$" may be meaningful syntax for some `shell` and `CLI` tools, so may need to add escape character or quotes for them in the command line, like: `"\$torch.device('cuda:1')"`. For more details: https://github.com/google/python-fire/blob/v0.4.0/fire/parser.py#L60.
```shell
python -m monai.bundle run training --config_file configs/train.json --device "\$torch.device('cuda:1')"
```
Override content from another config file.
```shell
python -m monai.bundle run training --config_file configs/train.json --network "%configs/test.json#network"
```

## Execute other bundle scripts

Besides `run`, there are also many other scripts for bundle functionalities. All the scripts are available at: https://docs.monai.io/en/latest/bundle.html#scripts.

Here is some typical examples:

1. Initialize a bundle directory based on the template and pretrained checkpoint weights.
```shell
python -m monai.bundle init_bundle --bundle_dir <target dir> --ckpt_file <checkpoint path>
```

2. Export the model checkpoint to a `TorchScript` model at the given filepath with metadata and config included as JSON files.
```shell
python -m monai.bundle ckpt_export network --filepath <export path> --ckpt_file <checkpoint path> --config_file <config path>
```

3. Verify the format of provided `metadata` file based on the predefined `schema`.
```shell
python -m monai.bundle verify_metadata --meta_file <meta path>
```

4. Verify the input and output data shape and data type of network defined in the metadata. It will test with fake Tensor data according to the required data shape in `metadata`.
```shell
python -m monai.bundle verify_net_in_out network --meta_file <metadata path> --config_file <config path>
```
The acceptable data shape in the metadata can support `"*"` for any size, or use an expression with Python mathematical operators and one character variables to represent dependence on an unknown quantity, for example, `"2**p"` represents a size which must be a power of 2, `"2**p*n"` must be a multiple of a power of 2. `"spatial_shape": [ "32 * n", "2 ** p * n", "*"]`.


5. Download a bundle from Github release or URL.
```shell
python -m monai.bundle download --name <bundle_name> --version "0.1.0" --bundle_dir "./"
```

## Hybrid programming with config and python code

A MONAI bundle supports flexible customized logic, there are several ways to achieve this:

- If defining own components like transform, loss, trainer, etc. in a python file, just use its module path in `_target_` within the config file.
- Parse the config in your own python program and do lazy instantiation with customized logic.

Here we show an example to parse the config in python code and execute the training.


```python
from monai.bundle import ConfigParser

parser = ConfigParser()
parser.read_config(f="configs/train.json")
parser.read_meta(f="configs/metadata.json")
```

`get`/`set` configuration content, the `set` method should happen before calling `parse()`.


```python
# original input channels 1
print(parser["network_def"]["in_channels"])
# change input channels to 4
parser["network_def"]["in_channels"] = 4
print(parser["network_def"]["in_channels"])
```

Parse the config content and instantiate components.


```python
# parse the structured config content
parser.parse()
# instantiate the network component and print the network structure
net = parser.get_parsed_content("network")
print(net)

# execute training
trainer = parser.get_parsed_content("train#trainer")
trainer.run()
```
