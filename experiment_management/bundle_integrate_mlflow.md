# Description
Add mlflow handler to an bundle.

# Overview
This tutorial shows how to add mlflow handler to a bundle to do experiment management. A mlflow handler is a handler to do experiment management during training and evaluation process in deep learning. Some default parameters and metrics will be logged by this handler. Users can also customize their own parameter, artifacts, experiment name and so on with mlflow handler.

## Usage 1
 Mlflow hander can be easily add to a bundle by modified corresponding json file. Take [spleen_ct_segmentation bundle](https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_ct_segmentation) for example, handler list in `train.json` is shown below.
```
"handlers": [
    {
        "_target_": "ValidationHandler",
        "validator": "@validate#evaluator",
        "epoch_level": true,
        "interval": "@val_interval"
    },
    {
        "_target_": "StatsHandler",
        "tag_name": "train_loss",
        "output_transform": "$monai.handlers.from_engine(['loss'], first=True)"
    },
    {
        "_target_": "TensorBoardStatsHandler",
        "log_dir": "@output_dir",
        "tag_name": "train_loss",
        "output_transform": "$monai.handlers.from_engine(['loss'], first=True)"
    }
],
``` 
After adding mlflow handler to this bundle, handler list should look like below. A MLFlowHandler was added to the end of handler list with a specified experiment name and output transform to get loss scalar.
```
"handlers": [
    {
        "_target_": "ValidationHandler",
        "validator": "@validate#evaluator",
        "epoch_level": true,
        "interval": "@val_interval"
    },
    {
        "_target_": "StatsHandler",
        "tag_name": "train_loss",
        "output_transform": "$monai.handlers.from_engine(['loss'], first=True)"
    },
    {
        "_target_": "TensorBoardStatsHandler",
        "log_dir": "@output_dir",
        "tag_name": "train_loss",
        "output_transform": "$monai.handlers.from_engine(['loss'], first=True)"
    },
    {
        "_target_": "MLFlowHandler",
        "output_transform": "$monai.handlers.from_engine(['loss'], first=True)",
        "experiment_name": "spleen_ct_segmentation"

    }
],
```
When using mlflow handler in evaluation process, the modify is same with training process.

## Usage 2
Mlflow handler can also be added to a bundle through python script without changing the original bundle json file. The code of this usage is shown below.

```
from monai.bundle import ConfigParser
from monai.handlers import MLFlowHandler

parser = ConfigParser()
parser.read_config("spleen_ct_segmentation/configs/train.json")
trainer = parser.get_parsed_content("train#trainer")
mlflow_handler = MLFlowHandler(output_transform=monai.handlers.from_engine(["loss"], first=True),
experiment_name="spleen_ct_segmentation")
mlflow_handler.attach(trainer)
trainer.run()
```

## Add artifacts to mlflow handler
This part shows how to use mlflow handler to record artifacts like images. As an example, we will save some prediction results. `inference.json` in spleen ct segmentation bundle is used here to demo the process. We only need to add a mlflow handler to `handlers` in it and specified `artifacts` parameter in mlflow handler with coresponding path. Then, all files in given path will be recorded in mlflow.

```
    "handlers": [
        {
            "_target_": "CheckpointLoader",
            "load_path": "$@bundle_root + '/models/model.pt'",
            "load_dict": {
                "model": "@network"
            }
        },
        {
            "_target_": "StatsHandler",
            "iteration_log": false
        },
        {
            "_target_": "MLFlowHandler",
            "output_transform": "$monai.handlers.from_engine(['loss'], first=True)",
            "experiment_name": "spleen_ct_segmentation",
            "artifacts":"$[@output_dir]"
        
        }
    ],
```
