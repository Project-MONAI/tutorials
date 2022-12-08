# Overview
This tutorial shows how to extend the features of workflow in the model-zoo bundles based on `event-handler` mechanism.

## Event-handler mechanism
placeholder

## Download example monai bundle from model-zoo
```
python -m monai.bundle download --name spleen_ct_segmentation --version "0.1.1" --bundle_dir "./"
```

## Extend the workflow to print the execution time for every iteration, every epoch and total time
placeholder

## Commands example
To run the workflow with customized components, `PYTHONPATH` should be revised to include the path to the customized component:
```
export PYTHONPATH=$PYTHONPATH:"<path to 'spleen_ct_segmentation/scripts'>"
```
And please make sure the folder `spleen_ct_segmentation/scripts` is a valid python module (it has a `__init__.py` file in the folder).

Execute training:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```
