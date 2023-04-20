
# Write a Bundle workflow in Python

MONAI provides the `BundleWorkflow` interface from version 1.2, which is used as the standard API for other applications to access and operate a bundle workflow, like training, evaluation, or inference. `BundleWorkflow` can help reduce the effort of application development and define standard interface and required properties for bundle development. Once a bundle workflow inherits from `BundleWorkflow`, we can accept it in the associated applications, no matter the bundle workflow is implemented by JSON / YAML or python code.

MONAI already implemented `ConfigWorkflow` for the common JSON / YAML config-based bundle workflow, this example shows how to write python-based training and inference workflows, and execute them with bundle CLI command.

## Execute training and inference with bundle script - `run_workflow`


There are several predefined scripts in the MONAI bundle module, here we leverage the `run_workflow` script to execute the python-based workflows in `scripts/train.py` and `scripts/inference.py`.

To run the workflows, `PYTHONPATH` should be revised to include the path to the scripts:
```
export PYTHONPATH=$PYTHONPATH:"<path to 'python_bundle_workflow/scripts'>"
```
And please make sure the folder `python_bundle_workflow/scripts` is a valid python module (it has a `__init__.py` file in the folder).

```shell
python -m monai.bundle run_workflow "scripts.train.TrainWorkflow"
```

```shell
python -m monai.bundle run --config_file configs/train.json
```
