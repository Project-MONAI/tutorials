
# Write a Bundle workflow in Python

MONAI provides the `BundleWorkflow` interface from version 1.2, which is used as the standard API for other applications to access and operate a bundle workflow, like training, evaluation, or inference. `BundleWorkflow` can help reducing the effort of application development and defining standard interface and required properties for the bundle development. Once a bundle workflow inherits from `BundleWorkflow`, we can accept it in the associated applications, no matter whether the bundle workflow is implemented by JSON / YAML or python code.

MONAI already implemented `ConfigWorkflow` for the common JSON / YAML config-based bundle workflow. This example shows how to write python-based training and inference workflows and implement the interfaces: `initialize`, `run`, `finalize`, `_set_property`, `_get_property` and all the required properties in: https://github.com/Project-MONAI/MONAI/blob/dev/monai/bundle/properties.py, then execute them with bundle CLI command.

If users want to check whether all the required properties are existing in the bundle workflow, they can call the `check_properties` API.

## Execute training and inference with bundle script - `run_workflow`

There are several predefined scripts in the MONAI bundle module, here we leverage the `run_workflow` script to execute the python-based workflows in `scripts/train.py` and `scripts/inference.py`.

To run the workflows, `PYTHONPATH` should be revised to include the path to the scripts:
```
export PYTHONPATH=$PYTHONPATH:"<path to 'python_bundle_workflow/scripts'>"
```
And please make sure the folder `python_bundle_workflow/scripts` is a valid python module (it has a `__init__.py` file in the folder).

Execute the training:
```shell
python -m monai.bundle run_workflow "scripts.train.TrainWorkflow"
```

Execute the inference with the trained model:
```shell
python -m monai.bundle run_workflow "scripts.inference.InferenceWorkflow"
```
