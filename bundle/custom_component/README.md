# Description
This example mainly shows a typical use case that brings customized python components (such as transform, network, metrics) in a configuration-based workflow.

Please note that this example depends on the `spleen_segmentation` bundle example and executes via overriding the config file of it.

## commands example
To run the workflow with customized components, `PYTHONPATH` should be revised to include the path to the customized component:
```
export PYTHONPATH=$PYTHONPATH:"<path to 'custom_component/scripts'>"
```
And please make sure the folder `custom_component/scripts` is a valid python module (it has a `__init__.py` file in the folder).

Override the `train` config with the customized `transform` and execute training:
```bash
python -m monai.bundle run training --meta_file <spleen_configs_path>/metadata.json \
    --config_file "['<spleen_configs_path>/train.json','configs/custom_train.json']" \
    --logging_file <spleen_configs_path>/logging.conf
```
