# Description
This example mainly shows a typical use case that applies customized python component(like: transform, network, metrics, etc.) in the `train` config.

Please note that this example depends on the `spleen_segmentation` bundle example and executes via overriding the config file of it.

## commands example
Export the customized python code to `PYTHONPATH`:
```
export PYTHONPATH=$PYTHONPATH:"<path to 'hybrid_programming/scripts'>"
```

Override the `train` config with one customized `transform` and execute training:
```
python -m monai.bundle run training --meta_file <spleen_configs_path>/metadata.json --config_file "['<spleen_configs_path>/train.json','configs/custom_train.json']" --logging_file <spleen_configs_path>/logging.conf
```
