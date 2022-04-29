# Description
This example mainly shows 2 typical use cases of hybrid programming with MONAI bundle:
- Apply customized python component(like: transform, network, metrics, etc.) in the `train` config.
- Parse the config in your own python program, instantiate necessary components with python program and execute the training.

Please note that this example depends on the `spleen_segmentation` bundle example and show the hybrid programming via overriding the config file of it.

## commands example
Override the `train` config with customized `transforms` and execute training:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file "['configs/train.json','configs/custom_train.json']" --logging_file configs/logging.conf
```

Parse the config in the python program and execute inference from the program:

```
python -m scripts.inference.run --config_file configs/inference.json
```
