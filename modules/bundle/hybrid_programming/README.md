# Description
This example mainly shows a typical use case that parses the config files in your own python program, instantiates necessary components with python program and executes the inference.

## commands example

Parse the config files in the Python program and execute inference from the python program, using `scripts/inference.py`:

```
python -m scripts.inference run --config_file "['configs/data_loading.json','configs/net_inferer.yaml','configs/post_processing.json']" --ckpt_path <path_to_checkpoint>
```

Parse the config file content in Python and overriding the default values, using the entry point defined in `scrips/train_demo.py`.
```bash
python -m scripts.train_demo run \
    --config_file=configs/net_inferer.yaml \
    --network#in_channels=10
```
