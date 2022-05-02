# Description
This example mainly shows a typical use case that parses the config files in your own python program, instantiates necessary components with python program and executes the inference.

## commands example

Parse the config files in the python program and execute inference from the python program:

```
python -m scripts.inference run --config_file "['configs/data_loading.json','configs/net_inferer.json','configs/post_processing.json']" --ckpt_path <path_to_checkpoint>
```
