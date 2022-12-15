
# WIP


# HECKTOR22 Dataset

HECKTOR22 tutorial is currently only supported for segresnet algo. 
Recommended to run on 8gpu machine.


## Running based on the input config

Default setting. By default Auto3Dseg uses all available GPUs. 

```bash
python -m monai.apps.auto3dseg AutoRunner run --input='./input.yaml' --algos='segresnet' 
```

## Running from python

You can also run it from python, where you can customize more options. Please see the comments in hecktor22.py 
```bash
python hecktor22.py 
```
