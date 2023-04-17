# FAQ

## Can I use a dictionary input instead of a "input.yaml" file?
Yes, ```nnUNetV2runner``` is relying on [Google Fire Python library](https://github.com/google/python-fire), which supports dictionary based input. The following is a concrete example.

```bash
## [pipeline] one-click solution with dict input
MODALITY="CT"
DATALIST="./msd_task09_spleen_folds.json"
DATAROOT="/workspace/data/Task09_Spleen"

python -m monai.apps.nnunet nnUNetRunner run --input "{'modality': '${MODALITY}', 'datalist': '${DATALIST}', 'dataroot': '${DATAROOT}'}"

```

## I want to know more command examples

Sure! Please check out [this documentation](commands.md).

## How to define multi-modal image inputs in the config file?

The `modality` key in the input `yaml` file accepts a list of strings as input.
[This example](input.yaml) is prepared for the multi-modal MSD prostate dataset.
