# FAQ

## Can I use a dictionary input instead of a "input.yaml" file?
Yes, ```nnUNetV2runner``` is relying on [Google Fire Python library](https://github.com/google/python-fire), which supports dictionary based input. The following is a concrete example.

```bash
## [pipeline] one-click solution with dict input
MODALITY="CT"
DATALIST="./msd_task09_spleen_folds.json"
DATAROOT="/workspace/data/nnunet_test/test09"

python -m monai.apps.nnunet nnUNetRunner run --input "{'modality': '${MODALITY}', 'datalist': '${DATALIST}', 'dataroot': '${DATAROOT}'}"


```
