# INSTANCE22 Dataset

This repository provides a benmarking guide and recipe to train the template algorithms, validation performance, and is tested and maintained by NVIDIA.

## Task Overview

The dataset is from MICCAI 2022 challenge **[INSTANCE22: The 2022 Intracranial Hemorrhage Segmentation Challenge on Non-Contrast head CT (NCCT)](https://instance.grand-challenge.org/)**. The solution described here won 2nd place (1st place in terms of Dice score).

100 non-contrast head CT volumes of clinically diagnosed patients with different kinds of ICH,
(including subdural hemorrhage, epidural hemorrhage, intraventricular hemorrhage, intraparenchymal
hemorrhage, and subarachnoid hemorrhage), are used for model training. The size of a CT volume is 512 x 512 x N, where N lies in [20, 70]. The pixel spacing of a CT volume is 0.42mm x 0.42mm x 5mm. The images will be stored in NIFTI files. Voxel-level segmentation annotations are: 0 - Background; 1 - ICH.

##### Validation performance: NVIDIA DGX-1 (4x V100 16G)

The complete command of **Auto3DSeg** can be found [here](../README.md#reference-python-apis-for-auto3dseg).

Our validation results can be obtained by running the
```
torchrun --nnodes=1 --nproc_per_node=4 scripts/train.py run --config_file configs/algo_config.yaml
```
training script in the MONAI 1.0.0 container on NVIDIA DGX-1 with (4x V100 16GB) GPUs. Performance numbers (in volumes per second) were averaged over an entire training epoch.

| Methods| Dimension | GPUs | Batch size / GPU | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg |
|:------:|:---------:|:----:|:----------------:|:------:|:------:|:------:|:------:|:------:|:---:|
| SwinUNETR  | 3 | 4 | 2 | 0. | 0. | 0. | 0. | 0. | 0. |
| SegResNet  | 3 | 4 | 2 | 0. | 0. | 0. | 0. | 0. | 0. |
| DiNTS      | 3 | 4 | 2 | 0. | 0. | 0. | 0. | 0. | 0. |
|SegResNet2d | 2 | 4 | 2 | 0. | 0. | 0. | 0. | 0. | 0. |
