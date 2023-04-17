# INSTANCE22 Dataset

This repository provides a benchmarking guide and recipe to train the template algorithms, validation performance, and is tested and maintained by NVIDIA.

## Task Overview

The dataset is from MICCAI 2022 challenge **[INSTANCE22: The 2022 Intracranial Hemorrhage Segmentation Challenge on Non-Contrast head CT (NCCT)](https://instance.grand-challenge.org/)**. The solution described here won 2nd place (1st place in terms of Dice score).

100 non-contrast head CT volumes of clinically diagnosed patients with different kinds of ICH,
(including subdural hemorrhage, epidural hemorrhage, intraventricular hemorrhage, intraparenchymal
hemorrhage, and subarachnoid hemorrhage), are used for model training. The size of a CT volume is 512 x 512 x N, where N lies in [20, 70]. The pixel spacing of a CT volume is 0.42mm x 0.42mm x 5mm. The images will be stored in NIFTI files. Voxel-level segmentation annotations are: 0 - Background; 1 - ICH.

##### Validation performance: NVIDIA DGX-1 (4x V100 16G)

The complete command of **Auto3DSeg** can be found [here](../../README.md#Reference-Python-APIs-for-Auto3DSeg). And our validation results can be obtained by running the training script with MONAI 1.0.0 on NVIDIA DGX-1 with (4x V100 16GB) GPUs.

| Methods| Dimension | GPUs | Batch size / GPU | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg |
|:------:|:---------:|:----:|:----------------:|:------:|:------:|:------:|:------:|:------:|:---:|
| SwinUNETR  | 3 | 4 | 2 | 0.4915 | 0.6457 | 0.6895 | 0.5256 | 0.5935 | 0.5891 |
| SegResNet  | 3 | 4 | 2 | 0.5992 | 0.7536 | 0.0088 | 0.6154 | 0.6985 | 0.5351 |
| DiNTS      | 3 | 4 | 2 | 0.6467 | 0.7491 | 0.7306	| 0.6638 | 0.6779 | 0.6936 |
| **SegResNet2d** | 2 | 4 | 2 | 0.6320 | 0.7778 | 0.7607 | 0.7006 | 0.7613 | **0.7265** |

The winning solution is fully based on 2D SegResNet because the network has a better average validation Dice score compared to other networks.
