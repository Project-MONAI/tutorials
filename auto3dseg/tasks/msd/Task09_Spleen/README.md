# MSD Dataset Task09 Spleen

This repository provides a benmarking guide and recipe to train the template algorithms, validation performance, and is tested and maintained by NVIDIA.


## Task Overview

The task is the volumetric (3D) segmentation of the spleen from CT image. The segmentation of spleen is formulated as the voxel-wise 2-class classification. Each voxel is predicted as either foreground (spleen) or background. And the model is optimized with both Dice loss and Cross Entropy loss between the predicted mask and ground truth segmentation. The dataset is from the 2018 MICCAI challenge [Medical Image Segmentation (MSD)](http://medicaldecathlon.com/).

- Target:
    1. spleen
- Modality: CT
- Size: 128x128x96 3D volumes (40 Training + 20 Testing)
- Challenge: MSD MICCAI Challenge

##### Validation performance: NVIDIA DGX-1 (4x V100 16)

The complete command of **Auto3DSeg** can be found [here](../../README.md#reference-python-apis-for-auto3dseg). And our validation results are obtained on NVIDIA DGX-1 with (4x V100 16GB) GPUs.

| Methods| Dimension | GPUs | Batch size / GPU | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg |
|:------:|:---------:|:----:|:----------------:|:------:|:------:|:------:|:------:|:------:|:---:|
| SegResNet   | 3 | 2 | 2 | 0.96427 | 0.95372 | 0.95498 | 0.95854 | 0.95636 | 0.95757 |
| DiNTS       | 3 | 4 | 2 | 0.93582 | 0.93904 | 0.95294 | 0.89958 | 0.92335 | 0.93015 |
| SegResNet2d | 3 | 4 | 2 | 0.78142 | 0.92268	| 0.89509 | 0.85007 | 0.91384 | 0.87262 |
| SwinUnetR | 3 | 2 | 2 | 0.73086 | 0.84109	| 0.85437 | 0.69816 | 0.75192 | 0.77528 |
