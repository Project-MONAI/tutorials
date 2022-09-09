# BTCV Dataset For Auto3dseg

This repository provides a benmarking guide and recipe to train the template algorithms, validation performance, and is tested and maintained by NVIDIA.


## Task Overview

The task is the volumetric (3D) segmentation of the prostate central gland and peripheral zone from the multi-contrast MRI (T2, ADC). The segmentation of prostate region is formulated as the voxel-wise 3-class classification. Each voxel is predicted as either foreground (prostate central gland, peripheral zone) or background. And the model is optimized with gradient descent method minimizing soft dice loss between the predicted mask and ground truth segmentation. The dataset is from the 2018 MICCAI challenge [Medical Image Segmentation (MSD)](http://medicaldecathlon.com/).

- Target:
    1. Prostate central gland
    2. Prostate peripheral zone
- Modality: MRI
- Size: 30 3D volumes (32 Training + 16 Testing)
- Challenge: MSD MICCAI Challenge

##### Validation performance: NVIDIA DGX-1 (4x V100 16)

Our results were obtained by running the
```
torchrun --nnodes=1 --nproc_per_node=4 scripts/train.py run --config_file configs/algo_config.yaml
```
training script in the MONAI 0.9.1 container on NVIDIA DGX-1 with (4x V100 32G) GPUs. Performance numbers (in volumes per second) were averaged over an entire training epoch.

| Methods| Dimension | GPUs | Batch size / GPU | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg |
|:------:|:---------:|:----:|:----------------:|:------:|:------:|:------:|:------:|:------:|:---:|
| SegResNet   | 3 | 4 | 2 | 0.76004 | 0.67638 | 0.68831 | 0.68003 | 0.75392 | 0.71174 |
| DiNTS       | 3 | 4 | 2 | 0.71309 | 0.71224 | 0.73416 | 0.70840 | 0.74065 | 0.72171 |
| SegResNet2d | 3 | 4 | 2 | 0.71290 | 0.65638	| 0.70347 | 0.62220 | 0.70122 | 0.67923 |
