# BTCV Dataset

This repository provides a benmarking guide and recipe to train the template algorithms, validation performance, and is tested and maintained by NVIDIA.


## Task Overview

For BTCV dataset, under Institutional Review Board (IRB) supervision, 50 abdomen CT scans of were randomly selected from a combination of an ongoing colorectal cancer chemotherapy trial, and a retrospective ventral hernia study. The 50 scans were captured during portal venous contrast phase with variable volume sizes (512 x 512 x 85 - 512 x 512 x 198) and field of views (approx. 280 x 280 x 280 mm3 - 500 x 500 x 650 mm3). The in-plane resolution varies from 0.54 x 0.54 mm2 to 0.98 x 0.98 mm2, while the slice thickness ranges from 2.5 mm to 5.0 mm.

- Target: 13 abdominal organs including
    1. Spleen
    2. Right Kidney
    3. Left Kidney
    4. Gallbladder
    5. Esophagus
    6. Liver
    7. Stomach
    8. Aorta
    9. IVC
    10. Portal and Splenic Veins
    11. Pancreas
    12. Right adrenal gland
    13. Left adrenal gland.
- Modality: CT
- Size: 30 3D volumes (24 Training + 6 Testing)
- Challenge: BTCV MICCAI Challenge

The following figure shows image patches with the organ sub-regions that are annotated in the CT (top left) and the final labels for the whole dataset (right).

Data, figures and resources are taken from:

![image](https://lh3.googleusercontent.com/pw/AM-JKLX0svvlMdcrchGAgiWWNkg40lgXYjSHsAAuRc5Frakmz2pWzSzf87JQCRgYpqFR0qAjJWPzMQLc_mmvzNjfF9QWl_1OHZ8j4c9qrbR6zQaDJWaCLArRFh0uPvk97qAa11HtYbD6HpJ-wwTCUsaPcYvM=w1724-h522-no?authuser=0)

- [High-resolution 3D abdominal segmentation with random patch network fusion (MIA)](https://www.sciencedirect.com/science/article/abs/pii/S1361841520302589)

- [Efficient multi-atlas abdominal segmentation on clinically acquired CT with SIMPLE context learning (MIA)](https://www.sciencedirect.com/science/article/abs/pii/S1361841515000766?via%3Dihub)

The image patches show anatomies of a subject, including:
1. large organs: spleen, liver, stomach.
2. Smaller organs: gallbladder, esophagus, kidneys, pancreas.
3. Vascular tissues: aorta, IVC, P&S Veins.
4. Glands: left and right adrenal gland

##### Validation performance: NVIDIA DGX-1 (4x V100 32G)

The complete command of **Auto3DSeg** can be found [here](../../README.md#reference-python-apis-for-auto3dseg). And our validation results are obtained on NVIDIA DGX-1 with (4x V100 16GB) GPUs.

| Methods| Dimension | GPUs | Batch size / GPU | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg |
|:------:|:---------:|:----:|:----------------:|:------:|:------:|:------:|:------:|:------:|:---:|
| SwinUNETR | 3 | 4 | 3 | 0.8111 | 0.8011 | 0.6712 | 0.6301 | 0.7239 | 0.7275 |
| SegResNet | 3 | 4 | 3 | 0.8212 | 0.8115 | 0.6848 | 0.6377 | 0.7368 | 0.7384 |
| DiNTS | 3 | 4 | 3 | 0.8058 | 0.7955 | 0.6880 | 0.6281 | 0.7008 | 0.7196 |
|SegResNet2d | 2 | 4 | 3 | 0.6803 | 0.7498 | 0.6188 | 0.6241 | 0.5848 | 0.6516 |
