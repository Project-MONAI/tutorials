
# KiTS23 

The KiTS dataset is from MICCAI 2023 challenge **[The 2023 Kidney and Kidney Tumor Segmentation Challenge (KiTS23)](https://kits-challenge.org/kits23/)**. The solution described here won the 1st place in the KiTS challenge [(NVAUTO team)](https://kits-challenge.org/kits23/#kits23-official-results):

Andriy Myronenko, Dong Yang, Yufan He and Daguang Xu: "Automated 3D Segmentation of Kidneys and Tumors in MICCAI KiTS 2023 Challenge". In MICCAI (2023). [arXiv](https://arxiv.org/abs/2310.04110)

![kits23_example](./kits23_example.png)

## Task overview

The task is to segment kidneys, tumors and cysts from 3D CTs.  The ground truth labels are provided for 489 cases with resolutions between 0.39x0.39x0.5 and 1x1x5 mm. 


## Auto3DSeg

The KiTS tutorial is only supported for **SegResNet** algo, Auto3DSeg runs a full workflow including data analysis, and multi-fold training. Please download the dataset into /data/kits23 folder first.


### Running based on the input config

The Auto3DSeg can be run using a config **input.yaml**

```bash
python -m monai.apps.auto3dseg AutoRunner run --input=./input.yaml --algos=segresnet
```

## Validation performance: NVIDIA DGX-1 (8x V100 32G)

The validation results can be obtained by running the training script with MONAI 1.3.0 on NVIDIA DGX-1 with (8x V100 32GB) GPUs. The results below are in terms of average dice.


| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg |
|:------:|:------:|:------:|:------:|:------:|:------:|:---:|
| SegResNet | 0.8997 | 0.8739 | 0.8923 |0.8911 | 0.8892 |0.88924 |


## Data

The KiTS23 challenge dataset [2,3] can be downloaded from [here](https://kits-challenge.org/kits23). Each user is responsible for checking the content of the datasets and the applicable licenses and determining if suitable for the intended use. The license for the KiTS23 dataset is different than the MONAI license.


## References
[1] Andriy Myronenko, Dong Yang, Yufan He and Daguang Xu: "Automated 3D Segmentation of Kidneys and Tumors in MICCAI KiTS 2023 Challenge". In MICCAI (2023). https://arxiv.org/abs/2310.04110


[2] Heller, N., Isensee, F., Maier-Hein, K.H., Hou, X., Xie, C., Li, F., Nan, Y., Mu, G., Lin, Z., Han, M., et al.: The state of the art in kidney and kidney tumor segmentation in contrast-enhanced ct imaging: Results of the kits19 challenge. Medical Image Analysis 67, 101821 (2021)

[3] Heller, N., Wood, A., Isensee, F., Radsch, T., Tejpaul, R., Papanikolopoulos, N.,Weight, C.: The 2023 kidney and kidney tumor segmentation challenge, https://kits-challenge.org/kits23/

