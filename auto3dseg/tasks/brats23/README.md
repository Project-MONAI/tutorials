
# BRATS23

The BRATS23 dataset is from MICCAI 2023 challenge **[The International Brain Tumor Segmentation 2023 (BraTS23)](https://www.synapse.org/brats)**, which includes 5 tumor segmentation sub-challenges: Adult Glioma, Metastases, Meningioma, Pediatric, Sub-saharan african. Each sub-challenge includes its own large dataset, where each case consists of 4 brain MRIs (T1, T1c, T2, FLAIR). The solution described here won the 1st place in the Metastasis, Meningioma and Sub-saharan african tumor segmentation challenges, and got 2nd place in the remaining Glioma (adult) and Pediatric glioma sub-challenges. 

Andriy Myronenko, Dong Yang, Yufan He and Daguang Xu: "Auto3DSeg for Brain Turmor Segmentation from 3D MRI in BraTS 2023 Challenge". In MICCAI (2023). [arXiv](https://arxiv.org/)

![brats23_example](./brats23_gli_0.png)

## Task overview

The task is to segment 3 brain tumor substructures: whole tumor (WT) - all labeled areas, tumor core (TC) - red and blue labels in the example, enhancing tumor (ET) - blue label. Please see the challenge description for the anatomical characteristics of each tumor sub-region [2]. Each case includes 4 MRI modalities spatially aligned and resampled to 1x1x1mm resolution by the organizers.  


## Auto3DSeg

The BRATS tutorial is only supported for **SegResNet** algo, Auto3DSeg runs a full workflow including data analysis, and multi-fold training. Please download the dataset into /data/brats23 folder first.


### Running based on the input config

The Auto3DSeg can be run using a config **input.yaml**

```bash
python -m monai.apps.auto3dseg AutoRunner run --input=./input.yaml --algos=segresnet
```

## Validation performance: NVIDIA DGX-1 (8x V100 16G)

The validation results can be obtained by running the training script with MONAI 1.3.0 on NVIDIA DGX-1 with (8x V100 32GB) GPUs. The results below are in terms of average dice.


| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg |
|:------:|:------:|:------:|:------:|:------:|:------:|:---:|
| SegResNet | 0.8997 | 0.8739 | 0.8923 |0.8911 | 0.8892 |0.88924 |


## Data

The BRATS23 challenge dataset [2,3] can be downloaded from [here](https://www.synapse.org/brats). Each user is responsible for checking the content of the datasets and the applicable licenses and determining if suitable for the intended use. The license for the KiTS23 dataset is different than the MONAI license.


## References
[1] Andriy Myronenko, Dong Yang, Yufan He and Daguang Xu: "Auto3DSeg for Brain Turmor Segmentation from 3D MRI in BraTS 2023 Challenge". In MICCAI (2023). https://arxiv.org/


[2] Baid, U., Ghodasara, S., Bilello, M., Mohan, S., Calabrese, E., Colak, E., Farahani, K., Kalpathy-Cramer, J., Kitamura, F.C., Pati, S., Prevedello, L.M., Rudie, J.D., Sako, C., Shinohara, R.T., Bergquist, T., Chai, R., Eddy, J., Elliott, J., Reade, W., Schaffter, T., Yu, T., Zheng, J., Annotators, B., Davatzikos, C., Mongan, J.,
Hess, C., Cha, S., Villanueva-Meyer, J.E., Freymann, J.B., Kirby, J.S., Wiestler, B., Crivellaro, P., Colen, R.R., Kotrotsou, A., Marcus, D.S., Milchenko, M., Naz-eri, A., Fathallah-Shaykh, H.M., Wiest, R., Jakab, A., Weber, M., Mahajan, A., Menze, B.H., Flanders, A.E., Bakas, S.: The RSNA-ASNR-MICCAI brats 2021 benchmark on brain tumor segmentation and radiogenomic classification. CoRR abs/2107.02314 (2021), https://arxiv.org/abs/2107.02314


