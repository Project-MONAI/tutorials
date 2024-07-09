# Medical AI for Synthetic Imaging (MAISI) Data Preparation

Disclaimer: We are not the host of the data. Please make sure to read the requirements and usage policies of the data and give credit to the authors of the dataset!

### 1 VAE training Data
The VAE training dataset used in MAISI contains 37243 CT training data and 1963 CT validation data from chest, abdomen, head and neck region; and 17887 MRI training data and 940 MRI validation data from brain, skul-stripped brain, chest, and below-abdomen region.

### 2 Diffusion model training Data
### 3 ControlNet model training Data

The ControlNet training dataset used in MAISI contains 6330 CT volumes (5058 and 1272 volumes are used for training and validation, respectively) across 20 datasets and covers different body regions and diseases. 

The table below summarizes the number of volumes for each dataset.

|Index| Dataset name|Number of volumes|
|:-----|:-----|:-----|
1 | AbdomenCT-1K | 789
2 | AeroPath | 15
3 | AMOS22 | 240
4 | Bone-Lesion	| 237
5 | BTCV | 48
6 | CT-ORG | 94
7 | CTPelvic1K-CLINIC | 94
8 | LIDC | 422
9 | MSD Task03 | 105
10 | MSD Task06 | 50
11 | MSD Task07 | 225
12 | MSD Task08 | 235
13 | MSD Task09 | 33
14 | MSD Task10 | 101
15 | Multi-organ-Abdominal-CT | 64
16 | Pancreas-CT | 51
17 | StonyBrook-CT | 1258
18 | TCIA_Colon | 1436
19 | TotalSegmentatorV2 | 654
20| VerSe | 179

### 4. Questions and bugs

- For questions relating to the use of MONAI, please use our [Discussions tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).

### Reference
[1] [Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." CVPR 2022.](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
