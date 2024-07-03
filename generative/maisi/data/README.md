# Medical AI for Synthetic Imaging (MAISI) Data Preparation

Disclaimer: We are not the host of the data. Please make sure to read the requirements and usage policies of the data and give credit to the authors of the dataset!

### 1 VAE training Data
The VAE training dataset used in MAISI contains 37243 CT training data and 1963 CT validation data from chest, abdomen, head and neck region; and 17887 MRI training data and 940 MRI validation data from brain, skul-stripped brain, chest, and below-abdomen region.

### 2 Diffusion model training Data
### 3 ControlNet model training Data
The preprocessed subset of [C4KC-KiTS](https://www.cancerimagingarchive.net/collection/c4kc-kits/) dataset used in this 
finetuning config is provided in `./dataset/C4KC-KiTS_subset`. The URL is specified at `large_files.yml`. The structure of example folder in the preprocessed dataset is: 

### 4. Questions and bugs

- For questions relating to the use of MONAI, please use our [Discussions tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).

### Reference
[1] [Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." CVPR 2022.](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
