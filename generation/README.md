# MONAI Generative Model Example
This folder contains examples for training and validating generative models in MONAI. The examples are designed as demonstration to showcase the training process for these types of networks. To achieve optimal performance, it is recommended that users to adjust the network and hyper-parameters based on their device and training dataset.

## Installation

Some tutorials may require extra components on top of what is installed with base MONAI:

```bash
pip install monai[lpips]
```

The MONAI GenerateModels package is no longer required, however this can be installed with `pip install monai-generative==0.2.3`.

----

## [MedNIST 2D latent diffusion model notebook](./2d_ldm/2d_ldm_tutorial.ipynb)
Example notebook demonstrating diffusion with MedNIST toy dataset.

## [Brats 2D latent diffusion model](./2d_ldm/README.md)
Example shows the use cases of training and validating a 2D Latent Diffusion Model on axial slices from Brats 2016&2017 data.

## [Brats 3D latent diffusion model notebook](./3d_ldm/3d_ldm_tutorial.ipynb)
Example shows the use cases of training and validating a 3D Latent Diffusion Model on Brats 2016&2017 data.

## [Brats 3D latent diffusion model](./3d_ldm/README.md)
Example shows the use cases of training and validating a 3D Latent Diffusion Model on Brats 2016&2017 data, expanding on the above notebook.

## [MAISI 3D latent diffusion model](./maisi/README.md)
Example shows the use cases of training and validating Nvidia MAISI (Medical AI for Synthetic Imaging) model, a 3D Latent Diffusion Model that can generate large CT images with paired segmentation masks, variable volume size and voxel size, as well as controllable organ/tumor size.

## [SPADE in VAE-GAN for Semantic Image Synthesis on 2D BraTS Data](./spade_gen/spade_gen.ipynb)
Example shows the use cases of applying SPADE, a VAE-GAN-based neural network for semantic image synthesis, to a subset of BraTS that was registered to MNI space and resampled to 2mm isotropic space, with segmentations obtained using Geodesic Information Flows (GIF).

## [Applying Latent Diffusion Models to 2D BraTS Data for Semantic Image Synthesis](./spade_ldm/spade_ldm_brats.ipynb)
Example shows the use cases of applying SPADE normalization to a latent diffusion model, following the methodology by Wang et al., for semantic image synthesis on a subset of BraTS registered to MNI space and resampled to 2mm isotropic space, with segmentations obtained using Geodesic Information Flows (GIF).

## [Diffusion Models for Implicit Image Segmentation Ensembles](./image_to_image_translation/tutorial_segmentation_with_ddpm.ipynb)
Example shows the use cases of how to use MONAI for 2D segmentation of images using DDPMs. The same structure can also be used for conditional image generation, or image-to-image translation.

## [Evaluate Realism and Diversity of the generated images](./realism_diversity_metrics/realism_diversity_metrics.ipynb)
Example shows the use cases of using MONAI to evaluate the performance of a generative model by computing metrics such as Frechet Inception Distance (FID) and Maximum Mean Discrepancy (MMD) for assessing realism, as well as MS-SSIM and SSIM for evaluating image diversity.
