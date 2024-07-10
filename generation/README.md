# MONAI Generative Model Example
This folder contains examples for training and validating a MONAI Generative Model.

The examples are designed as demonstration to showcase the training process for this type of network using MONAI Generative models. To achieve optimal performance, it is recommended that users to adjust the network and hyper-parameters based on their device and training dataset.

## Installation
```
pip install lpips
pip install monai-generative==0.2.3
```

## [Brats 3D latent diffusion model](./3d_ldm/README.md)
Example shows the use cases of training and validating a 3D Latent Diffusion Model on Brats 2016&2017 data.

## [Brats 2D latent diffusion model](./2d_ldm/README.md)
Example shows the use cases of training and validating a 2D Latent Diffusion Model on axial slices from Brats 2016&2017 data.

## [MAISI 3D latent diffusion model](./maisi/README.md)
Example shows the use cases of training and validating Nvidia MAISI (Medical AI for Synthetic Imaging) model, a 3D Latent Diffusion Model that can generate large CT images with paired segmentation masks, variable volume size and voxel size, as well as controllable organ/tumor size.
