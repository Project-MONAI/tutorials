# MONAI Generative Model Example
This folder contains examples for training and validating a MONAI Generative Model.

The examples are designed as demonstration to showcase the training process for this type of network using MONAI Generative models. To achieve optimal performance, it is recommended that users to adjust the network and hyper-parameters based on their device and training dataset.

## Installation
```
pip install lpips
pip install git+https://github.com/Project-MONAI/GenerativeModels.git@0.2.0
```

Or install it from source:
```
pip install lpips
git clone https://github.com/Project-MONAI/GenerativeModels.git
cd GenerativeModels/
python setup.py install
cd ..
```

## [Brats 3D latent diffusion model](./3d_ldm/README.md)
Example shows the use cases of training and validating a 3D Latent Diffusion Model on Brats 2016&2017 data.

## [Brats 2D latent diffusion model](./2d_ldm/README.md)
Example shows the use cases of training and validating a 2D Latent Diffusion Model on axial slices from Brats 2016&2017 data.
