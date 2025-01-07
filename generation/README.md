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

## [SPADE in VAE-GAN for Semantic Image Synthesis on 2D BraTS Data](./spade_gan/spade_gan.ipynb)
Example shows the use cases of applying SPADE, a VAE-GAN-based neural network for semantic image synthesis, to a subset of BraTS that was registered to MNI space and resampled to 2mm isotropic space, with segmentations obtained using Geodesic Information Flows (GIF).

## [Applying Latent Diffusion Models to 2D BraTS Data for Semantic Image Synthesis](./spade_ldm/spade_ldm_brats.ipynb)
Example shows the use cases of applying SPADE normalization to a latent diffusion model, following the methodology by Wang et al., for semantic image synthesis on a subset of BraTS registered to MNI space and resampled to 2mm isotropic space, with segmentations obtained using Geodesic Information Flows (GIF).

## [Diffusion Models for Implicit Image Segmentation Ensembles](./image_to_image_translation/tutorial_segmentation_with_ddpm.ipynb)
Example shows the use cases of how to use MONAI for 2D segmentation of images using DDPMs. The same structure can also be used for conditional image generation, or image-to-image translation.

## [Evaluate Realism and Diversity of the generated images](./realism_diversity_metrics/realism_diversity_metrics.ipynb)
Example shows the use cases of using MONAI to evaluate the performance of a generative model by computing metrics such as Frechet Inception Distance (FID) and Maximum Mean Discrepancy (MMD) for assessing realism, as well as MS-SSIM and SSIM for evaluating image diversity.

## [Training a 2D VQ-VAE + Autoregressive Transformers](./2d_vqvae_transformer/2d_vqvae_transformer_tutorial.ipynb):
Example shows how to train a Vector-Quantized Variation Autoencoder + Transformers on the MedNIST dataset.

## Training VQ-VAEs and VQ-GANs: [2D VAE](./2d_vqvae/2d_vqvae_tutorial.ipynb), [3D VAE](./3d_vqvae/3d_vqvae_tutorial.ipynb) and [2D GAN](./2d_vqgan/2d_vqgan_tutorial.ipynb)
Examples show how to train Vector Quantized Variation Autoencoder on [2D](./2d_vqvae/2d_vqvae_tutorial.ipynb) and [3D](./3d_vqvae/3d_vqvae_tutorial.ipynb), and how to use the PatchDiscriminator class to train a [VQ-GAN](./2d_vqgan/2d_vqgan_tutorial.ipynb) and improve the quality of the generated images.

## [Training a 2D Denoising Diffusion Probabilistic Model](./2d_ddpm/2d_ddpm_tutorial.ipynb):
Example shows how to easily train a DDPM on medical data (MedNIST).

## [Training a 3D Denoising Diffusion Probabilistic Model](./3d_ddpm/3d_ddpm_tutorial.ipynb):
Example shows how to easily train a DDPM on medical data (Decathlon Task 01).

## [Comparing different noise schedulers](./2d_ddpm/2d_ddpm_compare_schedulers.ipynb):
Example compares the performance of different noise schedulers. This shows how to sample a diffusion model using the DDPM, DDIM, and PNDM schedulers and how different numbers of timesteps affect the quality of the samples.

## [Training a 2D Denoising Diffusion Probabilistic Model with different parameterisation](./2d_ddpm/2d_ddpm_tutorial_v_prediction.ipynb):
Example shows how to train a DDPM using the v-prediction parameterization, which improves the stability and convergence of the model. MONAI supports different parameterizations for the diffusion model (epsilon, sample, and v-prediction).

## [Training a 2D DDPM using Pytorch Ignite](./2d_ddpm/2d_ddpm_compare_schedulers.ipynb):
Example shows how to train a DDPM on medical data using Pytorch Ignite. This shows how to use the DiffusionPrepareBatch to prepare the model inputs and MONAI's SupervisedTrainer and SupervisedEvaluator to train DDPMs.

## [Using a 2D DDPM to inpaint images](./2d_ddpm/2d_ddpm_inpainting.ipynb):
Example shows how to use a DDPM to inpaint of 2D images from the MedNIST dataset using the RePaint method.

## [Guiding the 2D diffusion synthesis using ControlNet](./controlnet/2d_controlnet.ipynb)
Example shows how to use ControlNet to condition a diffusion model trained on 2D brain MRI images on binary brain masks.

## [Spatial variational autoencoder for 2D modelling and synthesis](./2d_autoencoderkl)
Example shows the use cases of applying a spatial VAE to a 2D synthesis example. To obtain realistic results, the model is trained on the original VAE losses, as well as perceptual and adversarial ones.

## [Spatial variational autoencoder for 3D modelling and synthesis](./3d_autoencoderkl)
Example shows the use cases of applying a spatial VAE to a 3D synthesis example. To obtain realistic results, the model is trained on the original VAE losses, as well as perceptual and adversarial ones.

## Performing anomaly detection with diffusion models: [implicit guidance](./anomaly_detection/2d_classifierfree_guidance_anomalydetection_tutorial.ipynb), [using transformers](./anomaly_detection/anomaly_detection_with_transformers.ipynb) and [classifier free guidance](./anomaly_detection/anomalydetection_tutorial_classifier_guidance.ipynb)
Examples show how to perform anomaly detection in 2D, using implicit guidance [2D-classifier free guiance](./anomaly_detection/2d_classifierfree_guidance_anomalydetection_tutorial.ipynb), transformers [using transformers](./anomaly_detection/anomaly_detection_with_transformers.ipynb) and [classifier free guidance](./anomalydetection_tutorial_classifier_guidance).

## 2D super-resolution using diffusion models: [using torch](./2d_super_resolution/2d_sd_super_resolution.ipynb) and [using torch lightning](./2d_super_resolution/2d_sd_super_resolution_lightning.ipynb).
Examples show how to perform super-resolution in 2D, using PyTorch and PyTorch Lightning.

## [Guiding the synthetic process using a semantic encoder](./2d_diffusion_autoencoder/2d_diffusion_autoencoder.ipynb)
Example shows how to train a DDPM and an encoder simultaneously, resulting in the latents of the encoder guiding the inference process of the DDPM.
