# MONAI Tutorials
This repository hosts the MONAI tutorials.

### 1. Requirements
Most of the examples and tutorials require
[matplotlib](https://matplotlib.org/) and [Jupyter Notebook](https://jupyter.org/).

These could be installed by:
```bash
python -m pip install -U pip
python -m pip install -U matplotlib
python -m pip install -U notebook
```

Some of the examples may require optional dependencies. In case of any optional import errors,
please install the relevant packages according to the error message.
Or install all optional requirements by:
```
pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/master/requirements-dev.txt
```
### 2. List of notebooks
#### [3d_image_transforms](./3d_image_transforms.ipynb)
This notebook demonstrates the transformations on volumetric images.
#### [automatic_mixed_precision](./automatic_mixed_precision.ipynb)
This tutorial shows how to apply the automatic mixed precision(AMP) feature of PyTorch into training and evaluation programs.
And compares the training speed and memory usage with/without AMP.
#### [brats_segmentation_3d](./brats_segmentation_3d.ipynb)
This tutorial shows how to construct a training workflow of multi-labels segmentation task based on [MSD Brain Tumor dataset](http://medicaldecathlon.com).
#### [dataset_type_performance](./dataset_type_performance.ipynb)
This notebook compares the performance of `Dataset`, `CacheDataset` and `PersistentDataset`. These classes differ in how data is stored (in memory or on disk), and at which moment transforms are applied.
#### [integrate_3rd_party_transforms](./integrate_3rd_party_transforms.ipynb)
This tutorial shows how to integrate 3rd party transforms into MONAI program.
Mainly shows transforms from BatchGenerator, TorchIO, Rising and ITK.
#### [mednist_GAN_tutorial](./mednist_GAN_tutorial.ipynb)
This notebook illustrates the use of MONAI for training a network to generate images from a random input tensor.
A simple GAN is employed to do with a separate Generator and Discriminator networks.
#### [mednist_GAN_workflow](./mednist_GAN_workflow.ipynb)
This notebook shows the `GanTrainer`, a MONAI workflow engine for modularized adversarial learning. Train a medical image reconstruction network using the MedNIST hand CT scan dataset. Based on the tutorial.
#### [mednist_tutorial](./mednist_tutorial.ipynb)
This notebook shows how to easily integrate MONAI features into existing PyTorch programs.
It's based on the MedNIST dataset which is very suitable for beginners as a tutorial.
The content is also available as [a Colab tutorial](https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe).
#### [models_ensemble](./models_ensemble.ipynb)
This tutorial shows how to leverage `EnsembleEvaluator`, `MeanEnsemble` and `VoteEnsemble` modules in MONAI to set up ensemble program.
#### [multi_gpu_test](./multi_gpu_test.ipynb)
This notebook is a quick demo for devices, run the Ignite trainer engine on CPU, GPU and multiple GPUs.
#### [nifti_read_example](./nifti_read_example.ipynb)
Illustrate reading NIfTI files and iterating over image patches of the volumes loaded from them.
#### [post_transforms](./post_transforms.ipynb)
This notebook shows the usage of several post transforms based on the model output of spleen segmentation task.
#### [public_datasets](./public_datasets.ipynb)
This notebook shows how to quickly set up training workflow based on `MedNISTDataset` and `DecathlonDataset`, and how to create a new dataset.
#### [spleen_segmentation_3d](./spleen_segmentation_3d.ipynb)
This notebook is an end-to-end training and evaluation example of 3D segmentation based on [MSD Spleen dataset](http://medicaldecathlon.com).
The example shows the flexibility of MONAI modules in a PyTorch-based program:
- Transforms for dictionary-based training data structure.
- Load NIfTI images with metadata.
- Scale medical image intensity with expected range.
- Crop out a batch of balanced image patch samples based on positive / negative label ratio.
- Cache IO and transforms to accelerate training and validation.
- 3D UNet, Dice loss function, Mean Dice metric for 3D segmentation task.
- Sliding window inference.
- Deterministic training for reproducibility.
#### [spleen_segmentation_3d_lightning](./spleen_segmentation_3d_lightning.ipynb)
This notebook shows how MONAI may be used in conjunction with the [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework.
#### [unet_segmentation_3d_catalyst](./unet_segmentation_3d_catalyst.ipynb)
This notebook shows how MONAI may be used in conjunction with the [Catalyst](https://github.com/catalyst-team/catalyst) framework.
#### [transform_speed](./transform_speed.ipynb)
Illustrate reading NIfTI files and test speed of different transforms on different devices.
#### [transforms_demo_2d](./transforms_demo_2d.ipynb)
This notebook demonstrates the image transformations on histology images using
[the GlaS Contest dataset](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/).
#### [unet_segmentation_3d_ignite](./unet_segmentation_3d_ignite.ipynb)
This notebook is an end-to-end training & evaluation example of 3D segmentation based on synthetic dataset.
The example is a PyTorch Ignite program and shows several key features of MONAI, especially with medical domain specific transforms and event handlers.
