# MONAI Tutorials
This repository hosts the MONAI tutorials.

### 1. Requirements
Most of the examples and tutorials require
[matplotlib](https://matplotlib.org/) and [Jupyter Notebook](https://jupyter.org/).

These can be installed with:

```bash
python -m pip install -U pip
python -m pip install -U matplotlib
python -m pip install -U notebook
```

Some of the examples may require optional dependencies. In case of any optional import errors,
please install the relevant packages according to MONAI's [installation guide](https://docs.monai.io/en/latest/installation.html).
Or install all optional requirements with:

```bash
pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/master/requirements-dev.txt
```

#### Run the notebooks from Colab

Most of the Jupyter Notebooks have an "Open in Colab" button.
Please right-click on the button, and select "Open Link in New Tab" to start a Colab page with the corresponding notebook content.

To use GPU resources through Colab, please remember to change the runtime type to `GPU`:

1. From the `Runtime` menu select `Change runtime type`
1. Choose `GPU` from the drop-down menu
1. Click `SAVE`
This will reset the notebook and may ask you if you are a robot (these instructions assume you are not).

Running:

```bash
!nvidia-smi
```

in a cell will verify this has worked and show you what kind of hardware you have access to.

### 2. List of notebooks and examples
**2D classification**
#### [mednist_tutorial](./2d_classification/mednist_tutorial.ipynb)
This notebook shows how to easily integrate MONAI features into existing PyTorch programs.
It's based on the MedNIST dataset which is very suitable for beginners as a tutorial.
This tutorial also makes use of MONAI's in-built occlusion sensitivity functionality.

**2D segmentation**
#### [torch examples](./2d_segmentation/torch)
Training and evaluation examples of 2D segmentation based on UNet and synthetic dataset.
The examples are standard PyTorch programs and have both dictionary-based and array-based versions.

**3D classification**
#### [ignite examples](./3d_classification/ignite)
Training and evaluation examples of 3D classification based on DenseNet3D and [IXI dataset](https://brain-development.org/ixi-dataset).
The examples are PyTorch Ignite programs and have both dictionary-based and array-based transformation versions.
#### [torch examples](./3d_classification/torch)
Training and evaluation examples of 3D classification based on DenseNet3D and [IXI dataset](https://brain-development.org/ixi-dataset).
The examples are standard PyTorch programs and have both dictionary-based and array-based transformation versions.

**3D segmentation**
#### [ignite examples](./3d_segmentation/ignite)
Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset.
The examples are PyTorch Ignite programs and have both dictionary-base and array-based transformations.
#### [torch examples](./3d_segmentation/torch)
Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset.
The examples are standard PyTorch programs and have both dictionary-based and array-based versions.
#### [brats_segmentation_3d](./3d_segmentation/brats_segmentation_3d.ipynb)
This tutorial shows how to construct a training workflow of multi-labels segmentation task based on [MSD Brain Tumor dataset](http://medicaldecathlon.com).
#### [spleen_segmentation_3d_lightning](./3d_segmentation/spleen_segmentation_3d_lightning.ipynb)
This notebook shows how MONAI may be used in conjunction with the [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework.
#### [spleen_segmentation_3d](./3d_segmentation/spleen_segmentation_3d.ipynb)
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
#### [unet_segmentation_3d_catalyst](./3d_segmentation/unet_segmentation_3d_catalyst.ipynb)
This notebook shows how MONAI may be used in conjunction with the [Catalyst](https://github.com/catalyst-team/catalyst) framework.
#### [unet_segmentation_3d_ignite](./3d_segmentation/unet_segmentation_3d_ignite.ipynb)
This notebook is an end-to-end training & evaluation example of 3D segmentation based on synthetic dataset.
The example is a PyTorch Ignite program and shows several key features of MONAI, especially with medical domain specific transforms and event handlers.

**federated learning**
#### [Substra](./federated_learning/substra)
The example show how to execute the 3d segmentation torch tutorial on a federated learning platform, Substra.

**acceleration**
#### [distributed_training](./acceleration/distributed_training)
The examples show how to execute distributed training and evaluation based on 3 different frameworks:
- PyTorch native `DistributedDataParallel` module with `torch.distributed.launch`.
- Horovod APIs with `horovodrun`.
- PyTorch ignite and MONAI workflows.

They can run on several distributed nodes with multiple GPU devices on every node.
#### [automatic_mixed_precision](./acceleration/automatic_mixed_precision.ipynb)
And compares the training speed and memory usage with/without AMP.
#### [dataset_type_performance](./acceleration/dataset_type_performance.ipynb)
This notebook compares the performance of `Dataset`, `CacheDataset` and `PersistentDataset`. These classes differ in how data is stored (in memory or on disk), and at which moment transforms are applied.
#### [fast_training_tutorial](./acceleration/fast_training_tutorial.ipynb)
This tutorial compares the training performance of pure PyTorch program and optimized program in MONAI based on NVIDIA GPU device and latest CUDA library.
The optimization methods mainly include: `AMP`, `CacheDataset` and `Novograd`.
#### [multi_gpu_test](./acceleration/multi_gpu_test.ipynb)
This notebook is a quick demo for devices, run the Ignite trainer engine on CPU, GPU and multiple GPUs.
#### [transform_speed](./acceleration/transform_speed.ipynb)
Illustrate reading NIfTI files and test speed of different transforms on different devices.

**modules**
#### [engines](./modules/engines)
Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset with MONAI workflows, which contains engines, event-handlers, and post-transforms. And GAN training and evaluation example for a medical image generative adversarial network. Easy run training script uses `GanTrainer` to train a 2D CT scan reconstruction network. Evaluation script generates random samples from a trained network.

The examples are built with MONAI workflows, mainly contain: trainer/evaluator, handlers, post_transforms, etc.
#### [3d_image_transforms](./modules/3d_image_transforms.ipynb)
This notebook demonstrates the transformations on volumetric images.

#### [autoencoder_mednist](./modules/autoencoder_mednist)
This tutorial uses the MedNIST hand CT scan dataset to demonstrate MONAI's autoencoder class. The autoencoder is used with an identity encode/decode (i.e., what you put in is what you should get back), as well as demonstrating its usage for de-blurring and de-noising.

#### [dynunet_tutorial](./modules/dynunet_tutorial.ipynb)
This tutorial shows how to train 3D segmentation tasks on all the 10 decathlon datasets with the reimplementation of dynUNet in MONAI.
#### [integrate_3rd_party_transforms](./modules/integrate_3rd_party_transforms.ipynb)
This tutorial shows how to integrate 3rd party transforms into MONAI program.
Mainly shows transforms from BatchGenerator, TorchIO, Rising and ITK.
#### [load_medical_imagesl](./modules/load_medical_images.ipynb)
This notebook introduces how to easily load different formats of medical images in MONAI and execute many additional operations.
#### [mednist_GAN_tutorial](./modules/mednist_GAN_tutorial.ipynb)
This notebook illustrates the use of MONAI for training a network to generate images from a random input tensor.
A simple GAN is employed to do with a separate Generator and Discriminator networks.
#### [mednist_GAN_workflow_dict](./modules/mednist_GAN_workflow_dict.ipynb)
This notebook shows the `GanTrainer`, a MONAI workflow engine for modularized adversarial learning. Train a medical image reconstruction network using the MedNIST hand CT scan dataset. Dictionary version.
#### [mednist_GAN_workflow_array](./modules/mednist_GAN_workflow_array.ipynb)
This notebook shows the `GanTrainer`, a MONAI workflow engine for modularized adversarial learning. Train a medical image reconstruction network using the MedNIST hand CT scan dataset. Array version.
#### [models_ensemble](./modules/models_ensemble.ipynb)
This tutorial shows how to leverage `EnsembleEvaluator`, `MeanEnsemble` and `VoteEnsemble` modules in MONAI to set up ensemble program.
#### [nifti_read_example](./modules/nifti_read_example.ipynb)
Illustrate reading NIfTI files and iterating over image patches of the volumes loaded from them.
#### [dynunet_tutorial](./modules/dynunet_tutorial.ipynb)
This tutorial shows how to train 3D segmentation tasks on all the 10 decathlon datasets with the reimplementation of dynUNet in MONAI.
#### [post_transforms](./modules/post_transforms.ipynb)
This notebook shows the usage of several post transforms based on the model output of spleen segmentation task.
#### [public_datasets](./modules/public_datasets.ipynb)
This notebook shows how to quickly set up training workflow based on `MedNISTDataset` and `DecathlonDataset`, and how to create a new dataset.
#### [transforms_demo_2d](./modules/transforms_demo_2d.ipynb)
This notebook demonstrates the image transformations on histology images using
[the GlaS Contest dataset](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/).
