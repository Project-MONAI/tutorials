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
pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt
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

#### Data

Some notebooks will require additional data. They can be downloaded by running the [runexamples.sh](./runexamples.sh) script.

### 2. Questions and bugs

- For questions relating to the use of MONAI, please us our [Discussions tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).

### 3. Note to developers

During integration testing, we run these notebooks. To save time, we modify variables to avoid unecessary `for` loop iterations. Hence, during training please use the variables `max_epochs` and `val_interval` for the number of training epochs and validation interval, respectively.

If your notebook doesn't use the idea of epochs, then please add it to the variable `doesnt_contain_max_epochs` in `runner.sh`. This lets the runner know that it's not a problem if it doesn't find `max_epochs`.

If you have any other variables that would benefit by setting them to `1` during testing, add them to `strings_to_replace` in `runner.sh`.

### 4. List of notebooks and examples
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
Training, evaluation and inference examples of 3D segmentation based on UNet3D and synthetic dataset.
The examples are standard PyTorch programs and have both dictionary-based and array-based versions.
#### [brats_segmentation_3d](./3d_segmentation/brats_segmentation_3d.ipynb)
This tutorial shows how to construct a training workflow of multi-labels segmentation task based on [MSD Brain Tumor dataset](http://medicaldecathlon.com).
#### [spleen_segmentation_3d_aim](./3d_segmentation/spleen_segmentation_3d_visualization_basic.ipynb)
This notebook shows how MONAI may be used in conjunction with the [`aimhubio/aim`](https://github.com/aimhubio/aim).
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
The example is a PyTorch Ignite program and shows several key features of MONAI, especially with medical domain specific transforms and event handlers for profiling (logging, TensorBoard, MLFlow, etc.).
#### [COVID 19-20 challenge baseline](./3d_segmentation/challenge_baseline)
This folder provides a simple baseline method for training, validation, and inference for [COVID-19 LUNG CT LESION SEGMENTATION CHALLENGE - 2020](https://covid-segmentation.grand-challenge.org/COVID-19-20/) (a MICCAI Endorsed Event).
#### [unetr_btcv_segmentation_3d](./3d_segmentation/unetr_btcv_segmentation_3d.ipynb)
This notebook demonstrates how to construct a training workflow of UNETR on multi-organ segmentation task using the BTCV challenge dataset.
#### [unetr_btcv_segmentation_3d_lightning](./3d_segmentation/unetr_btcv_segmentation_3d_lightning.ipynb)
This tutorial demonstrates how MONAI can be used in conjunction with [PyTorch Lightning](https://www.pytorchlightning.ai/) framework to construct a training workflow of UNETR on multi-organ segmentation task using the BTCV challenge dataset.

**2D registration**
#### [registration using mednist](./2d_registration/registration_mednist.ipynb)
This notebook shows a quick demo for learning based affine registration of `64 x 64` X-Ray hands.

**3D registration**
#### [3D registration using paired lung CT](./3d_registration/paired_lung_ct.ipynb)
This tutorial shows how to use MONAI to register lung CT volumes acquired at different time points for a single patient.

#### [DeepAtlas](./deep_atlas/deep_atlas_tutorial.ipynb)
This tutorial demonstrates the use of MONAI for training of registration and segmentation models _together_. The DeepAtlas approach, in which the two models serve as a source of weakly supervised learning for each other, is useful in situations where one has many unlabeled images and just a few images with segmentation labels. The notebook works with 3D images from the OASIS-1 brain MRI dataset.

**deepgrow**
#### [Deepgrow](./deepgrow)
The example show how to train/validate a 2D/3D deepgrow model.  It also demonstrates running an inference for trained deepgrow models.

**DeepEdit**
#### [DeepEdit](./deepedit/ignite)
This example shows how to train/test a DeepEdit model. In this tutorial there is a Notebook that shows how to run
inference on a pretrained DeepEdit model.


**deployment**
#### [BentoML](./deployment/bentoml)
This is a simple example of training and deploying a MONAI network with [BentoML](https://www.bentoml.ai/) as a web server, either locally using the BentoML respository or as a containerized service.
#### [Ray](./deployment/ray)
This uses the previous notebook's trained network to demonstrate deployment a web server using [Ray](https://docs.ray.io/en/master/serve/index.html#rayserve).
#### [Triton](./deployment/Triton/)
This is example walks through using a Triton Server and Python client using MONAI on the MedNIST classification problem. The demo is self contained and the Readme explains how to use Triton "backends" to inject the MONAI code into the server.  [See Triton Inference Server/python_backend documentation](https://github.com/triton-inference-server/python_backend#usage)

**experiment management**
#### [Aim](./experiment_management/spleen_segmentation_aim.ipynb)
An example of experiment management with [Aim](https://aimstack.io/aim-monai-tutorial/), using 3D spleen segmentation as an example.
#### [MLFlow](./experiment_management/spleen_segmentation_mlflow.ipynb)
An example of experiment management with [MLFlow](https://www.mlflow.org/docs/latest/tracking.html), using 3D spleen segmentation as an example.

**Federated Learning**
#### [NVFlare](./federated_learning/nvflare)
The examples show how to train federated learning models with [NVFlare](https://pypi.org/project/nvflare/) and MONAI-based trainers.

#### [OpenFL](./federated_learning/openfl)
The examples show how to train federated learning models based on [OpenFL](https://github.com/intel/openfl) and MONAI.

#### [Substra](./federated_learning/substra)
The example show how to execute the 3d segmentation torch tutorial on a federated learning platform, Substra.

#### [Breast Density FL Challenge](./federated_learning/breast_density_challenge)
Reference implementation used in MICCAI 2022 [ACR-NVIDIA-NCI Breast Density FL challenge](http://breastdensityfl.acr.org).

**Digital Pathology**
#### [Whole Slide Tumor Detection](./pathology/tumor_detection)
The example show how to train and evaluate a tumor detection model (based on patch classification) on whole-slide histopathology images.

#### [Profiling Whole Slide Tumor Detection](./pathology/tumor_detection)
The example show how to use MONAI NVTX transforms to tag and profile pre- and post-processing transforms in the digital pathology whole slide tumor detection pipeline.

#### [NuClick:Interactive Annotation for Pathology](./pathology/nuclick)
The notebook demonstrates examples of training and inference pipelines with interactive annotation for pathology, NuClick is used for delineating nuclei, cells and a squiggle for outlining glands.

**acceleration**
#### [fast_model_training_guide](./acceleration/fast_model_training_guide.md)
The document introduces details of how to profile the training pipeline, how to analyze the dataset and select suitable algorithms, and how to optimize GPU utilization in single GPU, multi-GPUs or even multi-nodes.
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
The optimization methods mainly include: `AMP`, `CacheDataset`, `GPU transforms`, `ThreadDataLoader`, `DiceCELoss` and `SGD`.
#### [multi_gpu_test](./acceleration/multi_gpu_test.ipynb)
This notebook is a quick demo for devices, run the Ignite trainer engine on CPU, GPU and multiple GPUs.
#### [threadbuffer_performance](./acceleration/threadbuffer_performance.ipynb)
Demonstrates the use of the `ThreadBuffer` class used to generate data batches during training in a separate thread.
#### [transform_speed](./acceleration/transform_speed.ipynb)
Illustrate reading NIfTI files and test speed of different transforms on different devices.

**model_zoo**
#### [easy_integrate_bundle](./model_zoo/easy_integrate_bundle.py)
This tutorial shows a straightforward ensemble application to instruct users on how to integrate existing bundles in their own projects. By simply changing the data path and the path where the bundle is located, training and ensemble inference can be performed.

**computer_assisted_intervention**
#### [video segmentation](./computer_assisted_intervention/video_seg.ipynb)
This tutorial shows how to train a surgical tool segmentation model to locate tools in a given image. In addition, it also builds an example pipeline of an end-to-end video tool segmentation, with video input and video output.

**modules**
#### [bundle](./bundle)
Get started tutorial and concrete training / inference examples for MONAI bundle features.
#### [engines](./modules/engines)
Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset with MONAI workflows, which contains engines, event-handlers, and post-transforms. And GAN training and evaluation example for a medical image generative adversarial network. Easy run training script uses `GanTrainer` to train a 2D CT scan reconstruction network. Evaluation script generates random samples from a trained network.

The examples are built with MONAI workflows, mainly contain: trainer/evaluator, handlers, post_transforms, etc.
#### [3d_image_transforms](./modules/3d_image_transforms.ipynb)
This notebook demonstrates the transformations on volumetric images.
#### [2d_inference_3d_volume](./modules/2d_inference_3d_volume.ipynb)
Tutorial that demonstrates how monai `SlidingWindowInferer` can be used when a 3D volume input needs to be provided slice-by-slice to a 2D model and finally, aggregated into a 3D volume.
#### [autoencoder_mednist](./modules/autoencoder_mednist.ipynb)
This tutorial uses the MedNIST hand CT scan dataset to demonstrate MONAI's autoencoder class. The autoencoder is used with an identity encode/decode (i.e., what you put in is what you should get back), as well as demonstrating its usage for de-blurring and de-noising.
#### [batch_output_transform](./modules/batch_output_transform.py)
Tutorial to explain and show how to set `batch_transform` and `output_transform` of handlers to work with MONAI engines.
#### [compute_metric](./modules/compute_metric.py)
Example shows how to compute metrics from saved predictions and labels with PyTorch multi-processing support.
#### [csv_datasets](./modules/csv_datasets.ipynb)
Tutorial shows the usage of `CSVDataset` and `CSVIterableDataset`, load multiple CSV files and execute postprocessing logic.
#### [decollate_batch](./modules/decollate_batch.py)
Tutorial shows how to decollate batch data to simplify post processing transforms and execute more flexible following operations.
#### [image_dataset](./modules/image_dataset.py)
Notebook introduces basic usages of `monai.data.ImageDataset` module.
#### [dynunet_tutorial](./modules/dynunet_pipeline)
This tutorial shows how to train 3D segmentation tasks on all the 10 decathlon datasets with the reimplementation of dynUNet in MONAI.
#### [integrate_3rd_party_transforms](./modules/integrate_3rd_party_transforms.ipynb)
This tutorial shows how to integrate 3rd party transforms into MONAI program.
Mainly shows transforms from BatchGenerator, TorchIO, Rising and ITK.
#### [inverse transformations and test-time augmentations](./modules/inverse_transforms_and_test_time_augmentations.ipynb)
This notebook demonstrates the use of invertible transforms, and then leveraging inverse transformations to perform test-time augmentations.
#### [layer wise learning rate](./modules/layer_wise_learning_rate.ipynb)
This notebook demonstrates how to select or filter out expected network layers and set customized learning rate values.
#### [learning rate finder](./modules/learning_rate.ipynb)
This notebook demonstrates how to use `LearningRateFinder` API to tune the learning rate values for the network.
#### [load_medical_imagesl](./modules/load_medical_images.ipynb)
This notebook introduces how to easily load different formats of medical images in MONAI and execute many additional operations.
#### [mednist_GAN_tutorial](./modules/mednist_GAN_tutorial.ipynb)
This notebook illustrates the use of MONAI for training a network to generate images from a random input tensor.
A simple GAN is employed to do with a separate Generator and Discriminator networks.
#### [mednist_GAN_workflow_dict](./modules/mednist_GAN_workflow_dict.ipynb)
This notebook shows the `GanTrainer`, a MONAI workflow engine for modularized adversarial learning. Train a medical image reconstruction network using the MedNIST hand CT scan dataset. Dictionary version.
#### [mednist_GAN_workflow_array](./modules/mednist_GAN_workflow_array.ipynb)
This notebook shows the `GanTrainer`, a MONAI workflow engine for modularized adversarial learning. Train a medical image reconstruction network using the MedNIST hand CT scan dataset. Array version.
#### [cross_validation_models_ensemble](./modules/cross_validation_models_ensemble.ipynb)
This tutorial shows how to leverage `CrossValidation`, `EnsembleEvaluator`, `MeanEnsemble` and `VoteEnsemble` modules in MONAI to set up cross validation and ensemble program.
#### [nifti_read_example](./modules/nifti_read_example.ipynb)
Illustrate reading NIfTI files and iterating over image patches of the volumes loaded from them.
#### [network_api](./modules/network_api.ipynb)
This tutorial illustrates the flexible network APIs and utilities.
#### [postprocessing_transforms](./modules/postprocessing_transforms.ipynb)
This notebook shows the usage of several postprocessing transforms based on the model output of spleen segmentation task.
#### [public_datasets](./modules/public_datasets.ipynb)
This notebook shows how to quickly set up training workflow based on `MedNISTDataset` and `DecathlonDataset`, and how to create a new dataset.
#### [tcia_csv_processing](./modules/tcia_csv_processing.ipynb)
This notebook shows how to load the TCIA data with CSVDataset from CSV file and extract information for TCIA data to fetch DICOM images based on REST API.
#### [transforms_demo_2d](./modules/transforms_demo_2d.ipynb)
This notebook demonstrates the image transformations on histology images using
[the GlaS Contest dataset](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/).
#### [UNet_input_size_constrains](./modules/UNet_input_size_constrains.ipynb)
This tutorial shows how to determine a reasonable spatial size of the input data for MONAI UNet, which not only supports residual units, but also can use more hyperparameters (like `strides`, `kernel_size` and `up_kernel_size`) than the basic UNet implementation.
#### [TorchIO, MONAI, PyTorch Lightning](./modules/TorchIO_MONAI_PyTorch_Lightning.ipynb)
This notebook demonstrates how the three libraries from the official PyTorch Ecosystem can be used together to segment the hippocampus on brain MRIs from the Medical Segmentation Decathlon.
#### [varautoencoder_mednist](./modules/varautoencoder_mednist.ipynb)
This tutorial uses the MedNIST scan (or alternatively the MNIST) dataset to demonstrate MONAI's variational autoencoder class.
#### [interpretability](./modules/interpretability)
Tutorials in this folder demonstrate model visualisation and interpretability features of MONAI. Currently, it consists of class activation mapping and occlusion sensitivity for 3D classification model visualisations and analysis.
#### [Transfer learning with MMAR](./modules/transfer_mmar.ipynb)
This tutorial demonstrates a transfer learning pipeline from a pretrained model in [Clara Train's Medical Model Archive format](https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html).  The notebook also shows the use of LMDB-based dataset.

#### [Transform visualization](./modules/transform_visualization.ipynb)
This tutorial shows several visualization approaches for 3D image during transform augmentation.

#### [Auto3DSeg](./auto3dseg/)
This folder shows how to run the comprehensive Auto3Dseg pipeline with minimal inputs and customize the Auto3Dseg modules to meet different user requirements.
