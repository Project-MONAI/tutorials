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

这个笔记本展示了如何轻松地将 MONAI 功能集成到现有的 PyTorch 程序中。
它基于 MedNIST 数据集，非常适合初学者作为教程。
本教程还利用了 MONAI 内置的遮挡敏感度功能。

**2D segmentation**
#### [torch examples](./2d_segmentation/torch)
Training and evaluation examples of 2D segmentation based on UNet and synthetic dataset.
The examples are standard PyTorch programs and have both dictionary-based and array-based versions.

基于 UNet 和合成数据集的 2D 分割训练和评估示例。
这些示例是标准的 PyTorch 程序，具有基于字典和基于数组的版本。

**3D classification**
#### [ignite examples](./3d_classification/ignite)
Training and evaluation examples of 3D classification based on DenseNet3D and [IXI dataset](https://brain-development.org/ixi-dataset).
The examples are PyTorch Ignite programs and have both dictionary-based and array-based transformation versions.

基于 DenseNet3D 和 [IXI 数据集](https://brain-development.org/ixi-dataset) 的 3D 分类训练和评估示例。
这些示例是 PyTorch Ignite 程序，具有基于字典和基于数组的转换版本。

#### [torch examples](./3d_classification/torch)
Training and evaluation examples of 3D classification based on DenseNet3D and [IXI dataset](https://brain-development.org/ixi-dataset).
The examples are standard PyTorch programs and have both dictionary-based and array-based transformation versions.

基于 DenseNet3D 和 [IXI 数据集](https://brain-development.org/ixi-dataset) 的 3D 分类训练和评估示例。
这些示例是标准 PyTorch 程序，具有基于字典和基于数组的转换版本。

**3D segmentation**
#### [ignite examples](./3d_segmentation/ignite)
Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset.
The examples are PyTorch Ignite programs and have both dictionary-base and array-based transformations.

基于 UNet3D 和合成数据集的 3D 分割训练和评估示例。
这些示例是 PyTorch Ignite 程序，具有基于字典和基于数组的转换。

#### [torch examples](./3d_segmentation/torch)
Training, evaluation and inference examples of 3D segmentation based on UNet3D and synthetic dataset.
The examples are standard PyTorch programs and have both dictionary-based and array-based versions.

基于 UNet3D 和合成数据集的 3D 分割训练、评估和推理示例。
这些示例是标准的 PyTorch 程序，具有基于字典和基于数组的版本。

#### [brats_segmentation_3d](./3d_segmentation/brats_segmentation_3d.ipynb)
This tutorial shows how to construct a training workflow of multi-labels segmentation task based on [MSD Brain Tumor dataset](http://medicaldecathlon.com).

本教程展示了如何基于 [MSD 脑肿瘤数据集](http://medicaldecathlon.com) 构建多标签分割任务的训练工作流。

#### [spleen_segmentation_3d_aim](./3d_segmentation/spleen_segmentation_3d_visualization_basic.ipynb)
This notebook shows how MONAI may be used in conjunction with the [`aimhubio/aim`](https://github.com/aimhubio/aim).

本笔记本展示了如何将 MONAI 与 [`aimhubio/aim`](https://github.com/aimhubio/aim) 结合使用。

#### [spleen_segmentation_3d_lightning](./3d_segmentation/spleen_segmentation_3d_lightning.ipynb)
This notebook shows how MONAI may be used in conjunction with the [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework.

本笔记本展示了如何将 MONAI 与 [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) 框架结合使用。

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

这个notebook是一个基于[MSD脾脏数据集](http://medicaldecathlon.com)的3D分割的端到端训练和评估示例。
该示例展示了 MONAI 模块在基于 PyTorch 的程序中的灵活性：
- 转换为基于字典的训练数据结构。
- 加载带有元数据的 NIfTI 图像。
- 使用预期范围缩放医学图像强度。
- 根据正/负标签比率裁剪出一批平衡的图像补丁样本。
- 缓存 IO 和转换以加速训练和验证。
- 3D UNet、Dice 损失函数、用于 3D 分割任务的 Mean Dice 度量。
- 滑动窗口推断。
- 可重复性的确定性培训。

#### [unet_segmentation_3d_catalyst](./3d_segmentation/unet_segmentation_3d_catalyst.ipynb)
This notebook shows how MONAI may be used in conjunction with the [Catalyst](https://github.com/catalyst-team/catalyst) framework.

本笔记本展示了如何将 MONAI 与 [Catalyst](https://github.com/catalyst-team/catalyst) 框架结合使用。

#### [unet_segmentation_3d_ignite](./3d_segmentation/unet_segmentation_3d_ignite.ipynb)
This notebook is an end-to-end training & evaluation example of 3D segmentation based on synthetic dataset.
The example is a PyTorch Ignite program and shows several key features of MONAI, especially with medical domain specific transforms and event handlers for profiling (logging, TensorBoard, MLFlow, etc.).

该笔记本是基于合成数据集的 3D 分割的端到端训练和评估示例。
这个例子是一个 PyTorch Ignite 程序，展示了 MONAI 的几个关键特性，尤其是医疗领域特定的转换和用于分析的事件处理程序（日志记录、TensorBoard、MLFlow 等）。

#### [COVID 19-20 challenge baseline](./3d_segmentation/challenge_baseline)
This folder provides a simple baseline method for training, validation, and inference for [COVID-19 LUNG CT LESION SEGMENTATION CHALLENGE - 2020](https://covid-segmentation.grand-challenge.org/COVID-19-20/) (a MICCAI Endorsed Event).

此文件夹为 [COVID-19 肺部 CT 病变分割挑战 - 2020](https://covid-segmentation.grand-challenge.org/COVID-19-20/) 提供了一种简单的基线方法，用于训练、验证和推理（ MICCAI 认可的活动）。

#### [unetr_btcv_segmentation_3d](./3d_segmentation/unetr_btcv_segmentation_3d.ipynb)
This notebook demonstrates how to construct a training workflow of UNETR on multi-organ segmentation task using the BTCV challenge dataset.

本笔记本演示了如何使用 BTCV 挑战数据集构建多器官分割任务的 UNETR 训练工作流。

#### [unetr_btcv_segmentation_3d_lightning](./3d_segmentation/unetr_btcv_segmentation_3d_lightning.ipynb)
This tutorial demonstrates how MONAI can be used in conjunction with [PyTorch Lightning](https://www.pytorchlightning.ai/) framework to construct a training workflow of UNETR on multi-organ segmentation task using the BTCV challenge dataset.

本教程演示了如何将 MONAI 与 [PyTorch Lightning](https://www.pytorchlightning.ai/) 框架结合使用，使用 BTCV 挑战数据集构建多器官分割任务的 UNETR 训练工作流。

**2D registration**
#### [registration using mednist](./2d_registration/registration_mednist.ipynb)
This notebook shows a quick demo for learning based affine registration of `64 x 64` X-Ray hands.

此笔记本显示了一个快速演示，用于学习基于 64 x 64 的 X 射线手的仿射配准。

**3D registration**
#### [3D registration using paired lung CT](./3d_registration/paired_lung_ct.ipynb)
This tutorial shows how to use MONAI to register lung CT volumes acquired at different time points for a single patient.

本教程展示了如何使用 MONAI 记录单个患者在不同时间点采集的肺部 CT 体积。

#### [DeepAtlas](./deep_atlas/deep_atlas_tutorial.ipynb)
This tutorial demonstrates the use of MONAI for training of registration and segmentation models _together_. The DeepAtlas approach, in which the two models serve as a source of weakly supervised learning for each other, is useful in situations where one has many unlabeled images and just a few images with segmentation labels. The notebook works with 3D images from the OASIS-1 brain MRI dataset.

本教程演示了如何使用 MONAI 来训练配准和分割模型 _together_。 DeepAtlas 方法，其中两个模型作为弱监督学习的来源，在一个有许多未标记的图像和只有几个带有分割标签的图像的情况下很有用。 该笔记本使用来自 OASIS-1 大脑 MRI 数据集的 3D 图像。

**deepgrow**
#### [Deepgrow](./deepgrow)
The example show how to train/validate a 2D/3D deepgrow model.  It also demonstrates running an inference for trained deepgrow models.

该示例展示了如何训练/验证 2D/3D 深度增长模型。 它还演示了对训练有素的 deepgrow 模型进行推理。

**deployment**
#### [BentoML](./deployment/bentoml)
This is a simple example of training and deploying a MONAI network with [BentoML](https://www.bentoml.ai/) as a web server, either locally using the BentoML respository or as a containerized service.

这是一个使用 [BentoML](https://www.bentoml.ai/) 作为 Web 服务器训练和部署 MONAI 网络的简单示例，可以在本地使用 BentoML 存储库或作为容器化服务。

#### [Ray](./deployment/ray)
This uses the previous notebook's trained network to demonstrate deployment a web server using [Ray](https://docs.ray.io/en/master/serve/index.html#rayserve).

这使用之前笔记本的训练网络来演示使用 [Ray](https://docs.ray.io/en/master/serve/index.html#rayserve) 部署 Web 服务器。

**federated learning**
#### [NVFlare](./federated_learning/nvflare)
The examples show how to train federated learning models with [NVFlare](https://pypi.org/project/nvflare/) and MONAI-based trainers.

这些示例展示了如何使用 [NVFlare](https://pypi.org/project/nvflare/) 和基于 MONAI 的训练器训练联邦学习模型。

#### [OpenFL](./federated_learning/openfl)
The examples show how to train federated learning models based on [OpenFL](https://github.com/intel/openfl) and MONAI.

这些示例展示了如何基于 [OpenFL](https://github.com/intel/openfl) 和 MONAI 训练联邦学习模型。

#### [Substra](./federated_learning/substra)
The example show how to execute the 3d segmentation torch tutorial on a federated learning platform, Substra.

该示例展示了如何在联邦学习平台 Substra 上执行 3d 分割火炬教程。

**Digital Pathology**
#### [Whole Slide Tumor Detection](./pathology/tumor_detection)
The example show how to train and evaluate a tumor detection model (based on patch classification) on whole-slide histopathology images.

该示例展示了如何在全切片组织病理学图像上训练和评估肿瘤检测模型（基于补丁分类）。

#### [Profiling Whole Slide Tumor Detection](./pathology/tumor_detection)
The example show how to use MONAI NVTX transforms to tag and profile pre- and post-processing transforms in the digital pathology whole slide tumor detection pipeline.

该示例展示了如何使用 MONAI NVTX 转换在数字病理学全玻片肿瘤检测管道中标记和配置预处理和后处理转换。

**acceleration**
#### [fast_model_training_guide](./acceleration/fast_model_training_guide.md)
The document introduces details of how to profile the training pipeline, how to analyze the dataset and select suitable algorithms, and how to optimize GPU utilization in single GPU, multi-GPUs or even multi-nodes.

该文档详细介绍了如何分析训练管道、如何分析数据集和选择合适的算法，以及如何在单 GPU、多 GPU 甚至多节点中优化 GPU 利用率。

#### [distributed_training](./acceleration/distributed_training)
The examples show how to execute distributed training and evaluation based on 3 different frameworks:
- PyTorch native `DistributedDataParallel` module with `torch.distributed.launch`.
- Horovod APIs with `horovodrun`.
- PyTorch ignite and MONAI workflows.
- 
They can run on several distributed nodes with multiple GPU devices on every node.

这些示例展示了如何基于 3 个不同的框架执行分布式训练和评估：
- 带有 `torch.distributed.launch` 的 PyTorch 原生`DistributedDataParallel` 模块。
- 带有 `horovodrun` 的 Horovod API。
- PyTorch ignite 和 MONAI 工作流程。

它们可以在多个分布式节点上运行，每个节点上都有多个 GPU 设备。

#### [automatic_mixed_precision](./acceleration/automatic_mixed_precision.ipynb)
And compares the training speed and memory usage with/without AMP.

并比较使用/不使用 AMP 的训练速度和内存使用情况。

#### [dataset_type_performance](./acceleration/dataset_type_performance.ipynb)
This notebook compares the performance of `Dataset`, `CacheDataset` and `PersistentDataset`. These classes differ in how data is stored (in memory or on disk), and at which moment transforms are applied.

这个笔记本比较了`Dataset`、`CacheDataset`和`PersistentDataset`的性能。这些类的不同之处在于数据的存储方式（在内存中还是在磁盘上）以及应用转换的时间。

#### [fast_training_tutorial](./acceleration/fast_training_tutorial.ipynb)
This tutorial compares the training performance of pure PyTorch program and optimized program in MONAI based on NVIDIA GPU device and latest CUDA library.
The optimization methods mainly include: `AMP`, `CacheDataset` and `Novograd`.

本教程比较了纯 PyTorch 程序和 MONAI 中基于 NVIDIA GPU 设备和最新 CUDA 库的优化程序的训练性能。
优化方法主要有：`AMP`、`CacheDataset`和`Novograd`。

#### [multi_gpu_test](./acceleration/multi_gpu_test.ipynb)
This notebook is a quick demo for devices, run the Ignite trainer engine on CPU, GPU and multiple GPUs.

此笔记本是设备的快速演示，在 CPU、GPU 和多个 GPU 上运行 Ignite 训练引擎。

#### [threadbuffer_performance](./acceleration/threadbuffer_performance.ipynb)
Demonstrates the use of the `ThreadBuffer` class used to generate data batches during training in a separate thread.

演示使用 `ThreadBuffer` 类在单独的线程中训练期间生成数据批次。

#### [transform_speed](./acceleration/transform_speed.ipynb)
Illustrate reading NIfTI files and test speed of different transforms on different devices.

说明在不同设备上读取 NIfTI 文件和测试不同转换的速度。

**modules**
#### [engines](./modules/bundles)
Get started tutorial and concrete training / inference examples for MONAI bundle features.

MONAI 捆绑功能的入门教程和具体训练/推理示例。

#### [engines](./modules/engines)
Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset with MONAI workflows, which contains engines, event-handlers, and post-transforms. And GAN training and evaluation example for a medical image generative adversarial network. Easy run training script uses `GanTrainer` to train a 2D CT scan reconstruction network. Evaluation script generates random samples from a trained network.

The examples are built with MONAI workflows, mainly contain: trainer/evaluator, handlers, post_transforms, etc.

基于 UNet3D 和具有 MONAI 工作流的合成数据集的 3D 分割训练和评估示例，其中包含引擎、事件处理程序和后转换。以及用于医学图像生成对抗网络的 GAN 训练和评估示例。轻松运行的训练脚本使用“GanTrainer”来训练 2D CT 扫描重建网络。评估脚本从经过训练的网络生成随机样本。

示例使用 MONAI 工作流构建，主要包含：trainer/evaluator、handlers、post_transforms 等。

#### [3d_image_transforms](./modules/3d_image_transforms.ipynb)
This notebook demonstrates the transformations on volumetric images.

这个笔记本演示了体积图像的转换。

#### [2d_inference_3d_volume](./modules/2d_inference_3d_volume.ipynb)
Tutorial that demonstrates how monai `SlidingWindowInferer` can be used when a 3D volume input needs to be provided slice-by-slice to a 2D model and finally, aggregated into a 3D volume.

演示当需要将 3D 体积输入逐个切片提供给 2D 模型并最终聚合成 3D 体积时如何使用 monai `SlidingWindowInferer` 的教程。

#### [autoencoder_mednist](./modules/autoencoder_mednist.ipynb)
This tutorial uses the MedNIST hand CT scan dataset to demonstrate MONAI's autoencoder class. The autoencoder is used with an identity encode/decode (i.e., what you put in is what you should get back), as well as demonstrating its usage for de-blurring and de-noising.

本教程使用 MedNIST 手部 CT 扫描数据集来演示 MONAI 的自动编码器类。自动编码器与身份编码/解码一起使用（即，您输入的内容就是您应该返回的内容），以及演示其用于去模糊和去噪的用途。

#### [batch_output_transform](./modules/batch_output_transform.py)
Tutorial to explain and show how to set `batch_transform` and `output_transform` of handlers to work with MONAI engines.

解释和展示如何设置处理程序的 `batch_transform` 和 `output_transform` 以使用 MONAI 引擎的教程。

#### [compute_metric](./modules/compute_metric.py)
Example shows how to compute metrics from saved predictions and labels with PyTorch multi-processing support.

示例展示了如何使用 PyTorch 多处理支持从保存的预测和标签计算指标。

#### [csv_datasets](./modules/csv_datasets.ipynb)
Tutorial shows the usage of `CSVDataset` and `CSVIterableDataset`, load multiple CSV files and execute postprocessing logic.

教程展示了 `CSVDataset` 和 `CSVIterableDataset` 的用法，加载多个 CSV 文件并执行后处理逻辑。

#### [decollate_batch](./modules/decollate_batch.py)
Tutorial shows how to decollate batch data to simplify post processing transforms and execute more flexible following operations.

教程展示了如何去整理批处理数据以简化后处理转换并执行更灵活的后续操作。

#### [image_dataset](./modules/image_dataset.py)
Notebook introduces basic usages of `monai.data.ImageDataset` module.

Notebook 介绍了 `monai.data.ImageDataset` 模块的基本用法。

#### [dynunet_tutorial](./modules/dynunet_pipeline)
This tutorial shows how to train 3D segmentation tasks on all the 10 decathlon datasets with the reimplementation of dynUNet in MONAI.

本教程展示了如何在 MONAI 中重新实现 dynUNet，在所有 10 个十项全能数据集上训练 3D 分割任务。


#### [integrate_3rd_party_transforms](./modules/integrate_3rd_party_transforms.ipynb)
This tutorial shows how to integrate 3rd party transforms into MONAI program.
Mainly shows transforms from BatchGenerator, TorchIO, Rising and ITK.

本教程展示了如何将 3rd 方转换集成到 MONAI 程序中。
主要展示来自 BatchGenerator、TorchIO、Rising 和 ITK 的转换。

#### [inverse transformations and test-time augmentations](./modules/inverse_transforms_and_test_time_augmentations.ipynb)
This notebook demonstrates the use of invertible transforms, and then leveraging inverse transformations to perform test-time augmentations.

这个笔记本演示了可逆变换的使用，然后利用逆变换来执行测试时间增强。

#### [layer wise learning rate](./modules/layer_wise_learning_rate.ipynb)
This notebook demonstrates how to select or filter out expected network layers and set customized learning rate values.

此笔记本演示了如何选择或过滤出预期的网络层并设置自定义的学习率值。

#### [learning rate finder](./modules/learning_rate.ipynb)
This notebook demonstrates how to use `LearningRateFinder` API to tune the learning rate values for the network.

本笔记本演示了如何使用 `LearningRateFinder` API 来调整网络的学习率值。

#### [load_medical_imagesl](./modules/load_medical_images.ipynb)
This notebook introduces how to easily load different formats of medical images in MONAI and execute many additional operations.

本笔记本介绍了如何在 MONAI 中轻松加载不同格式的医学图像并执行许多附加操作。

#### [mednist_GAN_tutorial](./modules/mednist_GAN_tutorial.ipynb)
This notebook illustrates the use of MONAI for training a network to generate images from a random input tensor.
A simple GAN is employed to do with a separate Generator and Discriminator networks.

本笔记本说明了使用 MONAI 训练网络以从随机输入张量生成图像。
一个简单的 GAN 用于处理单独的生成器和鉴别器网络。

#### [mednist_GAN_workflow_dict](./modules/mednist_GAN_workflow_dict.ipynb)
This notebook shows the `GanTrainer`, a MONAI workflow engine for modularized adversarial learning. Train a medical image reconstruction network using the MedNIST hand CT scan dataset. Dictionary version.

这个笔记本展示了“GanTrainer”，一个用于模块化对抗学习的 MONAI 工作流引擎。使用 MedNIST 手部 CT 扫描数据集训练医学图像重建网络。字典版本。

#### [mednist_GAN_workflow_array](./modules/mednist_GAN_workflow_array.ipynb)
This notebook shows the `GanTrainer`, a MONAI workflow engine for modularized adversarial learning. Train a medical image reconstruction network using the MedNIST hand CT scan dataset. Array version.

这个笔记本展示了“GanTrainer”，一个用于模块化对抗学习的 MONAI 工作流引擎。使用 MedNIST 手部 CT 扫描数据集训练医学图像重建网络。数组版本。

#### [cross_validation_models_ensemble](./modules/cross_validation_models_ensemble.ipynb)
This tutorial shows how to leverage `CrossValidation`, `EnsembleEvaluator`, `MeanEnsemble` and `VoteEnsemble` modules in MONAI to set up cross validation and ensemble program.

本教程展示了如何利用 MONAI 中的 `CrossValidation`、`EnsembleEvaluator`、`MeanEnsemble` 和 `VoteEnsemble` 模块来设置交叉验证和集成程序。

#### [nifti_read_example](./modules/nifti_read_example.ipynb)
Illustrate reading NIfTI files and iterating over image patches of the volumes loaded from them.

说明读取 NIfTI 文件并迭代从它们加载的卷的图像补丁。

#### [network_api](./modules/network_api.ipynb)
This tutorial illustrates the flexible network APIs and utilities.

本教程说明了灵活的网络 API 和实用程序。

#### [postprocessing_transforms](./modules/postprocessing_transforms.ipynb)
This notebook shows the usage of several postprocessing transforms based on the model output of spleen segmentation task.

本笔记本展示了基于脾脏分割任务模型输出的几种后处理变换的用法。

#### [public_datasets](./modules/public_datasets.ipynb)
This notebook shows how to quickly set up training workflow based on `MedNISTDataset` and `DecathlonDataset`, and how to create a new dataset.

本笔记本展示了如何基于 `MedNISTDataset` 和 `DecathlonDataset` 快速设置训练工作流程，以及如何创建新数据集。

#### [tcia_csv_processing](./modules/tcia_csv_processing.ipynb)
This notebook shows how to load the TCIA data with CSVDataset from CSV file and extract information for TCIA data to fetch DICOM images based on REST API.

此笔记本展示了如何使用 CSVDataset 从 CSV 文件加载 TCIA 数据，并提取 TCIA 数据的信息以基于 REST API 获取 DICOM 图像。

#### [transforms_demo_2d](./modules/transforms_demo_2d.ipynb)
This notebook demonstrates the image transformations on histology images using
[the GlaS Contest dataset](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/).

这个笔记本演示了组织学图像上的图像转换，使用
[GlaS 竞赛数据集](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/)。

#### [UNet_input_size_constrains](./modules/UNet_input_size_constrains.ipynb)
This tutorial shows how to determine a reasonable spatial size of the input data for MONAI UNet, which not only supports residual units, but also can use more hyperparameters (like `strides`, `kernel_size` and `up_kernel_size`) than the basic UNet implementation.

本教程展示了如何为 MONAI UNet 确定输入数据的合理空间大小，它不仅支持残差单元，而且可以使用比基本 UNet 实现更多的超参数（如 `strides`、`kernel_size` 和 `up_kernel_size`）

#### [TorchIO, MONAI, PyTorch Lightning](./modules/TorchIO_MONAI_PyTorch_Lightning.ipynb)
This notebook demonstrates how the three libraries from the official PyTorch Ecosystem can be used together to segment the hippocampus on brain MRIs from the Medical Segmentation Decathlon.

本笔记本演示了如何使用来自官方 PyTorch 生态系统的三个库来分割来自 Medical Segmentation Decathlon 的大脑 MRI 上的海马体。

#### [varautoencoder_mednist](./modules/varautoencoder_mednist.ipynb)
This tutorial uses the MedNIST scan (or alternatively the MNIST) dataset to demonstrate MONAI's variational autoencoder class.

本教程使用 MedNIST 扫描（或 MNIST）数据集来演示 MONAI 的变分自动编码器类。

#### [interpretability](./modules/interpretability)
Tutorials in this folder demonstrate model visualisation and interpretability features of MONAI. Currently, it consists of class activation mapping and occlusion sensitivity for 3D classification model visualisations and analysis.

此文件夹中的教程演示了 MONAI 的模型可视化和可解释性功能。目前，它包括用于 3D 分类模型可视化和分析的类激活映射和遮挡敏感性。

#### [Transfer learning with MMAR](./modules/transfer_mmar.ipynb)
This tutorial demonstrates a transfer learning pipeline from a pretrained model in [Clara Train's Medical Model Archive format](https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html).  The notebook also shows the use of LMDB-based dataset.

本教程演示了来自 [Clara Train 的医学模型存档格式](https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html) 的预训练模型的迁移学习管道。该笔记本还展示了基于 LMDB 的数据集的使用。

#### [Transform visualization](./modules/transform_visualization.ipynb)
This tutorial shows several visualization approaches for 3D image during transform augmentation.

本教程展示了变换增强期间 3D 图像的几种可视化方法。
