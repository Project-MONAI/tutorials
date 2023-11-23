# MONAI and nnU-Net Integration

[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) is an open-source deep learning framework that has been specifically designed for medical image segmentation. And nnU-Net is a state-of-the-art deep learning framework that is tailored for medical image segmentation. It builds upon the popular U-Net architecture and incorporates various advanced features and improvements, such as cascaded networks, novel loss functions, and pre-processing steps. nnU-Net also provides an easy-to-use interface that allows users to train and evaluate their segmentation models quickly. nnU-Net has been widely used in various medical imaging applications, including brain segmentation, liver segmentation, and prostate segmentation, among others. The framework has consistently achieved state-of-the-art performance in various benchmark datasets and challenges, demonstrating its effectiveness and potential for advancing medical image analysis.

nnU-Net and MONAI are two powerful open-source frameworks that offer advanced tools and algorithms for medical image analysis. Both frameworks have gained significant popularity in the research community, and many researchers have been using these frameworks to develop new and innovative medical imaging applications.

nnU-Net is a framework that provides a standardized pipeline for training and evaluating neural networks for medical image segmentation tasks. MONAI, on the other hand, is a framework that provides a comprehensive set of tools for medical image analysis, including pre-processing, data augmentation, and deep learning models. It is also built on top of PyTorch and offers a wide range of pre-trained models, as well as tools for model training and evaluation. The integration between nnUNet and MONAI can offer several benefits to researchers in the medical imaging field. By combining the strengths of both frameworks, researchers can take advantage of the standardized pipeline provided by nnUNet and the comprehensive set of tools provided by MONAI.

Overall, the integration between nnU-Net and MONAI can offer significant benefits to researchers in the medical imaging field. By combining the strengths of both frameworks, researchers can accelerate their research and develop new and innovative solutions to complex medical imaging challenges.

## What's New in nnU-Net V2

nnU-Net has released a newer version, nnU-Net V2, recently. Some changes have been made as follows.
- Refactored repository: nnU-Net v2 has undergone significant changes in the repository structure, making it easier to navigate and understand. The codebase has been modularized, and the documentation has been improved, allowing for easier integration with other tools and frameworks.
- New features: nnU-Net v2 has introduced several new [features](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/changelog.md), including:
  - Region-based formulation with sigmoid activation;
  - Cross-platform support;
  - Multi-GPU training support.

Overall, nnU-Net v2 has introduced significant improvements and new features, making it a powerful and flexible deep learning framework for medical image segmentation. With its easy-to-use interface, modularized codebase, and advanced features, nnU-Net v2 is poised to advance the field of medical image analysis and improve patient outcomes.

## How does the integration work?
As part of the integration, we have introduced a new class called the `nnUNetV2Runner`, which utilizes the Python APIs available in the official nnU-Net repository. The `nnUNetV2Runner` provides several key features that are useful for general users of MONAI.
- The new class offers Python APIs at a high level to facilitate most of the components in nnU-Net, such as model training, validation, ensemble;
- Users are only required to provide the minimum input, as specified in most of the MONAI tutorials for 3D medical image segmentation. The new class will automatically handle data conversion to prepare data that meets the requirements of nnU-Net, which will largely save time for users to prepare the datasets;
- Additionally, we have enabled users with more GPU resources to automatically allocate model training jobs in parallel. As nnU-Net requires the training of 20 segmentation models by default, distributing model training to larger resources can significantly improve overall efficiency. For instance, users with 8 GPUs can increase model training speed by 6x to 8x automatically using the new class.

## Benchmarking Results on Public Datasets

In this session, we present the results of our `nnUNetV2Runner` and results from the official nnU-Net repository on various public datasets. The goal is to verify that our `nnUNetV2Runner` implementation achieves the same performance as native nnU-Net runs.

### Datasets

1. [BraTS21](http://braintumorsegmentation.org/): The RSNA-ASNR-MICCAI BraTS 2021 Challenge utilizes multi-institutional preoperative baseline multiparametric magnetic resonance imaging (mpMRI) scans and focuses on evaluating (task 1) state-of-the-art methods for segmentation of intrinsically heterogeneous brain glioblasts in mpMRI scans Tumor subregion.
2. [AMOS22](https://amos22.grand-challenge.org/): Task 1 focuses on the segmentation of abdominal organs using CT scans. The goal is to evaluate the performance of different segmentation methods on a diverse set of 500 cases, with annotations for 15 organs. Task 2 extends the scope of Task 1 by including MRI scans in addition to CT scans. Under this “Cross Modality” setting, a single algorithm must segment abdominal organs from both CT and MRI scans. This task provides an additional 100 MRI scans with the same type of annotation.

The table below shows the results of full-resolution 3D U-Net on fold 0 for each dataset. We can see that the performance of `nnUNetV2Runner` meets expectations.

| Tasks | native nnU-Net | `nnUNetV2Runner` |
|-----------------|-----------------|-----------------|
| BraTS21 | 0.92 | 0.94 |
| AMOS22 (Task 1) | 0.90 | 0.90 |
| AMOS22 (Task 2) | 0.89 | 0.89 |

## Steps

### Installation

The installation instruction is described [here](docs/install.md).

### Dataset and Datalist Preparation

The user needs to provide a data list (".json" file) for the new task and data root. In general, a valid data list needs to follow the format of the ones in [Medical Segmentation Decathlon (MSD)](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2).

In [this tutorial](../auto3dseg/notebooks/msd_datalist_generator.ipynb), we provided example steps to download the [MSD Spleen dataset](http://medicaldecathlon.com) and prepare a datalist.
Below we assume the dataset is downloaded to `/workspace/data/Task09_Spleen` and the datalist is in the current directory.

### Run with Minimal Input using ```nnUNetV2Runner```

After creating the data list, the user can create a simple "input.yaml" file (shown below) as the minimum input for **nnUNetV2Runner**.

```
modality: CT
datalist: "./msd_task09_spleen_folds.json"
dataroot: "/workspace/data/Task09_Spleen"
```

Note: For multi-modal inputs, please check the [Frequently Asked Questions section](#FAQ)

Users can also set values of directory variables as options in "input.yaml" if any directory needs to be specified.

```
dataset_name_or_id: 1 # task-specific integer index (optional)
nnunet_preprocessed: "./work_dir/nnUNet_preprocessed" # directory for storing pre-processed data (optional)
nnunet_raw: "./work_dir/nnUNet_raw_data_base" # directory for storing formated raw data (optional)
nnunet_results: "./work_dir/nnUNet_trained_models" # diretory for storing trained model checkpoints (optional)
```

Once the minimum input information is provided, the user can use the following commands to start the process of the entire nnU-Net pipeline automatically (from model training to model ensemble).

```bash
python -m monai.apps.nnunet nnUNetV2Runner run --input_config='./input.yaml'
```

For experiment and debugging purposes, users may want to set the number of epochs of training in the nnU-Net pipeline.
Our integration offers an optional argument `trainer_class_name` to specify the number of epochs as below:

```bash
python -m monai.apps.nnunet nnUNetV2Runner run --input_config='./input.yaml' --trainer_class_name nnUNetTrainer_1epoch
```

The supported `trainer_class_name` are:
- nnUNetTrainer (default)
- nnUNetTrainer_1epoch
- nnUNetTrainer_5epochs
- nnUNetTrainer_10epochs
- nnUNetTrainer_20epochs
- nnUNetTrainer_50epochs
- nnUNetTrainer_100epochs
- nnUNetTrainer_250epochs
- nnUNetTrainer_2000epochs
- nnUNetTrainer_4000epochs
- nnUNetTrainer_8000epochs

### Run nnU-Net modules using ```nnUNetV2Runner```

```nnUNetV2Runner``` offers the one-stop API to execute the pipeline, as well as the APIs to access the underlying components of nnU-Net V2. Below is the command for different components.

```bash
## [component] convert dataset
python -m monai.apps.nnunet nnUNetV2Runner convert_dataset --input_config "./input.yaml"

## [component] experiment planning and data pre-processing
python -m monai.apps.nnunet nnUNetV2Runner plan_and_process --input_config "./input.yaml"

## [component] use all available GPU(s) to train all 20 models
python -m monai.apps.nnunet nnUNetV2Runner train --input_config "./input.yaml"

## [component] use all available GPU(s) to train a single model
python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input_config "./input.yaml" \
    --config "3d_fullres" \
    --fold 0

## [component] distributed training of 20 models utilizing specified GPU devices 0 and 1
python -m monai.apps.nnunet nnUNetV2Runner train --input_config "./input.yaml" --gpu_id_for_all 0,1

## [component] find best configuration
python -m monai.apps.nnunet nnUNetV2Runner find_best_configuration --input_config "./input.yaml"

## [component] predict, ensemble, and postprocessing
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config "./input.yaml"

## [component] predict only
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config "./input.yaml" \
	--run_ensemble false --run_postprocessing false

## [component] ensemble only
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config "./input.yaml" \
	--run_predict false --run_postprocessing false

## [component] post-processing only
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config "./input.yaml" \
	--run_predict false --run_ensemble false
```

For utilizing PyTorch DDP in multi-GPU training, the subsequent command is offered to facilitate the training of a singlular model on a specific fold:

```bash
## [component] multi-gpu training for a single model
python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input_config "./input.yaml" \
    --config "3d_fullres" \
    --fold 0 \
    --gpu_id 0,1
```

We offer an alternative API for constructing datasets from [MSD challenge](http://medicaldecathlon.com/) to meet requirements of nnU-Net, as reference in the provided [link](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#how-to-use-decathlon-datasets).

```bash
## [component] converting msd datasets
python -m monai.apps.nnunet nnUNetV2Runner convert_msd_dataset --input_config "./input.yaml" --data_dir "/workspace/data/Task09_Spleen"
```

## FAQ

The common questions and answers can be found [here](docs/faq.md).
