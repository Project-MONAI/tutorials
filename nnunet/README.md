# MONAI and nnU-Net Integration

[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) is an open-source deep learning framework that has been specifically designed for medical image segmentation. And nnU-Net is a state-of-the-art deep learning framework that is tailored for medical image segmentation. It builds upon the popular U-Net architecture and incorporates various advanced features and improvements, such as cascaded networks, novel loss functions, and pre-processing steps. nnU-Net also provides an easy-to-use interface that allows users to train and evaluate their segmentation models quickly. nnU-Net has been widely used in various medical imaging applications, including brain segmentation, liver segmentation, and prostate segmentation, among others. The framework has consistently achieved state-of-the-art performance in various benchmark datasets and challenges, demonstrating its effectiveness and potential for advancing medical image analysis.

nnU-Net and MONAI are two powerful open-source frameworks that offer advanced tools and algorithms for medical image analysis. Both frameworks have gained significant popularity in the research community, and many researchers have been using these frameworks to develop new and innovative medical imaging applications.

nnU-Net is a framework that provides a standardized pipeline for training and evaluating neural networks for medical image segmentation tasks. MONAI, on the other hand, is a framework that provides a comprehensive set of tools for medical image analysis, including pre-processing, data augmentation, and deep learning models. It is also built on top of PyTorch and offers a wide range of pre-trained models, as well as tools for model training and evaluation. The integration between nnUNet and MONAI can offer several benefits to researchers in the medical imaging field. By combining the strengths of both frameworks, researchers can take advantage of the standardized pipeline provided by nnUNet and the comprehensive set of tools provided by MONAI.

Overall, the integration between nnU-Net and MONAI can offer significant benefits to researchers in the medical imaging field. By combining the strengths of both frameworks, researchers can accelerate their research and develop new and innovative solutions to complex medical imaging challenges.

## What's New in nnU-Net V2

nnU-Net has release a newer version, nnU-Net V2, recently. Some changes have been made as follows.
- Refactored repository: nnU-Net v2 has undergone significant changes in the repository structure, making it easier to navigate and understand. The codebase has been modularized, and the documentation has been improved, allowing for easier integration with other tools and frameworks.
- New features: nnU-Net v2 has introduced several new [features](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/changelog.md), including:
  - Region based formulation with sigmoid activation;
  - Cross-platform support;
  - Multi-GPU training support.

Overall, nnU-Net v2 has introduced significant improvements and new features, making it a powerful and flexible deep learning framework for medical image segmentation. With its easy-to-use interface, modularized codebase, and advanced features, nnU-Net v2 is poised to advance the field of medical image analysis and improve patient outcomes.

## How does the integration works?
As part of the integration, we have introduced a new class called the `nnUNetV2Runner`, which utilizes the Python APIs available in the official nnU-Net repository. The `nnUNetV2Runner` provides several key features that are useful for general users of MONAI.
- The new class offers Python APIs at a high level to facilitate most of the components in nnU-Net, such as model training, validation, ensemble;
- Users are only required to provide the minimum input, as specified in most of the MONAI tutorials for 3D medical image segmentation. The new class will automatically handle data conversion to prepare data that meets the requirements of nnU-Net, which will largely save time for users to prepare the datatsets;
- Additionally, we have enabled users with more GPU resources to automatically allocate model training jobs in parallel. As nnU-Net requires the training of 20 segmentation models by default, distributing model training to larger resources can significantly improve overall efficiency. For instance, users with 8 GPUs can increase model training speed by 6x to 8x automatically using the new class.

## Benchmarking Results on Public Datasets

In this session, we present the results of our `nnUNetV2Runner` and results from the official nnU-Net repository on various public datasets. The goal is to verify that our `nnUNetV2Runner` implementation achieves the same performance as native nnU-Net runs.

### Datasets

1. [BraTS21](http://braintumorsegmentation.org/): The RSNA-ASNR-MICCAI BraTS 2021 Challenge utilizes multi-institutional preoperative baseline multiparametric magnetic resonance imaging (mpMRI) scans and focuses on evaluating (task 1) state-of-the-art methods for segmentation of intrinsically heterogeneous brain glioblasts in mpMRI scans Tumor subregion.
2. [AMOS22](https://amos22.grand-challenge.org/): Task 1 focuses on the segmentation of abdominal organs using CT scans. The goal is to evaluate the performance of different segmentation methods on a diverse set of 500 cases, with annotations for 15 organs. Task 2 extends the scope of Task 1 by including MRI scans in addition to CT scans. Under this “Cross Modality” setting, a single algorithm must segment abdominal organs from both CT and MRI scans. This task provides an additional 100 MRI scans with the same type of annotation.

The table below shows the results of full-resolution 3D U-Net on fold 0 for each dataset. We can see that the performance of `nnUNetV2Runner` meets expectation.

| Tasks | native nnU-Net | `nnUNetV2Runner` |
|-----------------|-----------------|-----------------|
| BraTS21 | 0.92 | 0.94 |
| AMOS22 (Task 1) | 0.90 | 0.90 |
| AMOS22 (Task 2) | 0.89 | 0.89 |

## Steps

### 1. nnU-Net v2 installation

THe installation instruction is described [here](docs/install.md).

### 2. Run with Minimal Input using ```nnUNetV2Runner```

User needs to provide a data list (".json" file) for the new task and data root. In general, a valid data list needs to follow the format of ones in [Medical Segmentaiton Decathlon](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2). After creating the data list, the user can create a simple "input.yaml" file (shown below) as the minimum input for **nnUNetV2Runner**.

```
modality: CT
datalist: "./msd_task09_spleen_folds.json"
dataroot: "/workspace/data/nnunet_test/test09"
```

User can also set values of directory variables as options in "input.yaml" if any directory needs to be specified.

```
nnunet_preprocessed: "./work_dir/nnUNet_preprocessed" # optional
nnunet_raw: "./work_dir/nnUNet_raw_data_base" # optional
nnunet_results: "./work_dir/nnUNet_trained_models" # optional
```

Once the minimum input information is provided, user can use the following commands to start the process of the entire nnU-Net pipeline automatically (from model training to model ensemble).

```bash
python -m monai.apps.nnunet nnUNetV2Runner run --input_config='./input.yaml'
```

### 2. Run nnU-Net modules using ```nnUNetV2Runner```

```nnUNetV2Runner``` offers the one-stop API to execute the pipeline, as well as the APIs to access the underlying components of nnU-Net V2. Below are the command for different components.

```bash
## [component] convert dataset
python -m monai.apps.nnunet nnUNetV2Runner convert_dataset --input_config "./input_new.yaml"

## [component] converting msd datasets
python -m monai.apps.nnunet nnUNetV2Runner convert_msd_dataset --input_config "./input.yaml" --data_dir "/workspace/data/Task05_Prostate"

## [component] experiment planning and data pre-processing
python -m monai.apps.nnunet nnUNetV2Runner plan_and_process --input_config "./input.yaml"

## [component] single-gpu training for all 20 models
python -m monai.apps.nnunet nnUNetV2Runner train --input_config "./input.yaml"

## [component] single-gpu training for a single model
python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input_config "./input.yaml" \
    --config "3d_fullres" \
    --fold 0

## [component] multi-gpu training for all 20 models
export CUDA_VISIBLE_DEVICES=0,1 # optional
python -m monai.apps.nnunet nnUNetV2Runner train --input_config "./input.yaml" --num_gpus 2

## [component] multi-gpu training for a single model
export CUDA_VISIBLE_DEVICES=0,1 # optional
python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input_config "./input.yaml" \
    --config "3d_fullres" \
    --fold 0 \
    --num_gpus 2

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

## FAQ

THe common questions and answers can be found [here](docs/faq.md).
