<h1 align="center"> Auto3DSeg </h1>

<div align="center"> <img src="figures/workflow_v1.png" width="800"/> </div>

## Introduction

**Auto3DSeg** is a comprehensive solution for large-scale 3D medical image segmentation. It leverages the latest advances in **MONAI** and GPUs to efficiently develop and deploy algorithms with state-of-the-art performance for beginners or advanced researchers in the field. 3D medical image segmentation is an important task with great potential for clinical understanding, disease diagnosis, and surgical planning. According to the statistics of the recent [MICCAI](http://www.miccai.org/) conferences, more than 60% of the papers are applications of segmentation algorithms, and more than half of them use 3D datasets. After working in this field for many years, we have released the state-of-the-art segmentation solution **Auto3DSeg**, which requires minimal user input (e.g., data root and list).

**Auto3DSeg** first analyzes the global information such as intensity, data size, and data spacing of the dataset, and then generates algorithm folders in MONAI bundle format based on data statistics and algorithm templates. Next, all algorithms initiate model training to obtain checkpoints with the best validation accuracy. Finally, the ensemble module selects the algorithms via ranking trained checkpoints and creates ensemble predictions. Meanwhile, the solution offers different levels of user experience for beginners and advanced researchers. It has been tested on large-scale 3D medical imaging datasets in several different modalities.

<details open>
<summary>Major features</summary>

- **Unified Framework**

  **Auto3DSeg** is a self-contained solution for 3D medical image segmentation with minimal user input.

- **Flexible Modular Design**

  **Auto3DSeg** components can be used independently to meet different needs of users.

- **Support of Bring-Your-Own-Algorithm (BYOA)**

  We have introduced an efficient way to introduce users' own algorithms into the **Auto3DSeg** framework.

- **High Accuracy and Efficiency**

  **Auto3DSeg** achieves state-of-the-art performance in most applications of 3D medical image segmentation.

</details>

## Auto3DSeg Leaderboard Performance

- **Auto3DSeg** won 1st place in MICCAI 2022 challenge **[HECKTOR 2022: Head and neck tumor segmentation and outcome prediction in PET/CT images](https://hecktor.grand-challenge.org/)**
  - 1st place  (segmentation task)
- **Auto3DSeg** got 2nd place in MICCAI 2022 challenge **[INSTANCE22: The 2022 Intracranial Hemorrhage Segmentation Challenge on Non-Contrast head CT (NCCT)](https://instance.grand-challenge.org/)**
  - 1st rank in terms of Dice score
- **Auto3DSeg** got 2nd place in MICCAI 2022 challenge **[ISLES'22: Ischemic Stroke Lesion Segmentation Challenge](https://isles22.grand-challenge.org/)**
  - 1st rank in terms of Dice score

We have demonstrated preliminary results of public datasets are described in the [tasks](tasks) folder.

- [HECKTOR22](tasks/hecktor22/README.md)
- [INSTANCE22](tasks/instance22/README.md)
- [Beyond the Cranial Vault (BTCV) Abdomen Dataset](tasks/btcv/README.md)
- Medical Segmentation Decathlon (MSD) Dataset
	- [Task04 Task04_Hippocampus](tasks/msd/Task04_Hippocampus/README.md)
	- [Task05 Prostate](tasks/msd/Task05_Prostate/README.md)
	- [Task09 Spleen](tasks/msd/Task09_Spleen/README.md)

## A Two-Minute "Hello-World" Example

We provide [a two-minute example](notebooks/auto3dseg_hello_world.ipynb) for users to get initial impressions of **Auto3DSeg**. The example covers the entire pipeline from start to finish, and can be done in two minutes using a single GPU (GPU RAM >= 8GB). Each fold of cross validation consumes about 1.2 GB of disk space, and in this example, about 2.4 GB in total. All the results will be written to user's working directory.

## A "Real-World" Example

To further demonstrate the capabilities of **Auto3DSeg**, [here](tasks/instance22) is the detailed performance of the algorithm in **Auto3DSeg**, which won 2nd place in the MICCAI 2022 challenge** [INSTANCE22: The 2022 Intracranial Hemorrhage Segmentation Challenge on Non-Contrast Head CT (NCCT)](https://instance.grand-challenge.org/)**

## Reference Python APIs for Auto3DSeg

**Auto3DSeg** offers users different levels of APIs to run pipelines that suit their needs.

### 1. Run with Minimal Input using ```AutoRunner```

The user needs to provide a data list (".json" file) for the new task and data root. A typical data list is as this [example](tasks/msd/Task05_Prostate/msd_task05_prostate_folds.json). A sample datalist for an existing MSD formatted dataset can be created using [this notebook](notebooks/msd_datalist_generator.ipynb). After creating the data list, the user can create a simple "task.yaml" file (shown below) as the minimum input for **Auto3DSeg**.

```
modality: CT
datalist: "./task.json"
dataroot: "/workspace/data/task"
```

User needs to define the modality of data. Currently **Auto3DSeg** supports both CT and MRI (single- or multi-modality MRI). Then user can run the pipeline further from start to finish using the following simple bash command with the ```AutoRunner``` class.

```bash
python -m monai.apps.auto3dseg AutoRunner run --input='./task.yaml'
```

An example with detailed description is discussed [here](docs/run_with_minimal_input.md). And we demonstrate the entire pipeline with all necessary components in this example [notebook](notebooks/auto_runner.ipynb) using the AutoRunner class.

### 2. Run with Module APIs

**Auto3DSeg** offers the one-stop AutoRunner API to execute the pipeline, as well as the APIs to access the underlying components built to support the AutoRunner. In this [notebook](notebooks/auto3dseg_autorunner_ref_api.ipynb), AutoRunner is broken down by the step-by-step and we will introduce the API calls in Python and CLI commands. Particularly, we will map the AutoRunner commands and configurations to each of the **Auto3DSeg** module APIs.

## Demystifying Auto3DSeg Components

Each module of **Auto3DSeg** in different steps can be individually used for different purposes. And functions/methods in the components can be customized by users.

- Step 1: [Data analyzer](docs/data_analyzer.md)
- Step 2: [Algorithm generation](docs/algorithm_generation.md)
- Step 3: [Model training, validation, and inference](docs/bundle.md)
- Step 4: [Hyper-parameter optimization](docs/hpo.md)
- Step 5: [Model ensemble](docs/ensemble.md)

## Performance Benchmarking

The subsequent section presents the benchmarking outcomes of the Auto3DSeg algorithms concerning computational efficiency. Dataset [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) has been selected for demonstration purposes, as it is among the largest publicly available 3D medical image datasets, containing over 1,000 CT images and their corresponding 104 foreground classes of segmentation annotations. This dataset features a substantial variations in field-of-view and organ/bone shapes.

To ensure equitable comparisons, we adhere to the original methodology employed in TotalSegmentator, dividing the 104 foreground classes into five segments and utilizing one segment, comprised of [17 foreground classes](https://github.com/wasserth/TotalSegmentator/blob/bc9092164e6025a185473026613f38a4177e7a09/totalsegmentator/map_to_binary.py#L398-L417), for model training. We have provided numerical results for **each fold** in a 5-fold cross-validation of three algorithms (DiNTS, 3D SegResNet, SwinUNETR). It is important to note that, for this particular dataset, 2D SegResNet is not employed in the model training process due to the data spacing distribution and the internal algorithm selection logic we utilize. The GPU utilization and memory usage are assessed utilizing the widely recognized [DCGM](https://developer.nvidia.com/dcgm) library.

<div align="center">

|    Algorithm   |    GPU    | GPU Numbers | Model Training Time (Hours) | GPU Utilization Rate |
|:--------------:|:---------:|:-----------:|:---------------------------:|:--------------------:|
|      DiNTS     | 80GB A100 |      1      |             19.0            |          92%         |
|      DiNTS     | 80GB A100 |      8      |             2.5             |          92%         |
|      DiNTS     | 80GB A100 |      16     |             1.5             |          89%         |
|      DiNTS     | 80GB A100 |      32     |             0.9             |          84%         |
| SegResNet (3D) | 80GB A100 |      1      |             13.8            |          92%         |
| SegResNet (3D) | 80GB A100 |      8      |             2.8             |          91%         |
| SegResNet (3D) | 80GB A100 |      16     |             1.5             |          89%         |
| SegResNet (3D) | 80GB A100 |      32     |             0.8             |          88%         |
|    SwinUNETR   | 80GB A100 |      1      |             15.6            |          95%         |
|    SwinUNETR   | 80GB A100 |      8      |             2.2             |          94%         |
|    SwinUNETR   | 80GB A100 |      16     |             1.0             |          93%         |
|    SwinUNETR   | 80GB A100 |      32     |             0.6             |          91%         |

</div>

The table illustrates that when GPU number exceeds or is equal to 8, multi-node training is executed with 8 GPUs allocated per node. As demonstrated by the results, employing a larger number of GPUs significantly diminishes the model training duration. However, there is a minor decrease in GPU utilization rate, which can be attributed to the increased communication costs associated with a greater number of GPUs.

## GPU utilization optimization

Given the variety of GPU devices users have, we provide an automated way to optimize the GPU utilization (e.g., memory usage) of algorithms in Auto3DSeg.
During algorithm generation, users can enable the optimization option.
Auto3DSeg can then further automatically tune the hyper-parameters to fully utilize the available GPU capacity.
Concrete examples can be found [here](docs/gpu_opt.md).

## FAQ

Please refer to [FAQ](docs/faq.md) for frequently asked questions.
