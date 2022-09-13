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

## Get Started

We offer users different ways to use **Auto3DSeg** to suit their needs.

### 1. Run with Minimal Input

The user needs to provide a data list (".json" file) for the new task and data root. A typical data list is as this [example](tasks/msd/Task05_Prostate/msd_task05_prostate_folds.json). After creating the data list, the user can create a simple "task.yaml" file (shown below) as the minimum input for **Auto3DSeg**.

```
modality: CT
datalist: "./task.json"
dataroot: "/workspace/data/task"
```

User needs to define the modality of data. Currently **Auto3DSeg** supports both CT and MRI (single- or multi-modality MRI). Then user can run the pipeline further from start to finish using the following simple bash command.

```bash
python -m monai.apps.auto3dseg AutoRunner run --input='./task.yaml'
```

An example with detailed description is discussed in this [tutorial](docs/run_with_minimal_input.md).

### 2. Run with Components

We demonstrate the entire pipeline with all necessary componets in the [tutorial notebooks](notebooks/auto_runner.ipynb) using the AutoRunner class. And each component can be individually used for different purposes. And functions/methods in the components can be customized by users.

- Step 1: [Data analyzer](docs/data_analyzer.md)
- Step 2: [Algorithm generation](docs/algorithm_generation.md)
- Step 3: [Model training, validation, and inference](docs/bundle.md)
- Step 4: [Hyper-parameter optimization](docs/hpo.md)
- Step 5: [Model ensemble](docs/ensemble.md)

### 3. Run with Customization / Bring-Your-Own-Algorithm (BYOA)

**Auto3DSeg** also gives users the option to bring their own segmentation algorithm to **Auto3DSeg**. Users can add custom algorithms or custom algorithm templates. The details of adding customized algorithms can be found [here](docs/bring_your_own_algorithm.md).

## Benchmarks

Some benchmark results of public datasets are described in the [tasks](tasks) folder.

- [Beyond the Cranial Vault (BTCV) Abdomen Dataset](tasks/btcv)
- Medical Segmentation Decathlon (MSD) Dataset
	- [Task05 Prostate](tasks/msd/Task05_Prostate)
	- [Task09 Spleen](tasks/msd/Task09_Spleen)

## FAQ

Please refer to [FAQ](docs/faq.md) for frequently asked questions.
