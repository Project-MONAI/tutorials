## Run with Minimal Input

To get initial impressions of **Auto3DSeg**, users can try this [two-minute example](../notebooks/auto3dseg_hello_world.ipynb). The example covers the entire pipeline from start to finish and can be done in two minutes using a single GPU (GPU RAM >= 8GB).

Here are detailed steps to quickly launch **Auto3DSeg** for general medical image segmentation.

**Step 0.** Download public data or prepare internal data in a custom data root. For data from [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (MSD), users can use the following Python script to download it.

```python
import os
from monai.apps import download_and_extract

root = "./"
msd_task = "Task05_Prostate"
resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/" + msd_task + ".tar"
compressed_file = os.path.join(root, msd_task + ".tar")
if os.path.exists(root):
    download_and_extract(resource, compressed_file, root)
```

**Step 1.** Provide the following data list (a ".json" file) for a new task and the data root. The typical data list is shown as follows.

```
{
    "training": [
        {
            "fold": 0,
            "image": "image_001.nii.gz",
            "label": "label_001.nii.gz"
        },
        {
            "fold": 0,
            "image": "image_002.nii.gz",
            "label": "label_002.nii.gz"
        },
        {
            "fold": 1,
            "image": "image_003.nii.gz",
            "label": "label_001.nii.gz"
        },
        {
            "fold": 2,
            "image": "image_004.nii.gz",
            "label": "label_002.nii.gz"
        },
        {
            "fold": 3,
            "image": "image_005.nii.gz",
            "label": "label_003.nii.gz"
        },
        {
            "fold": 4,
            "image": "image_006.nii.gz",
            "label": "label_004.nii.gz"
        }
    ],
    "testing": [
        {
            "image": "image_010.nii.gz"
        }
    ]
}
```

**Step 2.** Prepare "task.yaml" with the necessary information as follows.

```
modality: CT
datalist: "./task.json"
dataroot: "/workspace/data/task"
```

**Step 3.** Run the following bash command to start the pipeline without any further intervention.

```bash
python -m monai.apps.auto3dseg AutoRunner run --input='./task.yaml'
```

## Input

A typical example of an input folder structure with all necessary components is as follows. Components can be located anywhere in the machine as long as the paths in **task.yaml** are correct.

```
./Task/
├─ Data/
├─ task.json
└─ task.yaml
```

## Output

When the pipeline finishes, all output files will be saved in the directory "./workdir" by default. And the output folder structure is shown as follows.

```
./Task/
├── Data/
├── task.json
├── task.yaml
└── workdir/
    ├── datastats.yaml
    ├── algorithm_templates
    │   ├── dints
    │   ├── segresnet
    │   ├── segresnet2d
    │   ├── swinunetr
    ├── dints_0
    │   ├── configs
    │   ├── model_fold0
    │   └── scripts
	...
    ├── segresnet_0
    │   ├── configs
    │   ├── model_fold0
    │   └── scripts
	...
    ├── segresnet2d_0
    │   ├── configs
    │   ├── model_fold0
    │   └── scripts
	...
    ├── swinunetr_0
    │   ├── configs
    │   ├── model_fold0
    │   └── scripts
    ...
    └── ensemble_output
```

Several important components are generated along the way.

1. "datastats.yaml" is a summary of the dataset from the [data analyzer](../docs/data_analyzer.md). The summary report includes information such as data size, spacing, intensity distribution, etc., for a better understanding of the dataset. An example "datastats.yaml" is shown as follows.

```
...
stats_summary:
  image_foreground_stats:
    intensity: {max: 1326.0, mean: 353.68545989990236, median: 339.03333333333336,
      min: 0.0, percentile_00_5: 94.70366643269857, percentile_10_0: 210.9, percentile_90_0: 518.3333333333334,
      percentile_99_5: 734.7439453125, stdev: 122.72876790364583}
  image_stats:
    channels:
      max: 2
      mean: 2.0
      median: 2.0
      min: 2
      percentile: [2, 2, 2, 2]
      percentile_00_5: 2
      percentile_10_0: 2
      percentile_90_0: 2
      percentile_99_5: 2
      stdev: 0.0
    intensity: {max: 2965.0, mean: 307.1866872151693, median: 239.9, min: 0.0, percentile_00_5: 1.5333333333333334,
      percentile_10_0: 54.53333333333333, percentile_90_0: 649.3333333333334, percentile_99_5: 1044.0333333333333,
      stdev: 238.39599100748697}
    shape:
      max: [384, 384, 24]
      mean: [317.8666666666667, 317.8666666666667, 18.8]
...
```

2."algorithm_templates" are the [algorithm templates](../docs/algorithm_generation.md#algorithm-templates) used to generate actual algorithm bundle folders with information from data statistics.

3."dints_x", "segresnet_x", "segresnet2d_x", and "swinunetr_x" are automatically generated 5-fold MONAI bundle based on [established networks and well-tuned training recipes](../docs/algorithm_generation.md#algorithms). They are self-contained folders, which can be used for model training, inference, and validation via executing commands in the README document of each bundle folder. More information can be referred to via this [link](https://docs.monai.io/en/latest/mb_specification.html)](https://docs.monai.io/en/latest/mb_specification.html). And "model_foldx" is where checkpoints after training are saved together with training history and tensorboard event files.

Note: if users would like to run model training parallel with more computing resources, they can stop the pipeline after bundle folders are generated and executed model training via commands in the README document of each bundle folder.

4."predictions_testing" are the predictions for the test data (with the "testing" key in the data list) from the model ensemble. By default, We select the best model/algorithm from each fold for the ensemble.
