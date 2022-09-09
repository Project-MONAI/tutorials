## Run with Minimal Input

Here are steps to quickly launch **Auto3DSeg** for general medical image segmentation.

**Step 0.** Download public data or prepare internal data in a custom data root. For data from [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (MSD), user can use the following Python script to download it.

```python
import os
from monai.apps import download_and_extract

root = "./"
msd_task = "Task05_Prostate"
resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/" + msd_task + ".tar"
compressed_file = os.path.join(root, msd_task + ".tar")
if os.path.exists(root):
    download_and_extract(resource, compressed_file, args.root)
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

**Step 2.** Prepare "task.yaml" with necessary information as follows.

```
modality: CT
datalist: "./task.json"
dataroot: "/workspace/data/task"
```

**Step 3.** Get Python script **run_auto3dseg.py** [here](../scripts/run_auto3dseg.py).

**Step 4.** Run the follow bash command to start the pipeline without any further intervetion.

```bash
python run_auto3dseg.py --input "task.yaml"
```
