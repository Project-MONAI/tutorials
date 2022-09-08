## Run with Minimal Input

Here are steps to quickly launch **Auto3DSeg** for general medical image segmentation.

**Step 0.** Download public data or prepare internal data in a custom data root.

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

