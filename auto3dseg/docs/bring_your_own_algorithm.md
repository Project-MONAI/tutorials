## Bring-Your-Own-Algorithm (BYOA)

**Auto3DSeg** provides two ways for users to customize segmentation algorithms or bring their own algorithm to join the pipeline.

### 1. Modify existing algorithm (or template)

The easy way to add customized algorithm is through modifying existing [algorithm templates](https://github.com/Project-MONAI/research-contributions/tree/main/auto3dseg/algorithm_templates). The default templates are implemented following the hybrid programming fashion. Thus user can create a copy of any existing template and modify python scripts ```scripts/train.py``` or configuration files in ```configs``` folder. If user would like to change the way of algorithm generation, user can update the "fill_template_config" function (shown below) in the "Algo" class of the script "scripts/algo.py". An example is shown below.

```python
class NetAlgo(BundleAlgo):
    def fill_template_config(self, data_stats_file, output_path, **kwargs):
			...
            # increase patch_size from [128, 128, 96]
            patch_size = [160, 160, 96]
            max_shape = data_stats["stats_summary#image_stats#shape#max"]
            patch_size = [
                max(32, shape_k // 32 * 32) if shape_k < p_k else p_k for p_k, shape_k in zip(patch_size, max_shape)
            ]

			...

            modality = data_src_cfg.get("modality", "ct").lower()
            spacing = data_stats["stats_summary#image_stats#spacing#median"]

            # change intensity range
            intensity_upper_bound = float(data_stats["stats_summary#image_foreground_stats#intensity#max"])
            intensity_lower_bound = float(data_stats["stats_summary#image_foreground_stats#intensity#min"])

            ct_intensity_xform = {
                "_target_": "Compose",
                "transforms": [
                    {
                        "_target_": "ScaleIntensityRanged",
                        "keys": "@image_key",
                        "a_min": intensity_lower_bound,
                        "a_max": intensity_upper_bound,
                        "b_min": -1.0, # change intensity range after normalizaiton
                        "b_max": 1.0,
                        "clip": True,
                    },
                    {"_target_": "CropForegroundd", "keys": ["@image_key", "@label_key"], "source_key": "@image_key"},
                ],
            }

            mr_intensity_transform = {
                "_target_": "NormalizeIntensityd",
                "keys": "@image_key",
                "nonzero": True,
                "channel_wise": True,
            }
			...
        return fill_records
```

After creating a new template, the user can add the template and start the process by following the tutorial "[Running **Auto3DSeg** with Components](../notebooks/pipeline.ipynb)". A algorithm template example with 3D U-Net can be found [here](../scripts/byoa/). Then, the user can start the process by following the tutorial "[Running **Auto3DSeg** with Components](../notebooks/pipeline.ipynb)".

### 2. Create customized algorithm (or template)

Users can also introduce a new algorithm to **Auto3DSeg**. The minimum requirements for the new algorithm are shown below.

1. A complete [Algorithm class](https://github.com/Project-MONAI/MONAI/blob/dev/monai/auto3dseg/algo_gen.py#L15-L49) with function implementaions;
2. A complete [Algorithm generation class](https://github.com/Project-MONAI/MONAI/blob/dev/monai/auto3dseg/algo_gen.py#L52-L112) with function implementaions.

Examples of algorithm classes and algorithm generation classes can be found [here](../scripts/add_new_algo.py).
