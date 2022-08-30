# Tutorial to visualize 3D (detection) boxes using 3D Slicer

Visualizing box prediction/annotation in 3D medical image detection is not straightforward. To better understand model predictions, we provide a way to further visualize detection results/ground truth using the free and open source software [3D Slicer](https://www.slicer.org/). 3D Slicer enables visualization of 3D images, 3D segmentation masks, and 3D meshes with 3D orthographic views. And we convert 3D box annotations/predictions into meshes that can be visualized by cross-sectional curves in three 2D planes.

## Prerequisite

- The Version of 3D Slicer should be **4.11.20210226** or later.
- Box information should be stored in a ".json" file. The information of *N* boxes is stored under the key "box" as *N* lists. The six values could be in different 3D [modes](https://github.com/Project-MONAI/MONAI/blob/edf3b742a4ae85d1f30462ed0c7511c520fae888/monai/data/box_utils.py#L447-L456) (e.g., corner or center coordinates). All the coordinate values are in either world coordinate system or image coordinate system. The "data\_sample.json" file is an example with X/Y/Z-axial center coordinates and box lengths (box mode **cccwhd**).

```
"box": [
    [
        -100.312255859375,
        67.529541015625,
        -231.97265625,
        6.328125,
        6.328125,
        6.4453125
    ],
    [
        -19.013427734375,
        10.576416015625,
        -48.28125,
        5.09765625,
        5.09765625,
        5.3125
    ]
]
```

## Steps

### 1. Create ".obj" file for predictions/annotation using the "save\_obj.sh" script.

User needs to specify the box mode and whether to use image coordinates. If image coordinates are specified, the image root directory needs to be provided.

#### Example of boxes in world coordinate

```
INPUT_DATASET_JSON="./data_sample.json"
OUTPUT_DIR="./out"

python save_obj.py  --input_box_mode "cccwhd" \
                    --input_dataset_json ${INPUT_DATASET_JSON} \
                    --output_dir ${OUTPUT_DIR}
```

#### Example of boxes in image coordinate

```
IMAGE_DATA_ROOT="/data_root"
INPUT_DATASET_JSON="./data_sample_xyzxyz_image-coordinate.json"
OUTPUT_DIR="./out_image_coord"

python save_obj.py  --image_coordinate \
                    --image_data_root ${IMAGE_DATA_ROOT} \
                    --input_box_mode "xyzxyz" \
                    --input_dataset_json ${INPUT_DATASET_JSON} \
                    --output_dir ${OUTPUT_DIR}
```

### 2. Load the original 3D image and the resulting ".obj" file into 3D Slicer.

Image is visualized in 3 2D planes, while boxes are visualized in 3D space.

![Step 2](figures/step-1.png)

### 3. Opens the "Data" tab for box property editing.

![Step 3](figures/step-2.png)

### 4. Change the box color by double-clicking the color tab (for better visualization).

![Step 3](figures/step-3.png)

### 5. Enable in-plane visibility of the box by selecting the "Visible" option.

The following example shows predicted boxed (red) and ground truth boxes (green).

![Step 4](figures/step-4.png)

After completing these steps, the user can analyze the boxes using 3D Slicer's basic visualization capabilities (e.g., zoom in, zoom out).
