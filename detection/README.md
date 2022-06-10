# Detection Example
This folder contains an example to run train and validate a 3D detection RetinaNet model.
The workflow of MONAI detection module is shown in the following figure.
<p align="center">
  <img src="https://github.com/Project-MONAI/MONAI/blob/dev/docs/images/detection.png" alt="detection scheme")
</p>

MONAI implementation is based on:

Lin, Tsung-Yi, et al. "Focal loss for dense object detection." ICCV 2017. https://arxiv.org/abs/1708.02002

### 1. Data

The dataset we are experimenting in this example is LUNA16 (https://luna16.grand-challenge.org/Home/).
LUNA16 is a public dataset of CT lung nodule detection. Using raw CT scans, the goal is to identify locations of possible nodules, and to assign a probability for being a nodule to each location.

Disclaimer: We are not the host of the data. Please make sure to read the requirements and usage policies of the data and give credit to the authors of the dataset!

### 2. Questions and bugs

- For questions relating to the use of MONAI, please us our [Discussions tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).

### 3. Run the example
#### [Prepare Your Data](./luna16_prepare_images.py)

The raw CT images in LUNA16 have various of voxel sizes. The first step is to resample them to the same size.

Please open ./config/environment_luna16_prepare.json, change the value of "orig_data_base_dir" to the directory where you store the downloaded images, and change the value of "data_base_dir" to the target directory where you will save the resampled images.

Then resample the images by running
```bash
# Resample images to the spacing defined in ./config/environment_luna16_fold0.json
python luna16_prepare_images.py
```

The original images are with mhd/raw format, the resampled images will be with Nifti format.

#### [3D Detection Training](./luna16_training.py)

The LUNA16 dataset was splitted into 10-fold by LUNA16 challenge organizers to run cross-fold training and validation.

Taking fold 0 as an example, the first step is to open ./config/environment_luna16_fold0.json,
and change the value of "data_base_dir" to the directory where you saved the resampled images.
Then run:

```bash
python3 luna16_training.py \
    -e ./config/environment_luna16_fold0.json \
    -c ./config/config_train_luna16_16g.json
```

This python script uses batch size and patch size defined in ./config/config_train_luna16_16g.json, which works for a 16G GPU.
If you have a different GPU memory size, please change "batch_size", "patch_size", and "val_patch_size" to fit the GPU you use.

For fold i, please change ./config/environment_luna16_fold{i}.json, and run
```bash
python3 luna16_training.py \
    -e ./config/environment_luna16_fold${fold}.json \
    -c ./config/config_train_luna16_16g.json
```

#### [3D Detection Inference](./luna16_testing.py)

If you have a different GPU memory size than 16G, please maximize "val_patch_size" to fit the GPU you use.

For fold i, please run
```bash
python3 luna16_testing.py \
    -e ./config/environment_luna16_fold${fold}.json \
    -c ./config/config_train_luna16_16g.json
```


#### [LUNA16 Detection Evaluation](./run_luna16_offical_eval.sh)

Please first make sure the 10 resulted json files result_luna16_fold{i}.json are in ./result folder.
Then run:
```bash
./run_luna16_offical_eval.sh
```

It first combine result json files from the 10 folds into one csv file, 
then run the official LUNA16 evaluation script downloaded from https://luna16.grand-challenge.org/Evaluation/

The evaluation scores for will be stored in ./result/eval_luna16_scores