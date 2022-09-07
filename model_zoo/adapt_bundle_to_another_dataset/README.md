# Description
Adapt a bundle to another dataset.

# Overview
This tutorial shows how to adapt an example monai bundle from MONAI model-zoo to a new dataset.

## Data
The new dataset is BTCV challenge dataset (https://www.synapse.org/#!Synapse:syn3193805/wiki/217752). It has 24 Training + 6 Validation CT abdominal scans.
Introduction of BTCV dataset can be found in https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb.

There are 13 organs labeld in BTCV. In this example, we focus on spleen segmentation only.

Step 1: Download BTCV dataset RawData.zip following the instruction in https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb. Extract it as `./data/RawData`.

Step 2: Download the the json file for data splits in https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb, and save it to `./data/dataset_0.json`.

Step 3: The segmentation labels in BTCV dataset contain 13 organs. Therefore there are 14 intensity levels in the label images.
In this experiment, we focus on spleen only, which has intensity = 1 in the label images. We split the labels and create binary spleen masks by running `python ./split_spleen_labels.py`.

Step 4: `cp -avr ./data/RawData/Training/img ./data/btcv_spleen/imagesTr`

## Download example monai bundle from model-zoo
```
python -m monai.bundle download --name spleen_ct_segmentation --version "0.1.1" --bundle_dir "./"
```

## Training configuration
The training was performed with at least 12GB-memory GPUs.
During training, 19 out of the 24 scans are used for training, while the rest 5 is used for validation and model selection.

Actual Model Input: 96 x 96 x 96

## Modify ./spleen_ct_segmentation/configs/train.json from the downloaded example
| Old json config | Updated json config |
| --- | --- |
| "bundle_root": "/workspace/data/tutorials/modules/bundle/spleen_segmentation", | "bundle_root": "./spleen_ct_segmentation", |
| "dataset_dir": "/workspace/data/Task09_Spleen",| "data_file_base_dir": "./data/btcv_spleen", |
| "images": "$list(sorted(glob.glob(@dataset_dir + '/imagesTr/*.nii.gz')))",| "data_list_file_path": "./data/dataset_0.json", |
| "labels": "$list(sorted(glob.glob(@dataset_dir + '/labelsTr/*.nii.gz')))",| "train_datalist": "$monai.data.load_decathlon_datalist(@data_list_file_path, is_segmentation=True, data_list_key='training', base_dir=@data_file_base_dir)", |
| (In "train#dataset",) "data": "$[{'image': i, 'label': l} for i, l in zip(@images[:-9], @labels[:-9])]",| "data": "$@train_datalist[: int(0.8 * len(@train_datalist))]", |
| (In "validate#dataset",) "data": "$[{'image': i, 'label': l} for i, l in zip(@images[-9:], @labels[-9:])]",| "data": "$@train_datalist[int(0.8 * len(@train_datalist)):]", |
| (In "train#trainer",) "max_epochs": 100,| "max_epochs": 600,|
| (In "optimizer",) "lr": 0.0001,| "lr": 0.0002,|

## Modify ./spleen_ct_segmentation/configs/evaluate.json
| Old json config | Updated json config |
| --- | --- |
| (Add a new line to the json file)| "test_datalist": "$monai.data.load_decathlon_datalist(@data_list_file_path, is_segmentation=True, data_list_key='validation', base_dir=@data_file_base_dir)", |
| (Add a new line to the json file)| "validate#dataset": {"_target_": "Dataset","data": "$@test_datalist","transform": "@validate#preprocessing"},|


## Scores
This model achieves the following Dice score on the validation data:
Mean Dice = 0.9294.


## Commands example

Execute training:

```
python -m monai.bundle run training --meta_file ./spleen_ct_segmentation/configs/metadata.json --config_file ./spleen_ct_segmentation/configs/train.json --logging_file ./spleen_ct_segmentation/configs/logging.conf
```

Override the `train` config to execute evaluation:

```
python -m monai.bundle run evaluating --meta_file ./spleen_ct_segmentation/configs/metadata.json --config_file "['./spleen_ct_segmentation/configs/train.json','./spleen_ct_segmentation/configs/evaluate.json']" --logging_file ./spleen_ct_segmentation/configs/logging.conf
```



# Disclaimer
This is an example, not to be used for diagnostic purposes.

# References
[1] Xia, Yingda, et al. "3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training." arXiv preprint arXiv:1811.12506 (2018). https://arxiv.org/abs/1811.12506.

[2] Kerfoot E., Clough J., Oksuz I., Lee J., King A.P., Schnabel J.A. (2019) Left-Ventricle Quantification Using Residual U-Net. In: Pop M. et al. (eds) Statistical Atlases and Computational Models of the Heart. Atrial Segmentation and LV Quantification Challenges. STACOM 2018. Lecture Notes in Computer Science, vol 11395. Springer, Cham. https://doi.org/10.1007/978-3-030-12029-0_40
