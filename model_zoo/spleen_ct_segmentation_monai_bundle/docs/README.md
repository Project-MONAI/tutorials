# Description
Finetune a pre-trained model for volumetric (3D) segmentation of CT spleen from MSD dataset and apply it to BTCV dataset.

# Overview
This tutorial shows how to finetune a pretrained model from MONAI model zoo on a new dataset using monai bundle.

## Data
The new dataset is BTCV challenge dataset (https://www.synapse.org/#!Synapse:syn3193805/wiki/217752). It has 24 Training + 6 Validation CT abdominal scans.
Introduction of BTCV dataset can be found in https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb

Step 1: Download BTCV dataset and format in the folder following the instruction in https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb

Step 2: Download the the json file for data splits in https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb

Step 3: The labels in BTCV dataset contains 13 organ. So we split the labels and extract spleen label out. This is done by running script/split_spleen_labels.py.

Step 4: move ./data/imagesTr to ./data/spleen/imagesTr

## Pretrained model
The pretrained model is from MONAI model zoo,
 https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/spleen_ct_segmentation_v0.1.1.zip
It was trained using the runner-up [1] awarded pipeline of the "Medical Segmentation Decathlon Challenge 2018" using the UNet architecture [2] with 32

Please download the pretrained model from MONAI model-zoo https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/spleen_ct_segmentation_v0.1.1.zip, and move model.pt to ./models/model.pt

## Training configuration
The training was performed with at least 12GB-memory GPUs.
During training, 19 out of the 24 scans are used for training, while the rest 5 is used for validation and model selection.

Actual Model Input: 96 x 96 x 96

## Modify train.json
| Old json config | Updated json config |
| --- | --- |
| "bundle_root": "/workspace/data/tutorials/modules/bundle/spleen_segmentation", | "bundle_root": Your work directory, |
| "dataset_dir": "/workspace/data/Task09_Spleen",| "data_file_base_dir": "./data/spleen", |
| "images": "$list(sorted(glob.glob(@dataset_dir + '/imagesTr/*.nii.gz')))",| "data_list_file_path": "./data/dataset_0.json", |
| "labels": "$list(sorted(glob.glob(@dataset_dir + '/labelsTr/*.nii.gz')))",| "train_datalist": "$monai.data.load_decathlon_datalist(@data_list_file_path, is_segmentation=True, data_list_key='training', base_dir=@data_file_base_dir)", |
| (In "train#dataset",) "data": "$[{'image': i, 'label': l} for i, l in zip(@images[:-9], @labels[:-9])]",| "data": "$@train_datalist[: int(0.8 * len(@train_datalist))]", |
| (In "validate#dataset",) "data": "$[{'image': i, 'label': l} for i, l in zip(@images[-9:], @labels[-9:])]",| "data": "$@train_datalist[int(0.8 * len(@train_datalist)):]", |
| (In "validate#handlers",) "key_metric_filename": "model.pt"| (train from scratch) "key_metric_filename": "model_from scratch.pt"|
| (In "validate#handlers",) "key_metric_filename": "model.pt"| (train from pretrained model) "key_metric_filename": "model_transfer.pt"|
| (In ""train#handlers"", add)| (train from pretrained model) {"_target_": "CheckpointLoader","load_path": "$@ckpt_dir + '/model.pt'","load_dict": {"model": "@network"}},|

## Modify evaluate.json
| Old json config | Updated json config |
| --- | --- |
| (add)| "test_datalist": "$monai.data.load_decathlon_datalist(@data_list_file_path, is_segmentation=True, data_list_key='validation', base_dir=@data_file_base_dir)", |
| (add)| "validate#dataset": {"_target_": "Dataset","data": "$@test_datalist","transform": "@validate#preprocessing"},|
| (In "validate#handlers",) "load_path": "$@ckpt_dir + '/model.pt'",|(train from scratch) "load_path": "$@ckpt_dir + '/model_from_scratch.pt'", |
| (In "validate#handlers",) "load_path": "$@ckpt_dir + '/model.pt'",|(train from scratch) "load_path": "$@ckpt_dir + '/model_transfer.pt'", |


## Scores
This model achieves the following Dice score on the validation data:

When training with BTCV data from scratch, we got mean Dice = 0.9294.
When finetuning with BTCV data from the pretrained model, we got mean Dice = 0.9488

The Dice of finetuning result is better than training from scratch for every subject in the validation set.

## commands example for training from scratch

Execute training from scratch:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```

Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json']" --logging_file configs/logging.conf
```

## commands example for finetuning

Override the `train` config to execute finetuning from pretrained model:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file "['configs/train.json','configs/finetune.json']" --logging_file configs/logging.conf
```

Override the `train` config to execute evaluation with the trained model:
First modify the "ckpt_file": "model_btcv.pt" to "ckpt_file": "model_transfer.pt" in configs/evaluate.json, then run

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json']" --logging_file configs/logging.conf
```


# Disclaimer
This is an example, not to be used for diagnostic purposes.

# References
[1] Xia, Yingda, et al. "3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training." arXiv preprint arXiv:1811.12506 (2018). https://arxiv.org/abs/1811.12506.

[2] Kerfoot E., Clough J., Oksuz I., Lee J., King A.P., Schnabel J.A. (2019) Left-Ventricle Quantification Using Residual U-Net. In: Pop M. et al. (eds) Statistical Atlases and Computational Models of the Heart. Atrial Segmentation and LV Quantification Challenges. STACOM 2018. Lecture Notes in Computer Science, vol 11395. Springer, Cham. https://doi.org/10.1007/978-3-030-12029-0_40
