# Description
Load the pre-trained weights from a MONAI model-zoo model and execute the transfer-learning in a typical PyTroch training program based on it.

# Overview
This tutorial shows how to load a pretrained U-net from MONAI model-zoo and train it on BTCV dataset using pytorch. The pretrained U-net was trained on volumetric (3D) segmentation of CT spleen from MSD dataset.

## Data
The description of data and data preparation can be found in [adapt_bundle_to_another_dataset](../adapt_bundle_to_another_dataset)

## Pretrained model
The pretrained model is from MONAI model-zoo spleen_ct_segmentation.
It was trained using the runner-up [1] awarded pipeline of the "Medical Segmentation Decathlon Challenge 2018" using the UNet architecture [2] with 32 training images and 9 validation images.

It can automatically download the bundle and load the model with the following code, which can be found in `train.py`.
```
import monai
pretrained_model = monai.bundle.load(
    name="spleen_ct_segmentation", bundle_dir="./", version="0.1.1"
)
model.load_state_dict(pretrained_model)
```

## Training configuration
The training was performed with at least 12GB-memory GPUs.
During training, 19 out of the 24 scans are used for training, while the rest 5 is used for validation and model selection.

Actual Model Input: 96 x 96 x 96

## Input and output formats
Input: 1 channel CT image

Output: 2 channels: Label 1: spleen; Label 0: everything else

## Scores
This model achieves the following Dice score on the validation data:

When training with BTCV data from scratch, we got mean Dice = 0.9294.
When finetuning with BTCV data from the pretrained model, we got mean Dice = 0.9488

The Dice of finetuning result is better than training from scratch for every subject in the validation set.

## Commands example
Train and evaluate from scratch:
```
python3 train.py
python3 evaluate.py
```

Load pre-trained model and do transfer learning:
```
python3 train.py --load_pretrained_ckpt
python3 evaluate.py --load_pretrained_ckpt
```

# Disclaimer
This is an example, not to be used for diagnostic purposes.

# References
[1] Xia, Yingda, et al. "3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training." arXiv preprint arXiv:1811.12506 (2018). https://arxiv.org/abs/1811.12506.

[2] Kerfoot E., Clough J., Oksuz I., Lee J., King A.P., Schnabel J.A. (2019) Left-Ventricle Quantification Using Residual U-Net. In: Pop M. et al. (eds) Statistical Atlases and Computational Models of the Heart. Atrial Segmentation and LV Quantification Challenges. STACOM 2018. Lecture Notes in Computer Science, vol 11395. Springer, Cham. https://doi.org/10.1007/978-3-030-12029-0_40
