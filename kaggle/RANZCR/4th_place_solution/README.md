# Overview
This pipeline is modified from the 4th place solution of RANZCR CLiP - Catheter and Line Position Challenge in Kaggle:
https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification

The original solution is produced by Team `Watercooled`, and the authors are Dieter (https://www.kaggle.com/christofhenkel) and Psi (https://www.kaggle.com/philippsinger).

# Preparations

Recommend to use MONAI docker:

```
docker pull projectmonai/monai:latest
```

Otherwise, except for monai required libraries, please ensure that `opencv-python` and `scipy` are installed. The command could be:
```
pip install opencv-python
pip install scipy
```

Please download the dataset from the competition site and modify the dataset path in `configs/default_config.py`.

In this pipeline, the data split file `train_folds.csv` is achieved from the following public kernel which is different from the original solution:
https://www.kaggle.com/underwearfitting/how-to-properly-split-folds
Please download and put it into the dataset's path

# GPU used:

A single NVIDIA Tesla V100 32G

# Inference

To do inference locally, the command could be:

```
python train.py -c cfg_seg_40_1024d_full -i True -p your_actual_local_weight_folder_path
```

In Kaggle, since the kernel for submission does not support the internet access, you may need to copy the code of `MONAI/monai/` (if using MONAI docker, the path is `/opt/monai/monai/`) into this directory and upload the whole folder as a Kaggle dataset to use. Please take the `run_infer` function in `train.py` for reference and write the script.

# Places uncovered

The original solution is consisted with 6 different models/training strategies:

1. Backbone: EfficientNet-b8, with imagenet pretrained weights
2. Backbone: EfficientNet-b8, with AdvProp (https://rwightman.github.io/pytorch-image-models/models/advprop/) pretrained weights
3. Backbone: EfficientNet-b7, with imagenet pretrained weights
4. Backbone: EfficientNet-b7, with AdvProp pretrained weights
5. Backbone: EfficientNet-b7, with noisy student(https://rwightman.github.io/pytorch-image-models/models/noisy-student/) pretrained weights
6. Backbone: EfficientNet-b7, with noisy student pretrained weights (different training strategies)

However, only the 2nd, 3rd and 4th methods are covered in the reproduction. The reason is that the EfficientNet implemented in MONAI refered to `EfficientNet-PyTorch
` (https://github.com/lukemelas/EfficientNet-PyTorch) rather than `pytorch-image-models` (https://github.com/rwightman/pytorch-image-models Team `Watercooled` refers to this repository). `EfficientNet-PyTorch` does not provide the uncovered pretrained weights, and due to some differences in efficientnet structure, we cannot make use of weights from `pytorch-image-models` instead.

# Performance

According to the above mentioned reasons, this pipeline does not fully reproduce the original solution. As for the models that have the suitable pretrained weights, comparable performances are reachable.

Training with a single efficientnet-b8 model with AdvProp pretrained weights can achieve:

1. one seed can reach to around `0.97013` for private LB and `0.97092` for public LB.
2. four seeds can reach to around `0.97429` for private LB and `0.97460` for public LB.

The commands are:

```
python train.py -c cfg_seg_40_1024d_full -s 657028
python train.py -c cfg_seg_40_1024d_full -s 770799
python train.py -c cfg_seg_40_1024d_full -s 825460
python train.py -c cfg_seg_40_1024d_full -s 962001
```

In addition, ensemble with more models can get a higher score:
```
python train.py -c cfg_seg_philipp_16_ch_1024_ap_full -s 868472
python train.py -c cfg_seg_philipp_16_ch_1024_ap_full -s 183105

python train.py -c cfg_seg_philipp_16_ch_1024_nons_full -s 701922
python train.py -c cfg_seg_philipp_16_ch_1024_nons_full -s 7259
```
