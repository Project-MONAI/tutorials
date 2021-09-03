# Overview
This pipeline is modified from the 4th place solution of RANZCR CLiP - Catheter and Line Position Challenge in Kaggle:
https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification

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

# Performance

Training with a single efficientnet-b8 model with:

1. one seed can reach to around `0.97013` for private LB and `0.97092` for public LB.
2. four seeds can reach to around `0.97429` for private LB and `0.97460` for public LB.

```
python train.py -c cfg_seg_40_1024d_full -s 657028
python train.py -c cfg_seg_40_1024d_full -s 770799
python train.py -c cfg_seg_40_1024d_full -s 825460
python train.py -c cfg_seg_40_1024d_full -s 962001
```

Ensemble with more models can get a higher score:
```
python train.py -c cfg_seg_philipp_16_ch_1024_ap_full -s 868472
python train.py -c cfg_seg_philipp_16_ch_1024_ap_full -s 183105

python train.py -c cfg_seg_philipp_16_ch_1024_nons_full -s 701922
python train.py -c cfg_seg_philipp_16_ch_1024_nons_full -s 7259
```

# Inference

To do inference locally, the command could be:

```
python train.py -c cfg_seg_40_1024d_full -i True -p your_actual_local_weight_folder_path

```

In Kaggle, since the kernel for submission does not support the internet access, you may need to copy the code of `MONAI/monai/` (if using MONAI docker, the path is `/opt/monai/monai/`) into this directory and upload the whole folder as a Kaggle dataset to use. Please take the `run_infer` function in `train.py` for reference and write the script.
