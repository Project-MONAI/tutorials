# Overview
This pipeline is modified from the 4th place solution of RANZCR CLiP - Catheter and Line Position Challenge in Kaggle:
https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification

# Prepared data

Please download the dataset from the competition site and modify the dataset path in `configs/default_config.py`. 

In this pipeline, the data split file `train_folds.csv` is achieved from the following public kernel which is different from the original solution:
https://www.kaggle.com/underwearfitting/how-to-properly-split-folds
Please download and put it into the dataset's path

# GPU used:

A single NVIDIA Tesla V100 32G

# Required Environment:

(docker) projectmonai/monai:latest

# Performance

Training with a single efficientnet-b8 model with one seed (using the following command) can reach to around 0.97014 for private LB and 0.97244 for public LB:

```
python train.py -c cfg_seg_40_1024d_full -s 657028
```

Training with more models can get a higher score:
```
python train.py -c cfg_seg_40_1024d_full -s 657028
python train.py -c cfg_seg_40_1024d_full -s 770799
python train.py -c cfg_seg_40_1024d_full -s 825460
python train.py -c cfg_seg_40_1024d_full -s 962001

python train.py -c cfg_seg_philipp_16_ch_1024_ap_full -s 868472
python train.py -c cfg_seg_philipp_16_ch_1024_ap_full -s 183105

python train.py -c cfg_seg_philipp_16_ch_1024_nons_full -s 701922
python train.py -c cfg_seg_philipp_16_ch_1024_nons_full -s 7259

```

(the corresponding ensembled scores as well as the inference kernel will be updated later)