# Overview
This pipeline is modified from NNUnet [1][2] which wins the "Medical Segmentation Decathlon Challenge 2018" and open sourced from https://github.com/MIC-DKFZ/nnUNet.

## Data
The source decathlon datasets can be found from http://medicaldecathlon.com/.

After getting the dataset, please run `create_datalist.py` to get the datalists (please check the command line arguments first). The default seed can help to get the same 5 folds data splits as NNUnet has, and the created datalist will be in `config/`

My running environment:

- OS: Ubuntu 20.04.1 LTS
- Python: 3.8.5
- Pytorch: 1.8.0

To prevent the inconsistency, all json files are included in `config/` already.

## Training
Please run `train.py` for training. Please modify the command line arguments according
to the actual situation, such as `determinism_flag` for deterministic training, `amp` for automatic mixed precision.

## Validation
Please run `train.py` and set the argument `mode` to `val` for validation.

## Inference
Please run `inference.py` for inference.

## Examples
All training scripts for 10 tasks are included in `commands/`.
For instance:

- `train.sh` is used for training.
- `finetune.sh` is used for finetuning.
- `val.sh` is used for validation.
- `infer.sh` is used for inference.
- If you need to use multiple GPUs, please run scripts that contain `multi_gpu`.

You can take task 04's scripts for reference since for other tasks, only the training parts are included. A task folder that contains `train.sh` means it only needs to use 1 GPU for training, and `train_multi_gpu.sh` means it needs at least 2 GPUs for training.

The devices I used for training for all tasks are shown as follow:

| task | number of GPUs used (Tesla V100 32GB) |
|:----:|:-------------------------------------:|
|   1  |                   2                   |
|   2  |                   1                   |
|   3  |                   4                   |
|   4  |                   1                   |
|   5  |                   1                   |
|   6  |                   1                   |
|   7  |                   2                   |
|   8  |                   2                   |
|   9  |                   1                   |
|  10  |                   1                   |

I used these scripts and trained for all 5 folds for all 10 tasks. As for the test set, I did the ensemble by average the 5 feature maps (coming from 5 folds' models) before the `argmax` manipulation (for task 03, since the feature maps are very large, I just did voting for 5 final predictions). By submitting the ensembled results to the Decathlon Challenge's Leaderboard, I got the following results:

|         | DynUNet class 1 |   2  |   3  | NNUNet class 1 |   2  |   3  |
|:-------:|:---------------:|:----:|:----:|:--------------:|:----:|:----:|
| task 01 |       0.68      | 0.47 | 0.69 |      0.68      | 0.47 | 0.68 |


|         | DynUNet class 1 | NNUNet class 1 |
|:-------:|:---------------:|:--------------:|
| task 02 |       0.93      |      0.93      |
| task 06 |       0.67      |      0.74      |
| task 09 |       0.96      |      0.97      |
| task 10 |       0.55      |      0.58      |


|         | DynUNet class 1 |   2  | NNUNet class 1 |   2  |
|:-------:|:---------------:|:----:|:--------------:|:----:|
| task 03 |       0.95      | 0.72 |      0.96      | 0.76 |
| task 04 |       0.90      | 0.88 |      0.90      | 0.89 |
| task 05 |       0.71      | 0.87 |      0.77      | 0.90 |
| task 07 |       0.81      | 0.54 |      0.82      | 0.53 |
| task 08 |       0.66      | 0.71 |      0.66      | 0.72 |

Comments:
- The results of DynUNet come from the re-implemented `3D_fullres` version in MONAI and without postprocessing.

- The results of NNUnet come from different versions (`3D_fullres` for task 01, 02 and 04, `3D_cascade` for task 10, and ensembled two versions for other tasks) and may have postprocessing [1]. 

- Therefore, the two results may not be fully comparable and the above tables are just for reference. 

- After implementing this repository, I re-trained on task 04 and attached the validation results as follow, and the comparisons between DynUNet and NNUnet are all for the single `3D_fullres` version.

As for task 04, with the default settings in `train.sh` and `finetune.sh`, you can get around the following validation results:

|         | 0      | 1      | 2      | 3      | 4      | Mean   | NNUNet val |
|---------|--------|--------|--------|--------|--------|--------|------------|
| class 1 | 0.9007 | 0.8930 | 0.8985 | 0.8979 | 0.9015 | 0.8983 | 0.8975     |
| class 2 | 0.8835 | 0.8774 | 0.8826 | 0.8818 | 0.8828 | 0.8816 | 0.8807     |


# References
[1] Isensee F, Jäger P F, Kohl S A A, et al. Automated design of deep learning methods for biomedical image segmentation[J]. arXiv preprint arXiv:1904.08128, 2019.

[2] Isensee F, Petersen J, Klein A, et al. nnu-net: Self-adapting framework for u-net-based medical image segmentation[J]. arXiv preprint arXiv:1809.10486, 2018.
