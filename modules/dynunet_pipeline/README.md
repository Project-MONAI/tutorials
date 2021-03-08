# Overview
This pipeline is modified from NNUnet [1][2] which wins the "Medical Segmentation Decathlon Challenge 2018" and open sourced from https://github.com/MIC-DKFZ/nnUNet.

## Data
The source decathlon datasets can be found from http://medicaldecathlon.com/.

After getting the dataset, please run `create_datalist.py` to get the datalists (please check the command line arguments first). The default seed can help to get the same 5 folds data splits as NNUnet has, and the created datalist will be in `config/`

## Training
Please run `train.py` for training. Please modify the command line arguments according
to the actual situation, such as `determinism_flag` for deterministic training, `amp` for automatic mixed precision.

## Validation
Please run `train.py` and set the argument `mode` to `val` for validation.

## Inference
Please run `inference.py` for inference.

## Examples
There are some examples in `commands/` and based on task 04 (fold 0 for validation).
For instance:

- `train.sh` is used for training.
- `finetune.sh` is used for finetuning.
- `val.sh` is used for validation.
- `infer.sh` is used for inference.
- If you need to use multiple GPUs, please run scripts that contain `multi_gpu`.

With the default settings in `train.sh` and `finetune.sh`, you can get around the following validation results:

|         | 0      | 1      | 2      | 3      | 4      | Mean   | NNUNet val |
|---------|--------|--------|--------|--------|--------|--------|------------|
| class 1 | 0.9007 | 0.8930 | 0.8985 | 0.8979 | 0.9015 | 0.8983 | 0.8975     |
| class 2 | 0.8835 | 0.8774 | 0.8826 | 0.8818 | 0.8828 | 0.8816 | 0.8807     |

Comments:
- The results of NNUNet come from `3D_fullres` version, which is also the DynUNet used version. The scores are gotten from [1].

# References
[1] Isensee F, Jäger P F, Kohl S A A, et al. Automated design of deep learning methods for biomedical image segmentation[J]. arXiv preprint arXiv:1904.08128, 2019.

[2] Isensee F, Petersen J, Klein A, et al. nnu-net: Self-adapting framework for u-net-based medical image segmentation[J]. arXiv preprint arXiv:1809.10486, 2018.
