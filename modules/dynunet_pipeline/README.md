# Overview
This pipeline is modified from NNUnet [1,2] which wins the "Medical Segmentation Decathlon Challenge 2018" and open sourced from https://github.com/MIC-DKFZ/nnUNet.

## Data
The source decathlon datasets can be found from http://medicaldecathlon.com/.

After getting the dataset, please run `create_datalist.py` to get the datalists (please check the command line arguments first). The default seed can help to get the same 5 folds data splits as NNUnet has, and the created datalist will be in `config/`

## Training
Please run `train.py` for training. Please modify the command line arguments according
to the actual situation.

A sample training script is shown in `commands/run_task04.sh`, it runs on task 04 and use
fold 0 for validation. You can use `bash commands/run_task04.sh` to run this script.

## Validation
Please run `train.py` and set the argument `mode` to `val` for validation.

# References
[1] Isensee F, Jäger P F, Kohl S A A, et al. Automated design of deep learning methods for biomedical image segmentation[J]. arXiv preprint arXiv:1904.08128, 2019.

[2] Isensee F, Petersen J, Klein A, et al. nnu-net: Self-adapting framework for u-net-based medical image segmentation[J]. arXiv preprint arXiv:1809.10486, 2018.
