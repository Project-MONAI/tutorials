# Description
A pre-trained model for volumetric (3D) segmentation of the spleen from CT image.

# Model Overview
This model is trained using the runner-up [1] awarded pipeline of the "Medical Segmentation Decathlon Challenge 2018" using the UNet architecture [2] with 32 training images and 9 validation images.

## Data
The training dataset is Task09_Spleen.tar from http://medicaldecathlon.com/.

## Training configuration
The training was performed with at least 12GB-memory GPUs.

Actual Model Input: 96 x 96 x 96

## Input and output formats
Input: 1 channel CT image

Output: 2 channels: Label 1: spleen; Label 0: everything else

## Scores
This model achieves the following Dice score on the validation data (our own split from the training dataset):

Mean Dice = 0.96

## commands example
Execute training:

```bash
python -m monai.bundle run training \
    --meta_file configs/metadata.json \
    --config_file configs/train.json \
    --logging_file configs/logging.conf
```

Override the `train` config to execute multi-GPU training:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run training \
    --meta_file configs/metadata.json \
    --config_file "['configs/train.json','configs/multi_gpu_train.json']" \
    --logging_file configs/logging.conf
```

Override the `train` config to execute evaluation with the trained model:

```bash
python -m monai.bundle run evaluating \
    --meta_file configs/metadata.json \
    --config_file "['configs/train.json','configs/evaluate.json']" \
    --logging_file configs/logging.conf
```

Execute inference:

```bash
python -m monai.bundle run evaluating \
    --meta_file configs/metadata.json \
    --config_file configs/inference.json \
    --logging_file configs/logging.conf
```

Verify the metadata format:

```
python -m monai.bundle verify_metadata --meta_file configs/metadata.json --filepath eval/schema.json
```

Verify the data shape of network:

```
python -m monai.bundle verify_net_in_out network_def --meta_file configs/metadata.json --config_file configs/inference.json
```

Export checkpoint to TorchScript file:

```bash
python -m monai.bundle ckpt_export network_def \
    --filepath models/model.ts \
    --ckpt_file models/model.pt \
    --meta_file configs/metadata.json \
    --config_file configs/inference.json
```

# Disclaimer
This is an example, not to be used for diagnostic purposes.

# References
[1] Xia, Yingda, et al. "3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training." arXiv preprint arXiv:1811.12506 (2018). https://arxiv.org/abs/1811.12506.

[2] Kerfoot E., Clough J., Oksuz I., Lee J., King A.P., Schnabel J.A. (2019) Left-Ventricle Quantification Using Residual U-Net. In: Pop M. et al. (eds) Statistical Atlases and Computational Models of the Heart. Atrial Segmentation and LV Quantification Challenges. STACOM 2018. Lecture Notes in Computer Science, vol 11395. Springer, Cham. https://doi.org/10.1007/978-3-030-12029-0_40
