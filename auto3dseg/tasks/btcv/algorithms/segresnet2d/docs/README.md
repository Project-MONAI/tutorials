# Description

A 2D neural network based algorithm for volumetric segmentation of 3D medical images.

# Model Overview

This model is trained using the state-of-the-art algorithm [1] of the "Brain Tumor Segmentation (BraTS) Challenge 2018".

## Training configuration

The training was performed with at least 16GB-memory GPUs.

## commands example

Execute model training:

```
python -m scripts.train run --config_file configs/algo_config.yaml
```

Execute multi-GPU model training (recommended):

```
torchrun --nnodes=1 --nproc_per_node=8 -m scripts.train run --config_file configs/algo_config.yaml
```

Execute validation:

```
python -m scripts.validate run --config_file configs/algo_config.yaml
```

Execute inference:

```
python -m scripts.infer run --config_file configs/algo_config.yaml
```

# Disclaimer

This is an example, not to be used for diagnostic purposes.

# References

[1] Myronenko, A., 2018, September. 3D MRI brain tumor segmentation using autoencoder regularization. In International MICCAI Brainlesion Workshop (pp. 311-320). Springer, Cham.
