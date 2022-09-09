# Description

A neural architecture search algorithm for volumetric segmentation of 3D medical images.

# Model Overview

This model is trained using the state-of-the-art algorithm [1] of the "Medical Segmentation Decathlon Challenge 2018".

## Training configuration

The training was performed with at least 16GB-memory GPUs.

## commands example

Execute model searching:

```
python -m scripts.search run --config_file configs/algo_config.yaml
```

Execute multi-GPU model searching (recommended):

```
torchrun --nnodes=1 --nproc_per_node=8 -m scripts.search run --config_file configs/algo_config.yaml
```

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

[1] He, Y., Yang, D., Roth, H., Zhao, C. and Xu, D., 2021. Dints: Differentiable neural network topology search for 3d medical image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5841-5850).
