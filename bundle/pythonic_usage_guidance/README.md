# Pythonic Bundle Access Tutorial

A MONAI bundle contains the stored weights of a model, training, inference, post-processing transform sequences and other information. This tutorial aims to explore how to access a bundle in Python and use it in your own application. We'll cover the following topics:
1. Downloading the Bundle.
2. Creating a `BundleWorkflow`.
3. Getting Properties from the Bundle.
4. Updating Properties.
5. Using Components in Your Own Pipeline.
6. Utilizing Pretrained Weights from the Bundle.
7. A Simple Comparison of the Usage between `ConfigParser` and `BundleWorkflow`.

The example training dataset is Task09_Spleen.tar from http://medicaldecathlon.com/.

## Requirements

The script is tested with:

- `Ubuntu 20.04` | `Python 3.8.10` | `CUDA 12.2` | `Pytorch 1.13.1`

- it is tested on 24gb single-gpu machine

## Dependencies and installation

### MONAI

You can conda environments to install the dependencies.

or you can just use MONAI docker.
```bash
docker pull projectmonai/monai:latest
```

For more information please check out [the installation guide](https://docs.monai.io/en/latest/installation.html).

## Questions and bugs

- For questions relating to the use of MONAI, please use our [Discussions tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).
