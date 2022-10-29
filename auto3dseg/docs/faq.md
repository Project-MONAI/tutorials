# Frequently Asked Questions (FAQ)

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue in [MONAI GitHub repository](https://github.com/Project-MONAI).

## Installation

**Auto3DSeg** is a GPU based application. The minimum requirement is a single GPU with more than 16GB RAM, which supports PyTorch version 1.7+.

**Auto3DSeg** is a direct application of the MONAI library. To use **Auto3DSeg**, MONAI needs to be installed following the [official instructions](https://docs.monai.io/en/stable/installation.html). Alternatively, user can run scripts inside the [MONAI docker container](https://hub.docker.com/r/projectmonai/monai). Meanwhile, additional Python libraries need to be installed as well with the following command.

```bash
pip install nibabel==4.0.2 fire==0.4.0 scikit-image==0.19.3
```
