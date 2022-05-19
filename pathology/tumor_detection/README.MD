# Example pipelines for detecting tumors on histopathology whole slide images

## Description

Here we use a classification model to classify small batches extracted from very large whole-slide histopathology images. Since the patches are very small compare to the whole image, we can then use this model for detection of tumor in different area of a whole-slide pathology image.

## Model Overview

The model is based on ResNet18 with the last fully connected layer replaced by a 1x1 convolution layer.

## Data

All the data used to train and validate this model is from [Camelyon-16 Challenge](https://camelyon16.grand-challenge.org/). You can download all the images for "CAMELYON16" data set from various sources listed [here](https://camelyon17.grand-challenge.org/Data/).

Location information for training/validation patches (the location on the whole slide image where patches are extracted) are adopted from [NCRF/coords](https://github.com/baidu-research/NCRF/tree/master/coords). The reformatted coordinations and labels are stored in a json file (`dataset_0.json`), and can be downloaded from [here](https://drive.google.com/file/d/1m2pwko6hxwsxeDWZY2oSOV-_KT97Ol0o/view?usp=sharing)

This pipeline expects the training/validation data (whole slide images) reside in `cfg["data_root"]/training/images`. By default `data_root` is pointing to `/workspace/data/medical/pathology/` You can easily modify it to point to a different directory by passing the following argument in the runtime: `--data-root /other/data/root/dir/`.

> `dataset_0_subset_0.json` is also provided [here](https://drive.google.com/file/d/1NCd0y4FR42maQpfZjzKlFSIX4oeKgysg/view?usp=sharing) to check the functionality of the pipeline using only two of the whole slide images: `tumor_001` and `tumor_101`. <br/>
> This dataset should not be used for the real training or any perfomance evaluation.

### Input and output formats

Input for the training pipeline is a json file (dataset.json) which includes path to each WSI, the location and the label information for each training patch.

Output of the network is the probability of whether the input patch contains tumor or not.

## Disclaimer

This is an example, not to be used for diagnostic purposes.
