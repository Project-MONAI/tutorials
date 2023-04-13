# Example pipelines for detecting tumors on histopathology whole slide images

## Description

Here we use a classification model to classify small batches extracted from very large whole-slide histopathology images. Since the patches are very small compare to the whole image, we can then use this model for the detection of tumors in a different area of a whole-slide pathology image.

## Model Overview

The model is based on ResNet18 with the last fully connected layer replaced by a 1x1 convolution layer.

## Data

All the data used to train and validate this model is from the [Camelyon-16 Challenge](https://camelyon16.grand-challenge.org/). You can download all the images for the "CAMELYON16" data set from various sources listed [here](https://camelyon17.grand-challenge.org/Data/).

Location information for training/validation patches (the location on the whole slide image where patches are extracted) is adopted from [NCRF/coords](https://github.com/baidu-research/NCRF/tree/master/coords). The reformatted coordinations and labels in CSV format for training (`training.csv`) can be found [here](https://drive.google.com/file/d/1httIjgji6U6rMIb0P8pE0F-hXFAuvQEf/view?usp=sharing) and for validation (`validation.csv`) can be found [here](https://drive.google.com/file/d/1tJulzl9m5LUm16IeFbOCoFnaSWoB6i5L/view?usp=sharing).

This pipeline expects the training/validation data (whole slide images) reside in `cfg["data_root"]/training/images`. By default `data_root` is pointing to the code folder `./`; however, you can easily modify it to point to a different directory by passing the following argument in the runtime: `--data-root /other/data/root/dir/`.

> [`training_sub.csv`](https://drive.google.com/file/d/1rO8ZY-TrU9nrOsx-Udn1q5PmUYrLG3Mv/view?usp=sharing) and [`validation_sub.csv`](https://drive.google.com/file/d/130pqsrc2e9wiHIImL8w4fT_5NktEGel7/view?usp=sharing) is also provided to check the functionality of the pipeline using only two of the whole slide images: `tumor_001` (for training) and `tumor_101` (for validation). This dataset should not be used for the real training or any performance evaluation.

### Input and output formats

Input for the training pipeline is a JSON file (dataset.json) which includes the path to each WSI, the location and the label information for each training patch.

The output of the network is the probability of whether the input patch contains the tumor or not.

## Disclaimer

This is an example, not to be used for diagnostic purposes.
