# MetricsReloaded
These scipts show how to use the [MetricsReloaded package](https://github.com/Project-MONAI/MetricsReloaded) with MONAI to compute a range of metrics for a binary segmentation task.

## Install
Besides having installed MONAI, make sure to install the MetricsReloaded package by, e.g:
```sh
pip install git+https://github.com/Project-MONAI/MetricsReloaded@monai-support
```

## Run
First, run the training script:
```sh
python unet_training.py
```
to train a UNet on synthetic data. This script shows you how to use MetricsReloaded during validation.

Next, run the evaluation script:
```sh
python unet_evaluation.py
```
to predict on unsen cases and compute MetricsReloaded metrics from the predictions and references, which have been saved on disk. The requested metrics are printed to screen as well as saved to `results_metrics_reloaded.csv`.
