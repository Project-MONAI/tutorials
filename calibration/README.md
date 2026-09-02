# Segmentation model calibration

This folder contains a self-contained tutorial for evaluating and improving the marginal class-wise calibration of
semantic segmentation models. It trains on complete 3D volumes from the Medical Segmentation Decathlon
`Task04_Hippocampus` MRI dataset. The notebook downloads the approximately 28 MB archive automatically and reuses
the directory configured by `MONAI_DATA_DIRECTORY`.

The notebook demonstrates MONAI's calibration metrics, low-level bin statistics, Ignite handler, and hard- and
soft-binned L1 Average Calibration Error losses. It uses complete-volume train/validation/test cohorts and compares
a segmentation baseline with one hard L1-ACE configuration chosen in validation-only preliminary experiments for
its calibration improvement with minimal Dice reduction. The focused comparison discusses the associated
publication's 1:1:1 objective, finite-bin estimates, and the limitations of auxiliary calibration training.

For a one-epoch CI smoke test, run:

```bash
export MONAI_DATA_DIRECTORY=/path/to/persistent/monai-data
./runner.sh -t calibration/segmentation_calibration.ipynb
```

`runner.sh` rewrites `max_epochs` and `val_interval` to one. To reproduce the saved full experiment, open the notebook
in Jupyter and run all cells without that rewrite. A CUDA GPU is strongly recommended for the two 3D training runs.

The notebook requires a MONAI build containing `HardL1ACELoss` and `SoftL1ACELoss`. Until those APIs are available in
an official MONAI package, run it in an environment with the corresponding MONAI core contribution installed editable.
