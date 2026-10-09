# Multimodal Early-Fusion Network: Radiographs + Clinical Tabular Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/multimodal/nakaseke_multimodal_early_fusion/multimodal_early_fusion_tutorial.ipynb)

This tutorial demonstrates an **early-fusion** architecture that combines a 2D medical image stream
with a low-dimensional clinical tabular stream in a single MONAI dictionary-based pipeline, using
[`multimodal_early_fusion_tutorial.ipynb`](./multimodal_early_fusion_tutorial.ipynb).

## Motivation

Most MONAI tutorials focus on a single imaging modality. In real deployments, especially in
resource-constrained clinics, an image is rarely read in isolation -- a clinician also has vitals,
labs, or a short structured history on hand, and a model that ignores that context is throwing away
signal it doesn't have to. This tutorial shows the minimum end-to-end MONAI/PyTorch pattern for
combining the two without hand-rolling a custom `Dataset`.

The clinical feature schema (age, BMI, salivary pH, systolic blood pressure) and the binary screening
task are modeled after a hypertension-screening workflow explored at Nakaseke Hospital, Uganda. The
underlying patient data is confidential and is **not** included in or downloaded by this tutorial.

## Dataset

**Fully synthetic, generated locally, no download required.** The notebook's
`simulate_nakaseke_multimodal_dataset()` function creates, on the fly:

- one synthetic 2D radiograph per patient, saved as a NIfTI (`.nii.gz`) file with `nibabel`
- a 4-dimensional tabular vector `[age, bmi, salivary_ph, systolic_bp]`
- a binary label

Both modalities are generated from a shared hidden "risk factor" per patient, so neither the image
nor the tabular vector is fully predictive of the label on its own -- this is what motivates fusing
them. There is no claim that the notebook's results reflect real-world diagnostic performance; the
synthetic cohort exists only to exercise the pipeline end-to-end without any real or downloadable
data.

## What the notebook covers

1. Reproducible setup with `monai.utils.set_determinism`.
2. Synthetic multimodal data generation and a MONAI-style data manifest
   (`{"image": ..., "nakaseke_tabular": ..., "label": ...}`).
3. A dictionary-based `Compose` pipeline where image-only transforms (`LoadImaged`,
   `EnsureChannelFirstd`, `ScaleIntensityRanged`) touch only the `"image"` key, while the tabular and
   label keys are cast to tensors with `EnsureTyped` and otherwise left untouched.
4. `ResilientMultimodalClassifier`: a MONAI `DenseNet121` visual stream (512-d embedding) fused via
   `torch.cat` with a small tabular projection stream (16-d embedding) into a 528-d representation,
   followed by a dropout-regularized classification head.
5. A short training loop (`max_epochs`, `val_interval`) showing validation accuracy improve as the
   model learns to use both streams.

## Requirements

Everything needed is installed by the notebook's own `Setup environment` cell
(`monai-weekly[nibabel, tqdm]`, `matplotlib`). No GPU is required -- the default image size (64x64)
and cohort size (200 patients) are chosen to train in well under a minute on CPU.
