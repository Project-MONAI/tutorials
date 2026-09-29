# MRI Reconstruction Tutorials

This folder contains educational Jupyter notebooks that introduce the fundamentals of MRI reconstruction.

## Tutorials

### 01 - K-Space Basics with fastMRI Knee Data
**Notebook:** [01_kspace_basics_fastmri_knee.ipynb](./01_kspace_basics_fastmri_knee.ipynb)

An introductory tutorial covering:
- What k-space is and its relationship to MRI images via the Fourier transform
- How low and high spatial frequencies contribute to image content
- Why undersampling k-space causes aliasing artifacts
- How MONAI's reconstruction transforms (`RandomKspaceMaskd`, `EquispacedKspaceMaskd`, etc.) process k-space data
- The zero-filled reconstruction problem that deep learning methods aim to solve

**Dataset:** [fastMRI](https://fastmri.org/dataset) knee single-coil validation set (requires registration, non-commercial license). Only one `.h5` file is needed.

**Prerequisites:** Basic Python and NumPy. No MRI experience required.

## Related Production Tutorials

For training-focused tutorials using the brain multi-coil dataset, see:
- [U-Net Demo](../unet_demo/) - BasicUNet for MRI reconstruction
- [VarNet Demo](../varnet_demo/) - End-to-end Variational Network for MRI reconstruction
