# Installation

Users need to install both MONAI and nnU-Net to utilize the nnunet runner.

## MONAI

Users can follow the [link](https://docs.monai.io/en/stable/installation.html#option-1-as-a-part-of-your-system-wide-module) to install dev branch of MONAI.
The following command shows the example to install MONAI and Necessary dependencies.

```bash
# install latest monai (pip install monai)
pip install git+https://github.com/Project-MONAI/MONAI#egg=monai

# install dependencies
pip install fire nibabel
pip install "scikit-image>=0.19.0"
```

## nnU-Net (V2)

To run components of nnU-Net V2, users need to properly install PyTorch on their own or adopt [Pytorch docker containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) maintained by NVIDIA.
And other dependent libraries can be installed by running basic commands.

```bash
# install dependencies
pip install --upgrade git+https://github.com/MIC-DKFZ/acvl_utils.git
pip install --upgrade git+https://github.com/MIC-DKFZ/dynamic-network-architectures.git

# install nnunet
pip install nnunetv2

# install hiddenlayer (optional)
pip install --upgrade git+https://github.com/julien-blanchon/hiddenlayer.git
```

The official instruction can be found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).
