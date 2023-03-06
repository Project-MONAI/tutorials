# Installation

Users need to install both MONAI and nnU-Net before running any experiments.

## MONAI

Users can follow the [link](https://https://docs.monai.io/en/stable/installation.html#option-1-as-a-part-of-your-system-wide-module) to install dev branch of MONAI.
The following command shows the example to install MONAI and Necessary dependencies.

```bash
# install monai (pip install monai)
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
python setup.py develop

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
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
git checkout nnunet_remake
git pull # just for good measure
pip install -e .

# install hiddenlayer (optional)
pip install --upgrade git+https://github.com/julien-blanchon/hiddenlayer.git
```

The official instruction can be found [here](https://github.com/MIC-DKFZ/nnUNet/blob/nnunet_remake/documentation/installation_instructions.md).
