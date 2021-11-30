#!/bin/bash
clear

pip install nibabel
pip install pandas

# Update pip
python -m pip install -U pip
# Install scikit-image
python -m pip install -U scikit-image

pip install git+https://github.com/Project-MONAI/MONAI#egg=monai
