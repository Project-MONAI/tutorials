# Accelerated MRI reconstruction with the end-to-end variational network (e2e-VarNet)

<p align="center"><img src="./figures/workflow.PNG" width="800" height="400"></p>


This folder contains code to train and validate an e2e-VarNet ([https://arxiv.org/pdf/2004.06688.pdf](https://arxiv.org/pdf/2004.06688.pdf)) for accelerated MRI reconstruction. Accelerated MRI reconstruction is a compressed sensing task where the goal is to recover a ground-truth image from an under-sampled measurement. The under-sampled measurement is based on the frequency domain and is often called the $k$-space.

***

### List of contents

* [Questions and bugs](#Questions-and-bugs)

* [Dataset](#Dataset)

* [Model checkpoint](#Model-checkpoint)

* [Training](#Training)

* [Inference](#Inference)

* [Acknowledgment](#Acknowledgement)

***

# Questions and bugs

- For questions relating to the use of MONAI, please use our [Discussions](https://github.com/Project-MONAI/MONAI/discussions) tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).

# Dataset

Please see the [dataset description](../unet_demo/README.md#dataset) for our dataset preparation.


# Model checkpoint

We have already provided a model checkpoint [varnet_mri_reconstruction.pt](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/varnet_mri_reconstruction.pt) for a VarNet with `30,069,558` parameters. To obtain this checkpoint, we trained
a VarNet with the default hyper-parameters in `train.py` on our T2 subset of the brain dataset. The user can train their model on an arbitrary portion of the dataset.

The training dynamics for our checkpoint are depicted in the figure below.

<p align="center"><img src="./figures/dynamics.PNG" width="800" height="250"></p>

# Training

Running `train.py` trains a VarNet. The default setup automatically detects a GPU for training; if not available, the CPU will be used.

    # Run this to get a full list of training arguments
    python ./train.py -h

    # This is an example of calling train.py
    python ./train.py
        --data_path_train train_dir \
        --data_path_val val_dir \
        --exp varnet_mri_recon \
        --exp_dir ./ \
        --mask_type equispaced \
        --num_epochs 50 \
        --num_workers 0 \
        --lr 0.00001

# Inference

The notebook `inference.ipynb` contains an example to perform inference. The average SSIM score over the test subset is computed and then
one sample is picked for visualization.

Our checkpoint achieves `0.9650` SSIM on our test subset which is comparable to the original result reported on the
[fastMRI public leaderboard](https://fastmri.org/leaderboards/) (which is `0.9606` SSIM). Note that the results reported
on the leaderboard are for the unreleased test set. Moreover, the leaderboard model is trained on the validation set.

# Acknowledgment
Data used in the preparation of this tutorial were obtained from the NYU fastMRI Initiative database (fastmri.med.nyu.edu).[citation of Knoll et al Radiol Artif Intell. 2020 Jan 29;2(1):e190007.
doi: 10.1148/ryai.2020190007. (https://pubs.rsna.org/doi/10.1148/ryai.2020190007), and the arXiv paper: https://arxiv.org/abs/1811.08839] As such, NYU fastMRI investigators provided data but did not participate in analysis or writing of this tutorial. A listing of NYU fastMRI investigators, subject to updates, can be found at:fastmri.med.nyu.edu. The primary goal of fastMRI is to test whether machine learning can aid in the reconstruction of medical images.
