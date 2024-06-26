# MAISI Example
This example shows the use cases of training and validating Nvidia MAISI (Medical AI for Synthetic Imaging), a 3D Latent Diffusion Model that can generate large CT images with paired segmentation masks, variable volume size and voxel size, as well as controllable organ/tumor size.

The training and inference workflow of MAISI is depicted in the figure below. It begins by training an autoencoder in pixel space to encode images into latent features. Following that, it trains a diffusion model in the latent space to denoise the noisy latent features. During inference, it first generates latent features from random noise by applying multiple denoising steps using the trained diffusion model. Finally, it decodes the denoised latent features into images using the trained autoencoder.
<p align="center">
  <img src="./figs/maisi_train.png" alt="MAISI training scheme")
</p>

<p align="center">
  <img src="./figs/maisi_infer.png" alt="MAISI inference scheme")
</p>

MAISI is based on the following papers:

[**Latent Diffusion:** Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." CVPR 2022.](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)

### 1. Data
Disclaimer: We are not the host of the data. Please make sure to read the requirements and usage policies of the data and give credit to the authors of the dataset!

### 1.1 VAE training Data

The VAE training dataset used in MAISI contains.

| Dataset                                  | Num of training data | Num of validation data |
|------------------------------------------|----------------------|------------------------|
| Covid 19 Chest CT                        | 722                  | 49                     |
| TCIA Colon Abdomen CT                    | 1522                 | 77                     |
| MSD03 Liver Abdomen CT                   | 104                  | 0                      |
| LIDC chest CT                            | 450                  | 24                     |
| TCIA Stony Brook Covid Chest CT          | 2644                 | 139                    |
| NLST Chest CT                            | 31801                | 1647                   |
| TCIA Upenn GBM Brain MR, skull-stripped  | 2550                 | 134                    |
| Aomic Brain MR                           | 2630                 | 138                    |
| Aomic Brain MR, skull-stripped           | 2630                 | 138                    |
| QTIM Brain MR                            | 1275                 | 67                     |
| QTIM Brain MR, skull-stripped            | 1275                 | 67                     |
| Acrin Chest MR                           | 6599                 | 347                    |
| TCIA Prostate MR Below-Abdomen MR        | 928                  | 49                     |
| Total CT                                 | 37243                | 17887                  |
| Total MR                                 | 1963                 | 940                    |

### 1.2 Unconditional Diffusion model training Data
### 1.3 ControNet model training Data

### 2. Installation
Please refer to the [Installation of MONAI Generative Model](../README.md)

### 3. Run the example

#### [3.1 3D Autoencoder Training](./train_autoencoder.py)

The network configuration files are located in [./config/config_train_32g.json](./config/config_train_32g.json) for 32G GPU
and [./config/config_train_16g.json](./config/config_train_16g.json) for 16G GPU.
You can modify the hyperparameters in these files to suit your requirements.

The training script resamples the brain images based on the voxel spacing specified in the `"spacing"` field of the configuration files. For instance, `"spacing": [1.1, 1.1, 1.1]` resamples the images to a resolution of 1.1x1.1x1.1 mm. If you have a GPU with larger memory, you may consider changing the `"spacing"` field to `"spacing": [1.0, 1.0, 1.0]`.

The training script uses the batch size and patch size defined in the configuration files. If you have a different GPU memory size, you should adjust the `"batch_size"` and `"patch_size"` parameters in the `"autoencoder_train"` to match your GPU. Note that the `"patch_size"` needs to be divisible by 4.

Before you start training, please set the path in [./config/environment.json](./config/environment.json).

- `"model_dir"`: where it saves the trained models
- `"tfevent_path"`: where it saves the tensorboard events
- `"output_dir"`: where you store the generated images during inference.
- `"resume_ckpt"`: whether to resume training from existing checkpoints.
- `"data_base_dir"`: where you store the Brats dataset.

Below is the the training command for single GPU.

```bash
python train_autoencoder.py -c ./config/config_train_32g.json -e ./config/environment.json -g 1
```

The training script also enables multi-GPU training. For instance, if you are using eight 32G GPUs, you can run the training script with the following command:
```bash
export NUM_GPUS_PER_NODE=8
torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=1 \
    --master_addr=localhost --master_port=1234 \
    train_autoencoder.py -c ./config/config_train_32g.json -e ./config/environment.json -g ${NUM_GPUS_PER_NODE}
```

<p align="center">
  <img src="./figs/train_recon.png" alt="autoencoder train curve" width="45%" >
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="./figs/val_recon.png" alt="autoencoder validation curve" width="45%" >
</p>

With eight DGX1V 32G GPUs, it took around 55 hours to train 1000 epochs.

#### [3.2 3D Latent Diffusion Training](./train_diffusion.py)
The training script uses the batch size and patch size defined in the configuration files. If you have a different GPU memory size, you should adjust the `"batch_size"` and `"patch_size"` parameters in the `"diffusion_train"` to match your GPU. Note that the `"patch_size"` needs to be divisible by 16.

To train with single 32G GPU, please run:
```bash
python train_diffusion.py -c ./config/config_train_32g.json -e ./config/environment.json -g 1
```

The training script also enables multi-GPU training. For instance, if you are using eight 32G GPUs, you can run the training script with the following command:
```bash
export NUM_GPUS_PER_NODE=8
torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=1 \
    --master_addr=localhost --master_port=1234 \
    train_diffusion.py -c ./config/config_train_32g.json -e ./config/environment.json -g ${NUM_GPUS_PER_NODE}
```
<p align="center">
  <img src="./figs/train_diffusion.png" alt="latent diffusion train curve" width="45%" >
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="./figs/val_diffusion.png" alt="latent diffusion validation curve" width="45%" >
</p>

#### [3.3 Inference](./inference.py)
To generate one image during inference, please run the following command:
```bash
python inference.py -c ./config/config_train_32g.json -e ./config/environment.json --num 1
```
`--num` defines how many images it would generate.

An example output is shown below.
<p align="center">
  <img src="./figs/syn_axial.png" width="30%" >
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="./figs/syn_sag.png" width="30%" >
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="./figs/syn_cor.png" width="30%" >
</p>

### 4. Questions and bugs

- For questions relating to the use of MONAI, please use our [Discussions tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).

### Reference
[1] [Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." CVPR 2022.](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)

[2] [Menze, Bjoern H., et al. "The multimodal brain tumor image segmentation benchmark (BRATS)." IEEE transactions on medical imaging 34.10 (2014): 1993-2024.](https://ieeexplore.ieee.org/document/6975210)

[3] [Pinaya et al. "Brain imaging generation with latent diffusion models"](https://arxiv.org/abs/2209.07162)
