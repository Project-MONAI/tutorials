# Medical AI for Synthetic Imaging (MAISI)
This example shows the use cases of training and validating Nvidia MAISI (Medical AI for Synthetic Imaging), a 3D Latent Diffusion Model that can generate large CT images with paired segmentation masks, variable volume size and voxel size, as well as controllable organ/tumor size.

## MAISI Model Highlight
- A Foundation VAE model for latent feature compression that works for both CT and MRI with flexible volume size and voxel size
- A Foundation Diffusion model that can generate large CT volumes up to 512x512x768 size, with flexible volume size and voxel size
- A ControlNet to generate image/mask pairs that can improve downstream tasks, with controllable organ/tumor size

## Example Results and Evaluation

## MAISI Model Workflow
The training and inference workflow of MAISI is depicted in the figure below. It begins by training an autoencoder in pixel space to encode images into latent features. Following that, it trains a diffusion model in the latent space to denoise the noisy latent features. During inference, it first generates latent features from random noise by applying multiple denoising steps using the trained diffusion model. Finally, it decodes the denoised latent features into images using the trained autoencoder.
<p align="center">
  <img src="./figures/maisi_train.jpg" alt="MAISI training scheme">
  <br>
  <em>Figure 1: MAISI training scheme</em>
</p>

<p align="center">
  <img src="./figures/maisi_infer.jpg" alt="MAISI inference scheme")
  <br>
  <em>Figure 2: MAISI inference scheme</em>
</p>
MAISI is based on the following papers:

[**Latent Diffusion:** Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." CVPR 2022.](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)

[**ControlNet:**  Lvmin Zhang, Anyi Rao, Maneesh Agrawala; “Adding Conditional Control to Text-to-Image Diffusion Models.” ICCV 2023.](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf)

### 1. Installation
Please refer to the [Installation of MONAI Generative Model](../README.md).

Note: MAISI depends on [xFormers](https://github.com/facebookresearch/xformers) library, which unfortunately does not yet support ARM64.
We will update after xFormers supports ARM64.

### 2. Model inference and example outputs
Please refer to [maisi_inference_tutorial.ipynb](maisi_inference_tutorial.ipynb) for the tutorial for MAISI model inference.

### 3. Training example
Training data preparation can be found in [./data/README.md](./data/README.md)

#### [3.1 3D Autoencoder Training](./train_autoencoder.py)
Please refer to [maisi_train_vae_tutorial.ipynb](maisi_train_vae_tutorial.ipynb) for the tutorial for MAISI VAE model training.

#### [3.2 3D Latent Diffusion Training](./train_diffusion.py)
The training script uses the batch size and patch size defined in the configuration files. If you have a different GPU memory size, you should adjust the `"batch_size"` and `"patch_size"` parameters in the `"diffusion_train"` to match your GPU. Note that the `"patch_size"` needs to be divisible by 16.

To train with single 32G GPU, please run:
```bash
python train_diffusion.py -c ./config/config_maisi.json -e ./config/environment.json -g 1
```

The training script also enables multi-GPU training. For instance, if you are using eight 32G GPUs, you can run the training script with the following command:
```bash
export NUM_GPUS_PER_NODE=8
torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=1 \
    --master_addr=localhost --master_port=1234 \
    train_diffusion.py -c ./config/config_maisi.json -e ./config/environment.json -g ${NUM_GPUS_PER_NODE}
```
<p align="center">
  <img src="./figs/train_diffusion.png" alt="latent diffusion train curve" width="45%" >
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="./figs/val_diffusion.png" alt="latent diffusion validation curve" width="45%" >
</p>

#### [3.3 3D ControNet Training](./scripts/train_controlnet.py)

We provide a [training config](./configs/config_maisi_controlnet_train.json) executing finetuning for pretrained ControlNet with with a new class (i.e., Kidney Tumor). When finetuning with other new class names, please update the `weighted_loss_label` in training config  and [label_dict.json](./configs/label_dict.json) accordingly. There are 8 dummy labels as placeholders in default `label_dict.json` that can be used for finetuning. Preprocessed dataset for controNet training and more deatils anout data preparation can be found in the [README](./data/README.md).

#### Training configuration
The training was performed with the following:
- GPU: at least 60GB GPU memory for 512 x 512 x 512 volume
- Actual Model Input (the size of image embedding in latent space): 128 x 128 x 128
- AMP: True

#### Execute training:
To train with a single GPU, please run:
```bash
python ./scripts/train_controlnet.py -c ./config/config_maisi.json -t ./config/config_maisi_controlnet_train.json -e ./config/environment_maisi_controlnet_train.json -g 1
```

The training script also enables multi-GPU training. For instance, if you are using eight GPUs, you can run the training script with the following command:
```bash
export NUM_GPUS_PER_NODE=8
torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=1 \
    --master_addr=localhost --master_port=1234 \
    ./scripts/train_controlnet.py -c ./config/config_maisi.json -t ./config/config_maisi_controlnet_train.json -e ./config/environment_maisi_controlnet_train.json -g ${NUM_GPUS_PER_NODE}
```
### 4. Questions and bugs

- For questions relating to the use of MONAI, please use our [Discussions tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).
