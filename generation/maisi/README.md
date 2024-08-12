# Medical AI for Synthetic Imaging (MAISI)
This example demonstrates the applications of training and validating NVIDIA MAISI, a 3D Latent Diffusion Model (LDM) capable of generating large CT images accompanied by corresponding segmentation masks. It supports variable volume size and voxel spacing and allows for the precise control of organ/tumor size.

## MAISI Model Highlight
- A Foundation Variational Auto-Encoder (VAE) model for latent feature compression that works for both CT and MRI with flexible volume size and voxel size
- A Foundation Diffusion model that can generate large CT volumes up to 512 &times; 512 &times; 768 size, with flexible volume size and voxel size
- A ControlNet to generate image/mask pairs that can improve downstream tasks, with controllable organ/tumor size

## Example Results and Evaluation

## MAISI Model Workflow
The training and inference workflows of MAISI are depicted in the figure below. It begins by training an autoencoder in pixel space to encode images into latent features. Following that, it trains a diffusion model in the latent space to denoise the noisy latent features. During inference, it first generates latent features from random noise by applying multiple denoising steps using the trained diffusion model. Finally, it decodes the denoised latent features into images using the trained autoencoder.
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

Note: MAISI depends on [xFormers](https://github.com/facebookresearch/xformers) library.
ARM64 users can build xFormers from the [source](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers) if the available wheel does not meet their requirements.

### 2. Model inference and example outputs
The information for the inference input, like body region and anatomy to generate, is stored in "./configs/config_infer.json". Please feel free to play with it.
- `"num_output_samples"`: int, the number of output image/mask pairs it will generate.
- `"spacing"`: voxel size of generated images. E.g., if set to `[1.5, 1.5, 2.0]`, it will generate images with a resolution of 1.5x1.5x2.0 mm.
- `"output_size"`: volume size of generated images. E.g., if set to `[512, 512, 256]`, it will generate images with size of 512x512x256. They need to be divisible by 16. If you have a small GPU memory size, you should adjust it to small numbers.
- `"controllable_anatomy_size"`: a list of controllable anatomy and its size scale (0--1). E.g., if set to `[["liver", 0.5],["hepatic tumor", 0.3]]`, the generated image will contain liver that have a median size, with size around 50% percentile, and hepatic tumor that is relatively small, with around 30% percentile. The output will contain paired image and segmentation mask for the controllable anatomy.
- `"body_region"`: If "controllable_anatomy_size" is not specified, "body_region" will be used to constrain the region of generated images. It needs to be chosen from "head", "chest", "thorax", "abdomen", "pelvis", "lower".
- `"anatomy_list"`: If "controllable_anatomy_size" is not specified, the output will contain paired image and segmentation mask for the anatomy in "./configs/label_dict.json".
- `"autoencoder_sliding_window_infer_size"`: in order to save GPU memory, we use sliding window inference when decoding latents to image when `"output_size"` is large. This is the patch size of the sliding window. Small value will reduce GPU memory but increase time cost. They need to be divisible by 16. 
- `"autoencoder_sliding_window_infer_overlap"`: float between 0 and 1. Large value will reduce the stitching artifacts when stitching patches during sliding window inference, but increase time cost. If you do not observe seam lines in the generated image result, you can use a smaller value to save inference time.


Please refer to [maisi_inference_tutorial.ipynb](maisi_inference_tutorial.ipynb) for the tutorial for MAISI model inference.

To run the inferenc script, please run:
```bash
python -m scripts.inference -c ./configs/config_maisi.json -i ./configs/config_infer.json -e ./configs/environment.json --random-seed 0
```

### 3. Training example
Training data preparation can be found in [./data/README.md](./data/README.md)

#### [3.1 3D Autoencoder Training](./maisi_train_vae_tutorial.ipynb)

Please refer to [maisi_train_vae_tutorial.ipynb](maisi_train_vae_tutorial.ipynb) for the tutorial for MAISI VAE model training.

#### [3.2 3D Latent Diffusion Training](./scripts/diff_model_train.py)

Please refer to [maisi_diff_unet_training_tutorial.ipynb](maisi_diff_unet_training_tutorial.ipynb) for the tutorial for MAISI diffusion model training.

#### [3.3 3D ControlNet Training](./scripts/train_controlnet.py)

We provide a [training config](./configs/config_maisi_controlnet_train.json) executing finetuning for pretrained ControlNet with a new class (i.e., Kidney Tumor).
When finetuning with other new class names, please update the `weighted_loss_label` in training config
and [label_dict.json](./configs/label_dict.json) accordingly. There are 8 dummy labels as deletable placeholders in default `label_dict.json` that can be used for finetuning. Users may apply any placeholder labels for fine-tuning purpose. If there are more than 8 new labels needed in finetuning, users can freely define numeric label indices less than 256. The current ControlNet implementation can support up to 256 labels (0~255).
Preprocessed dataset for ControlNet training and more details anout data preparation can be found in the [README](./data/README.md).

#### Training Configuration
The training was performed with the following:
- GPU: at least 60GB GPU memory for 512 &times; 512 &times; 512 volume
- Actual Model Input (the size of 3D image feature in latent space) for the latent diffusion model: 128 &times; 128 &times; 128 for 512 &times; 512 &times; 512 volume
- AMP: True

#### Execute Training:
To train with a single GPU, please run:
```bash
python -m scripts.train_controlnet -c ./configs/config_maisi.json -t ./configs/config_maisi_controlnet_train.json -e ./configs/environment_maisi_controlnet_train.json -g 1
```

The training script also enables multi-GPU training. For instance, if you are using eight GPUs, you can run the training script with the following command:
```bash
export NUM_GPUS_PER_NODE=8
torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=1 \
    --master_addr=localhost --master_port=1234 \
    -m scripts.train_controlnet -c ./configs/config_maisi.json -t ./configs/config_maisi_controlnet_train.json -e ./configs/environment_maisi_controlnet_train.json -g ${NUM_GPUS_PER_NODE}
```
Please also check [maisi_train_controlnet_tutorial.ipynb](./maisi_train_controlnet_tutorial.ipynb) for more details about data preparation and training parameters.

### 4. License

The code is released under Apache 2.0 License.

The model weight is released under [NSCLv1 License](./LICENSE.weights).

### 5. Questions and Bugs

- For questions relating to the use of MONAI, please use our [Discussions tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).
