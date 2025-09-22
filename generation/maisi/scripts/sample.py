# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import random
import time
from datetime import datetime
import warnings
import gc

import monai
import torch
from monai.data import MetaTensor
from monai.inferers.inferer import DiffusionInferer
from monai.transforms import Compose, SaveImage
from monai.utils import set_determinism
from tqdm import tqdm
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler, DDPMScheduler

from .augmentation import augmentation
from .find_masks import find_masks
from .quality_check import is_outlier
from .utils import (
    binarize_labels,
    general_mask_generation_post_process,
    get_body_region_index_from_mask,
    remap_labels,
    dynamic_infer,
)


class ReconModel(torch.nn.Module):
    """
    A PyTorch module for reconstructing images from latent representations.

    Attributes:
        autoencoder: The autoencoder model used for decoding.
        scale_factor: Scaling factor applied to the input before decoding.
    """

    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        """
        Decode the input latent representation to an image.

        Args:
            z (torch.Tensor): The input latent representation.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        recon_pt_nda = self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)
        return recon_pt_nda


def initialize_noise_latents(latent_shape, device):
    """
    Initialize random noise latents for image generation with float16.

    Args:
        latent_shape (tuple): The shape of the latent space.
        device (torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: Initialized noise latents.
    """
    return (
        torch.randn(
            [
                1,
            ]
            + list(latent_shape)
        )
        .half()
        .to(device)
    )


def ldm_conditional_sample_one_mask(
    autoencoder,
    diffusion_unet,
    noise_scheduler,
    scale_factor,
    anatomy_size,
    device,
    latent_shape,
    label_dict_remap_json,
    num_inference_steps=1000,
    autoencoder_sliding_window_infer_size=[96, 96, 96],
    autoencoder_sliding_window_infer_overlap=0.6667,
):
    """
    Generate a single synthetic mask using a latent diffusion model.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        diffusion_unet (nn.Module): The diffusion U-Net model.
        noise_scheduler: The noise scheduler for the diffusion process.
        scale_factor (float): Scaling factor for the latent space.
        anatomy_size (torch.Tensor): Tensor specifying the desired anatomy sizes.
        device (torch.device): The device to run the computation on.
        latent_shape (tuple): The shape of the latent space.
        label_dict_remap_json (str): Path to the JSON file for label remapping.
        num_inference_steps (int): Number of inference steps for the diffusion process.
        autoencoder_sliding_window_infer_size (list, optional): Size of the sliding window for inference. Defaults to [96, 96, 96].
        autoencoder_sliding_window_infer_overlap (float, optional): Overlap ratio for sliding window inference. Defaults to 0.6667.

    Returns:
        torch.Tensor: The generated synthetic mask.
    """
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        # Generate random noise
        latents = initialize_noise_latents(latent_shape, device)
        anatomy_size = torch.FloatTensor(anatomy_size).unsqueeze(0).unsqueeze(0).half().to(device)
        # synthesize latents
        if isinstance(noise_scheduler, DDPMScheduler) and num_inference_steps < noise_scheduler.num_train_timesteps:
            warnings.warn(
                "**************************************************************\n"
                "* WARNING: Mask noise_scheduler is a DDPMScheduler.\n"
                "* We expect num_inference_steps = noise_scheduler.num_train_timesteps"
                f" = {noise_scheduler.num_train_timesteps}.\n"
                f"* Yet got num_inference_steps = {num_inference_steps}.\n"
                "* The generated image quality is not guaranteed.\n"
                "**************************************************************"
            )

        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        # mask generator is DDPM
        inferer_ddpm = DiffusionInferer(noise_scheduler)
        latents = inferer_ddpm.sample(
            input_noise=latents,
            diffusion_model=diffusion_unet,
            scheduler=noise_scheduler,
            verbose=True,
            conditioning=anatomy_size.to(device),
        )

        inferer = SlidingWindowInferer(
            roi_size=autoencoder_sliding_window_infer_size,
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=autoencoder_sliding_window_infer_overlap,
            sw_device=device,
            device=torch.device("cpu"),
        )
        synthetic_mask = dynamic_infer(inferer, recon_model, latents)
        synthetic_mask = torch.softmax(synthetic_mask, dim=1)
        synthetic_mask = torch.argmax(synthetic_mask, dim=1, keepdim=True)
        # mapping raw index to 132 labels
        synthetic_mask = remap_labels(synthetic_mask, label_dict_remap_json)

        ###### post process #####
        data = synthetic_mask.squeeze().cpu().detach().numpy()

        labels = [23, 24, 26, 27, 128]
        target_tumor_label = None
        for index, size in enumerate(anatomy_size[0, 0, 5:10]):
            if size.item() != -1.0:
                target_tumor_label = labels[index]

        logging.info(f"target_tumor_label for postprocess:{target_tumor_label}")
        data = general_mask_generation_post_process(data, target_tumor_label=target_tumor_label, device=device)
        synthetic_mask = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)

    return synthetic_mask


def ldm_conditional_sample_one_image(
    autoencoder,
    diffusion_unet,
    controlnet,
    noise_scheduler,
    scale_factor,
    device,
    combine_label_or,
    spacing_tensor,
    latent_shape,
    output_size,
    noise_factor,
    top_region_index_tensor=None,
    bottom_region_index_tensor=None,
    modality_tensor=None,
    num_inference_steps=1000,
    autoencoder_sliding_window_infer_size=[96, 96, 96],
    autoencoder_sliding_window_infer_overlap=0.6667,
):
    """
    Generate a single synthetic image using a latent diffusion model with controlnet.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        diffusion_unet (nn.Module): The diffusion U-Net model.
        controlnet (nn.Module): The controlnet model.
        noise_scheduler: The noise scheduler for the diffusion process.
        scale_factor (float): Scaling factor for the latent space.
        device (torch.device): The device to run the computation on.
        combine_label_or (torch.Tensor): The combined label tensor.
        spacing_tensor (torch.Tensor): Tensor specifying the spacing.
        latent_shape (tuple): The shape of the latent space.
        output_size (tuple): The desired output size of the image.
        noise_factor (float): Factor to scale the initial noise.
        top_region_index_tensor (torch.Tensor): Tensor specifying the top region index. Defaults to None.
        bottom_region_index_tensor (torch.Tensor): Tensor specifying the bottom region index. Defaults to None.
        modality_tensor (torch.Tensor): Int Tensor specifying the modality.
        num_inference_steps (int): Number of inference steps for the diffusion process.
        autoencoder_sliding_window_infer_size (list, optional): Size of the sliding window for inference. Defaults to [96, 96, 96].
        autoencoder_sliding_window_infer_overlap (float, optional): Overlap ratio for sliding window inference. Defaults to 0.6667.

    Returns:
        tuple: A tuple containing the synthetic image and its corresponding label.
    """
    # CT image intensity range
    a_min = -1000
    a_max = 1000
    # autoencoder output intensity range
    b_min = 0.0
    b_max = 1

    include_body_region = diffusion_unet.include_top_region_index_input
    include_modality = diffusion_unet.num_class_embeds is not None

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        logging.info("---- Start generating latent features... ----")
        start_time = time.time()
        # generate segmentation mask
        combine_label = combine_label_or.to(device)
        if (
            output_size[0] != combine_label.shape[2]
            or output_size[1] != combine_label.shape[3]
            or output_size[2] != combine_label.shape[4]
        ):
            logging.info(
                "output_size is not a desired value. Need to interpolate the mask to match with output_size. The result image will be very low quality."
            )
            combine_label = torch.nn.functional.interpolate(combine_label, size=output_size, mode="nearest")

        controlnet_cond_vis = binarize_labels(combine_label.as_tensor().long()).half()

        # Generate random noise
        latents = initialize_noise_latents(latent_shape, device) * noise_factor

        # synthesize latents
        if isinstance(noise_scheduler, RFlowScheduler):
            noise_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                input_img_size_numel=torch.prod(torch.tensor(latents.shape[2:])),
            )
        else:
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        if isinstance(noise_scheduler, DDPMScheduler) and num_inference_steps < noise_scheduler.num_train_timesteps:
            warnings.warn(
                "**************************************************************\n"
                "* WARNING: Image noise_scheduler is a DDPMScheduler.\n"
                "* We expect num_inference_steps = noise_scheduler.num_train_timesteps"
                f" = {noise_scheduler.num_train_timesteps}.\n"
                f"* Yet got num_inference_steps = {num_inference_steps}.\n"
                "* The generated image quality is not guaranteed.\n"
                "**************************************************************"
            )

        all_timesteps = noise_scheduler.timesteps
        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
        progress_bar = tqdm(
            zip(all_timesteps, all_next_timesteps),
            total=min(len(all_timesteps), len(all_next_timesteps)),
        )
        for t, next_t in progress_bar:
            # get controlnet output
            # Create a dictionary to store the inputs
            controlnet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
                "controlnet_cond": controlnet_cond_vis,
            }
            if include_modality:
                controlnet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

            # get diffusion network output
            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual": mid_block_res_sample,
            }
            # Add extra arguments if include_body_region is True
            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )
            if include_modality:
                unet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            model_output = diffusion_unet(**unet_inputs)

            if not isinstance(noise_scheduler, RFlowScheduler):
                latents, _ = noise_scheduler.step(model_output, t, latents)  # type: ignore
            else:
                latents, _ = noise_scheduler.step(model_output, t, latents, next_t)  # type: ignore
        end_time = time.time()
        logging.info(f"---- DM/ControlNet Latent features generation time: {end_time - start_time} seconds ----")
        del (
            unet_inputs,
            controlnet_inputs,
            model_output,
            controlnet_cond_vis,
            down_block_res_samples,
            mid_block_res_sample,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # decode latents to synthesized images
        logging.info("---- Start decoding latent features into images... ----")
        start_time = time.time()

        inferer = SlidingWindowInferer(
            roi_size=autoencoder_sliding_window_infer_size,
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=autoencoder_sliding_window_infer_overlap,
            sw_device=device,
            device=torch.device("cpu"),
        )
        synthetic_images = dynamic_infer(inferer, recon_model, latents)
        synthetic_images = torch.clip(synthetic_images, b_min, b_max).cpu()
        end_time = time.time()
        logging.info(f"---- Image VAE decoding time: {end_time - start_time} seconds ----")

        ## post processing:
        # project output to [0, 1]
        synthetic_images = (synthetic_images - b_min) / (b_max - b_min)
        # project output to [-1000, 1000]
        synthetic_images = synthetic_images * (a_max - a_min) + a_min
        # regularize background intensities
        synthetic_images = crop_img_body_mask(synthetic_images, combine_label)
        torch.cuda.empty_cache()

    return synthetic_images, combine_label


def filter_mask_with_organs(combine_label, anatomy_list):
    """
    Filter a mask to only include specified organs.

    Args:
        combine_label (torch.Tensor): The input mask.
        anatomy_list (list): List of organ labels to keep.

    Returns:
        torch.Tensor: The filtered mask.
    """
    # final output mask file has shape of output_size, contains labels in anatomy_list
    # it is already interpolated to target size
    combine_label = combine_label.long()
    # filter out the organs that are not in anatomy_list
    for i in range(len(anatomy_list)):
        organ = anatomy_list[i]
        # replace it with a negative value so it will get mixed
        combine_label[combine_label == organ] = -(i + 1)
    # zero-out voxels with value not in anatomy_list
    combine_label[combine_label > 0] = 0
    # output positive values
    combine_label = -combine_label
    return combine_label


def crop_img_body_mask(synthetic_images, combine_label):
    """
    Crop the synthetic image using a body mask.

    Args:
        synthetic_images (torch.Tensor): The synthetic images.
        combine_label (torch.Tensor): The body mask.

    Returns:
        torch.Tensor: The cropped synthetic images.
    """
    synthetic_images[combine_label == 0] = -1000
    return synthetic_images


def check_input(
    body_region,
    anatomy_list,
    label_dict_json,
    output_size,
    spacing,
    controllable_anatomy_size=[("pancreas", 0.5)],
):
    """
    Validate input parameters for image generation.

    Args:
        body_region (list): List of body regions.
        anatomy_list (list): List of anatomical structures.
        label_dict_json (str): Path to the label dictionary JSON file.
        output_size (tuple): Desired output size of the image.
        spacing (tuple): Desired voxel spacing.
        controllable_anatomy_size (list): List of tuples specifying controllable anatomy sizes.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # check output_size and spacing format
    if output_size[0] != output_size[1]:
        raise ValueError(f"The first two components of output_size need to be equal, yet got {output_size}.")
    if (output_size[0] not in [256, 384, 512]) or (output_size[2] not in [128, 256, 384, 512, 640, 768]):
        raise ValueError(
            f"The output_size[0] have to be chosen from [256, 384, 512], and output_size[2] have to be chosen from [128, 256, 384, 512, 640, 768], yet got {output_size}."
        )

    if spacing[0] != spacing[1]:
        raise ValueError(f"The first two components of spacing need to be equal, yet got {spacing}.")
    if spacing[0] < 0.5 or spacing[0] > 3.0 or spacing[2] < 0.5 or spacing[2] > 5.0:
        raise ValueError(
            f"spacing[0] have to be between 0.5 and 3.0 mm, spacing[2] have to be between 0.5 and 5.0 mm, yet got {spacing}."
        )

    if output_size[0] * spacing[0] < 256:
        FOV = [output_size[axis] * spacing[axis] for axis in range(3)]
        raise ValueError(
            f"`'spacing'({spacing}mm) and 'output_size'({output_size}) together decide the output field of view (FOV). The FOV will be {FOV}mm. We recommend the FOV in x and y axis to be at least 256mm for head, and at least 384mm for other body regions like abdomen. There is no such restriction for z-axis."
        )

    if controllable_anatomy_size == None:
        logging.info(f"`controllable_anatomy_size` is not provided.")
        return

    # check controllable_anatomy_size format
    if len(controllable_anatomy_size) > 10:
        raise ValueError(
            f"The length of list controllable_anatomy_size has to be less than 10. Yet got length equal to {len(controllable_anatomy_size)}."
        )
    available_controllable_organ = [
        "liver",
        "gallbladder",
        "stomach",
        "pancreas",
        "colon",
    ]
    available_controllable_tumor = [
        "hepatic tumor",
        "bone lesion",
        "lung tumor",
        "colon cancer primaries",
        "pancreatic tumor",
    ]
    available_controllable_anatomy = available_controllable_organ + available_controllable_tumor
    controllable_tumor = []
    controllable_organ = []
    for controllable_anatomy_size_pair in controllable_anatomy_size:
        if controllable_anatomy_size_pair[0] not in available_controllable_anatomy:
            raise ValueError(
                f"The controllable_anatomy have to be chosen from {available_controllable_anatomy}, yet got {controllable_anatomy_size_pair[0]}."
            )
        if controllable_anatomy_size_pair[0] in available_controllable_tumor:
            controllable_tumor += [controllable_anatomy_size_pair[0]]
        if controllable_anatomy_size_pair[0] in available_controllable_organ:
            controllable_organ += [controllable_anatomy_size_pair[0]]
        if controllable_anatomy_size_pair[1] == -1:
            continue
        if controllable_anatomy_size_pair[1] < 0 or controllable_anatomy_size_pair[1] > 1.0:
            raise ValueError(
                f"The controllable size scale have to be between 0 and 1,0, or equal to -1, yet got {controllable_anatomy_size_pair[1]}."
            )
    if len(controllable_tumor + controllable_organ) != len(list(set(controllable_tumor + controllable_organ))):
        raise ValueError(f"Please do not repeat controllable_anatomy. Got {controllable_tumor + controllable_organ}.")
    if len(controllable_tumor) > 1:
        raise ValueError(f"Only one controllable tumor is supported. Yet got {controllable_tumor}.")

    if len(controllable_anatomy_size) > 0:
        logging.info(
            f"`controllable_anatomy_size` is not empty.\nWe will ignore `body_region` and `anatomy_list` and synthesize based on `controllable_anatomy_size`: ({controllable_anatomy_size})."
        )
    else:
        logging.info(
            f"`controllable_anatomy_size` is empty.\nWe will synthesize based on `body_region`: ({body_region}) and `anatomy_list`: ({anatomy_list})."
        )
        # check body_region format
        available_body_region = [
            "head",
            "chest",
            "thorax",
            "abdomen",
            "pelvis",
            "lower",
        ]
        for region in body_region:
            if region not in available_body_region:
                raise ValueError(
                    f"The components in body_region have to be chosen from {available_body_region}, yet got {region}."
                )

        # check anatomy_list format
        with open(label_dict_json) as f:
            label_dict = json.load(f)
        for anatomy in anatomy_list:
            if anatomy not in label_dict.keys():
                raise ValueError(
                    f"The components in anatomy_list have to be chosen from {label_dict.keys()}, yet got {anatomy}."
                )
    logging.info(f"The generate results will have voxel size to be {spacing}mm, volume size to be {output_size}.")

    return


class LDMSampler:
    """
    A sampler class for generating synthetic medical images and masks using latent diffusion models.

    Attributes:
        Various attributes related to model configuration, input parameters, and generation settings.
    """

    def __init__(
        self,
        body_region,
        anatomy_list,
        all_mask_files_json,
        all_anatomy_size_condtions_json,
        all_mask_files_base_dir,
        label_dict_json,
        label_dict_remap_json,
        autoencoder,
        diffusion_unet,
        controlnet,
        noise_scheduler,
        scale_factor,
        mask_generation_autoencoder,
        mask_generation_diffusion_unet,
        mask_generation_scale_factor,
        mask_generation_noise_scheduler,
        device,
        latent_shape,
        mask_generation_latent_shape,
        output_size,
        output_dir,
        controllable_anatomy_size,
        image_output_ext=".nii.gz",
        label_output_ext=".nii.gz",
        real_img_median_statistics="./configs/image_median_statistics.json",
        spacing=[1, 1, 1],
        modality=1,
        num_inference_steps=None,
        mask_generation_num_inference_steps=None,
        random_seed=None,
        autoencoder_sliding_window_infer_size=[96, 96, 96],
        autoencoder_sliding_window_infer_overlap=0.6667,
    ) -> None:
        """
        Initialize the LDMSampler with various parameters and models.

        Args:
            Various parameters related to model configuration, input settings, and output specifications.
        """
        self.random_seed = random_seed
        if random_seed is not None:
            set_determinism(seed=random_seed)

        with open(label_dict_json, "r") as f:
            label_dict = json.load(f)
        self.all_anatomy_size_condtions_json = all_anatomy_size_condtions_json

        # initialize variables
        self.body_region = body_region
        self.anatomy_list = [label_dict[organ] for organ in anatomy_list]
        self.all_mask_files_json = all_mask_files_json
        self.data_root = all_mask_files_base_dir
        self.label_dict_remap_json = label_dict_remap_json
        self.autoencoder = autoencoder
        self.diffusion_unet = diffusion_unet
        self.controlnet = controlnet
        self.noise_scheduler = noise_scheduler
        self.scale_factor = scale_factor
        self.mask_generation_autoencoder = mask_generation_autoencoder
        self.mask_generation_diffusion_unet = mask_generation_diffusion_unet
        self.mask_generation_scale_factor = mask_generation_scale_factor
        self.mask_generation_noise_scheduler = mask_generation_noise_scheduler
        self.device = device
        self.latent_shape = latent_shape
        self.mask_generation_latent_shape = mask_generation_latent_shape
        self.output_size = output_size
        self.output_dir = output_dir
        self.noise_factor = 1.0
        self.controllable_anatomy_size = controllable_anatomy_size
        if len(self.controllable_anatomy_size):
            logging.info("controllable_anatomy_size is given, mask generation is triggered!")
            # overwrite the anatomy_list by given organs in self.controllable_anatomy_size
            self.anatomy_list = [label_dict[organ_and_size[0]] for organ_and_size in self.controllable_anatomy_size]
        self.image_output_ext = image_output_ext
        self.label_output_ext = label_output_ext
        # Set the default value for number of inference steps to 1000
        self.num_inference_steps = num_inference_steps if num_inference_steps is not None else 1000
        self.mask_generation_num_inference_steps = (
            mask_generation_num_inference_steps if mask_generation_num_inference_steps is not None else 1000
        )

        if any(size % 16 != 0 for size in autoencoder_sliding_window_infer_size):
            raise ValueError(
                f"autoencoder_sliding_window_infer_size must be divisible by 16.\n Got {autoencoder_sliding_window_infer_size}"
            )
        if not (0 <= autoencoder_sliding_window_infer_overlap <= 1):
            raise ValueError(
                f"Value of autoencoder_sliding_window_infer_overlap must be between 0 and 1.\n Got {autoencoder_sliding_window_infer_overlap}"
            )
        self.autoencoder_sliding_window_infer_size = autoencoder_sliding_window_infer_size
        self.autoencoder_sliding_window_infer_overlap = autoencoder_sliding_window_infer_overlap

        # quality check args
        self.max_try_time = 2  # if not pass quality check, will try self.max_try_time times
        with open(real_img_median_statistics, "r") as json_file:
            self.median_statistics = json.load(json_file)
        self.label_int_dict = {
            "liver": [1],
            "spleen": [3],
            "pancreas": [4],
            "kidney": [5, 14],
            "lung": [28, 29, 30, 31, 31],
            "brain": [22],
            "hepatic tumor": [26],
            "bone lesion": [128],
            "lung tumor": [23],
            "colon cancer primaries": [27],
            "pancreatic tumor": [24],
            "bone": list(range(33, 57)) + list(range(63, 98)) + [120, 122, 127],
        }

        # networks
        self.autoencoder.eval()
        self.diffusion_unet.eval()
        self.controlnet.eval()
        self.mask_generation_autoencoder.eval()
        self.mask_generation_diffusion_unet.eval()

        self.spacing = spacing
        self.modality_tensor = modality * torch.ones((1,), dtype=torch.long).to(device)
        self.include_body_region = self.diffusion_unet.include_top_region_index_input
        self.include_modality = self.diffusion_unet.num_class_embeds is not None

        val_transforms_list = [
            monai.transforms.LoadImaged(keys=["pseudo_label"]),
            monai.transforms.EnsureChannelFirstd(keys=["pseudo_label"]),
            monai.transforms.Orientationd(keys=["pseudo_label"], axcodes="RAS"),
            monai.transforms.EnsureTyped(keys=["pseudo_label"], dtype=torch.uint8),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
        ]
        if self.include_body_region:
            val_transforms_list += [
                monai.transforms.Lambdad(keys="top_region_index", func=lambda x: torch.FloatTensor(x)),
                monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: torch.FloatTensor(x)),
                monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
                monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
            ]

        self.val_transforms = Compose(val_transforms_list)
        logging.info("LDM sampler initialized.")

    def sample_multiple_images(self, num_img):
        """
        Generate multiple synthetic images and masks.

        Args:
            num_img (int): Number of images to generate.
        """
        modality_tensor = self.modality_tensor
        output_filenames = []
        if len(self.controllable_anatomy_size) > 0:
            # we will use mask generation instead of finding candidate masks
            # create a dummy selected_mask_files for placeholder
            selected_mask_files = list(range(num_img))
            # prerpare organ size conditions
            anatomy_size_condtion = self.prepare_anatomy_size_condtion(self.controllable_anatomy_size)
        else:
            need_resample = False
            # find candidate mask and save to candidate_mask_files
            candidate_mask_files = find_masks(
                self.body_region,
                self.anatomy_list,
                self.spacing,
                self.output_size,
                True,
                self.all_mask_files_json,
                self.data_root,
            )
            if len(candidate_mask_files) < num_img:
                # if we cannot find enough masks based on the exact match of anatomy list, spacing, and output size,
                # then we will try to find the closest mask in terms of  spacing, and output size.
                logging.info("Resample mask file to get desired output size and spacing")
                candidate_mask_files = self.find_closest_masks(num_img)
                need_resample = True

            selected_mask_files = self.select_mask(candidate_mask_files, num_img)
            logging.info(f"Images will be generated based on {selected_mask_files}.")
            if len(selected_mask_files) < num_img:
                raise ValueError(
                    (
                        f"len(selected_mask_files) ({len(selected_mask_files)}) < num_img ({num_img}). "
                        "This should not happen. Please revisit function select_mask(self, candidate_mask_files, num_img)."
                    )
                )

        num_generated_img = 0
        for index_s in range(len(selected_mask_files)):
            item = selected_mask_files[index_s]
            if num_generated_img >= num_img:
                break
            logging.info("---- Start preparing masks... ----")
            start_time = time.time()
            if len(self.controllable_anatomy_size) > 0:
                # generate a synthetic mask
                (
                    combine_label_or,
                    top_region_index_tensor,
                    bottom_region_index_tensor,
                    spacing_tensor,
                ) = self.prepare_one_mask_and_meta_info(anatomy_size_condtion)
            else:
                # read in mask file
                mask_file = item["mask_file"]
                if_aug = item["if_aug"]
                (
                    combine_label_or,
                    top_region_index_tensor,
                    bottom_region_index_tensor,
                    spacing_tensor,
                ) = self.read_mask_information(mask_file)
                if need_resample:
                    combine_label_or = self.ensure_output_size_and_spacing(combine_label_or)
                # mask augmentation
                if if_aug:
                    combine_label_or = augmentation(combine_label_or, self.output_size, self.random_seed)
            end_time = time.time()
            logging.info(f"---- Mask preparation time: {end_time - start_time} seconds ----")
            torch.cuda.empty_cache()
            # generate image/label pairs
            to_generate = True
            try_time = 0
            # start generation
            synthetic_images, synthetic_labels = self.sample_one_pair(
                combine_label_or,
                top_region_index_tensor,
                bottom_region_index_tensor,
                spacing_tensor,
                modality_tensor,
            )
            # synthetic image quality check
            pass_quality_check = self.quality_check(
                synthetic_images.cpu().detach().numpy(), combine_label_or.cpu().detach().numpy()
            )
            print(num_img - num_generated_img, (len(selected_mask_files) - index_s))
            if pass_quality_check or (num_img - num_generated_img) >= (len(selected_mask_files) - index_s):
                if not pass_quality_check:
                    logging.info(
                        "Generated image/label pair did not pass quality check, but will still save them. "
                        "Please consider changing spacing and output_size to facilitate a more realistic setting."
                    )
                num_generated_img = num_generated_img + 1
                # save image/label pairs
                output_postfix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                synthetic_labels.meta["filename_or_obj"] = "sample.nii.gz"
                synthetic_images = MetaTensor(synthetic_images, meta=synthetic_labels.meta)
                img_saver = SaveImage(
                    output_dir=self.output_dir,
                    output_postfix=output_postfix + "_image",
                    output_ext=self.image_output_ext,
                    separate_folder=False,
                )
                img_saver(synthetic_images[0])
                synthetic_images_filename = os.path.join(
                    self.output_dir, "sample_" + output_postfix + "_image" + self.image_output_ext
                )
                # filter out the organs that are not in anatomy_list
                synthetic_labels = filter_mask_with_organs(synthetic_labels, self.anatomy_list)
                label_saver = SaveImage(
                    output_dir=self.output_dir,
                    output_postfix=output_postfix + "_label",
                    output_ext=self.label_output_ext,
                    separate_folder=False,
                )
                label_saver(synthetic_labels[0])
                synthetic_labels_filename = os.path.join(
                    self.output_dir, "sample_" + output_postfix + "_label" + self.label_output_ext
                )
                output_filenames.append([synthetic_images_filename, synthetic_labels_filename])
                to_generate = False
            else:
                logging.info("Generated image/label pair did not pass quality check, will re-generate another pair.")
        return output_filenames

    def select_mask(self, candidate_mask_files, num_img):
        """
        Select mask files for image generation.

        Args:
            candidate_mask_files (list): List of candidate mask files.
            num_img (int): Number of images to generate.

        Returns:
            list: Selected mask files with augmentation flags.
        """
        selected_mask_files = []
        random.shuffle(candidate_mask_files)

        for n in range(len(candidate_mask_files)):
            mask_file = candidate_mask_files[n % len(candidate_mask_files)]
            selected_mask_files.append({"mask_file": mask_file, "if_aug": True})
        return selected_mask_files

    def sample_one_pair(
        self,
        combine_label_or_aug,
        top_region_index_tensor,
        bottom_region_index_tensor,
        spacing_tensor,
        modality_tensor,
    ):
        """
        Generate a single pair of synthetic image and mask.

        Args:
            combine_label_or_aug (torch.Tensor): Combined label tensor or augmented label.
            top_region_index_tensor (torch.Tensor): Tensor specifying the top region index.
            bottom_region_index_tensor (torch.Tensor): Tensor specifying the bottom region index.
            spacing_tensor (torch.Tensor): Tensor specifying the spacing.
            modality_tensor (torch.Tensor): Int Tensor specifying the modality.

        Returns:
            tuple: A tuple containing the synthetic image and its corresponding label.
        """
        # generate image/label pairs
        synthetic_images, synthetic_labels = ldm_conditional_sample_one_image(
            autoencoder=self.autoencoder,
            diffusion_unet=self.diffusion_unet,
            controlnet=self.controlnet,
            noise_scheduler=self.noise_scheduler,
            scale_factor=self.scale_factor,
            device=self.device,
            combine_label_or=combine_label_or_aug,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
            modality_tensor=modality_tensor,
            latent_shape=self.latent_shape,
            output_size=self.output_size,
            noise_factor=self.noise_factor,
            num_inference_steps=self.num_inference_steps,
            autoencoder_sliding_window_infer_size=self.autoencoder_sliding_window_infer_size,
            autoencoder_sliding_window_infer_overlap=self.autoencoder_sliding_window_infer_overlap,
        )
        return synthetic_images, synthetic_labels

    def prepare_anatomy_size_condtion(
        self,
        controllable_anatomy_size,
    ):
        """
        Prepare anatomy size conditions for mask generation.

        Args:
            controllable_anatomy_size (list): List of tuples specifying controllable anatomy sizes.

        Returns:
            list: Prepared anatomy size conditions.
        """
        anatomy_size_idx = {
            "gallbladder": 0,
            "liver": 1,
            "stomach": 2,
            "pancreas": 3,
            "colon": 4,
            "lung tumor": 5,
            "pancreatic tumor": 6,
            "hepatic tumor": 7,
            "colon cancer primaries": 8,
            "bone lesion": 9,
        }
        provide_anatomy_size = [None for _ in range(10)]
        logging.info(f"controllable_anatomy_size: {controllable_anatomy_size}")
        for element in controllable_anatomy_size:
            anatomy_name, anatomy_size = element
            provide_anatomy_size[anatomy_size_idx[anatomy_name]] = anatomy_size

        with open(self.all_anatomy_size_condtions_json, "r") as f:
            all_anatomy_size_condtions = json.load(f)

        # loop through the database and find closest combinations
        candidate_list = []
        for anatomy_size in all_anatomy_size_condtions:
            size = anatomy_size["organ_size"]
            diff = 0
            for db_size, provide_size in zip(size, provide_anatomy_size):
                if provide_size is None:
                    continue
                diff += abs(provide_size - db_size)
            candidate_list.append((size, diff))
        candidate_condition = sorted(candidate_list, key=lambda x: x[1])[0][0]

        # overwrite the anatomy size provided by users
        for element in controllable_anatomy_size:
            anatomy_name, anatomy_size = element
            candidate_condition[anatomy_size_idx[anatomy_name]] = anatomy_size

        return candidate_condition

    def prepare_one_mask_and_meta_info(self, anatomy_size_condtion):
        """
        Prepare a single mask and its associated meta information.

        Args:
            anatomy_size_condtion (list): Anatomy size conditions.

        Returns:
            tuple: A tuple containing the prepared mask and associated tensors.
        """
        combine_label_or = self.sample_one_mask(anatomy_size=anatomy_size_condtion)
        # TODO: current mask generation model only can generate 256^3 volumes with 1.5 mm spacing.
        affine = torch.zeros((4, 4))
        affine[0, 0] = 1.5
        affine[1, 1] = 1.5
        affine[2, 2] = 1.5
        affine[3, 3] = 1.0  # dummy
        combine_label_or = MetaTensor(combine_label_or, affine=affine)
        combine_label_or = self.ensure_output_size_and_spacing(combine_label_or)

        top_region_index, bottom_region_index = get_body_region_index_from_mask(combine_label_or)

        spacing_tensor = torch.FloatTensor(self.spacing).unsqueeze(0).half().to(self.device) * 1e2
        top_region_index_tensor = torch.FloatTensor(top_region_index).unsqueeze(0).half().to(self.device) * 1e2
        bottom_region_index_tensor = torch.FloatTensor(bottom_region_index).unsqueeze(0).half().to(self.device) * 1e2

        return combine_label_or, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor

    def sample_one_mask(self, anatomy_size):
        """
        Generate a single synthetic mask.

        Args:
            anatomy_size (list): Anatomy size specifications.

        Returns:
            torch.Tensor: The generated synthetic mask.
        """
        # generate one synthetic mask
        synthetic_mask = ldm_conditional_sample_one_mask(
            self.mask_generation_autoencoder,
            self.mask_generation_diffusion_unet,
            self.mask_generation_noise_scheduler,
            self.mask_generation_scale_factor,
            anatomy_size,
            self.device,
            self.mask_generation_latent_shape,
            label_dict_remap_json=self.label_dict_remap_json,
            num_inference_steps=self.mask_generation_num_inference_steps,
            autoencoder_sliding_window_infer_size=self.autoencoder_sliding_window_infer_size,
            autoencoder_sliding_window_infer_overlap=self.autoencoder_sliding_window_infer_overlap,
        )
        return synthetic_mask

    def ensure_output_size_and_spacing(self, labels, check_contains_target_labels=True):
        """
        Ensure the output mask has the correct size and spacing.

        Args:
            labels (torch.Tensor): Input label tensor.
            check_contains_target_labels (bool): Whether to check if the resampled mask contains target labels.

        Returns:
            torch.Tensor: Resampled label tensor.

        Raises:
            ValueError: If the resampled mask doesn't contain required class labels.
        """
        current_spacing = [labels.affine[0, 0], labels.affine[1, 1], labels.affine[2, 2]]
        current_shape = list(labels.squeeze().shape)

        need_resample = False
        # check spacing
        for i, j in zip(current_spacing, self.spacing):
            if i != j:
                need_resample = True
        # check output size
        for i, j in zip(current_shape, self.output_size):
            if i != j:
                need_resample = True
        # resample to target size and spacing
        if need_resample:
            logging.info("Resampling mask to target shape and spacing")
            logging.info(f"Resize Spacing: {current_spacing} -> {self.spacing}")
            logging.info(f"Output size: {current_shape} -> {self.output_size}")
            spacing = monai.transforms.Spacing(pixdim=tuple(self.spacing), mode="nearest")
            pad_crop = monai.transforms.ResizeWithPadOrCrop(spatial_size=tuple(self.output_size))
            labels = pad_crop(spacing(labels.squeeze(0))).unsqueeze(0).to(labels.dtype)

            contained_labels = torch.unique(labels)
            if check_contains_target_labels:
                # check if the resampled mask still contains those target labels
                for anatomy_label in self.anatomy_list:
                    if anatomy_label not in contained_labels:
                        raise ValueError(
                            f"Resampled mask does not contain required class labels {anatomy_label}. Please tune spacing and output size."
                        )
        return labels

    def read_mask_information(self, mask_file):
        """
        Read mask information from a file.

        Args:
            mask_file (str): Path to the mask file.

        Returns:
            tuple: A tuple containing the mask tensor and associated information.
        """
        val_data = self.val_transforms(mask_file)

        for key in ["pseudo_label", "spacing", "top_region_index", "bottom_region_index"]:
            if isinstance(val_data[key], torch.Tensor):
                val_data[key] = val_data[key].unsqueeze(0).to(self.device)
            else:
                val_data[key] = None

        return (
            val_data["pseudo_label"],
            val_data["top_region_index"],
            val_data["bottom_region_index"],
            val_data["spacing"],
        )

    def find_closest_masks(self, num_img):
        """
        Find the closest matching masks from the database.

        Args:
            num_img (int): Number of images to generate.

        Returns:
            list: List of closest matching mask candidates.

        Raises:
            ValueError: If suitable candidates cannot be found.
        """
        # first check the database based on anatomy list
        candidates = find_masks(
            self.body_region,
            self.anatomy_list,
            self.spacing,
            self.output_size,
            False,
            self.all_mask_files_json,
            self.data_root,
        )

        if len(candidates) < num_img:
            raise ValueError(f"candidate masks are less than {num_img}).")

        # loop through the database and find closest combinations
        new_candidates = []
        for c in candidates:
            diff = 0
            include_c = True
            for axis in range(3):
                if abs(c["dim"][axis]) < self.output_size[axis] - 64:
                    # we cannot upsample the mask too much
                    include_c = False
                    break
                # check diff in FOV, major metric
                diff += abs(
                    (abs(c["dim"][axis] * c["spacing"][axis]) - self.output_size[axis] * self.spacing[axis]) / 10
                )
                # check diff in dim
                diff += abs((abs(c["dim"][axis]) - self.output_size[axis]) / 100)
                # check diff in spacing
                diff += abs(abs(c["spacing"][axis]) - self.spacing[axis])
            if include_c:
                new_candidates.append((c, diff))

        # choose top-2*num_img candidates (at least 5)
        num_candidates = max(self.max_try_time * num_img, 5)
        new_candidates = sorted(new_candidates, key=lambda x: x[1])

        final_candidates = []
        # check top-2*num_img candidates and update spacing after resampling
        for c, _ in new_candidates:
            c = self.resample_mask_check_organ_list(c)
            if c is not None:
                final_candidates.append(c)
            if len(final_candidates) >= num_candidates:
                break
        if len(final_candidates) == 0:
            raise ValueError("Cannot find body region with given organ list.")
        return final_candidates

    def resample_mask_check_organ_list(self, mask):
        """
        Resample mask and check if the resampled mask contains the required organ list.

        Args:
            mask (dict): input mask.

        Returns:
            dict: resampled mask. If None, means the resampled mask does not contain the required organ list

        Raises:
            ValueError: If suitable candidates cannot be found.
        """

        image_loader = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True)
        label = image_loader(mask["pseudo_label"])
        try:
            label = self.ensure_output_size_and_spacing(label.unsqueeze(0))
        except ValueError as e:
            if "Resampled mask does not contain required class labels" in str(e):
                return None
            else:
                raise e
        # get region_index after resample
        top_region_index, bottom_region_index = get_body_region_index_from_mask(label)
        mask["top_region_index"] = top_region_index
        mask["bottom_region_index"] = bottom_region_index
        mask["spacing"] = self.spacing
        mask["dim"] = self.output_size
        return mask

    def quality_check(self, image_data, label_data):
        """
        Perform a quality check on the generated image.
        Args:
            image_data (np.ndarray): The generated image.
            label_data (np.ndarray): The corresponding whole body mask.
        Returns:
            bool: True if the image passes the quality check, False otherwise.
        """
        outlier_results = is_outlier(self.median_statistics, image_data, label_data, self.label_int_dict)
        for label, result in outlier_results.items():
            if result.get("is_outlier", False):
                logging.info(
                    f"Generated image quality check for label '{label}' failed: median value {result['median_value']} is outside the acceptable range ({result['low_thresh']} - {result['high_thresh']})."
                )
                return False
        return True
