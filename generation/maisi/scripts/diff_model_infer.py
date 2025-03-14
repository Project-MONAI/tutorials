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

from __future__ import annotations

import argparse
import logging
import os
import random
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from monai.inferers import sliding_window_inference
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism
from tqdm import tqdm

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .sample import ReconModel, check_input
from .utils import define_instance, dynamic_infer


def set_random_seed(seed: int) -> int:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.

    Returns:
        int: Set random seed.
    """
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def load_models(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> tuple:
    """
    Load the autoencoder and UNet models.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load models on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        tuple: Loaded autoencoder, UNet model, and scale factor.
    """
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(checkpoint_autoencoder)
    except Exception:
        logger.error("The trained_autoencoder_path does not exist!")

    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(f"{args.model_dir}/{args.model_filename}", map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=True)
    logger.info(f"checkpoints {args.model_dir}/{args.model_filename} loaded.")

    scale_factor = checkpoint["scale_factor"]
    logger.info(f"scale_factor -> {scale_factor}.")

    return autoencoder, unet, scale_factor


def prepare_tensors(args: argparse.Namespace, device: torch.device) -> tuple:
    """
    Prepare necessary tensors for inference.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load tensors on.

    Returns:
        tuple: Prepared top_region_index_tensor, bottom_region_index_tensor, and spacing_tensor.
    """
    top_region_index_tensor = np.array(args.diffusion_unet_inference["top_region_index"]).astype(float) * 1e2
    bottom_region_index_tensor = np.array(args.diffusion_unet_inference["bottom_region_index"]).astype(float) * 1e2
    spacing_tensor = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2

    top_region_index_tensor = torch.from_numpy(top_region_index_tensor[np.newaxis, :]).half().to(device)
    bottom_region_index_tensor = torch.from_numpy(bottom_region_index_tensor[np.newaxis, :]).half().to(device)
    spacing_tensor = torch.from_numpy(spacing_tensor[np.newaxis, :]).half().to(device)
    modality_tensor = args.diffusion_unet_inference["modality"] * torch.ones(
        (len(spacing_tensor)), dtype=torch.long
    ).to(device)

    return top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor


def run_inference(
    args: argparse.Namespace,
    device: torch.device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: float,
    top_region_index_tensor: torch.Tensor,
    bottom_region_index_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor,
    modality_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        top_region_index_tensor (torch.Tensor): Top region index tensor.
        bottom_region_index_tensor (torch.Tensor): Bottom region index tensor.
        spacing_tensor (torch.Tensor): Spacing tensor.
        modality_tensor (torch.Tensor): Modality tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    noise = torch.randn(
        (
            1,
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}")

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    autoencoder.eval()
    unet.eval()

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    progress_bar = tqdm(
        zip(all_timesteps, all_next_timesteps),
        total=min(len(all_timesteps), len(all_next_timesteps)),
    )
    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in progress_bar:
            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": image,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
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
            model_output = unet(**unet_inputs)
            if not isinstance(noise_scheduler, RFlowScheduler):
                image, _ = noise_scheduler.step(model_output, t, image)  # type: ignore
            else:
                image, _ = noise_scheduler.step(model_output, t, image, next_t)  # type: ignore

        inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80],
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=0.4,
            sw_device=device,
            device=device,
        )
        synthetic_images = dynamic_infer(inferer, recon_model, image)
        data = synthetic_images.squeeze().cpu().detach().numpy()
        a_min, a_max, b_min, b_max = -1000, 1000, 0, 1
        data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        data = np.clip(data, a_min, a_max)
        return np.int16(data)


def save_image(
    data: np.ndarray,
    output_size: tuple,
    out_spacing: tuple,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Save the generated synthetic image to a file.

    Args:
        data (np.ndarray): Synthetic image data.
        output_size (tuple): Output size of the image.
        out_spacing (tuple): Spacing of the output image.
        output_path (str): Path to save the output image.
        logger (logging.Logger): Logger for logging information.
    """
    out_affine = np.eye(4)
    for i in range(3):
        out_affine[i, i] = out_spacing[i]

    new_image = nib.Nifti1Image(data, affine=out_affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_image, output_path)
    logger.info(f"Saved {output_path}.")


@torch.inference_mode()
def diff_model_infer(env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int) -> None:
    """
    Main function to run the diffusion model inference.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("inference")
    random_seed = set_random_seed(
        args.diffusion_unet_inference["random_seed"] + local_rank
        if args.diffusion_unet_inference["random_seed"]
        else None
    )
    logger.info(f"Using {device} of {world_size} with random seed: {random_seed}")

    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])
    output_prefix = args.output_prefix
    ckpt_filepath = f"{args.model_dir}/{args.model_filename}"

    if local_rank == 0:
        logger.info(f"[config] ckpt_filepath -> {ckpt_filepath}.")
        logger.info(f"[config] random_seed -> {random_seed}.")
        logger.info(f"[config] output_prefix -> {output_prefix}.")
        logger.info(f"[config] output_size -> {output_size}.")
        logger.info(f"[config] out_spacing -> {out_spacing}.")

    check_input(None, None, None, output_size, out_spacing, None)

    autoencoder, unet, scale_factor = load_models(args, device, logger)
    num_downsample_level = max(
        1,
        (
            len(args.diffusion_unet_def["num_channels"])
            if isinstance(args.diffusion_unet_def["num_channels"], list)
            else len(args.diffusion_unet_def["attention_levels"])
        ),
    )
    divisor = 2 ** (num_downsample_level - 2)
    logger.info(f"num_downsample_level -> {num_downsample_level}, divisor -> {divisor}.")

    top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor = prepare_tensors(args, device)
    data = run_inference(
        args,
        device,
        autoencoder,
        unet,
        scale_factor,
        top_region_index_tensor,
        bottom_region_index_tensor,
        spacing_tensor,
        modality_tensor,
        output_size,
        divisor,
        logger,
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = "{0}/{1}_seed{2}_size{3:d}x{4:d}x{5:d}_spacing{6:.2f}x{7:.2f}x{8:.2f}_{9}_rank{10}.nii.gz".format(
        args.output_dir,
        output_prefix,
        random_seed,
        output_size[0],
        output_size[1],
        output_size[2],
        out_spacing[0],
        out_spacing[1],
        out_spacing[2],
        timestamp,
        local_rank,
    )
    save_image(data, output_size, out_spacing, output_path, logger)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_maisi.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for distributed inference",
    )

    args = parser.parse_args()
    diff_model_infer(args.env_config, args.model_config, args.model_def, args.num_gpus)
