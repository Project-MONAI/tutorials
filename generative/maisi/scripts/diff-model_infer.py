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

import os
import random
import torch
import nibabel as nib
import numpy as np
from datetime import datetime
import fire

from monai.utils import set_determinism
from inferer import DiffusionInferer, LatentDiffusionInferer
from custom_network import AutoencoderKLCKModified
from custom_network_tp import AutoencoderKLCKModified_TP
from custom_network_diffusion import CustomDiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler


def main(
    ckpt_filepath="",
    random_seed=random.randint(0, 99999),
    output_prefix="unet_3d",
    output_size=512,
    amp=True,
    a_min=-1000,
    a_max=1000,
    b_min=0,
    b_max=1,
):
    """
    Main function to run the diffusion model.

    Args:
        ckpt_filepath (str): Path to the checkpoint file.
        random_seed (int): Random seed for reproducibility.
        output_prefix (str): Prefix for output filenames.
        output_size (int or tuple): Output size of the images.
        amp (bool): Flag to enable automatic mixed precision.
        a_min (int): Minimum intensity value for scaling.
        a_max (int): Maximum intensity value for scaling.
        b_min (int): Minimum value for normalization.
        b_max (int): Maximum value for normalization.
    """
    print(f"a_min: {a_min}, a_max: {a_max}, b_min: {b_min}, b_max: {b_max}.")

    # Initialize distributed processing
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    print(f"Using {device}.")
    print(f"world_size -> {world_size}.")

    # Set random seed for reproducibility
    rand_seed = random_seed + local_rank
    set_determinism(rand_seed)
    print(f"random seed: {rand_seed}")

    list_tuples = [
        (128, 128, 128, 2.0, 2.0),
        (128, 128, 128, 3.0, 3.0),
    ]
    selected_tuple = random.choice(list_tuples)
    output_size = (selected_tuple[0], selected_tuple[1], selected_tuple[2])
    out_spacing = (selected_tuple[3], selected_tuple[3], selected_tuple[4])

    if local_rank == 0:
        print(f"[config] amp -> {amp}.")
        print(f"[config] ckpt_filepath -> {ckpt_filepath}.")
        print(f"[config] random_seed -> {random_seed}.")
        print(f"[config] output_prefix -> {output_prefix}.")
        print(f"[config] output_size -> {output_size}.")
        print(f"[config] out_spacing -> {out_spacing}.")

    # Initialize autoencoder
    autoencoder = AutoencoderKLCKModified_TP(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256),
        latent_channels=4,
        attention_levels=(False, False, False),
        num_res_blocks=(2, 2, 2),
        norm_num_groups=32,
        norm_eps=1e-06,
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
    )
    autoencoder.to(device)

    # Initialize UNet model
    unet = CustomDiffusionModelUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        num_channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_head_channels=(0, 0, 32, 32),
        num_res_blocks=2,
        use_flash_attention=True,
        input_top_region_index=True,
        input_bottom_region_index=True,
        input_spacing=True,
    )
    unet.to(device)

    # Load the saved states into the model and optimizer
    checkpoint = torch.load(f"{ckpt_filepath}", map_location=device)
    unet.load_state_dict(checkpoint["unet_state_dict"])
    print(f"checkpoints {ckpt_filepath} loaded.")

    num_train_timesteps = checkpoint["num_train_timesteps"]
    print(f"num_train_timesteps -> {num_train_timesteps}.")

    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=num_train_timesteps)

    scale_factor = checkpoint["scale_factor"]
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    print(f"scale_factor -> {scale_factor}.")

    checkpoint_autoencoder = torch.load("/workspace/monai/generative/from_canz/autoencoder_epoch273.pt")
    new_state_dict = {}
    for k, v in checkpoint_autoencoder.items():
        if "decoder" in k and "conv" in k:
            new_key = (
                k.replace("conv.weight", "conv.conv.weight")
                if "conv.weight" in k
                else k.replace("conv.bias", "conv.conv.bias")
            )
        elif "encoder" in k and "conv" in k:
            new_key = (
                k.replace("conv.weight", "conv.conv.weight")
                if "conv.weight" in k
                else k.replace("conv.bias", "conv.conv.bias")
            )
        else:
            new_key = k
        new_state_dict[new_key] = v
    autoencoder.load_state_dict(new_state_dict)
    print("checkpoint_autoencoder loaded.")

    autoencoder.eval()
    unet.eval()

    def recon1(z, autoencoder, scale_factor):
        """
        Reconstruct the input using the autoencoder.

        Args:
            z (torch.Tensor): Latent space tensor.
            autoencoder (torch.nn.Module): Autoencoder model.
            scale_factor (float): Scaling factor.

        Returns:
            torch.Tensor: Reconstructed tensor.
        """
        recon_pt_nda = autoencoder.decode_stage_2_outputs(z / scale_factor)
        return recon_pt_nda

    if not amp:
        torch.set_float32_matmul_precision("highest")
        print("torch.set_float32_matmul_precision -> highest.")

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            noise = torch.randn(
                (
                    1,
                    4,
                    output_size[0] // 4,
                    output_size[1] // 4,
                    output_size[2] // 4,
                )
            ).to(device)
            print("noise:", noise.device, noise.dtype, type(noise))

            _factor = 1.0
            noise = noise * _factor
            print(f"scale noise by {_factor}")

            top_region_index_tensor = np.array([0, 1, 0, 0]).astype(float)
            bottom_region_index_tensor = np.array([0, 0, 1, 0]).astype(float)
            spacing_tensor = np.array(out_spacing).astype(float)

            top_region_index_tensor = top_region_index_tensor * 1e2
            print(f"top_region_index_tensor: {top_region_index_tensor}.")
            bottom_region_index_tensor = bottom_region_index_tensor * 1e2
            print(f"bottom_region_index_tensor: {bottom_region_index_tensor}.")
            spacing_tensor = spacing_tensor * 1e2
            print(f"spacing_tensor: {spacing_tensor}.")

            top_region_index_tensor = top_region_index_tensor[np.newaxis, :]
            bottom_region_index_tensor = bottom_region_index_tensor[np.newaxis, :]
            spacing_tensor = spacing_tensor[np.newaxis, :]

            if amp:
                top_region_index_tensor = torch.from_numpy(top_region_index_tensor).half().to(device)
                bottom_region_index_tensor = torch.from_numpy(bottom_region_index_tensor).half().to(device)
                spacing_tensor = torch.from_numpy(spacing_tensor).half().to(device)
            else:
                top_region_index_tensor = torch.from_numpy(top_region_index_tensor).float().to(device)
                bottom_region_index_tensor = torch.from_numpy(bottom_region_index_tensor).float().to(device)
                spacing_tensor = torch.from_numpy(spacing_tensor).float().to(device)

            outputs = DiffusionInferer.sample(
                inferer,
                input_noise=noise,
                diffusion_model=unet,
                scheduler=scheduler,
                save_intermediates=False,
                intermediate_steps=False,
                top_region_index_tensor=top_region_index_tensor,
                bottom_region_index_tensor=bottom_region_index_tensor,
                spacing_tensor=spacing_tensor,
            )
            print("outputs", outputs.size())

        with torch.cuda.amp.autocast(enabled=True):
            target_shape = output_size
            recon_pt_nda = torch.zeros(
                (1, 1, target_shape[0], target_shape[1], target_shape[2]),
                dtype=outputs.dtype,
            ).to("cuda")
            _count = torch.zeros(
                (1, 1, target_shape[0], target_shape[1], target_shape[2]),
                dtype=outputs.dtype,
            ).to("cuda")
            z = outputs

            _temp = recon1(
                z[
                    ...,
                    : target_shape[0] // 2,
                    : target_shape[1] // 2,
                    : target_shape[2] // 2,
                ],
                autoencoder,
                scale_factor,
            )
            recon_pt_nda[
                ...,
                : target_shape[0] // 2,
                : target_shape[1] // 2,
                : target_shape[2] // 2,
            ] += _temp[
                ...,
                : target_shape[0] // 2,
                : target_shape[1] // 2,
                : target_shape[2] // 2,
            ]
            _count[
                ...,
                : target_shape[0] // 2,
                : target_shape[1] // 2,
                : target_shape[2] // 2,
            ] += 1.0
            _temp = recon1(
                z[
                    ...,
                    : target_shape[0] // 2,
                    target_shape[1] // 2 :,
                    : target_shape[2] // 2,
                ],
                autoencoder,
                scale_factor,
            )
            recon_pt_nda[
                ...,
                : target_shape[0] // 2,
                target_shape[1] // 2 :,
                : target_shape[2] // 2,
            ] += _temp[
                ...,
                : target_shape[0] // 2,
                target_shape[1] // 2 :,
                : target_shape[2] // 2,
            ]
            _count[
                ...,
                : target_shape[0] // 2,
                target_shape[1] // 2 :,
                : target_shape[2] // 2,
            ] += 1.0
            _temp = recon1(
                z[
                    ...,
                    target_shape[0] // 2 :,
                    : target_shape[1] // 2,
                    : target_shape[2] // 2,
                ],
                autoencoder,
                scale_factor,
            )
            recon_pt_nda[
                ...,
                target_shape[0] // 2 :,
                : target_shape[1] // 2,
                : target_shape[2] // 2,
            ] += _temp[
                ...,
                target_shape[0] // 2 :,
                : target_shape[1] // 2,
                : target_shape[2] // 2,
            ]
            _count[
                ...,
                target_shape[0] // 2 :,
                : target_shape[1] // 2,
                : target_shape[2] // 2,
            ] += 1.0
            _temp = recon1(
                z[
                    ...,
                    target_shape[0] // 2 :,
                    target_shape[1] // 2 :,
                    : target_shape[2] // 2,
                ],
                autoencoder,
                scale_factor,
            )
            recon_pt_nda[
                ...,
                target_shape[0] // 2 :,
                target_shape[1] // 2 :,
                : target_shape[2] // 2,
            ] += _temp[
                ...,
                target_shape[0] // 2 :,
                target_shape[1] // 2 :,
                : target_shape[2] // 2,
            ]
            _count[
                ...,
                target_shape[0] // 2 :,
                target_shape[1] // 2 :,
                : target_shape[2] // 2,
            ] += 1.0
            _temp = recon1(
                z[
                    ...,
                    : target_shape[0] // 2,
                    : target_shape[1] // 2,
                    target_shape[2] // 2 :,
                ],
                autoencoder,
                scale_factor,
            )
            recon_pt_nda[
                ...,
                : target_shape[0] // 2,
                : target_shape[1] // 2,
                target_shape[2] // 2 :,
            ] += _temp[
                ...,
                : target_shape[0] // 2,
                : target_shape[1] // 2,
                target_shape[2] // 2 :,
            ]
            _count[
                ...,
                : target_shape[0] // 2,
                : target_shape[1] // 2,
                target_shape[2] // 2 :,
            ] += 1.0
            _temp = recon1(
                z[
                    ...,
                    : target_shape[0] // 2,
                    target_shape[1] // 2 :,
                    target_shape[2] // 2 :,
                ],
                autoencoder,
                scale_factor,
            )
            recon_pt_nda[
                ...,
                : target_shape[0] // 2,
                target_shape[1] // 2 :,
                target_shape[2] // 2 :,
            ] += _temp[
                ...,
                : target_shape[0] // 2,
                target_shape[1] // 2 :,
                target_shape[2] // 2 :,
            ]
            _count[
                ...,
                : target_shape[0] // 2,
                target_shape[1] // 2 :,
                target_shape[2] // 2 :,
            ] += 1.0
            _temp = recon1(
                z[
                    ...,
                    target_shape[0] // 2 :,
                    : target_shape[1] // 2,
                    target_shape[2] // 2 :,
                ],
                autoencoder,
                scale_factor,
            )
            recon_pt_nda[
                ...,
                target_shape[0] // 2 :,
                : target_shape[1] // 2,
                target_shape[2] // 2 :,
            ] += _temp[
                ...,
                target_shape[0] // 2 :,
                : target_shape[1] // 2,
                target_shape[2] // 2 :,
            ]
            _count[
                ...,
                target_shape[0] // 2 :,
                : target_shape[1] // 2,
                target_shape[2] // 2 :,
            ] += 1.0
            _temp = recon1(
                z[
                    ...,
                    target_shape[0] // 2 :,
                    target_shape[1] // 2 :,
                    target_shape[2] // 2 :,
                ],
                autoencoder,
                scale_factor,
            )
            recon_pt_nda[
                ...,
                target_shape[0] // 2 :,
                target_shape[1] // 2 :,
                target_shape[2] // 2 :,
            ] += _temp[
                ...,
                target_shape[0] // 2 :,
                target_shape[1] // 2 :,
                target_shape[2] // 2 :,
            ]
            _count[
                ...,
                target_shape[0] // 2 :,
                target_shape[1] // 2 :,
                target_shape[2] // 2 :,
            ] += 1.0

            synthetic_images = recon_pt_nda / _count

            data = synthetic_images.squeeze().cpu().detach().numpy()

            if True:
                data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
                data = np.clip(data, a_min, a_max)
                data = np.int16(data)

            out_affine = np.eye(4)
            for _k in range(3):
                out_affine[_k, _k] = out_spacing[_k]
            new_image = nib.Nifti1Image(data, affine=out_affine)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        nib.save(
            new_image,
            f"./predictions/{output_prefix}_seed{rand_seed}_size{output_size[0]:d}x{output_size[1]:d}x{output_size[2]:d}_spacing{out_spacing[0]:.2f}x{out_spacing[1]:.2f}x{out_spacing[2]:.2f}_{timestamp}.nii.gz",
        )

    return


if __name__ == "__main__":
    fire.Fire(main)
