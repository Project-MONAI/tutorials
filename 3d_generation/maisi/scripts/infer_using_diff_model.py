#!/usr/bin/env python

import fire
import nibabel as nib
import numpy as np
import os
import random
import shutil
import tempfile

import torch
import torch.nn.functional as F

from custom_network import AutoencoderKLCKModified
from custom_network_tp import AutoencoderKLCKModified_TP
from datetime import datetime
from monai import transforms
from monai.apps import DecathlonDataset

from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from torch.distributed._tensor import (
    DeviceMesh,
    Shard,
    distribute_tensor,
    distribute_module,
)
from tqdm import tqdm

from inferer import DiffusionInferer, LatentDiffusionInferer

from custom_network_diffusion import CustomDiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from PIL import Image


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
    print(f"a_min: {a_min}, a_max: {a_max}, b_min: {b_min}, b_max: {b_max}.")
    # print_config()

    # # for reproducibility purposes set a seed
    # set_determinism(42)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    print(f"Using {device}.")
    print(f"world_size -> {world_size}.")

    # for reproducibility purposes set a seed
    if False:
        rand_seed = 2
    else:
        rand_seed = random_seed
        # rand_seed = random.randint(0, 1e4)
    rand_seed += local_rank
    set_determinism(rand_seed)
    print(f"random seed: {rand_seed}")

    if False:
        list_tuples = [(512, 512, 96), (512, 512, 224), (512, 512, 480)]
        output_size = random.choice(list_tuples)

        list_spacing_xy = [0.59, 0.78, 0.97]
        list_spacing_z = [2.49, 1.24, 0.63]
        spacing_xy = random.choice(list_spacing_xy)
        spacing_z = random.choice(list_spacing_z)
        out_spacing = (spacing_xy, spacing_xy, spacing_z)
    else:
        # list_tuples = [
        #     (512, 512, 128, 0.73, 2.5),
        #     (512, 512, 128, 0.73, 2.5),
        #     (512, 512, 224, 0.77, 1.24),
        #     # (512, 512, 224, 0.78, 1.24),
        #     (512, 512, 224, 0.78, 1.25),
        #     (512, 512, 288, 0.68, 1.0),
        #     (512, 512, 288, 0.68, 1.0),
        #     (512, 512, 480, 0.79, 0.63),
        # ]
        list_tuples = [
            # # (512, 512, 128, 0.98, 2.0),
            # # (512, 512, 128, 0.94, 1.43),
            # # (512, 512, 128, 0.77, 3.95),
            # # (512, 512, 256, 0.86, 1.05),
            # # (512, 512, 256, 0.82, 2.66),
            # # (512, 512, 256, 0.81, 1.19),
            # (256, 256, 256, 1.55, 1.60),
            # (256, 256, 256, 1.56, 1.24),
            # (256, 256, 256, 1.59, 1.18),
            # (256, 256, 256, 1.61, 1.31),
            # (256, 256, 128, 1.46, 1.98),
            # (256, 256, 128, 1.38, 1.17),
            # (256, 256, 128, 1.76, 1.10),
            # (512, 512, 512, 1.0, 1.0),
            # (256, 256, 128, 1.0, 2.0),
            # (256, 256, 128, 1.5, 3.0),
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

    if True:
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
    else:
        autoencoder = AutoencoderKLCKModified(
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

    # scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
    num_train_timesteps = checkpoint["num_train_timesteps"]
    # num_train_timesteps = 1
    print(f"num_train_timesteps -> {num_train_timesteps}.")

    # scheduler_method = checkpoint['scheduler_method']
    # print(f'scheduler_method -> {scheduler_method}.')

    # if scheduler_method == 'ddpm':
    #     scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
    # elif scheduler_method == 'ddim':
    #     scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)

    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=num_train_timesteps)

    # print(f"Scaling factor set to {1/torch.std(z)}")
    # scale_factor = 1 / torch.std(z)
    # scale_factor = 1.00900
    scale_factor = checkpoint["scale_factor"]
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    print(f"scale_factor -> {scale_factor}.")

    # load checkpoint
    checkpoint_autoencoder = torch.load("/workspace/monai/generative/from_canz/autoencoder_epoch273.pt")
    if True:
        new_state_dict = {}
        for k, v in checkpoint_autoencoder.items():
            if "decoder" in k and "conv" in k:
                new_key = (
                    k.replace("conv.weight", "conv.conv.weight")
                    if "conv.weight" in k
                    else k.replace("conv.bias", "conv.conv.bias")
                )
                new_state_dict[new_key] = v
            elif "encoder" in k and "conv" in k:
                new_key = (
                    k.replace("conv.weight", "conv.conv.weight")
                    if "conv.weight" in k
                    else k.replace("conv.bias", "conv.conv.bias")
                )
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        checkpoint_autoencoder = new_state_dict
    autoencoder.load_state_dict(checkpoint_autoencoder)
    print("checkpoint_autoencoder loaded.")

    autoencoder.eval()
    unet.eval()

    def recon1(z, autoencoder, scale_factor):
        recon_pt_nda = autoencoder.decode_stage_2_outputs(z / scale_factor)
        return recon_pt_nda

    if not amp:
        torch.set_float32_matmul_precision("highest")
        print("torch.set_float32_matmul_precision -> highest.")

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            if False:
                noise = torch.randn((1, 4, 64, 64, 64))
            else:
                noise = torch.randn(
                    (
                        1,
                        4,
                        output_size[0] // 4,
                        output_size[1] // 4,
                        output_size[2] // 4,
                    )
                )
            noise = noise.to(device)
            print("noise:", noise.device, noise.dtype, type(noise))

            # scale noise
            _factor = 1.0
            # _factor = 0.5
            noise = noise * _factor
            print(f"scale noise by {_factor}")

            # top_region_index_tensor = np.array([1, 0, 0, 0]).astype(float)
            top_region_index_tensor = np.array([0, 1, 0, 0]).astype(float)
            # top_region_index_tensor = np.array([0, 0, 0, 1]).astype(float)
            bottom_region_index_tensor = np.array([0, 0, 1, 0]).astype(float)
            # bottom_region_index_tensor = np.array([0, 0, 0, 1]).astype(float)

            # out_spacing = [1.0, 1.0, 1.0]
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

            # synthetic_images = autoencoder.decode_stage_2_outputs(outputs / scale_factor)
            # print("synthetic_images:", synthetic_images.size(), synthetic_images.device)

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
                    : target_shape[0] // 4 // 4 * 3,
                    : target_shape[1] // 4 // 4 * 3,
                    : target_shape[2] // 4 // 4 * 3,
                ],
                autoencoder,
                scale_factor,
            )
            # recon_pt_nda[..., :target_shape[0]//2, :target_shape[1]//2, :target_shape[2]//2] = _temp[..., :target_shape[0]//2, :target_shape[1]//2, :target_shape[2]//2]
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
                    : target_shape[0] // 4 // 4 * 3,
                    target_shape[1] // 4 // 4 * 1 :,
                    : target_shape[2] // 4 // 4 * 3,
                ],
                autoencoder,
                scale_factor,
            )
            # recon_pt_nda[..., :target_shape[0]//2, target_shape[1]//2:, :target_shape[2]//2] = _temp[..., :target_shape[0]//2, target_shape[1]//4*1:, :target_shape[2]//2]
            recon_pt_nda[
                ...,
                : target_shape[0] // 2,
                target_shape[1] // 2 :,
                : target_shape[2] // 2,
            ] += _temp[
                ...,
                : target_shape[0] // 2,
                target_shape[1] // 4 * 1 :,
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
                    target_shape[0] // 4 // 4 * 1 :,
                    : target_shape[1] // 4 // 4 * 3,
                    : target_shape[2] // 4 // 4 * 3,
                ],
                autoencoder,
                scale_factor,
            )
            # recon_pt_nda[..., target_shape[0]//2:, :target_shape[1]//2, :target_shape[2]//2] = _temp[..., target_shape[0]//4*1:, :target_shape[1]//2, :target_shape[2]//2]
            recon_pt_nda[
                ...,
                target_shape[0] // 2 :,
                : target_shape[1] // 2,
                : target_shape[2] // 2,
            ] += _temp[
                ...,
                target_shape[0] // 4 * 1 :,
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
                    target_shape[0] // 4 // 4 * 1 :,
                    target_shape[1] // 4 // 4 * 1 :,
                    : target_shape[2] // 4 // 4 * 3,
                ],
                autoencoder,
                scale_factor,
            )
            # recon_pt_nda[..., target_shape[0]//2:, target_shape[1]//2:, :target_shape[2]//2] = _temp[..., target_shape[0]//4*1:, target_shape[1]//4*1:, :target_shape[2]//2]
            recon_pt_nda[
                ...,
                target_shape[0] // 2 :,
                target_shape[1] // 2 :,
                : target_shape[2] // 2,
            ] += _temp[
                ...,
                target_shape[0] // 4 * 1 :,
                target_shape[1] // 4 * 1 :,
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
                    : target_shape[0] // 4 // 4 * 3,
                    : target_shape[1] // 4 // 4 * 3,
                    target_shape[2] // 4 // 4 * 1 :,
                ],
                autoencoder,
                scale_factor,
            )
            # recon_pt_nda[..., :target_shape[0]//2, :target_shape[1]//2, target_shape[2]//2:] = _temp[..., :target_shape[0]//2, :target_shape[1]//2, target_shape[2]//4*1:]
            recon_pt_nda[
                ...,
                : target_shape[0] // 2,
                : target_shape[1] // 2,
                target_shape[2] // 2 :,
            ] += _temp[
                ...,
                : target_shape[0] // 2,
                : target_shape[1] // 2,
                target_shape[2] // 4 * 1 :,
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
                    : target_shape[0] // 4 // 4 * 3,
                    target_shape[1] // 4 // 4 * 1 :,
                    target_shape[2] // 4 // 4 * 1 :,
                ],
                autoencoder,
                scale_factor,
            )
            # recon_pt_nda[..., :target_shape[0]//2, target_shape[1]//2:, target_shape[2]//2:] = _temp[..., :target_shape[0]//2, target_shape[1]//4*1:, target_shape[2]//4*1:]
            recon_pt_nda[
                ...,
                : target_shape[0] // 2,
                target_shape[1] // 2 :,
                target_shape[2] // 2 :,
            ] += _temp[
                ...,
                : target_shape[0] // 2,
                target_shape[1] // 4 * 1 :,
                target_shape[2] // 4 * 1 :,
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
                    target_shape[0] // 4 // 4 * 1 :,
                    : target_shape[1] // 4 // 4 * 3,
                    target_shape[2] // 4 // 4 * 1 :,
                ],
                autoencoder,
                scale_factor,
            )
            # recon_pt_nda[..., target_shape[0]//2:, :target_shape[1]//2, target_shape[2]//2:] = _temp[..., target_shape[0]//4*1:, :target_shape[1]//2, target_shape[2]//4*1:]
            recon_pt_nda[
                ...,
                target_shape[0] // 2 :,
                : target_shape[1] // 2,
                target_shape[2] // 2 :,
            ] += _temp[
                ...,
                target_shape[0] // 4 * 1 :,
                : target_shape[1] // 2,
                target_shape[2] // 4 * 1 :,
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
                    target_shape[0] // 4 // 4 * 1 :,
                    target_shape[1] // 4 // 4 * 1 :,
                    target_shape[2] // 4 // 4 * 1 :,
                ],
                autoencoder,
                scale_factor,
            )
            # recon_pt_nda[..., target_shape[0]//2:, target_shape[1]//2:, target_shape[2]//2:] = _temp[..., target_shape[0]//4*1:, target_shape[1]//4*1:, target_shape[2]//4*1:]
            recon_pt_nda[
                ...,
                target_shape[0] // 2 :,
                target_shape[1] // 2 :,
                target_shape[2] // 2 :,
            ] += _temp[
                ...,
                target_shape[0] // 4 * 1 :,
                target_shape[1] // 4 * 1 :,
                target_shape[2] // 4 * 1 :,
            ]
            _count[
                ...,
                target_shape[0] // 2 :,
                target_shape[1] // 2 :,
                target_shape[2] // 2 :,
            ] += 1.0

            synthetic_images = recon_pt_nda / _count

            data = synthetic_images.squeeze().cpu().detach().numpy()

            # data = data * 2.0 - 1.0
            # data = np.clip(data, -1.0, 1.0) * 1000.0

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

        # data_2d = data[..., data.shape[2] // 2].squeeze().astype(float)
        # data_2d = (data_2d + 1000.0) / 2000 * 255 + 1e-6
        # data_2d = np.floor(data_2d).astype(np.uint8)
        # png_img = Image.fromarray(data_2d)
        # png_img.save(f'{output_prefix}_seed{rand_seed}_{timestamp}.png')

    return


if __name__ == "__main__":
    fire.Fire(main)
