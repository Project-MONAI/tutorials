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

import argparse
import json
import os
import random
import torch
import torch.distributed as dist
import nibabel as nib
import numpy as np

from datetime import datetime, timedelta
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from utils import define_instance, load_autoencoder_ckpt


class ReconModel(torch.nn.Module):
    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        recon_pt_nda = self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)
        return recon_pt_nda


def diff_model_infer(env_config_path: str, model_config_path: str):
    """
    Main function to run the diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
    """
    args = argparse.Namespace()

    # Load environment configuration
    with open(env_config_path, "r") as f:
        env_config = json.load(f)

    for k, v in env_config.items():
        setattr(args, k, v)

    # Load model configuration
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    for k, v in model_config.items():
        setattr(args, k, v)

    a_min = -1000
    a_max = 1000
    b_min = 0
    b_max = 1
    print(f"a_min: {a_min}, a_max: {a_max}, b_min: {b_min}, b_max: {b_max}.")

    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=7200))

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(dist.get_world_size())
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    print(f"Using {device} of {world_size}")

    # Set random seed for reproducibility
    if args.diffusion_unet_inference["random_seed"] == None:
        random_seed = random.randint(0, 99999)
    else:
        random_seed = args.diffusion_unet_inference["random_seed"] + local_rank
    set_determinism(random_seed)
    print(f"random seed: {random_seed}")

    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])

    ckpt_filepath = f"{args.model_dir}/{args.model_filename}"
    output_prefix = args.output_prefix

    if local_rank == 0:
        print(f"[config] ckpt_filepath -> {ckpt_filepath}.")
        print(f"[config] random_seed -> {random_seed}.")
        print(f"[config] output_prefix -> {output_prefix}.")
        print(f"[config] output_size -> {output_size}.")
        print(f"[config] out_spacing -> {out_spacing}.")

    # Initialize autoencoder
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = load_autoencoder_ckpt(args.trained_autoencoder_path)
        autoencoder.load_state_dict(checkpoint_autoencoder)
    except:
        print("The trained_autoencoder_path does not exist!")

    # Initialize UNet model
    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(f"{ckpt_filepath}", map_location=device)
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=True)
    print(f"checkpoints {ckpt_filepath} loaded.")

    num_downsample_level = 1
    if isinstance(args.diffusion_unet_def["num_channels"], list):
        num_downsample_level = max(num_downsample_level, len(args.diffusion_unet_def["num_channels"]))
    elif isinstance(args.diffusion_unet_def["attention_levels"], list):
        num_downsample_level = max(num_downsample_level, len(args.diffusion_unet_def["attention_levels"]))
    print(f"num_downsample_level -> {num_downsample_level}.")
    divisor = 2 ** (num_downsample_level - 2)
    print(f"divisor -> {divisor}.")

    num_train_timesteps = checkpoint["num_train_timesteps"]
    print(f"num_train_timesteps -> {num_train_timesteps}.")

    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    scale_factor = checkpoint["scale_factor"]
    print(f"scale_factor -> {scale_factor}.")

    autoencoder.eval()
    unet.eval()

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            noise = torch.randn(
                (
                    1,
                    model_config["latent_channels"],
                    output_size[0] // divisor,
                    output_size[1] // divisor,
                    output_size[2] // divisor,
                )
            ).to(device)
            print("noise:", noise.device, noise.dtype, type(noise))

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

            top_region_index_tensor = torch.from_numpy(top_region_index_tensor).half().to(device)
            bottom_region_index_tensor = torch.from_numpy(bottom_region_index_tensor).half().to(device)
            spacing_tensor = torch.from_numpy(spacing_tensor).half().to(device)

            image = noise

            # synthesize latents
            for t in tqdm(noise_scheduler.timesteps, ncols=110):
                model_output = unet(
                    x=image,
                    timesteps=torch.Tensor((t,)).to(device),
                    top_region_index_tensor=top_region_index_tensor,
                    bottom_region_index_tensor=bottom_region_index_tensor,
                    spacing_tensor=spacing_tensor,
                )
                image, _ = noise_scheduler.step(model_output, t, image)

            synthetic_images = sliding_window_inference(
                inputs=image,
                roi_size=(
                    min(output_size[0] // divisor // 4 * 3, 96),
                    min(output_size[1] // divisor // 4 * 3, 96),
                    min(output_size[2] // divisor // 4 * 3, 96),
                ),
                sw_batch_size=1,
                predictor=recon_model,
                mode="gaussian",
                overlap=2.0 / 3.0,
                sw_device=device,
                device=device,
            )

            data = synthetic_images.squeeze().cpu().detach().numpy()
            data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
            data = np.clip(data, a_min, a_max)
            data = np.int16(data)

            out_affine = np.eye(4)
            for _k in range(3):
                out_affine[_k, _k] = out_spacing[_k]
            new_image = nib.Nifti1Image(data, affine=out_affine)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"{args.output_dir}/{output_prefix}_seed{random_seed}_size{output_size[0]:d}x{output_size[1]:d}x{output_size[2]:d}_spacing{out_spacing[0]:.2f}x{out_spacing[1]:.2f}x{out_spacing[2]:.2f}_{timestamp}.nii.gz"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(new_image, output_path)
        print(f"Saved {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument("--env_config", type=str, required=True, help="Path to environment configuration file")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model configuration file")

    args = parser.parse_args()

    diff_model_infer(args.env_config, args.model_config)
