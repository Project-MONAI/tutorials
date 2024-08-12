# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# &nbsp;&nbsp;&nbsp;&nbsp;http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# # MAISI Inference Script
import argparse
import json
import logging
import os
import tempfile
import sys

import monai
import torch
from monai.apps import download_url
from monai.config import print_config
from monai.transforms import LoadImage, Orientation
from monai.utils import set_determinism
from scripts.sample import LDMSampler, check_input
from scripts.utils import define_instance, load_autoencoder_ckpt, load_diffusion_ckpt
from scripts.utils_plot import find_label_center_loc, get_xyz_plot, show_image


def main():
    parser = argparse.ArgumentParser(description="maisi.controlnet.training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./configs/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./configs/config_maisi.json",
        help="config json file that stores network hyper-parameters",
    )
    parser.add_argument(
        "-i",
        "--inference-file",
        default="./configs/config_infer.json",
        help="config json file that stores inference hyper-parameters",
    )
    parser.add_argument(
        "-s",
        "--random-seed",
        default=None,
        help="random seed, can be None or int",
    )
    args = parser.parse_args()
    # Step 0: configuration
    logger = logging.getLogger("maisi.inference")

    # ## Set deterministic training for reproducibility
    if args.random_seed is not None:
        set_determinism(seed=args.random_seed)

    # ## Setup data directory
    # You can specify a directory with the `MONAI_DATA_DIRECTORY` environment variable.
    # This allows you to save results and reuse downloads.
    # If not specified a temporary directory will be used.

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    files = [
        {
            "path": "models/autoencoder_epoch273.pt",
            "url": "https://drive.google.com/file/d/1jQefG0yJPzSvTG5rIJVHNqDReBTvVmZ0/view?usp=drive_link",
        },
        {
            "path": "models/input_unet3d_data-all_steps1000size512ddpm_random_current_inputx_v1.pt",
            "url": "https://drive.google.com/file/d/1FtOHBGUF5dLZNHtiuhf5EH448EQGGs-_/view?usp=sharing",
        },
        {
            "path": "models/controlnet-20datasets-e20wl100fold0bc_noi_dia_fsize_current.pt",
            "url": "https://drive.google.com/file/d/1izr52Whkk56OevNTk2QzI86eJV9TTaLk/view?usp=sharing",
        },
        {
            "path": "models/mask_generation_autoencoder.pt",
            "url": "https://drive.google.com/file/d/1FzWrpv6ornYUaPiAWGOOxhRx2P9Wnynm/view?usp=drive_link",
        },
        {
            "path": "models/mask_generation_diffusion_unet.pt",
            "url": "https://drive.google.com/file/d/11SA9RUZ6XmCOJr5v6w6UW1kDzr6hlymw/view?usp=drive_link",
        },
        {
            "path": "configs/candidate_masks_flexible_size_and_spacing_3000.json",
            "url": "https://drive.google.com/file/d/1yMkH-lrAsn2YUGoTuVKNMpicziUmU-1J/view?usp=sharing",
        },
        {
            "path": "configs/all_anatomy_size_condtions.json",
            "url": "https://drive.google.com/file/d/1AJyt1DSoUd2x2AOQOgM7IxeSyo4MXNX0/view?usp=sharing",
        },
        {
            "path": "datasets/all_masks_flexible_size_and_spacing_3000.zip",
            "url": "https://drive.google.com/file/d/16MKsDKkHvDyF2lEir4dzlxwex_GHStUf/view?usp=sharing",
        },
    ]

    for file in files:
        file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(root_dir, file["path"])
        download_url(url=file["url"], filepath=file["path"])

    # ## Read in environment setting, including data directory, model directory, and output directory
    # The information for data directory, model directory, and output directory are saved in ./configs/environment.json
    env_dict = json.load(open(args.environment_file, "r"))
    for k, v in env_dict.items():
        # Update the path to the downloaded dataset in MONAI_DATA_DIRECTORY
        val = v if "datasets/" not in v else os.path.join(root_dir, v)
        setattr(args, k, val)
        print(f"{k}: {val}")
    print("Global config variables have been loaded.")

    # ## Read in configuration setting, including network definition, body region and anatomy to generate, etc.
    #
    # The information used for both training and inference, like network definition, is stored in "./configs/config_maisi.json". Training and inference should use the same "./configs/config_maisi.json".
    #
    # The information for the inference input, like body region and anatomy to generate, is stored in "./configs/config_infer.json". Please feel free to play with it.
    # - `"num_output_samples"`: int, the number of output image/mask pairs it will generate.
    # - `"spacing"`: voxel size of generated images. E.g., if set to `[1.5, 1.5, 2.0]`, it will generate images with a resolution of 1.5x1.5x2.0 mm.
    # - `"output_size"`: volume size of generated images. E.g., if set to `[512, 512, 256]`, it will generate images with size of 512x512x256. They need to be divisible by 16. If you have a small GPU memory size, you should adjust it to small numbers.
    # - `"controllable_anatomy_size"`: a list of controllable anatomy and its size scale (0--1). E.g., if set to `[["liver", 0.5],["hepatic tumor", 0.3]]`, the generated image will contain liver that have a median size, with size around 50% percentile, and hepatic tumor that is relatively small, with around 30% percentile. The output will contain paired image and segmentation mask for the controllable anatomy.
    # - `"body_region"`: If "controllable_anatomy_size" is not specified, "body_region" will be used to constrain the region of generated images. It needs to be chosen from "head", "chest", "thorax", "abdomen", "pelvis", "lower".
    # - `"anatomy_list"`: If "controllable_anatomy_size" is not specified, the output will contain paired image and segmentation mask for the anatomy in "./configs/label_dict.json".
    # - `"autoencoder_sliding_window_infer_size"`: in order to save GPU memory, we use sliding window inference when decoding latents to image when `"output_size"` is large. This is the patch size of the sliding window. Small value will reduce GPU memory but increase time cost. They need to be divisible by 16.
    # - `"autoencoder_sliding_window_infer_overlap"`: float between 0 and 1. Large value will reduce the stitching artifacts when stitching patches during sliding window inference, but increase time cost. If you do not observe seam lines in the generated image result, you can use a smaller value to save inference time.
    config_dict = json.load(open(args.config_file, "r"))
    for k, v in config_dict.items():
        setattr(args, k, v)

    # check the format of inference inputs
    config_infer_dict = json.load(open(args.inference_file, "r"))
    for k, v in config_infer_dict.items():
        setattr(args, k, v)
        print(f"{k}: {v}")

    check_input(
        args.body_region,
        args.anatomy_list,
        args.label_dict_json,
        args.output_size,
        args.spacing,
        args.controllable_anatomy_size,
    )
    latent_shape = [args.latent_channels, args.output_size[0] // 4, args.output_size[1] // 4, args.output_size[2] // 4]
    print("Network definition and inference inputs have been loaded.")

    # ## Initialize networks and noise scheduler, then load the trained model weights.
    # The networks and noise scheduler are defined in `config_file`. We will read them in and load the model weights.
    noise_scheduler = define_instance(args, "noise_scheduler")
    mask_generation_noise_scheduler = define_instance(args, "mask_generation_noise_scheduler")

    device = torch.device("cuda")

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = load_autoencoder_ckpt(args.trained_autoencoder_path)
    autoencoder.load_state_dict(checkpoint_autoencoder)

    diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path)
    new_dict = load_diffusion_ckpt(diffusion_unet.state_dict(), checkpoint_diffusion_unet["unet_state_dict"])
    diffusion_unet.load_state_dict(new_dict, strict=True)
    scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)

    controlnet = define_instance(args, "controlnet_def").to(device)
    checkpoint_controlnet = torch.load(args.trained_controlnet_path)
    monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
    controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"], strict=True)

    mask_generation_autoencoder = define_instance(args, "mask_generation_autoencoder_def").to(device)
    checkpoint_mask_generation_autoencoder = load_autoencoder_ckpt(args.trained_mask_generation_autoencoder_path)
    mask_generation_autoencoder.load_state_dict(checkpoint_mask_generation_autoencoder)

    mask_generation_diffusion_unet = define_instance(args, "mask_generation_diffusion_def").to(device)
    checkpoint_mask_generation_diffusion_unet = torch.load(args.trained_mask_generation_diffusion_path)
    mask_generation_diffusion_unet.load_old_state_dict(checkpoint_mask_generation_diffusion_unet)
    mask_generation_scale_factor = args.mask_generation_scale_factor

    print("All the trained model weights have been loaded.")

    # ## Define the LDM Sampler, which contains functions that will perform the inference.
    ldm_sampler = LDMSampler(
        args.body_region,
        args.anatomy_list,
        args.all_mask_files_json,
        args.all_anatomy_size_conditions_json,
        args.all_mask_files_base_dir,
        args.label_dict_json,
        args.label_dict_remap_json,
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
        args.mask_generation_latent_shape,
        args.output_size,
        args.output_dir,
        args.controllable_anatomy_size,
        image_output_ext=args.image_output_ext,
        label_output_ext=args.label_output_ext,
        spacing=args.spacing,
        num_inference_steps=args.num_inference_steps,
        mask_generation_num_inference_steps=args.mask_generation_num_inference_steps,
        random_seed=args.random_seed,
        autoencoder_sliding_window_infer_size=args.autoencoder_sliding_window_infer_size,
        autoencoder_sliding_window_infer_overlap=args.autoencoder_sliding_window_infer_overlap,
    )

    print(f"The generated image/mask pairs will be saved in {args.output_dir}.")
    output_filenames = ldm_sampler.sample_multiple_images(args.num_output_samples)
    print("MAISI image/mask generation finished")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
