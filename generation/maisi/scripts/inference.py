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
import sys
import tempfile

import monai
import torch
from monai.apps import download_url
from monai.config import print_config
from monai.transforms import LoadImage, Orientation
from monai.utils import set_determinism

from scripts.sample import LDMSampler, check_input
from scripts.utils import define_instance
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
        "-x",
        "--extra-config-file",
        default=None,
        help="config json file that stores inference extra parameters",
    )
    parser.add_argument(
        "-s",
        "--random-seed",
        default=None,
        help="random seed, can be None or int",
    )
    parser.add_argument(
        "--version",
        default="maisi3d-rflow",
        type=str,
        help="maisi_version, choose from ['maisi3d-ddpm', 'maisi3d-rflow']",
    )
    args = parser.parse_args()
    # Step 0: configuration
    logger = logging.getLogger("maisi.inference")

    maisi_version = args.version

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

    # TODO: remove the `files` after the files are uploaded to the NGC
    files = [
        {
            "path": "models/autoencoder_epoch273.pt",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials"
            "/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt",
        },
        {
            "path": "models/mask_generation_autoencoder.pt",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai"
            "/tutorials/mask_generation_autoencoder.pt",
        },
        {
            "path": "models/mask_generation_diffusion_unet.pt",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai"
            "/tutorials/model_zoo/model_maisi_mask_generation_diffusion_unet_v2.pt",
        },
        {
            "path": "configs/all_anatomy_size_condtions.json",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/all_anatomy_size_condtions.json",
        },
        {
            "path": "datasets/all_masks_flexible_size_and_spacing_4000.zip",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai"
            "/tutorials/all_masks_flexible_size_and_spacing_4000.zip",
        },
    ]

    if maisi_version == "maisi3d-ddpm":
        files += [
            {
                "path": "models/diff_unet_3d_ddpm.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo"
                "/model_maisi_input_unet3d_data-all_steps1000size512ddpm_random_current_inputx_v1_alternative.pt",
            },
            {
                "path": "models/controlnet_3d_ddpm.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo"
                "/model_maisi_controlnet-20datasets-e20wl100fold0bc_noi_dia_fsize_current_alternative.pt",
            },
            {
                "path": "configs/candidate_masks_flexible_size_and_spacing_3000.json",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai"
                "/tutorials/candidate_masks_flexible_size_and_spacing_3000.json",
            },
        ]
    elif maisi_version == "maisi3d-rflow":
        files += [
            {
                "path": "models/diff_unet_3d_rflow.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/"
                "diff_unet_ckpt_rflow_epoch19350.pt",
            },
            {
                "path": "models/controlnet_3d_rflow.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/"
                "controlnet_rflow_epoch60.pt",
            },
            {
                "path": "configs/candidate_masks_flexible_size_and_spacing_4000.json",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai"
                "/tutorials/candidate_masks_flexible_size_and_spacing_4000.json",
            },
        ]
    else:
        raise ValueError(
            f"maisi_version has to be chosen from ['maisi3d-ddpm', 'maisi3d-rflow'], yet got {maisi_version}."
        )

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
    # The information for the inference input, like body region and anatomy to generate, is stored in "./configs/config_infer.json".
    # Please refer to README.md for the details.
    config_dict = json.load(open(args.config_file, "r"))
    for k, v in config_dict.items():
        setattr(args, k, v)

    # check the format of inference inputs
    config_infer_dict = json.load(open(args.inference_file, "r"))
    # override num_split if asked
    if "autoencoder_tp_num_splits" in config_infer_dict:
        args.autoencoder_def["num_splits"] = config_infer_dict["autoencoder_tp_num_splits"]
        args.mask_generation_autoencoder_def["num_splits"] = config_infer_dict["autoencoder_tp_num_splits"]
    for k, v in config_infer_dict.items():
        setattr(args, k, v)
        print(f"{k}: {v}")

    #
    # ## Read in optional extra configuration setting - typically acceleration options (TRT)
    #
    #
    if args.extra_config_file is not None:
        extra_config_dict = json.load(open(args.extra_config_file, "r"))
        for k, v in extra_config_dict.items():
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

    autoencoder = define_instance(args, "autoencoder").to(device)
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
    autoencoder.load_state_dict(checkpoint_autoencoder)

    diffusion_unet = define_instance(args, "diffusion_unet").to(device)
    checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path, weights_only=False)
    diffusion_unet.load_state_dict(checkpoint_diffusion_unet["unet_state_dict"], strict=True)
    scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)

    controlnet = define_instance(args, "controlnet").to(device)
    checkpoint_controlnet = torch.load(args.trained_controlnet_path, weights_only=False)
    monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
    controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"], strict=True)

    mask_generation_autoencoder = define_instance(args, "mask_generation_autoencoder").to(device)
    checkpoint_mask_generation_autoencoder = torch.load(
        args.trained_mask_generation_autoencoder_path, weights_only=True
    )
    mask_generation_autoencoder.load_state_dict(checkpoint_mask_generation_autoencoder)

    mask_generation_diffusion_unet = define_instance(args, "mask_generation_diffusion").to(device)
    checkpoint_mask_generation_diffusion_unet = torch.load(
        args.trained_mask_generation_diffusion_path, weights_only=False
    )
    mask_generation_diffusion_unet.load_state_dict(checkpoint_mask_generation_diffusion_unet["unet_state_dict"])
    mask_generation_scale_factor = checkpoint_mask_generation_diffusion_unet["scale_factor"]

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
        modality=args.modality,
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
    torch.cuda.reset_peak_memory_stats()
    main()
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
    print(f"Peak GPU memory usage: {peak_memory_gb:.2f} GB")
