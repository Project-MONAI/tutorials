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

import random
from datetime import datetime
import json
import logging
import os
import sys
import argparse
from pathlib import Path


import monai
import torch
from monai.transforms import Compose, SaveImage
from monai.data import MetaTensor
from monai.utils import set_determinism
from tqdm import tqdm
from generative.metrics import FIDMetric
from generative.inferers import LatentDiffusionInferer
from utils import binarize_labels, MapLabelValue, general_mask_generation_post_process, get_body_region_index_from_mask, define_instance, load_autoencoder_ckpt
from find_masks import find_masks
from augmentation import augmentation
from sample import LDMSampler, check_input

def main():
    parser = argparse.ArgumentParser(description="PyTorch Latent Diffusion Model Inference")
    parser.add_argument(
        "-c",
        "--config-file",
        default="../configs/config_maisi.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument(
        "-e",
        "--environment-file",
        default="../configs/environment.json",
        help="environment json file that stores environment path",
    )
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))
    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    Path(args.output_dir).mkdir(exist_ok=True)

    print("Check input")
    check_input(args.body_region,args.anatomy_list,args.label_dict_json,args.output_size,args.spacing,args.controllable_anatomy_size)

    print("Start loading networks.")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = load_autoencoder_ckpt(args.trained_autoencoder_path)
    autoencoder.load_state_dict(checkpoint_autoencoder)

    diffusion_unet = define_instance(args, "difusion_unet_def").to(device)
    checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path)
    diffusion_unet.load_state_dict(checkpoint_diffusion_unet['unet_state_dict'])

    controlnet = define_instance(args, "controlnet_def").to(device)
    monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
    checkpoint_controlnet = torch.load(args.trained_controlnet_path)
    controlnet.load_state_dict(checkpoint_controlnet['controlnet_state_dict'], strict=True)
    scale_factor = checkpoint_controlnet['scale_factor'].to(device)

    mask_generation_autoencoder = define_instance(args, "mask_generation_autoencoder_def").to(device)
    checkpoint_mask_generation_autoencoder = torch.load(args.trained_mask_generation_autoencoder_path)
    mask_generation_autoencoder.load_state_dict(checkpoint_mask_generation_autoencoder, strict=True)
    
    mask_generation_diffusion_unet = define_instance(args, "mask_generation_diffusion_def").to(device)
    checkpoint_mask_generation_difusion_unet = torch.load(args.trained_mask_generation_diffusion_path)
    mask_generation_diffusion_unet.load_state_dict(checkpoint_mask_generation_difusion_unet, strict=True)
    
    
    print("Initialize MAISI sampler.")
    latent_shape = [args.latent_channels,
        args.output_size[0]//4,
        args.output_size[1]//4,
        args.output_size[2]//4
    ]
    ldm_sampler = LDMSampler(
        body_region=args.body_region,
        anatomy_list=args.anatomy_list,
        all_mask_files_json=args.all_mask_files_json,
        all_anatomy_size_condtions_json=args.all_anatomy_size_condtions_json,
        all_mask_files_base_dir=args.all_mask_files_base_dir,
        label_dict_json=args.label_dict_json,
        label_dict_remap_json=args.label_dict_remap_json,
        autoencoder=autoencoder,
        difusion_unet=diffusion_unet,
        controlnet=controlnet,
        scale_factor=scale_factor,
        noise_scheduler=args.noise_scheduler,
        mask_generation_autoencoder=mask_generation_autoencoder,
        mask_generation_difusion_unet=mask_generation_diffusion_unet,
        mask_generation_scale_factor=args.mask_generation_scale_factor,
        mask_generation_noise_scheduler=args.mask_generation_noise_scheduler,
        controllable_anatomy_size=args.controllable_anatomy_size,
        output_ext=args.output_ext,
        device=args.device,
        latent_shape=args.latent_shape,
        mask_generation_latent_shape=args.mask_generation_latent_shape,
        output_size=args.output_size,
        quality_check_args=args.quality_check_args,
        spacing=args.spacing,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        mask_generation_num_inference_steps=args.mask_generation_num_inference_steps,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
    
    
