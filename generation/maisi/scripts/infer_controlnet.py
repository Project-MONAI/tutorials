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
import logging
import os
import sys
from datetime import datetime

import torch
import torch.distributed as dist
from monai.data import MetaTensor, decollate_batch
from monai.networks.utils import copy_model_state
from monai.transforms import SaveImage
from monai.utils import RankFilter

from .sample import check_input, ldm_conditional_sample_one_image
from .utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="maisi.controlnet.infer")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./configs/environment_maisi_controlnet_train.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./configs/config_maisi.json",
        help="config json file that stores network hyper-parameters",
    )
    parser.add_argument(
        "-t",
        "--training-config",
        default="./configs/config_maisi_controlnet_train.json",
        help="config json file that stores training hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")

    args = parser.parse_args()

    # Step 0: configuration
    logger = logging.getLogger("maisi.controlnet.infer")
    # whether to use distributed data parallel
    use_ddp = args.gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World_size: {world_size}")

    with open(args.environment_file, "r") as env_file:
        env_dict = json.load(env_file)
    with open(args.config_file, "r") as config_file:
        config_dict = json.load(config_file)
    with open(args.training_config, "r") as training_config_file:
        training_config_dict = json.load(training_config_file)

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in training_config_dict.items():
        setattr(args, k, v)

    # Step 1: set data loader
    _, val_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
    )

    # Step 2: define AE, diffusion model and controlnet
    # define AE
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    # load trained autoencoder model
    if args.trained_autoencoder_path is not None:
        if not os.path.exists(args.trained_autoencoder_path):
            raise ValueError("Please download the autoencoder checkpoint.")
        autoencoder_ckpt = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(autoencoder_ckpt)
        logger.info(f"Load trained diffusion model from {args.trained_autoencoder_path}.")
    else:
        logger.info("trained autoencoder model is not loaded.")

    # define diffusion Model
    unet = define_instance(args, "diffusion_unet_def").to(device)
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    # load trained diffusion model
    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError("Please download the trained diffusion unet checkpoint.")
        diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
        unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"])
        # load scale factor from diffusion model checkpoint
        scale_factor = diffusion_model_ckpt["scale_factor"]
        logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")
        logger.info(f"loaded scale_factor from diffusion model ckpt -> {scale_factor}.")
    else:
        logger.info("trained diffusion model is not loaded.")
        scale_factor = 1.0
        logger.info(f"set scale_factor -> {scale_factor}.")

    # define ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    # copy weights from the DM to the controlnet
    copy_model_state(controlnet, unet.state_dict())
    # load trained controlnet model if it is provided
    if args.trained_controlnet_path is not None:
        if not os.path.exists(args.trained_controlnet_path):
            raise ValueError("Please download the trained ControlNet checkpoint.")
        controlnet.load_state_dict(
            torch.load(args.trained_controlnet_path, map_location=device, weights_only=False)["controlnet_state_dict"]
        )
        logger.info(f"load trained controlnet model from {args.trained_controlnet_path}")
    else:
        logger.info("trained controlnet is not loaded.")

    noise_scheduler = define_instance(args, "noise_scheduler")

    # Step 3: inference
    autoencoder.eval()
    controlnet.eval()
    unet.eval()

    for batch in val_loader:
        # get label mask
        labels = batch["label"].to(device)
        # get corresponding conditions
        if include_body_region:
            top_region_index_tensor = batch["top_region_index"].to(device)
            bottom_region_index_tensor = batch["bottom_region_index"].to(device)
        else:
            top_region_index_tensor = None
            bottom_region_index_tensor = None
        spacing_tensor = batch["spacing"].to(device)
        modality_tensor = args.controlnet_infer["modality"] * torch.ones((len(labels),), dtype=torch.long).to(device)
        out_spacing = tuple((batch["spacing"].squeeze().numpy() / 100).tolist())
        # get target dimension
        dim = batch["dim"]
        output_size = (dim[0].item(), dim[1].item(), dim[2].item())
        latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)
        # check if output_size and out_spacing are valid.
        check_input(None, None, None, output_size, out_spacing, None)
        # generate a single synthetic image using a latent diffusion model with controlnet.
        synthetic_images, _ = ldm_conditional_sample_one_image(
            autoencoder=autoencoder,
            diffusion_unet=unet,
            controlnet=controlnet,
            noise_scheduler=noise_scheduler,
            scale_factor=scale_factor,
            device=device,
            combine_label_or=labels,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
            modality_tensor=modality_tensor,
            latent_shape=latent_shape,
            output_size=output_size,
            noise_factor=1.0,
            num_inference_steps=args.controlnet_infer["num_inference_steps"],
            autoencoder_sliding_window_infer_size=args.controlnet_infer["autoencoder_sliding_window_infer_size"],
            autoencoder_sliding_window_infer_overlap=args.controlnet_infer["autoencoder_sliding_window_infer_overlap"],
        )
        # save image/label pairs
        labels = decollate_batch(batch)[0]["label"]
        output_postfix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        labels.meta["filename_or_obj"] = "sample.nii.gz"
        synthetic_images = MetaTensor(synthetic_images.squeeze(0), meta=labels.meta)
        img_saver = SaveImage(
            output_dir=args.output_dir,
            output_postfix=output_postfix + "_image",
            separate_folder=False,
        )
        img_saver(synthetic_images)
        label_saver = SaveImage(
            output_dir=args.output_dir,
            output_postfix=output_postfix + "_label",
            separate_folder=False,
        )
        label_saver(labels)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
