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
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import set_determinism

from utils import define_instance


def main():
    parser = argparse.ArgumentParser(description="PyTorch Latent Diffusion Model Inference")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1,
        help="number of generated images",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    # load trained networks
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
    autoencoder.load_state_dict(torch.load(trained_g_path))

    diffusion_model = define_instance(args, "diffusion_def").to(device)
    trained_diffusion_path = os.path.join(args.model_dir, "diffusion_unet.pt")
    diffusion_model.load_state_dict(torch.load(trained_diffusion_path))

    scheduler = DDPMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
    )
    inferer = LatentDiffusionInferer(scheduler, scale_factor=1.0)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    latent_shape = [p // 4 for p in args.diffusion_train["patch_size"]]
    noise_shape = [1, args.latent_channels] + latent_shape

    for _ in range(args.num):
        noise = torch.randn(noise_shape, dtype=torch.float32).to(device)
        with torch.no_grad():
            synthetic_images = inferer.sample(
                input_noise=noise,
                autoencoder_model=autoencoder,
                diffusion_model=diffusion_model,
                scheduler=scheduler,
            )
        filename = os.path.join(args.output_dir, datetime.now().strftime("synimg_%Y%m%d_%H%M%S"))
        final_img = nib.Nifti1Image(synthetic_images[0, 0, ...].unsqueeze(-1).cpu().numpy(), np.eye(4))
        nib.save(final_img, filename)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
