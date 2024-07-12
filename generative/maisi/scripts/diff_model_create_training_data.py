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
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist

import monai
from monai.transforms import Compose
from monai.utils import set_determinism

from utils import define_instance, load_autoencoder_ckpt

# Set the random seed for reproducibility
set_determinism(seed=0)


def create_transforms(dim: tuple = None) -> Compose:
    """
    Create a set of MONAI transforms for preprocessing.

    Args:
        dim (tuple, optional): New dimensions for resizing. Defaults to None.

    Returns:
        Compose: Composed MONAI transforms.
    """
    if dim:
        return Compose(
            [
                monai.transforms.LoadImaged(keys="image"),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.Orientationd(keys="image", axcodes="RAS"),
                monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
                monai.transforms.ScaleIntensityRanged(
                    keys="image",
                    a_min=-1000,
                    a_max=1000,
                    b_min=0,
                    b_max=1,
                    clip=True,
                ),
                monai.transforms.Resized(keys="image", spatial_size=dim, mode="trilinear"),
            ]
        )
    else:
        return Compose(
            [
                monai.transforms.LoadImaged(keys="image"),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            ]
        )


def round_number(number: int, base_number: int = 128) -> int:
    """
    Round the number to the nearest multiple of the base number, with a minimum value of the base number.

    Args:
        number (int): Number to be rounded.
        base_number (int): Number to be common divisor.

    Returns:
        int: Rounded number.
    """
    new_number = max(round(float(number) / float(base_number)), 1.0) * float(base_number)
    return int(new_number)


def diff_model_create_training_data(env_config_path: str, model_config_path: str) -> None:
    """
    Create training data for the diffusion model.

    Args:
        env_config (dict): Environment configuration.
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

    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    print(f"Using device {device}")

    # Load autoencoder if saving embeddings
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = load_autoencoder_ckpt(args.trained_autoencoder_path)
        autoencoder.load_state_dict(checkpoint_autoencoder)
    except:
        print("The trained_autoencoder_path does not exist!")

    if not os.path.exists(args.embedding_base_dir):
        os.makedirs(args.embedding_base_dir)

    with open(args.json_data_list, "r") as file:
        json_data = json.load(file)
    filenames_raw = json_data["training"]
    filenames_raw = [_item["image"] for _item in filenames_raw]
    print(f"filenames_raw: {filenames_raw}")

    plain_transforms = create_transforms(dim=None)

    for _iter in range(len(filenames_raw)):
        if _iter % world_size != local_rank:
            continue

        filepath = filenames_raw[_iter]
        out_filename_base = filepath.replace(".gz", "").replace(".nii", "")
        out_filename_base = os.path.join(args.embedding_base_dir, out_filename_base)
        out_filename = out_filename_base + f"_emb.nii.gz"

        if os.path.isfile(out_filename):
            continue

        test_data = {"image": os.path.join(args.data_base_dir, filepath)}
        transformed_data = plain_transforms(test_data)
        nda = transformed_data["image"]

        dim = nda.meta["dim"]
        dim = dim[1:4]
        dim = [int(dim[_i]) for _i in range(3)]

        spacing = nda.meta["pixdim"]
        spacing = spacing[1:4]
        spacing = [float(spacing[_i]) for _i in range(3)]

        print("old", dim, spacing)

        new_dim = [round_number(dim[_i]) for _i in range(3)]
        new_dim = tuple(new_dim)
        new_transforms = create_transforms(new_dim)

        new_data = new_transforms(test_data)
        nda_image = new_data["image"]

        new_affine = nda_image.meta["affine"]
        new_affine = new_affine.numpy()

        nda_image = nda_image.numpy().squeeze()
        print("new", nda_image.shape, new_affine)

        try:
            out_path = Path(out_filename)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"out_filename: {out_filename}")

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    pt_nda = torch.from_numpy(nda_image).float().to(device)
                    pt_nda.unsqueeze_(0).unsqueeze_(0)

                    z = autoencoder.encode_stage_2_inputs(pt_nda)
                    print(f"z: {z.size()}, {z.dtype}")

                    out_nda = z.squeeze().cpu().detach().numpy()
                    out_nda = out_nda.transpose(1, 2, 3, 0)
                    out_img = nib.Nifti1Image(np.float32(out_nda), affine=new_affine)
                    nib.save(out_img, out_filename)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training Data Creation")
    parser.add_argument("--env_config", type=str, required=True, help="Path to environment configuration file")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model configuration file")

    args = parser.parse_args()
    diff_model_create_training_data(args.env_config, args.model_config)
