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

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import monai
from monai.transforms import Compose
from monai.utils import set_determinism

from custom_network_tp import AutoencoderKLCKModified_TP

# Set the random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
set_determinism(seed=0)


def load_autoencoder(autoencoder_root, device):
    """
    Load the autoencoder model and its checkpoints.

    Args:
        autoencoder_root (str): Path to the autoencoder root directory.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: Loaded autoencoder model.
    """
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

    checkpoint_path = os.path.join(autoencoder_root, "autoencoder_epoch273.pt")
    checkpoint_autoencoder = torch.load(checkpoint_path, map_location=device)
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
    print("checkpoints loaded.")
    return autoencoder


def get_filenames(filenames_filepath):
    """
    Get the list of filenames from the provided filepath.

    Args:
        filenames_filepath (str): Path to the filenames file.

    Returns:
        list: List of filenames.
    """
    if filenames_filepath.endswith(".txt"):
        with open(filenames_filepath, "r") as file:
            lines = file.readlines()
        filenames_raw = [_item.strip() for _item in lines]
    else:
        with open(filenames_filepath, "r") as file:
            data = json.load(file)
        filenames_raw = data
    return filenames_raw


def create_transforms(dim=None):
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
                monai.transforms.LoadImaged(keys=["image", "label"]),
                monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
                monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                monai.transforms.EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.short]),
                monai.transforms.Resized(
                    keys=["image", "label"],
                    spatial_size=dim,
                    mode=["trilinear", "nearest"],
                ),
            ]
        )
    else:
        return Compose(
            [
                monai.transforms.LoadImaged(keys="image"),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.Orientationd(keys="image", axcodes="RAS"),
                monai.transforms.ScaleIntensityRanged(
                    keys="image",
                    a_min=-1000,
                    a_max=1000,
                    b_min=0,
                    b_max=1,
                    clip=True,
                ),
                monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
            ]
        )


def round_number(number):
    """
    Round the number to the nearest multiple of 128, with a minimum value of 128.

    Args:
        number (float): Number to be rounded.

    Returns:
        int: Rounded number.
    """
    new_number = max(round(float(number) / 128.0), 1.0) * 128.0
    return int(new_number)


def process_file(filepath, dataroot, output_dir, pl_root, transforms):
    """
    Process a single file for data transformation and saving.

    Args:
        filepath (str): Filepath to process.
        dataroot (str): Root directory for the data.
        output_dir (str): Output directory for the transformed data.
        pl_root (str): Root directory for the pseudo labels.
        transforms (Compose): MONAI transforms to apply.
    """
    print(f"Processing: {filepath}")

    out_filepath_base = os.path.join(output_dir, filepath.replace(".gz", "").replace(".nii", ""))
    if os.path.isfile(out_filepath_base + "_image.nii.gz") and os.path.isfile(out_filepath_base + "_label.nii.gz"):
        return

    test_data = {"image": os.path.join(dataroot, filepath)}

    data = transforms(test_data)
    nda = data["image"]

    dim = nda.meta["dim"]
    dim = dim[1:4]
    dim = [int(dim[_i]) for _i in range(3)]

    spacing = nda.meta["pixdim"]
    spacing = spacing[1:4]
    spacing = [float(spacing[_i]) for _i in range(3)]

    print(dim, spacing)

    new_dim = [round_number(dim[_i]) for _i in range(3)]
    new_dim = tuple(new_dim)
    new_transforms = create_transforms(new_dim)

    new_test_data = {
        "image": os.path.join(dataroot, filepath),
        "label": os.path.join(pl_root, filepath),
    }

    new_data = new_transforms(new_test_data)
    nda_image = new_data["image"]
    nda_label = new_data["label"]

    affine = nda_image.meta["affine"]
    affine = affine.numpy()

    print("new", nda_image.size(), nda_label.size(), affine)

    nda_image = nda_image.numpy().squeeze().astype(np.int16)
    out_filename = out_filepath_base + f"_image.nii.gz"
    out_path = Path(out_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(nda_image, affine=affine)
    nib.save(out_img, out_filename)
    print(f"out_filename: {out_filename}")

    nda_label = nda_label.numpy().squeeze().astype(np.uint8)
    out_filename = out_filepath_base + f"_label.nii.gz"
    out_path = Path(out_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(nda_label, affine=affine)
    nib.save(out_img, out_filename)
    print(f"out_filename: {out_filename}")

    return f"Finished {filepath}"


def create_training_data(
    dataroot, filenames_filepath, output_root_embedding, autoencoder_root, list_filepath, output_dir, pl_root
):
    # Load autoencoder if saving embeddings
    autoencoder = load_autoencoder(autoencoder_root, device)

    if not os.path.exists(output_root_embedding):
        os.makedirs(output_root_embedding)

    filenames_raw = get_filenames(filenames_filepath)
    transforms = create_transforms()

    for _iter in range(len(filenames_raw)):
        filepath = filenames_raw[_iter]
        out_filename_base = filepath.replace("_image.nii.gz", "")
        out_filename_base = os.path.join(output_root_embedding, out_filename_base)
        out_filename = out_filename_base + f"_emb.nii.gz"

        if os.path.isfile(out_filename):
            continue
        else:
            print(f"{out_filename} does not exist.")

        test_data = {"image": os.path.join(dataroot, filepath)}
        transformed_data = transforms(test_data)
        nda = transformed_data["image"]

        spacing = nda.meta["pixdim"][1:4]
        spacing = [float(spacing[_i]) for _i in range(3)]

        nda = nda.numpy().squeeze()
        print(f"nda: {nda.shape} {nda.dtype} {np.amax(nda)} {np.amin(nda)}")

        affine = np.eye(4)
        for _s in range(3):
            affine[_s, _s] = spacing[_s]

        try:
            out_path = Path(out_filename)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"out_filename: {out_filename}")

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    pt_nda = torch.from_numpy(nda).float().to(device)
                    pt_nda.unsqueeze_(0).unsqueeze_(0)

                    z = autoencoder.encode_stage_2_inputs(pt_nda)
                    print(f"z: {z.size()}, {z.dtype}, {z.is_cuda} {1 / torch.std(z)} {torch.mean(z)}")

                    out_nda = z.squeeze().cpu().detach().numpy()
                    out_nda = out_nda.transpose(1, 2, 3, 0)
                    out_img = nib.Nifti1Image(np.float32(out_nda), affine=affine)
                    nib.save(out_img, out_filename)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    with open(list_filepath, "r") as file:
        filepaths = file.readlines()
    filepaths = [_item.strip() for _item in filepaths]

    for filepath in filepaths:
        process_file(filepath, dataroot, output_dir, pl_root, transforms)


if __name__ == "__main__":
    dataroot = "/dataroot"
    filenames_filepath = "/filenames_image_nii.txt"
    output_root_embedding = "/dataroot/encoding_128"
    autoencoder_root = "/workspace/monai/generative/from_canz"
    list_filepath = "/dataroot/filenames_nii_common.txt"
    output_dir = "/mnt/drive2/data_128"
    pl_root = "/mnt/drive2/V2_pseudo_12Feb2024"

    create_training_data(
        dataroot, filenames_filepath, output_root_embedding, autoencoder_root, list_filepath, output_dir, pl_root
    )
