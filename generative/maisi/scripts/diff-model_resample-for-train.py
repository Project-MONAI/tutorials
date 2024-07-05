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

import copy
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.transforms import Compose
from monai.utils import set_determinism

from custom_network_tp import AutoencoderKLCKModified_TP

# Set the random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
set_determinism(seed=0)

# Configuration parameters
save_embedding = True
dataroot = "/mnt/drive2/data_128"
filenames_filepath = "/mnt/drive2/data_128/filenames_image_nii_autopet.txt"
output_root_embedding = "/mnt/drive2/encoding_128"
autoencoder_root = "/workspace/monai/generative/from_canz"

# Initialize distributed processing
dist.init_process_group(backend="nccl", init_method="env://")
local_rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

print(f"Using {device}.")
print(f"world_size -> {world_size}.")

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
            new_key = k.replace("conv.weight", "conv.conv.weight") if "conv.weight" in k else k.replace("conv.bias", "conv.conv.bias")
        elif "encoder" in k and "conv" in k:
            new_key = k.replace("conv.weight", "conv.conv.weight") if "conv.weight" in k else k.replace("conv.bias", "conv.conv.bias")
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

def create_transforms():
    """
    Create a set of MONAI transforms for preprocessing.

    Returns:
        Compose: Composed MONAI transforms.
    """
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

def main():
    # Load autoencoder if saving embeddings
    autoencoder = None
    if save_embedding:
        autoencoder = load_autoencoder(autoencoder_root, device)

    if local_rank == 0:
        if not os.path.exists(output_root_embedding):
            os.makedirs(output_root_embedding)

    filenames_raw = get_filenames(filenames_filepath)
    transforms = create_transforms()

    for _iter in range(len(filenames_raw)):
        if _iter % world_size != local_rank:
            continue

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

        if save_embedding:
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

if __name__ == "__main__":
    main()
