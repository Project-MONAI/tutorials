#!/usr/bin/env python

import copy
import json
import monai
import nibabel as nib
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

from custom_network_tp import AutoencoderKLCKModified_TP
from monai.transforms import Compose
from monai.utils import first, set_determinism
from pathlib import Path


torch.manual_seed(0)
np.random.seed(0)
set_determinism(seed=0)

############
save_embedding = True
# if False:
#     dataroot = "/lustre/fsw/portfolios/convai/projects/convai_monai_genai3dct/datasets/dongy/data/data_128"
#     filenames_filepath = "/lustre/fsw/portfolios/convai/projects/convai_monai_genai3dct/datasets/dongy/exps/temp3/filenames_image_nii_v1_1.txt"
#     output_root_embedding = f"/lustre/fsw/portfolios/convai/projects/convai_monai_genai3dct/datasets/dongy/data/encoding_128"
#     autoencoder_root = "/lustre/fsw/portfolios/convai/projects/convai_monai_genai3dct/datasets/dongy/from_canz"
# else:
#     dataroot = "/mnt/drive2/data_128"
#     filenames_filepath = (
#         "/workspace/monai/generative/utils/lists/filenames_image_nii_v1.txt"
#     )
#     output_root_embedding = f"/mnt/drive2/encoding_128"
#     autoencoder_root = "/workspace/monai/generative/from_canz"

dataroot = "/mnt/drive2/data_128"
filenames_filepath = "/mnt/drive2/data_128/filenames_image_nii_autopet.txt"
output_root_embedding = "/mnt/drive2/encoding_128"
autoencoder_root = "/workspace/monai/generative/from_canz"
############

# rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
# local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
# world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
# device = torch.device("cuda", local_rank)
# torch.cuda.set_device(device)
# print(f"Using {device} of {world_size}")

dist.init_process_group(backend="nccl", init_method="env://")
local_rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

print(f"Using {device}.")
print(f"world_size -> {world_size}.")

if save_embedding:
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

    checkpoint_autoencoder = torch.load(
        os.path.join(autoencoder_root, "autoencoder_epoch273.pt"),
        map_location=torch.device(f"cuda:{local_rank}"),
    )
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
    print("checkpoints loaded.")

if local_rank == 0:
    if not os.path.exists(output_root_embedding):
        os.makedirs(output_root_embedding)

if filenames_filepath.endswith(".txt"):
    with open(filenames_filepath, "r") as file:
        lines = file.readlines()
    filenames_raw = [_item.strip() for _item in lines]
else:
    with open(filenames_filepath, "r") as file:
        data = json.load(file)
    filenames_raw = data

transforms = Compose(
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

for _iter in range(len(filenames_raw)):
    if _iter % world_size != local_rank:
        continue

    filepath = filenames_raw[_iter]

    out_filename_base = filepath.replace("_image.nii.gz", "")
    out_filename_base = os.path.join(output_root_embedding, out_filename_base)
    # print(f"out_filename_base: {out_filename_base}")

    out_filename = out_filename_base + f"_emb.nii.gz"
    if os.path.isfile(out_filename):
        continue
    else:
        print(f"{out_filename} does not exist.")

    test_data = {"image": os.path.join(dataroot, filepath)}
    transformed_data = transforms(test_data)
    nda = transformed_data["image"]

    spacing = nda.meta["pixdim"]
    spacing = spacing[1:4]
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

                    # if mode == 'nearest':
                    #     resized_pt_nda = F.interpolate(pt_nda, size=target_shape, mode='nearest')
                    # elif mode == 'trilinear':
                    #     resized_pt_nda = F.interpolate(pt_nda, size=target_shape, mode='trilinear', align_corners=False)

                    z = autoencoder.encode_stage_2_inputs(pt_nda)
                    print(f"z: {z.size()}, {z.dtype}, {z.is_cuda} {1 / torch.std(z)} {torch.mean(z)}")

                    out_nda = z.squeeze().cpu().detach().numpy()
                    # print(f'out_nda: {out_nda.shape} {out_nda.dtype}')
                    out_nda = out_nda.transpose(1, 2, 3, 0)
                    out_img = nib.Nifti1Image(np.float32(out_nda), affine=affine)
                    nib.save(out_img, out_filename)
        except:
            print("something wrong")
