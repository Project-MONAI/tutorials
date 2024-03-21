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

import numpy as np
import torch
import warnings
from fastmri_ssim import skimage_ssim

from monai.transforms import (
    Compose,
    SpatialCrop,
    LoadImaged,
    EnsureTyped,
)

from monai.apps.reconstruction.transforms.dictionary import (
    ExtractDataKeyFromMetaKeyd,
    RandomKspaceMaskd,
    EquispacedKspaceMaskd,
)

from monai.apps.reconstruction.fastmri_reader import FastMRIReader
from monai.apps.reconstruction.networks.nets.varnet import VariationalNetworkModel
from monai.apps.reconstruction.networks.nets.complex_unet import ComplexUnet
from monai.apps.reconstruction.networks.nets.coil_sensitivity_model import CoilSensitivityModel
from monai.losses.ssim_loss import SSIMLoss

from pathlib import Path
import argparse
from monai.data import CacheDataset, DataLoader, decollate_batch
from torch.utils.tensorboard import SummaryWriter

import logging
import os
import sys
from datetime import datetime
import time
from collections import defaultdict
import random

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

warnings.filterwarnings("ignore")


def trainer(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    outpath = os.path.join(args.exp_dir, args.exp)
    Path(outpath).mkdir(parents=True, exist_ok=True)  # create output directory to store model checkpoints
    now = datetime.now()
    date = now.strftime("%m-%d-%y_%H-%M")
    writer = SummaryWriter(
        outpath + "/" + date
    )  # create a date directory within the output directory for storing training logs

    # create training-validation data loaders
    train_files = list(Path(args.data_path_train).iterdir())
    random.shuffle(train_files)
    train_files = train_files[
        : int(args.sample_rate * len(train_files))
    ]  # select a subset of the data according to sample_rate
    train_files = [dict([("kspace", train_files[i])]) for i in range(len(train_files))]
    print(f"#train files: {len(train_files)}")

    val_files = list(Path(args.data_path_val).iterdir())
    random.shuffle(val_files)
    val_files = val_files[
        : int(args.sample_rate * len(val_files))
    ]  # select a subset of the data according to sample_rate
    val_files = [dict([("kspace", val_files[i])]) for i in range(len(val_files))]
    print(f"#validation files: {len(val_files)}")

    # define mask transform type (e.g., whether it is equispaced or random)
    if args.mask_type == "random":
        MaskTransform = RandomKspaceMaskd(
            keys=["kspace"],
            center_fractions=args.center_fractions,
            accelerations=args.accelerations,
            spatial_dims=2,
            is_complex=True,
        )
    elif args.mask_type == "equispaced":
        MaskTransform = EquispacedKspaceMaskd(
            keys=["kspace"],
            center_fractions=args.center_fractions,
            accelerations=args.accelerations,
            spatial_dims=2,
            is_complex=True,
        )

    train_transforms = Compose(
        [
            LoadImaged(keys=["kspace"], reader=FastMRIReader, image_only=False, dtype=np.complex64),
            # user can also add other random transforms but remember to disable randomness for val_transforms
            ExtractDataKeyFromMetaKeyd(keys=["reconstruction_rss", "mask"], meta_key="kspace_meta_dict"),
            MaskTransform,
            EnsureTyped(keys=["kspace", "kspace_masked_ifft", "reconstruction_rss"]),
        ]
    )

    train_ds = CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # since there's no randomness in train_transforms, we use it for val_transforms as well
    val_ds = CacheDataset(
        data=val_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # create the model
    coil_sens_model = CoilSensitivityModel(spatial_dims=2, features=args.sensitivity_model_features)
    refinement_model = ComplexUnet(spatial_dims=2, features=args.features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VariationalNetworkModel(coil_sens_model, refinement_model, num_cascades=args.num_cascades).to(device)
    print("#model_params:", np.sum([len(p.flatten()) for p in model.parameters()]))

    if args.resume_checkpoint:
        model.load_state_dict(torch.load(args.checkpoint_dir))
        print("resume training from a given checkpoint...")

    # create the loss function
    loss_function = SSIMLoss(spatial_dims=2).to(device)

    # create the optimizer and the learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    # start a typical PyTorch training loop
    val_interval = 2  # doing validation every 2 epochs
    best_metric = -1
    best_metric_epoch = -1
    tic = time.time()
    for epoch in range(args.num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            input, mask, target, max_value = (
                batch_data["kspace_masked"].to(device),
                batch_data["mask"][0].to(device),
                batch_data["reconstruction_rss"].to(device),
                batch_data["kspace_meta_dict"]["max"],
            )

            final_shape = target.shape[-2:]
            max_value = torch.tensor(max_value).unsqueeze(0).to(device)
            loss_function.ssim_metric.data_range = max_value

            # iterate through all slices
            slice_dim = 1  # change this if another dimension is your slice dimension
            num_slices = input.shape[slice_dim]
            for i in range(num_slices):
                step += 1
                optimizer.zero_grad()

                # forward pass
                inp = input[:, i, ...].unsqueeze(slice_dim)
                tar = target[:, i, ...].unsqueeze(slice_dim)
                output = model(inp[0], mask.bool())

                # crop output to match target size
                roi_center = tuple(i // 2 for i in output.shape[-2:])
                cropper = SpatialCrop(roi_center=roi_center, roi_size=final_shape)
                output_crp = cropper(output).unsqueeze(0)

                loss = loss_function(output_crp, tar)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"{step}, train_loss: {epoch_loss/step:.4f}", "\r", end="")
        scheduler.step()
        epoch_loss /= step
        writer.add_scalar("train_loss", epoch_loss, epoch + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} time elapsed: {(time.time()-tic)/60:.2f} mins")

        # validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_ssim = list()
                for val_data in val_loader:
                    input, mask, target, fname = (
                        val_data["kspace_masked"].to(device),
                        val_data["mask"][0].to(device),
                        val_data["reconstruction_rss"].to(device),
                        val_data["kspace_meta_dict"]["filename"],
                    )

                    final_shape = target.shape[-2:]

                    # iterate through all slices:
                    slice_dim = 1  # change this if another dimension is your slice dimension
                    num_slices = input.shape[slice_dim]
                    outputs = []
                    targets = []
                    for i in range(num_slices):
                        inp = input[:, i, ...].unsqueeze(slice_dim)
                        tar = target[:, i, ...].unsqueeze(slice_dim)

                        # forward pass
                        output = model(inp[0], mask.bool())

                        # crop output to match target size
                        roi_center = tuple(i // 2 for i in output.shape[-2:])
                        cropper = SpatialCrop(roi_center=roi_center, roi_size=final_shape)
                        output_crp = cropper(output).unsqueeze(0)

                        outputs.append(output_crp.data.cpu().numpy()[0][0])
                        targets.append(tar.data.cpu().numpy()[0][0])

                    outputs = np.stack(outputs)
                    targets = np.stack(targets)
                    val_ssim.append(skimage_ssim(targets, outputs))

                metric = np.mean(val_ssim)

                # save the best checkpoint so far
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(outpath, "varnet_mri_reconstruction.pt"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean ssim: {:.4f} best mean ssim: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_ssim", metric, epoch + 1)

    print(f"training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


def __main__():
    parser = argparse.ArgumentParser()

    # data loader arguments
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Data loader batch size (batch_size>1 is suitable for varying input size",
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers to use in data loader",
    )

    parser.add_argument(
        "--cache_rate",
        default=0.0,
        type=float,
        help="The fraction of the data to be cached when being loaded",
    )

    parser.add_argument(
        "--data_path_train",
        default=None,
        type=Path,
        help="Path to the fastMRI training set",
    )

    parser.add_argument(
        "--data_path_val",
        default=None,
        type=Path,
        help="Path to the fastMRI validation set",
    )

    parser.add_argument(
        "--sample_rate",
        default=1.0,
        type=float,
        help="what fraction of the dataset to use for training (also, what fraction of validation set to use)",
    )

    # Mask parameters
    parser.add_argument("--accelerations", default=[4], type=list, help="acceleration factors used during training")

    parser.add_argument(
        "--center_fractions",
        default=[0.08],
        type=list,
        help="center fractions used during training (center fraction denotes the center region to exclude from masking)",
    )

    # training params
    parser.add_argument("--num_epochs", default=50, type=int, help="number of training epochs")

    parser.add_argument("--exp_dir", default="./", type=Path, help="output directory to save training logs")

    parser.add_argument(
        "--exp",
        default="varnet_mri_recon",
        type=str,
        help="experiment name (a folder will be created with this name to store the results)",
    )

    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")

    parser.add_argument("--lr_step_size", default=40, type=int, help="decay learning rate every lr_step_size epochs")

    parser.add_argument(
        "--lr_gamma",
        default=0.1,
        type=float,
        help="every lr_step_size epochs, decay learning rate by a factor of lr_gamma",
    )

    parser.add_argument("--weight_decay", default=0.0, type=float, help="ridge regularization factor")

    parser.add_argument(
        "--mask_type", default="random", type=str, help="under-sampling mask type: ['random','equispaced']"
    )

    # model specific args
    parser.add_argument("--drop_prob", default=0.0, type=float, help="dropout probability for U-Net")

    parser.add_argument(
        "--features",
        default=[18, 36, 72, 144, 288, 18],
        type=list,
        help="six integers as numbers of features (see monai.networks.nets.basic_unet)",
    )

    parser.add_argument(
        "--sensitivity_model_features",
        default=[8, 16, 32, 64, 128, 8],
        type=list,
        help="six integers as numbers of sensitivity model features (see monai.networks.nets.basic_unet)",
    )

    parser.add_argument("--num_cascades", default=12, type=int, help="number of cascades")

    parser.add_argument(
        "--resume_checkpoint", default=False, type=bool, help="if True, training statrts from a model checkpoint"
    )

    parser.add_argument(
        "--checkpoint_dir", default=None, type=Path, help="model checkpoint path to resume training from"
    )

    args = parser.parse_args()
    trainer(args)


__main__()
