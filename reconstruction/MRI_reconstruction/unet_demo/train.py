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
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
    ThresholdIntensityd,
)

from monai.apps.reconstruction.transforms.dictionary import (
    ExtractDataKeyFromMetaKeyd,
    RandomKspaceMaskd,
    EquispacedKspaceMaskd,
    ReferenceBasedSpatialCropd,
    ReferenceBasedNormalizeIntensityd,
)

from monai.apps.reconstruction.fastmri_reader import FastMRIReader
from monai.networks.nets import BasicUNet

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

    val_files = list(Path(args.data_path_val).iterdir())
    ##################################################### temp: dummy file
    val_files = list(filter(lambda f: "file_brain_AXT2_201_2010294.h5" not in str(f), val_files))
    ##################################################### temp: dummy file
    random.shuffle(val_files)
    val_files = val_files[
        : int(args.sample_rate * len(val_files))
    ]  # select a subset of the data according to sample_rate
    val_files = [dict([("kspace", val_files[i])]) for i in range(len(val_files))]

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
            # user can also add other random transforms
            ExtractDataKeyFromMetaKeyd(keys=["reconstruction_rss", "mask"], meta_key="kspace_meta_dict"),
            MaskTransform,
            ReferenceBasedSpatialCropd(keys=["kspace_masked_ifft"], ref_key="reconstruction_rss"),
            ReferenceBasedNormalizeIntensityd(
                keys=["kspace_masked_ifft", "reconstruction_rss"], ref_key="kspace_masked_ifft", channel_wise=True
            ),
            ThresholdIntensityd(
                keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=6.0, above=False, cval=6.0
            ),
            ThresholdIntensityd(
                keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=-6.0, above=True, cval=-6.0
            ),
            EnsureTyped(keys=["kspace", "kspace_masked_ifft", "reconstruction_rss"]),
        ]
    )

    train_ds = CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = CacheDataset(
        data=val_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        features=args.features,
    ).to(device)
    print("#model_params:", np.sum([len(p.flatten()) for p in model.parameters()]))
    if args.resume_checkpoint:
        model.load_state_dict(torch.load(args.checkpoint_dir))
        print("resume training from a given checkpoint...")

    # create the loss function
    loss_function = torch.nn.L1Loss()

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
            input, target, mean, std = (
                batch_data["kspace_masked_ifft"],
                batch_data["reconstruction_rss"],
                batch_data["mean"],
                batch_data["std"],
            )

            # iterate through all slices
            slice_dim = 1  # change this if another dimension is your slice dimension
            num_slices = input.shape[slice_dim]
            for i in range(num_slices):
                step += 1
                optimizer.zero_grad()

                inp = input[:, i, ...].unsqueeze(slice_dim)
                tar = target[:, i, ...].unsqueeze(slice_dim)
                output = model(inp.to(device))

                loss = loss_function(output, tar.to(device))

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
            outputs = defaultdict(list)
            targets = defaultdict(list)
            with torch.no_grad():
                val_ssim = list()
                val_loss = list()
                for val_data in val_loader:
                    input, target, mean, std, fname = (
                        val_data["kspace_masked_ifft"],
                        val_data["reconstruction_rss"],
                        val_data["mean"],
                        val_data["std"],
                        val_data["kspace_meta_dict"]["filename"],
                    )

                    # iterate through all slices:
                    slice_dim = 1  # change this if another dimension is your slice dimension
                    num_slices = input.shape[slice_dim]
                    for i in range(num_slices):
                        inp = input[:, i, ...].unsqueeze(slice_dim)
                        tar = target[:, i, ...].unsqueeze(slice_dim)
                        output = model(inp.to(device))

                        vloss = loss_function(output, tar.to(device))
                        val_loss.append(vloss.item())

                        _std = std[0][i].item()
                        _mean = mean[0][i].item()
                        outputs[fname[0]].append(output.data.cpu().numpy()[0][0] * _std + _mean)
                        targets[fname[0]].append(tar.numpy()[0][0] * _std + _mean)

                # compute validation ssims
                for fname in outputs:
                    outputs[fname] = np.stack(outputs[fname])
                    targets[fname] = np.stack(targets[fname])
                    val_ssim.append(skimage_ssim(targets[fname], outputs[fname]))

                metric = np.mean(val_ssim)

                # save the best checkpoint so far
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(outpath, "unet_mri_reconstruction.pt"))
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

    parser.add_argument("--exp_dir", default=None, type=Path, help="output directory to save training logs")

    parser.add_argument(
        "--exp",
        default=None,
        type=str,
        help="experiment name (a folder will be created with this name to store the results)",
    )

    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")

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
        default=[32, 64, 128, 256, 512, 32],
        type=list,
        help="six integers as numbers of features (see monai.networks.nets.basic_unet)",
    )

    parser.add_argument(
        "--resume_checkpoint", default=False, type=bool, help="if True, training statrts from a model checkpoint"
    )

    parser.add_argument(
        "--checkpoint_dir", default=None, type=Path, help="model checkpoint path to resume training from"
    )

    args = parser.parse_args()
    trainer(args)


__main__()
