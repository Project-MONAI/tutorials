# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from ignite.engine import Engine
from torch.utils.data import DataLoader

from monai import config
from monai.data import ImageDataset, create_test_image_3d, decollate_batch
from monai.handlers import CheckpointLoader, MeanDice, StatsHandler
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import Activations, AddChannel, AsDiscrete, Compose, SaveImage, ScaleIntensity, EnsureType


def main(tempdir):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(5):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))

    # define transforms for image and segmentation
    imtrans = Compose([ScaleIntensity(), AddChannel(), EnsureType()])
    segtrans = Compose([AddChannel(), EnsureType()])
    ds = ImageDataset(images, segs, transform=imtrans, seg_transform=segtrans, image_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # define sliding window size and batch size for windows inference
    roi_size = (96, 96, 96)
    sw_batch_size = 4

    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    save_image = SaveImage(output_dir="tempdir", output_ext=".nii.gz", output_postfix="seg")

    def _sliding_window_processor(engine, batch):
        net.eval()
        with torch.no_grad():
            val_images, val_labels = batch[0].to(device), batch[1].to(device)
            seg_probs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
            seg_probs = [post_trans(i) for i in decollate_batch(seg_probs)]
            val_data = decollate_batch(batch[2])
            for seg_prob, data in zip(seg_probs, val_data):
                save_image(seg_prob, data)
            return seg_probs, val_labels

    evaluator = Engine(_sliding_window_processor)

    # add evaluation metric to the evaluator engine
    MeanDice().attach(evaluator, "Mean_Dice")

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't need to print loss for evaluator, so just print metrics, user can also customize print functions
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
    )
    val_stats_handler.attach(evaluator)

    # the model was trained by "unet_training_array" example
    ckpt_saver = CheckpointLoader(load_path="./runs_array/net_checkpoint_100.pt", load_dict={"net": net})
    ckpt_saver.attach(evaluator)

    # sliding window inference for one image at every iteration
    loader = DataLoader(ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    state = evaluator.run(loader)
    print(state)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
