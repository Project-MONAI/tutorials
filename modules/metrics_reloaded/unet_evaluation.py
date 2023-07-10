# Copyright 2023 MONAI Consortium
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

from monai import config
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.handlers import CheckpointLoader, MeanDice, StatsHandler
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import Activations, EnsureChannelFirst, AsDiscrete, Compose, SaveImage, ScaleIntensity

from MetricsReloaded.processes.overall_process import ProcessEvaluation


def get_metrics_reloaded_dict(pth_ref, pth_pred):
    """Prepare input dictionary for MetricsReloaded package."""
    preds = []
    refs = []
    names = []
    for r, p in zip(pth_ref, pth_pred):
        name = r.split(os.sep)[-1].split(".nii.gz")[0]
        names.append(name)

        ref = nib.load(r).get_fdata()
        pred = nib.load(p).get_fdata()
        refs.append(ref)
        preds.append(pred)

    dict_file = {}
    dict_file["pred_loc"] = preds
    dict_file["ref_loc"] = refs
    dict_file["pred_prob"] = preds
    dict_file["ref_class"] = refs
    dict_file["pred_class"] = preds
    dict_file["list_values"] = [1]
    dict_file["file"] = pth_pred
    dict_file["names"] = names

    return dict_file


def main(tempdir, img_size=96):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Set patch size
    patch_size = (int(img_size / 2.0),) * 3

    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(5):
        im, seg = create_test_image_3d(img_size, img_size, img_size, num_seg_classes=1)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"lab{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "lab*.nii.gz")))

    # define transforms for image and segmentation
    imtrans = Compose([ScaleIntensity(), EnsureChannelFirst()])
    segtrans = Compose([EnsureChannelFirst()])
    ds = ImageDataset(images, segs, transform=imtrans, seg_transform=segtrans, image_only=False)

    # Compute UNet levels and strides from image size
    min_size = 4  # minimum size allowed at coarsest resolution level
    num_levels = int(np.maximum(np.ceil(np.log2(np.min(img_size)) - np.log2(min_size)), 1))
    channels = [2 ** (i + 4) for i in range(num_levels)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=(2,) * (num_levels - 1),
        num_res_units=2,
    ).to(device)

    # define sliding window size and batch size for windows inference
    roi_size = patch_size
    sw_batch_size = 4

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    save_image = SaveImage(output_dir=tempdir, output_ext=".nii.gz", output_postfix="pred", separate_folder=False)

    def _sliding_window_processor(engine, batch):
        net.eval()
        with torch.no_grad():
            val_images, val_labels = batch[0].to(device), batch[1].to(device)
            seg_probs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
            seg_probs = [post_trans(i) for i in decollate_batch(seg_probs)]
            for seg_prob in seg_probs:
                save_image(seg_prob)
            return seg_probs, val_labels

    evaluator = Engine(_sliding_window_processor)

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't need to print loss for evaluator, so just print metrics, user can also customize print functions
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
    )
    val_stats_handler.attach(evaluator)

    # the model was trained by "unet_training_array" example
    cwd = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1])
    load_path = sorted(list(filter(os.path.isfile, glob(cwd + os.sep + "runs_array" + os.sep + "*.pt"))))
    ckpt_loader = CheckpointLoader(load_path=load_path[-1], load_dict={"net": net})
    ckpt_loader.attach(evaluator)

    # sliding window inference for one image at every iteration
    loader = DataLoader(ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    state = evaluator.run(loader)
    print(state)

    # Prepare MetricsReloaded input
    pth_ref = sorted(list(filter(os.path.isfile, glob(tempdir + os.sep + "lab*.nii.gz"))))
    pth_pred = sorted(list(filter(os.path.isfile, glob(tempdir + os.sep + "*_pred.nii.gz"))))

    # Prepare input dictionary for MetricsReloaded package
    dict_file = get_metrics_reloaded_dict(pth_ref, pth_pred)

    # Run MetricsReloaded evaluation process
    PE = ProcessEvaluation(
        dict_file,
        "SemS",
        localization="mask_iou",
        file=dict_file["file"],
        flag_map=True,
        assignment="greedy_matching",
        measures_overlap=[
            "numb_ref",
            "numb_pred",
            "numb_tp",
            "numb_fp",
            "numb_fn",
            "iou",
            "fbeta",
        ],
        measures_boundary=[
            "assd",
            "boundary_iou",
            "hd",
            "hd_perc",
            "masd",
            "nsd",
        ],
        case=True,
        thresh_ass=0.000001,
    )

    # Save results as CSV
    PE.resseg.to_csv(cwd + os.sep + "results_metrics_reloaded.csv")

    return


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
