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

"""
This example shows how to efficiently compute Dice scores for pairs of segmentation prediction
and references in multi-processing based on MONAI's metrics API.
It can even run on multi-nodes.
Main steps to set up the distributed data parallel:

- Execute `torchrun` to create processes on every node for every process.
  It receives parameters as below:
  `--nproc_per_node=NUM_PROCESSES_PER_NODE`
  `--nnodes=NUM_NODES`
  For more details, refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py.
  Alternatively, we can also use `torch.multiprocessing.spawn` to start program, but it that case, need to handle
  all the above parameters and compute `rank` manually, then set to `init_process_group`, etc.
  `torchrun` is even more efficient than `torch.multiprocessing.spawn`.
- Use `init_process_group` to initialize every process.
- Partition the saved predictions and labels into ranks for parallel computation.
- Compute `Dice Metric` on every process, reduce the results after synchronization.

Note:
    `torchrun` will launch `nnodes * nproc_per_node = world_size` processes in total.
    Example script to execute this program on a single node with 2 processes:
    `torchrun --nproc_per_node=2 compute_metric.py`

Referring to: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

"""

import argparse
import os
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist

from monai.data import create_test_image_3d, partition_dataset
from monai.handlers import write_metrics_reports
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    KeepLargestConnectedComponentd,
    LoadImaged,
    ScaleIntensityd,
    ToDeviced,
)
from monai.utils import string_list_all_gather


def compute(args):
    # generate synthetic data for the example
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0 and not os.path.exists(args.dir):
        # create 16 random pred, label paris for evaluation
        print(f"generating synthetic data to {args.dir} (this may take a while)")
        os.makedirs(args.dir)
        # if have multiple nodes, set random seed to generate same random data for every node
        np.random.seed(seed=0)
        for i in range(16):
            pred, label = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1, noise_max=0.5)
            n = nib.Nifti1Image(pred, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"pred{i:d}.nii.gz"))
            n = nib.Nifti1Image(label, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"label{i:d}.nii.gz"))

    # initialize the distributed evaluation process, change to gloo backend if computing on CPU
    dist.init_process_group(backend="nccl", init_method="env://")

    preds = sorted(glob(os.path.join(args.dir, "pred*.nii.gz")))
    labels = sorted(glob(os.path.join(args.dir, "label*.nii.gz")))
    datalist = [{"pred": pred, "label": label} for pred, label in zip(preds, labels)]

    # split data for every subprocess, for example, 16 processes compute in parallel
    data_part = partition_dataset(
        data=datalist,
        num_partitions=dist.get_world_size(),
        shuffle=False,
        even_divisible=False,
    )[dist.get_rank()]

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    # define transforms for predictions and labels
    transforms = Compose(
        [
            LoadImaged(keys=["pred", "label"]),
            ToDeviced(keys=["pred", "label"], device=device),
            EnsureChannelFirstd(keys=["pred", "label"]),
            ScaleIntensityd(keys="pred"),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        ]
    )
    data_part = [transforms(item) for item in data_part]

    # compute metrics for current process
    metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    metric(y_pred=[i["pred"] for i in data_part], y=[i["label"] for i in data_part])
    filenames = [item["pred_meta_dict"]["filename_or_obj"] for item in data_part]
    # all-gather results from all the processes and reduce for final result
    result = metric.aggregate().item()
    filenames = string_list_all_gather(strings=filenames)

    if local_rank == 0:
        print("mean dice: ", result)
        # generate metrics reports at: output/mean_dice_raw.csv, output/mean_dice_summary.csv, output/metrics.csv
        write_metrics_reports(
            save_dir="./output",
            images=filenames,
            metrics={"mean_dice": result},
            metric_details={"mean_dice": metric.get_buffer()},
            summary_ops="*",
        )

    metric.reset()

    dist.destroy_process_group()


# usage example(refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py):

# torchrun --standalone --nnodes=NUM_NODES --nproc_per_node=NUM_GPUS_PER_NODE compute_metric.py -d DIR_OF_OUTPUT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./output", type=str, help="root directory of labels and predictions.")
    args = parser.parse_args()

    compute(args=args)


if __name__ == "__main__":
    main()
