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
This example shows how to compute metric in multi-processing based on PyTorch native `DistributedDataParallel` module.
It can run on several nodes with multiple GPU devices on every node.
Main steps to set up the distributed data parallel:

- Execute `torch.distributed.launch` to create processes on every node for every GPU.
  It receives parameters as below:
  `--nproc_per_node=NUM_GPUS_PER_NODE`
  `--nnodes=NUM_NODES`
  `--node_rank=INDEX_CURRENT_NODE`
  `--master_addr="localhost"`
  `--master_port=1234`
  For more details, refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py.
  Alternatively, we can also use `torch.multiprocessing.spawn` to start program, but it that case, need to handle
  all the above parameters and compute `rank` manually, then set to `init_process_group`, etc.
  `torch.distributed.launch` is even more efficient than `torch.multiprocessing.spawn`.
- Use `init_process_group` to initialize every process, every GPU runs in a separate process with unique rank.
  Here we use `NVIDIA NCCL` as the backend and must set `init_method="env://"` if use `torch.distributed.launch`.
- Partition the saved predictions and labels into ranks for parallel computation.
- Compute `Dice Metric` on every process, reduce the results after synchronization.

Note:
    `torch.distributed.launch` will launch `nnodes * nproc_per_node = world_size` processes in total.
    Suggest setting exactly the same software environment for every node, especially `PyTorch`, `nccl`, etc.
    A good practice is to use the same MONAI docker image for all nodes directly.
    Example script to execute this program on every node:
    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
           --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
           --master_addr="localhost" --master_port=1234
           compute_metric.py

    This example was tested with [Ubuntu 16.04/20.04], [NCCL 2.6.3].

Referring to: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

"""

import argparse
import os
from glob import glob

import torch
import torch.distributed as dist

from monai.data import partition_dataset
from monai.handlers import write_metrics_reports
from monai.metrics import DiceMetric
from monai.transforms import EnsureChannelFirstd, Compose, LoadImaged, ToTensord
from monai.utils import string_list_all_gather


def compute(args):
    # initialize the distributed evaluation process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)

    preds = sorted(glob(os.path.join(args.dir, "im*", "im*_seg.nii.gz")))
    labels = sorted(glob(os.path.join(args.dir, "seg*.nii.gz")))
    datalist = [{"pred": pred, "label": label} for pred, label in zip(preds, labels)]

    # split data for every subprocess, for example, 16 processes compute in parallel
    data_part = partition_dataset(
        data=datalist,
        num_partitions=dist.get_world_size(),
        shuffle=False,
        even_divisible=False,
    )[dist.get_rank()]

    # define transforms for predictions and labels
    transforms = Compose(
        [
            LoadImaged(keys=["pred", "label"]),
            EnsureChannelFirstd(keys=["pred", "label"]),
            ToTensord(keys=["pred", "label"]),
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

    if args.local_rank == 0:
        print("mean dice: ", result)
        # generate metric report
        write_metrics_reports(
            save_dir="./output",
            images=filenames,
            metrics={"mean_dice": result},
            metric_details={"mean_dice": metric.get_buffer()},
            summary_ops="*",
        )

    metric.reset()

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./output", type=str, help="root directory of labels and predictions.")
    # must parse the command-line argument: ``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by DDP
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    compute(args=args)


if __name__ == "__main__":
    main()
