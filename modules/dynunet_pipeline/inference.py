import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.data.utils import to_affine_nd
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import DynUNet
from monai.transforms import AsDiscrete

from create_network import get_kernels_strides, get_network
from inferrer import DynUNetInferrer
from task_params import (deep_supr_num, patch_size,
                         task_name)
from transforms import get_task_transforms, recovery_prediction


def inference(args):
    # load hyper parameters
    root_dir = args.root_dir
    task_id = args.task_id
    val_num_workers = args.val_num_workers
    checkpoint = args.checkpoint
    val_output_dir = "./runs_{}_fold{}_{}/".format(
        args.task_id, args.fold, args.expr_name
    )
    sw_batch_size = args.sw_batch_size
    infer_output_dir = os.path.join(val_output_dir, task_name[task_id])
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    amp = args.amp
    tta_val = args.tta_val

    if not os.path.exists(infer_output_dir):
        os.makedirs(infer_output_dir)

    transform_params = (args.pos_sample_num, args.neg_sample_num, args.num_samples)
    test_transform = get_task_transforms("test", task_id, *transform_params)

    test_ds = DecathlonDataset(
        root_dir=root_dir,
        task=task_name[task_id],
        transform=test_transform,
        section="test",
        download=False,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=val_num_workers
    )
    # produce the network
    device = torch.device("cuda:0")
    properties = test_ds.get_properties(keys=["labels", "modality"])
    n_classes = len(properties["labels"])
    net = get_network(device, properties, task_id, val_output_dir, checkpoint)

    net.eval()

    inferrer = DynUNetInferrer(
        device=device,
        val_data_loader=test_loader,
        network=net,
        output_dir=val_output_dir,
        n_classes=n_classes,
        inferer=SlidingWindowInferer(
            roi_size=patch_size[task_id],
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
        ),
        amp=amp,
        tta_val=tta_val,
    )

    inferrer.run()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-fold", "--fold", type=int, default=0, help="0-5")
    parser.add_argument(
        "-task_id", "--task_id", type=str, default="02", help="task 01 to 10"
    )
    parser.add_argument(
        "-root_dir",
        "--root_dir",
        type=str,
        default="/workspace/data/medical/",
        help="dataset path",
    )
    parser.add_argument(
        "-expr_name",
        "--expr_name",
        type=str,
        default="expr",
        help="the suffix of the experiment's folder",
    )
    parser.add_argument(
        "-train_num_workers",
        "--train_num_workers",
        type=int,
        default=4,
        help="the num_workers parameter of training dataloader.",
    )
    parser.add_argument(
        "-val_num_workers",
        "--val_num_workers",
        type=int,
        default=1,
        help="the num_workers parameter of validation dataloader.",
    )
    parser.add_argument(
        "-interval",
        "--interval",
        type=int,
        default=5,
        help="the validation interval under epoch level.",
    )
    parser.add_argument(
        "-eval_overlap",
        "--eval_overlap",
        type=float,
        default=0.5,
        help="the overlap parameter of SlidingWindowInferer.",
    )
    parser.add_argument(
        "-sw_batch_size",
        "--sw_batch_size",
        type=int,
        default=4,
        help="the sw_batch_size parameter of SlidingWindowInferer.",
    )
    parser.add_argument(
        "-window_mode",
        "--window_mode",
        type=str,
        default="gaussian",
        choices=["constant", "gaussian"],
        help="the mode parameter for SlidingWindowInferer.",
    )
    parser.add_argument(
        "-num_samples",
        "--num_samples",
        type=int,
        default=3,
        help="the num_samples parameter of RandCropByPosNegLabeld.",
    )
    parser.add_argument(
        "-pos_sample_num",
        "--pos_sample_num",
        type=int,
        default=1,
        help="the pos parameter of RandCropByPosNegLabeld.",
    )
    parser.add_argument(
        "-neg_sample_num",
        "--neg_sample_num",
        type=int,
        default=1,
        help="the neg parameter of RandCropByPosNegLabeld.",
    )
    parser.add_argument(
        "-cache_rate",
        "--cache_rate",
        type=float,
        default=1.0,
        help="the cache_rate parameter of CacheDataset.",
    )
    parser.add_argument(
        "-checkpoint",
        "--checkpoint",
        type=str,
        default=None,
        help="the filename of weights.",
    )
    parser.add_argument("-multi_gpu", "--multi_gpu", type=bool, default=False)
    parser.add_argument(
        "-amp",
        "--amp",
        type=bool,
        default=False,
        help="whether to use automatic mixed precision.",
    )
    parser.add_argument(
        "-tta_val",
        "--tta_val",
        type=bool,
        default=False,
        help="whether to use test time augmentation.",
    )

    args = parser.parse_args()
    inference(args)
