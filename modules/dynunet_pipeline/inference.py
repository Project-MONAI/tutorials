import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import torch.distributed as dist
from monai.inferers import SlidingWindowInferer
from torch.nn.parallel import DistributedDataParallel

from create_dataset import get_data
from create_network import get_network
from inferrer import DynUNetInferrer
from task_params import patch_size, task_name


def inference(args):
    # load hyper parameters
    task_id = args.task_id
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
    multi_gpu_flag = args.multi_gpu
    local_rank = args.local_rank

    if not os.path.exists(infer_output_dir):
        os.makedirs(infer_output_dir)

    if multi_gpu_flag:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda")

    properties, test_loader = get_data(args, mode="test")

    net = get_network(properties, task_id, val_output_dir, checkpoint)
    net = net.to(device)

    if multi_gpu_flag:
        net = DistributedDataParallel(
            module=net, device_ids=[device], find_unused_parameters=True
        )

    net.eval()

    inferrer = DynUNetInferrer(
        device=device,
        val_data_loader=test_loader,
        network=net,
        output_dir=infer_output_dir,
        num_classes=len(properties["labels"]),
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
        "-datalist_path",
        "--datalist_path",
        type=str,
        default="config/",
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
    parser.add_argument(
        "-multi_gpu",
        "--multi_gpu",
        type=bool,
        default=False,
        help="whether to use multiple GPUs for training.",
    )
    parser.add_argument("-local_rank", "--local_rank", type=int, default=0)
    args = parser.parse_args()
    inference(args)
