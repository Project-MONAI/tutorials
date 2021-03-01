import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from monai.handlers import (CheckpointSaver, LrScheduleHandler, MeanDice,
                            StatsHandler, ValidationHandler)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss

from create_dataset import get_data
from create_network import get_network
from evaluator import DynUNetEvaluator
from task_params import data_loader_params, patch_size
from trainer import DynUNetTrainer


def validation(args):
    # load hyper parameters
    task_id = args.task_id
    sw_batch_size = args.sw_batch_size
    tta_val = args.tta_val
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    amp = args.amp

    properties, val_loader = get_data(args, mode="validation")
    n_classes = len(properties["labels"])
    # produce the network
    checkpoint = args.checkpoint
    val_output_dir = "./runs_{}_fold{}_{}/".format(task_id, args.fold, args.expr_name)
    device = torch.device("cuda:0")
    net = get_network(device, properties, task_id, val_output_dir, checkpoint)

    net.eval()

    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        n_classes=n_classes,
        inferer=SlidingWindowInferer(
            roi_size=patch_size[task_id],
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
        ),
        post_transform=None,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=False,
                output_transform=lambda x: (x["pred"], x["label"]),
            )
        },
        additional_metrics=None,
        amp=amp,
        tta_val=tta_val,
    )

    evaluator.run()
    print(evaluator.state.metrics)
    results = evaluator.state.metric_details["val_mean_dice"]
    if n_classes > 2:
        for i in range(n_classes - 1):
            print("mean dice for label {} is {}".format(i + 1, results[:, i].mean()))


def train(args):
    # load hyper parameters
    task_id = args.task_id
    fold = args.fold
    val_output_dir = "./runs_{}_fold{}_{}/".format(task_id, fold, args.expr_name)
    log_filename = "nnunet_task{}_fold{}.log".format(task_id, fold)
    log_filename = os.path.join(val_output_dir, log_filename)
    interval = args.interval
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    amp_flag = args.amp
    lr_decay_flag = args.lr_decay
    sw_batch_size = args.sw_batch_size
    tta_val = args.tta_val
    batch_dice = args.batch_dice
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap

    # transforms
    train_batch_size = data_loader_params[task_id]["batch_size"]

    properties, val_loader = get_data(args, mode="validation")
    _, train_loader = get_data(args, batch_size=train_batch_size, mode="train")

    # produce the network
    device = torch.device("cuda:0")
    checkpoint = args.checkpoint
    net = get_network(device, properties, task_id, val_output_dir, checkpoint)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9
    )
    # produce evaluator
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointSaver(
            save_dir=val_output_dir, save_dict={"net": net}, save_key_metric=True
        ),
    ]

    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        n_classes=len(properties["labels"]),
        inferer=SlidingWindowInferer(
            roi_size=patch_size[task_id],
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
        ),
        post_transform=None,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=False,
                output_transform=lambda x: (x["pred"], x["label"]),
            )
        },
        val_handlers=val_handlers,
        amp=amp_flag,
        tta_val=tta_val,
    )
    # produce trainer
    loss = DiceCELoss(to_onehot_y=True, softmax=True, batch=batch_dice)
    train_handlers = []
    if lr_decay_flag:
        train_handlers += [LrScheduleHandler(lr_scheduler=scheduler, print_lr=True)]

    train_handlers += [
        ValidationHandler(validator=evaluator, interval=interval, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
    ]

    trainer = DynUNetTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        post_transform=None,
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp_flag,
    )

    # run
    logger = logging.getLogger()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Setup file handler
    fhandler = logging.FileHandler(log_filename)
    fhandler.setLevel(logging.INFO)
    fhandler.setFormatter(formatter)

    # Configure stream handler for the cells
    chandler = logging.StreamHandler()
    chandler.setLevel(logging.INFO)
    chandler.setFormatter(formatter)

    # Add both handlers
    logger.addHandler(fhandler)
    logger.addHandler(chandler)
    logger.setLevel(logging.INFO)

    # Show the handlers
    logger.handlers

    # Log Something
    logger.info("Test info")
    logger.debug("Test debug")

    trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-fold", "--fold", type=int, default=0, help="0-5")
    parser.add_argument(
        "-task_id", "--task_id", type=str, default="04", help="task 01 to 10"
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
    parser.add_argument("-learning_rate", "--learning_rate", type=float, default=1e-2)
    parser.add_argument(
        "-max_epochs",
        "--max_epochs",
        type=int,
        default=1000,
        help="number of epochs of training.",
    )
    parser.add_argument(
        "-mode", "--mode", type=str, default="train", choices=["train", "val"]
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
        "-lr_decay",
        "--lr_decay",
        type=bool,
        default=False,
        help="whether to use learning rate decay.",
    )
    parser.add_argument(
        "-tta_val",
        "--tta_val",
        type=bool,
        default=False,
        help="whether to use test time augmentation.",
    )
    parser.add_argument(
        "-batch_dice",
        "--batch_dice",
        type=bool,
        default=False,
        help="the batch parameter of DiceCELoss.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "val":
        validation(args)
