import argparse
import distutils.util
import json
import logging
import os
import sys
import time

import torch

from monai.apps.deepgrow.interaction import Interaction
from monai.engines import SupervisedEvaluator
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    MeanDice)
from monai.inferers import SimpleInferer
from monai.utils import set_determinism
from .deepgrow_training_2d import (
    get_network,
    get_loaders,
    get_pre_transforms,
    get_click_transforms,
    get_post_transforms
)


def create_validator(args, click):
    set_determinism(seed=args.seed)

    device = torch.device("cuda" if args.use_gpu else "cpu")

    pre_transforms = get_pre_transforms(json.loads(args.roi_size))
    click_transforms = get_click_transforms(sigmoid=False)
    post_transform = get_post_transforms(sigmoid=False)

    # define training components
    network = get_network(args).to(device)

    logging.info('Loading Network...')
    map_location = {"cuda:0": "cuda:{}".format(args.local_rank)}

    checkpoint = torch.load(args.model_path, map_location=map_location)
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]

    network.load_state_dict(checkpoint)

    # define event-handlers for engine
    _, val_loader = get_loaders(args, pre_transforms, train=False)
    fold_size = int(len(val_loader.dataset) / args.batch / args.folds) if args.folds else 0
    logging.info('Using Fold-Size: {}'.format(fold_size))

    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=args.output, output_transform=lambda x: None),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        iteration_update=Interaction(
            transforms=click_transforms,
            max_interactions=click,
            train=False),
        inferer=SimpleInferer(),
        post_transform=post_transform,
        val_handlers=val_handlers,
        key_val_metric={
            f'clicks_{click}_val_dice': MeanDice(
                include_background=False,
                output_transform=lambda x: (x["pred"], x["label"])
            )
        }
    )
    return evaluator


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--seed', type=int, default=42)

    parser.add_argument('-n', '--network', default='bunet', choices=['native', 'bunet', 'foo'])
    parser.add_argument('-z', '--net_size', type=int, default=64)
    parser.add_argument('-f', '--folds', type=int, default=10)

    parser.add_argument('-d', '--dataset_root', default='/workspace/data/52432')
    parser.add_argument('-j', '--dataset_json', default='/workspace/data/52432/dataset.json')
    parser.add_argument('-i', '--input', default='/workspace/data/52432/2D')
    parser.add_argument('-o', '--output', default='output')

    parser.add_argument('-g', '--use_gpu', type=strtobool, default='true')
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-t', '--limit', type=int, default=20)
    parser.add_argument('-m', '--model_path', default="output/model.pt")
    parser.add_argument('--roi_size', default="[128, 128]")

    parser.add_argument('-iv', '--max_val_interactions', default="[1,2,5,10,15]")
    parser.add_argument('--multi_gpu', type=strtobool, default='false')
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    if args.local_rank == 0:
        for arg in vars(args):
            logging.info('USING:: {} = {}'.format(arg, getattr(args, arg)))
        print("")

    if not os.path.exists(args.output):
        logging.info('output path [{}] does not exist. creating it now.'.format(args.output))
        os.makedirs(args.output, exist_ok=True)

    clicks = json.loads(args.max_val_interactions)
    for click in clicks:
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logging.info('                CLICKS = {}'.format(click))
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
        trainer = create_validator(args, click)

        start_time = time.time()
        trainer.run()
        end_time = time.time()

        logging.info('Total Run Time {}'.format(end_time - start_time))


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
