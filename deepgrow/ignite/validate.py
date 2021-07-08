import argparse
import distutils.util
import json
import logging
import os
import sys
import time

import torch

import train

from monai.apps.deepgrow.interaction import Interaction
from monai.engines import SupervisedEvaluator
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    from_engine,
    MeanDice,
)
from monai.inferers import SimpleInferer
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    ToTensord,
    Activationsd,
    AsDiscreted,
    SaveImaged,
)


def create_validator(args, click):
    set_determinism(seed=args.seed)

    device = torch.device("cuda" if args.use_gpu else "cpu")

    pre_transforms = train.get_pre_transforms(args.roi_size, args.model_size, args.dimensions)
    click_transforms = train.get_click_transforms()

    # define training components
    network = train.get_network(args.network, args.channels, args.dimensions).to(device)

    logging.info('Loading Network...')
    map_location = {"cuda:0": "cuda:{}".format(args.local_rank)}

    checkpoint = torch.load(args.model_path, map_location=map_location)
    network.load_state_dict(checkpoint)
    network.eval()

    # define event-handlers for engine
    _, val_loader = train.get_loaders(args, pre_transforms, train=False)
    fold_size = int(len(val_loader.dataset) / args.batch / args.folds) if args.folds else 0
    logging.info('Using Fold-Size: {}'.format(fold_size))

    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=args.output, output_transform=lambda x: None),
    ]

    post_transform_list = [
        ToTensord(keys='pred'),
        Activationsd(keys='pred', sigmoid=True),
        AsDiscreted(keys='pred', threshold_values=True, logit_thresh=0.5)
    ]

    if args.save_seg:
        post_transform_list.append(
            SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=os.path.join(args.output, f'clicks_{click}_images'))
        )

    post_transform = Compose(post_transform_list)

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        iteration_update=Interaction(
            transforms=click_transforms,
            max_interactions=click,
            train=False),
        inferer=SimpleInferer(),
        postprocessing=post_transform,
        val_handlers=val_handlers,
        key_val_metric={
            f'clicks_{click}_val_dice': MeanDice(
                include_background=False,
                output_transform=from_engine(["pred", "label"]),
            )
        }
    )
    return evaluator


def run(args):
    args.roi_size = json.loads(args.roi_size)
    args.model_size = json.loads(args.model_size)

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
        evaluator = create_validator(args, click)

        start_time = time.time()
        evaluator.run()
        end_time = time.time()

        logging.info('Total Run Time {}'.format(end_time - start_time))


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--seed', type=int, default=23)
    parser.add_argument('--dimensions', type=int, default=2)

    parser.add_argument('-n', '--network', default='bunet', choices=['unet', 'bunet'])
    parser.add_argument('-c', '--channels', type=int, default=32)
    parser.add_argument('-f', '--folds', type=int, default=10)

    parser.add_argument('-i', '--input', default='/workspace/data/deepgrow/2D/MSD_Task09_Spleen/dataset.json')
    parser.add_argument('-o', '--output', default='eval')
    parser.add_argument('--save_seg', type=strtobool, default='false')
    parser.add_argument('--cache_dir', type=str, default=None)

    parser.add_argument('-g', '--use_gpu', type=strtobool, default='true')
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-t', '--limit', type=int, default=20)
    parser.add_argument('-m', '--model_path', default="output/model.pt")
    parser.add_argument('--roi_size', default="[256, 256]")
    parser.add_argument('--model_size', default="[256, 256]")

    parser.add_argument('-iv', '--max_val_interactions', default="[0,1,2,5,10,15]")
    parser.add_argument('--multi_gpu', type=strtobool, default='false')
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
