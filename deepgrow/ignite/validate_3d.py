import argparse
import distutils.util
import logging
import sys

import validate


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--seed', type=int, default=23)
    parser.add_argument('--dimensions', type=int, default=3)

    parser.add_argument('-n', '--network', default='bunet', choices=['unet', 'bunet'])
    parser.add_argument('-c', '--channels', type=int, default=32)
    parser.add_argument('-f', '--folds', type=int, default=10)

    parser.add_argument('-i', '--input', default='/workspace/data/deepgrow/3D/MSD_Task09_Spleen/dataset.json')
    parser.add_argument('-o', '--output', default='eval3D')
    parser.add_argument('--save_seg', type=strtobool, default='false')
    parser.add_argument('--cache_dir', type=str, default=None)

    parser.add_argument('-g', '--use_gpu', type=strtobool, default='true')
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-t', '--limit', type=int, default=20)
    parser.add_argument('-m', '--model_path', default="output3D/model.pt")
    parser.add_argument('--roi_size', default="[256, 256, 256]")
    parser.add_argument('--model_size', default="[128, 128, 128]")

    parser.add_argument('-iv', '--max_val_interactions', default="[0,1,2,5,10,15]")
    parser.add_argument('--multi_gpu', type=strtobool, default='false')
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    validate.run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
