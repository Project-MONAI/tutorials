import argparse
import distutils.util
import json
import logging
import os
import sys

from monai.apps.deepgrow.dataset import create_dataset
from monai.data import partition_dataset


def prepare_datalist(args):
    dimensions = args.dimensions
    dataset_json = os.path.join(args.output, 'dataset.json')
    if not os.path.exists(dataset_json):
        logging.info('Processing dataset...')
        with open(os.path.join(args.dataset_json)) as f:
            datalist = json.load(f)

        datalist = create_dataset(
            datalist=datalist[args.datalist_key],
            base_dir=args.dataset_root,
            output_dir=args.output,
            dimension=dimensions,
            pixdim=[1.0] * dimensions,
            limit=args.limit,
            relative_path=args.relative_path
        )

        with open(dataset_json, 'w') as fp:
            json.dump(datalist, fp, indent=2)
    else:
        logging.info('Pre-load existing dataset.json')

    dataset_json = os.path.join(args.output, 'dataset.json')
    with open(dataset_json) as f:
        datalist = json.load(f)
    logging.info('+++ Dataset File: {}'.format(dataset_json))
    logging.info('+++ Total Records: {}'.format(len(datalist)))
    logging.info('')

    train_ds, val_ds = partition_dataset(datalist, ratios=[args.split, (1 - args.split)], shuffle=True, seed=args.seed)
    dataset_json = os.path.join(args.output, 'dataset-0.json')
    with open(dataset_json, 'w') as fp:
        json.dump({'training': train_ds, 'validation': val_ds}, fp, indent=2)

    logging.info('*** Dataset File: {}'.format(dataset_json))
    logging.info('*** Total Records for Training: {}'.format(len(train_ds)))
    logging.info('*** Total Records for Validation: {}'.format(len(val_ds)))


def run(args):
    for arg in vars(args):
        logging.info('USING:: {} = {}'.format(arg, getattr(args, arg)))
    logging.info("")

    if not os.path.exists(args.output):
        logging.info('output path [{}] does not exist. creating it now.'.format(args.output))
        os.makedirs(args.output, exist_ok=True)
    prepare_datalist(args)


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-dims', '--dimensions', type=int, default=2)

    parser.add_argument('-d', '--dataset_root', default='/workspace/data/52432')
    parser.add_argument('-j', '--dataset_json', default='/workspace/data/52432/dataset.json')
    parser.add_argument('-k', '--datalist_key', default='training')

    parser.add_argument('-o', '--output', default='/workspace/data/52432/2D')
    parser.add_argument('-x', '--split', type=float, default=0.9)
    parser.add_argument('-t', '--limit', type=int, default=0)
    parser.add_argument('-r', '--relative_path', type=strtobool, default='false')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
