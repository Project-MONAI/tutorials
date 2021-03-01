import os

import torch.distributed as dist
from monai.data import (CacheDataset, DataLoader, load_decathlon_datalist,
                        load_decathlon_properties, partition_dataset)

from task_params import task_name
from transforms import get_task_transforms


def get_data(args, batch_size=1, mode="train"):
    # get necessary parameters:
    fold = args.fold
    task_id = args.task_id
    root_dir = args.root_dir
    datalist_path = args.datalist_path
    dataset_path = os.path.join(root_dir, task_name[task_id])
    transform_params = (args.pos_sample_num, args.neg_sample_num, args.num_samples)
    multi_gpu_flag = args.multi_gpu

    transform = get_task_transforms(mode, task_id, *transform_params)
    if mode == "test":
        list_key = "test"
    else:
        list_key = "{}_fold{}".format(mode, fold)
    datalist_name = "dataset_task{}.json".format(task_id)

    property_keys = [
        "name",
        "description",
        "reference",
        "licence",
        "tensorImageSize",
        "modality",
        "labels",
        "numTraining",
        "numTest",
    ]

    datalist = load_decathlon_datalist(
        os.path.join(datalist_path, datalist_name), True, list_key, dataset_path
    )

    properties = load_decathlon_properties(
        os.path.join(datalist_path, datalist_name), property_keys
    )
    if mode in ["validation", "test"]:
        if multi_gpu_flag:
            datalist = partition_dataset(
                data=datalist,
                shuffle=False,
                num_partitions=dist.get_world_size(),
                even_divisible=False,
            )[dist.get_rank()]

        val_ds = CacheDataset(
            data=datalist,
            transform=transform,
            num_workers=4,
        )

        data_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.val_num_workers,
        )
    elif mode == "train":
        if multi_gpu_flag:
            datalist = partition_dataset(
                data=datalist,
                shuffle=True,
                num_partitions=dist.get_world_size(),
                even_divisible=True,
            )[dist.get_rank()]

        train_ds = CacheDataset(
            data=datalist,
            transform=transform,
            num_workers=8,
            cache_rate=args.cache_rate,
        )
        data_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.train_num_workers,
            drop_last=True,
        )
    else:
        raise ValueError(f"mode should be train, validation or test.")

    return properties, data_loader
