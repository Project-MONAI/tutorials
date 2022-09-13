#!/usr/bin/env python

import argparse
import nibabel as nib
import numpy as np
import os
import sys
import tempfile
import torch
import unittest

from monai.apps.auto3dseg import (
    BundleGen,
    DataAnalyzer,
    AlgoEnsembleBestN,
    AlgoEnsembleBuilder,
)
from monai.apps.auto3dseg.ensemble_builder import AlgoEnsembleKeys
from monai.bundle.config_parser import ConfigParser


def main():
    parser = argparse.ArgumentParser(description="auto3dseg")
    parser.add_argument(
        "--input",
        type=str,
        default="./task.yaml",
        help="input information",
    )
    args = parser.parse_args()

    ## preparation
    test_path = "./"
    data_src_cfg = args.input
    cfg = ConfigParser.load_config_file(data_src_cfg)
    dataroot = cfg["dataroot"]
    datalist_filename = cfg["datalist"]
    datalist = ConfigParser.load_config_file(datalist_filename)

    work_dir = os.path.join(test_path, "workdir")
    da_output_yaml = os.path.join(work_dir, "datastats.yaml")

    if not os.path.isdir(dataroot):
        os.makedirs(dataroot)

    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)

    ## data analysis
    da = DataAnalyzer(datalist, dataroot, output_path=da_output_yaml)
    da.get_all_case_stats()

    ## algorithm generation
    bundle_generator = BundleGen(
        algo_path=work_dir,
        data_stats_filename=da_output_yaml,
        data_src_cfg_name=data_src_cfg,
    )

    bundle_generator.generate(work_dir, num_fold=5)
    history = bundle_generator.get_history()

    ## model training
    gpus = [_i for _i in range(torch.cuda.device_count())]

    train_param = {
        "CUDA_VISIBLE_DEVICES": gpus,
    }

    for i, record in enumerate(history):
        for name, algo in record.items():
            algo.train(train_param)

    ## model ensemble
    n_best = 1
    builder = AlgoEnsembleBuilder(history, data_src_cfg)
    builder.set_ensemble_method(AlgoEnsembleBestN(n_best=n_best))
    ensemble = builder.get_ensemble()
    pred = ensemble()
    print("ensemble picked the following best {0:d}:".format(n_best))
    for algo in ensemble.get_algo_ensemble():
        print(algo[AlgoEnsembleKeys.ID])


if __name__ == "__main__":
    main()
