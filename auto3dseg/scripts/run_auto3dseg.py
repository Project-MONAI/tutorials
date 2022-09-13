#!/usr/bin/env python

import argparse

from monai.apps.auto3dseg import AutoRunner
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

    data_src_cfg = args.input
    cfg = ConfigParser.load_config_file(data_src_cfg)

    work_dir = "./work_dir"
    runner = AutoRunner(work_dir=work_dir, input=cfg)
    runner.run()

if __name__ == "__main__":
    main()
