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

    work_dir = "./work_dir"
    runner = AutoRunner(work_dir=work_dir, input=args.input)
    runner.run()

if __name__ == "__main__":
    main()
