# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import distutils.util
import logging
import sys

import train


def strtobool(val):
    return bool(distutils.util.strtobool(val))


if __name__ == "__main__":
    # Single GPU (it will also export)
    # python train_3d.py
    #
    # Multi GPU (run export separate)
    # python -m torch.distributed.launch \
    #   --nproc_per_node=`nvidia-smi -L | wc -l` \
    #   --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1234 \
    #   -m train_3d --multi_gpu true -e 100
    #
    # python train.py --export
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=23)
    parser.add_argument("--dimensions", type=int, default=3)

    parser.add_argument("-n", "--network", default="bunet", choices=["unet", "bunet"])
    parser.add_argument("-c", "--channels", type=int, default=32)
    parser.add_argument(
        "-i",
        "--input",
        default="/workspace/data/deepgrow/3D/MSD_Task09_Spleen/dataset.json",
    )
    parser.add_argument("-o", "--output", default="output3D")

    parser.add_argument("-g", "--use_gpu", type=strtobool, default="true")
    parser.add_argument("-a", "--amp", type=strtobool, default="false")

    parser.add_argument("-e", "--epochs", type=int, default=200)
    parser.add_argument("-b", "--batch", type=int, default=1)
    parser.add_argument("-x", "--split", type=float, default=0.9)
    parser.add_argument("-t", "--limit", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("-r", "--resume", type=strtobool, default="false")
    parser.add_argument("-m", "--model_path", default="output3D/model.pt")
    parser.add_argument("--roi_size", default="[128, 192, 192]")
    parser.add_argument("--model_size", default="[128, 192, 192]")

    parser.add_argument("-f", "--val_freq", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-it", "--max_train_interactions", type=int, default=15)
    parser.add_argument("-iv", "--max_val_interactions", type=int, default=20)

    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--image_interval", type=int, default=5)
    parser.add_argument("--multi_gpu", type=strtobool, default="false")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--export", type=strtobool, default="false")

    args = parser.parse_args()
    train.run(args)
