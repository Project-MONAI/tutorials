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

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from monai.data import (
    Dataset,
    DatasetSummary,
    load_decathlon_datalist,
    load_decathlon_properties,
)
from monai.transforms import LoadImaged

from task_params import task_name


def get_task_params(args):
    """
    This function is used to achieve the spacings of decathlon dataset.
    In addition, for CT images (task 03, 06, 07, 08, 09 and 10), this function
    also prints the mean and std values (used for normalization), and the min (0.5 percentile)
    and max(99.5 percentile) values (used for clip).

    """
    task_id = args.task_id
    root_dir = args.root_dir
    datalist_path = args.datalist_path
    dataset_path = os.path.join(root_dir, task_name[task_id])
    datalist_name = "dataset_task{}.json".format(task_id)

    # get all training data
    datalist = load_decathlon_datalist(
        os.path.join(datalist_path, datalist_name), True, "training", dataset_path
    )

    # get modality info.
    properties = load_decathlon_properties(
        os.path.join(datalist_path, datalist_name), "modality"
    )

    dataset = Dataset(
        data=datalist,
        transform=LoadImaged(keys=["image", "label"]),
    )

    calculator = DatasetSummary(dataset, num_workers=4)
    target_spacing = calculator.get_target_spacing()
    print("spacing: ", target_spacing)
    if properties["modality"]["0"] == "CT":
        print("CT input, calculate statistics:")
        calculator.calculate_statistics()
        print("mean: ", calculator.data_mean, " std: ", calculator.data_std)
        calculator.calculate_percentiles(
            sampling_flag=True, interval=10, min_percentile=0.5, max_percentile=99.5
        )
        print(
            "min: ",
            calculator.data_min_percentile,
            " max: ",
            calculator.data_max_percentile,
        )
    else:
        print("non CT input, skip calculating.")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
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
        "-datalist_path",
        "--datalist_path",
        type=str,
        default="config/",
    )

    args = parser.parse_args()
    get_task_params(args)
